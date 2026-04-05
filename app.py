from flask import Flask, request, jsonify
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import io
import os

try:
    from transformers import (
        AutoImageProcessor,
        AutoModelForImageClassification,
        pipeline,
    )
except ImportError:
    AutoImageProcessor = None
    AutoModelForImageClassification = None
    pipeline = None

app = Flask(__name__)

CHECKPOINT_PATH = "accident_classifier_checkpoint.pth"
AI_DET_DIR      = os.getenv("AI_DET_DIR", "ai_detector")
IMAGE_SIZE      = 224
THRESHOLD       = 0.5
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HF_DEVICE       = 0 if torch.cuda.is_available() else -1

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def load_model_1(checkpoint_path):
    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(1280, 512),       # ← extra hidden layer
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, 2),          # ← output layer
    )
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    print(f"Model 1 loaded on {DEVICE}")
    return model, checkpoint.get("idx_to_class", {0: "Accident", 1: "Non Accident"})

def load_fraud_model(ai_det_dir):
    if pipeline is None:
        print("'transformers' is not installed. AI-generated check is disabled.")
        return None
    if not os.path.isdir(ai_det_dir):
        print(f"AI detector directory not found: {ai_det_dir}. AI-generated check is disabled.")
        return None

    ai_model = AutoModelForImageClassification.from_pretrained(ai_det_dir, local_files_only=True)
    ai_processor = AutoImageProcessor.from_pretrained(ai_det_dir, local_files_only=True)

    fraud_clf = pipeline(
        "image-classification",
        model=ai_model,
        image_processor=ai_processor,
        device=HF_DEVICE,
    )
    return fraud_clf

def load_pipelines():
    domain_clf, idx_map = load_model_1(CHECKPOINT_PATH)
    fraud_clf = load_fraud_model(AI_DET_DIR)

    if fraud_clf is not None:
        print("Both pipelines loaded from local disk (offline mode).")
    else:
        print("Accident pipeline loaded. AI-generated detector is unavailable.")

    return domain_clf, idx_map, fraud_clf

model, idx_to_class, fraud_clf = load_pipelines()

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def predict_model_1(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)
        accident_prob = probs[0, 0].item()
    prediction = 0 if accident_prob >= THRESHOLD else 1
    return prediction, accident_prob

def predict_ai_generated(image_bytes):
    if fraud_clf is None:
        return None

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    scores = fraud_clf(img)
    if not scores:
        return None

    top = max(scores, key=lambda x: float(x.get("score", 0.0)))
    top_label = str(top.get("label", "unknown"))
    top_score = float(top.get("score", 0.0))

    label_lower = top_label.lower()
    if "real" in label_lower or "human" in label_lower:
        is_ai_generated = False
    elif "fake" in label_lower or "ai" in label_lower or "synthetic" in label_lower:
        is_ai_generated = True
    else:
        is_ai_generated = top_score >= 0.5

    return {
        "ai_generated_prediction": int(is_ai_generated),
        "ai_generated_label": top_label,
        "ai_generated_score": round(top_score, 4),
    }

def get_model_1_label(prediction):
    if prediction in idx_to_class:
        return idx_to_class[prediction]
    if str(prediction) in idx_to_class:
        return idx_to_class[str(prediction)]
    return "Accident" if prediction == 0 else "Non Accident"

@app.route("/predict", methods=["POST"])
def predict_route():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Use key 'image'"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        return jsonify({"error": "Only JPG/PNG images are supported"}), 400
    try:
        image_bytes = file.read()
        prediction, accident_prob = predict_model_1(image_bytes)

        response = {
            "prediction": prediction,
            "label": get_model_1_label(prediction),
            "accident_probability": round(accident_prob, 4),
            "is_accident_model_1": bool(prediction == 0),
        }

        # Run the AI-generated detector only when model 1 says "accident" (prediction == 0).
        if prediction == 0:
            fraud_result = predict_ai_generated(image_bytes)
            if fraud_result is None:
                response["ai_generated_check"] = "unavailable"
            else:
                response.update(fraud_result)

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)