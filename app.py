from flask import Flask, request, jsonify
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import io

app = Flask(__name__)

CHECKPOINT_PATH = "accident_classifier_checkpoint.pth"
IMAGE_SIZE      = 224
THRESHOLD       = 0.5
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def load_model(checkpoint_path):
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
    print(f"✅ Model loaded on {DEVICE}")
    return model, checkpoint.get("idx_to_class", {0: "Accident", 1: "Non Accident"})

model, idx_to_class = load_model(CHECKPOINT_PATH)

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def predict(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)
        accident_prob = probs[0, 0].item()
    prediction = 1 if accident_prob >= THRESHOLD else 0
    return prediction, accident_prob

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
        prediction, accident_prob = predict(file.read())
        return jsonify({
            "prediction": prediction,
            "label": idx_to_class[prediction],
            "accident_probability": round(accident_prob, 4),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)