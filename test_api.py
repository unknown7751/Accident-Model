import requests
import os
import sys
from pathlib import Path

BASE_URL = "http://localhost:5000"

def print_result(test_name, response):
    print(f"\n{'='*50}")
    print(f"🧪 Test: {test_name}")
    print(f"   Status Code : {response.status_code}")
    try:
        data = response.json()
        print(f"   Response    : {data}")
        if "prediction" in data:
            label = data.get("label", "N/A")
            prob  = data.get("accident_probability", "N/A")
            pred  = data.get("prediction")
            icon  = "🚨" if pred == 1 else "✅"
            print(f"   Result      : {icon} {label} (accident_prob={prob})")
    except Exception:
        print(f"   Raw Response: {response.text}")
    print(f"{'='*50}")

def test_health():
    response = requests.get(f"{BASE_URL}/health")
    print_result("Health Check", response)
    assert response.status_code == 200
    print("✅ Health check passed!")

def test_no_image():
    response = requests.post(f"{BASE_URL}/predict")
    print_result("No Image Provided", response)
    assert response.status_code == 400
    print("✅ No image test passed!")

def test_wrong_file_type():
    fake_file = ("test.txt", b"this is not an image", "text/plain")
    response = requests.post(f"{BASE_URL}/predict", files={"image": fake_file})
    print_result("Wrong File Type", response)
    assert response.status_code == 400
    print("✅ Wrong file type test passed!")

def test_valid_image(image_path):
    if not os.path.exists(image_path):
        print(f"\n⚠️  Skipping — file not found: {image_path}")
        return
    with open(image_path, "rb") as f:
        response = requests.post(
            f"{BASE_URL}/predict",
            files={"image": (Path(image_path).name, f, "image/jpeg")}
        )
    print_result(f"Valid Image ({Path(image_path).name})", response)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in [0, 1]
    print("✅ Valid image test passed!")

def test_corrupt_image():
    fake_image = ("corrupt.jpg", b"not_real_image_bytes_123", "image/jpeg")
    response = requests.post(f"{BASE_URL}/predict", files={"image": fake_image})
    print_result("Corrupt Image", response)
    assert response.status_code == 500
    print("✅ Corrupt image test passed!")

if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "test_image.jpg"

    print("\n🚀 Starting API Tests...")
    print(f"   API URL    : {BASE_URL}")
    print(f"   Test Image : {image_path}")

    try:
        test_health()
        test_no_image()
        test_wrong_file_type()
        test_valid_image(image_path)
        test_corrupt_image()

        print("\n" + "🎉" * 20)
        print("✅ ALL TESTS PASSED!")
        print("🎉" * 20)

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("\n❌ Could not connect to API. Make sure app.py is running!")
        print("   Run: python app.py")
        sys.exit(1)