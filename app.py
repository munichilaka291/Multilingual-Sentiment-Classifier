from flask import Flask, render_template, request
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pytesseract
from PIL import Image

app = Flask(__name__)

# 🔹 Load your custom trained model (binary 0 = Neg, 1 = Pos)
MODEL_PATH = r"sentiment_model\content\sentiment_model_fallback"  # ✅ Use raw string for Windows paths
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# 🔹 Set Tesseract OCR Path (update if installed in another location)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Function: Predict Positive or Negative
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    return "Positive" if pred == 1 else "Negative"

# Home route
@app.route("/")
def home():
    return render_template("index.html")

# Route for text-based sentiment
@app.route("/predict_text", methods=["POST"])
def predict_text():
    review = request.form["review"].strip()
    if not review:
        return render_template("index.html", error="⚠️ Please enter text for sentiment analysis")
    prediction = predict_sentiment(review)
    return render_template("index.html", text_review=review, prediction=prediction, mode="Text")

# Route for image-based sentiment
@app.route("/predict_image", methods=["POST"])
def predict_image():
    file = request.files["image"]

    if not file:
        return render_template("index.html", error="⚠️ Please upload an image")

    image = Image.open(file.stream)

    # Extract text using OCR
    extracted_text = pytesseract.image_to_string(image).strip()
    if not extracted_text:
        return render_template("index.html", error="⚠️ No text detected in image")

    prediction = predict_sentiment(extracted_text)
    return render_template("index.html", image_text=extracted_text, prediction=prediction, mode="Image")

if __name__ == "__main__":
    app.run(debug=True)
