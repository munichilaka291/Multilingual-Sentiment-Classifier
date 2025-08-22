
# 📝 Sentiment Analysis with DistilBERT  
_A Transformer-based Multilingual Sentiment Classifier_  

---

## 📌 Project Overview  
This project implements a **sentiment classification system** using **DistilBERT**, a lightweight version of BERT.  
The goal is to classify customer reviews as **Positive 😀 (1)** or **Negative 😡 (0)**.  

The dataset contains product reviews across multiple categories (Books, Tablets, Fruits, Clothes, Shampoo, Hotel, Computers, Dairy, Water Heater, Mobile Phones).  

Additionally, the model supports **multilingual sentiment analysis** (Tamil, Kannada, Malayalam, Odia, Hindi, etc.) by translating text into English before classification.  

---

## ⚙️ Tech Stack / Libraries Used  
- **Model:** DistilBERT (`DistilBertForSequenceClassification`)  
- **Frameworks:** Transformers, PyTorch, Hugging Face Datasets  
- **Data Handling:** pandas, numpy  
- **Visualization (EDA):** matplotlib, seaborn  
- **Evaluation Metrics:** scikit-learn  
- **Experiment Tracking:** Weights & Biases (wandb)  
- **Optional (Multilingual Translation):** googletrans / MarianMT  

---

## 📂 Dataset  
- Source: E-commerce product reviews dataset (CSV format).  
- Columns:  
  - `cat_x`, `cat_y` → Original category (Chinese labels)  
  - `category` → Mapped English category (Books, Fruits, Clothes, etc.)  
  - `review_en` → Review in English  
  - `label` → Sentiment (0 = Negative, 1 = Positive)  

- Dataset Size: ~62,000 reviews  

---

## 🔍 Exploratory Data Analysis (EDA)  
1. **Sentiment Distribution:** Balanced dataset → ~31k Positive, ~31k Negative  
2. **Category Distribution:** Top categories include Clothes, Fruits, Tablets, Shampoo, Hotel  
3. **Review Length Analysis:** Reviews mostly 10–100 words  

---

## 🚀 Training Process  
1. Tokenization with `DistilBertTokenizerFast`  
2. Fine-tuned DistilBERT on dataset  
3. Optimized with Hugging Face `Trainer` API  
4. Saved trained model → `/content/sentiment_model_fallback`  

---

## 📊 Model Evaluation  
- ✅ Accuracy: ~93%  
- ✅ Precision: ~93%  
- ✅ Recall: ~92%  
- ✅ F1 Score: ~92%  

---

## 🌍 Multilingual Support  
Supports sentiment classification in: Tamil, Kannada, Malayalam, Odia, Hindi, and English.  

Example:  
```python
reviews = [
    "இந்த தயாரிப்பு மிகவும் சிறந்தது" ,   # Tamil
    "ಈ ಉತ್ಪನ್ನ ನನಗೆ ಇಷ್ಟವಾಗಲಿಲ್ಲ" ,       # Kannada
    "बहुत खराब सेवा है" ,                 # Hindi
    "Great product, I really liked it!"    # English
]
```

Output:  
- Tamil → Positive 😀  
- Kannada → Negative 😡  
- Hindi → Negative 😡  
- English → Positive 😀  

---

## 📌 How to Run  

### 1. Install Dependencies  
```bash
pip install transformers datasets torch scikit-learn matplotlib seaborn googletrans==4.0.0-rc1
```

### 2. Train Model (Colab/Jupyter)  
```python
from transformers import Trainer, TrainingArguments
# (see training notebook for full details)
```

### 3. Save Model  
```python
trainer.save_model("/content/sentiment_model_fallback")
```

### 4. Test Custom Reviews  
```python
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

model = DistilBertForSequenceClassification.from_pretrained("/content/sentiment_model_fallback")
tokenizer = DistilBertTokenizerFast.from_pretrained("/content/sentiment_model_fallback")

def predict_sentiment(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    preds = torch.argmax(logits, dim=-1).tolist()
    return ["Positive 😀" if p==1 else "Negative 😡" for p in preds]

print(predict_sentiment(["Worst service ever!", "This product is amazing!"]))
```

---

## 📌 Results & Insights  
- DistilBERT achieved **high accuracy** with **fast inference**.  
- Dataset was well-balanced across sentiments.  
- Adding **category-based analysis** gives insights into which products get more negative reviews.  
- Multilingual support ensures the model is useful in **real-world e-commerce platforms** with diverse customers.  

---

## 📌 Future Work  
- Deploy as a REST API (FastAPI / Flask)  
- Integrate into e-commerce feedback systems  
- Expand multilingual support with **native fine-tuning**  
- Add image + text multimodal sentiment analysis  

---

## 📌 Author  
👤 **Chilaka Bala Muneendra**  
📧 [munichilaka291@gmail.com]  
🌐 [https://github.com/munichilaka291]  
