from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import os

# Load model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

app = FastAPI(title="Email Spam Classifier API")

class EmailRequest(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Email Spam Classifier API is running."}

@app.post("/predict")
def predict_email(request: EmailRequest):
    text_tfidf = vectorizer.transform([request.text])
    
    prediction = model.predict(text_tfidf)[0]
    probabilities = model.predict_proba(text_tfidf)[0]
    
    spam_probability = float(probabilities[1])
    ham_probability = float(probabilities[0])
    
    result = "SPAM" if prediction == 1 else "HAM"
    
    return {
        "prediction": result,
        "spam_probability": round(spam_probability, 4),
        "ham_probability": round(ham_probability, 4),
        "confidence": round(max(spam_probability, ham_probability), 4)
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)