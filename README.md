# Email Spam Classifier API

Production-ready machine learning API for detecting spam messages using TF-IDF feature extraction and Naive Bayes classification.

## Model Performance
- Accuracy: 97.8%
- Precision (Spam): 1.00
- Recall (Spam): 0.84

## Tech Stack
- Python
- Scikit-learn
- FastAPI
- Uvicorn
- Deployed on Render

## API Endpoint

POST /predict

Example request:
{
  "text": "Congratulations! You won $1000. Click here now!"
}

Example response:
{
  "prediction": "SPAM",
  "spam_probability": 0.9821,
  "ham_probability": 0.0179,
  "confidence": 0.9821
}

## Live API
[https://email-spam-classifier-api.onrender.com/docs]
