import pickle

# Load model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

def predict_email(text):
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)[0]
    return "SPAM" if prediction == 1 else "HAM"

if __name__ == "__main__":
    user_input = input("Enter email text: ")
    result = predict_email(user_input)
    print("Prediction:", result)