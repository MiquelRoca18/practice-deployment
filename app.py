from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Cargar modelos en modo binario
with open("models/cv.pkl", "rb") as f:
    cv = pickle.load(f)

with open("models/clf.pkl", "rb") as f:
    clf = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction_text = None
    email = ""
    
    if request.method == "POST":
        email = request.form.get('content', '').strip()
        
        if not email:
            prediction_text = "Please enter some text."
        else:
            tokenized_email = cv.transform([email])
            prediction = clf.predict(tokenized_email)[0]
            
            if prediction == 1:
                prediction_text = "Spam"
            else:
                prediction_text = "Not Spam"
    
    return render_template("index.html", prediction=prediction_text, email=email)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)