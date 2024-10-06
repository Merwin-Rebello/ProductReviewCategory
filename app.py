from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the model and vectorizer
log_reg_classifier = joblib.load('log_reg_classifier.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    review_tfidf = tfidf_vectorizer.transform([review])
    predicted_category = log_reg_classifier.predict(review_tfidf)
    return render_template('index.html', review=review, category=predicted_category[0])

if __name__ == '__main__':
    app.run(debug=True)
