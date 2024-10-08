from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the SVM model and vectorizer
svm_classifier = joblib.load('svm_classifier_updated.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer_updated.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    review_tfidf = tfidf_vectorizer.transform([review])
    predicted_category = svm_classifier.predict(review_tfidf)  # Use the SVM classifier
    return render_template('index.html', review=review, category=predicted_category[0])

if __name__ == '__main__':
    app.run(debug=True)
