import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Example dataset of product reviews with categories
data = {
    'review': [
        'This phone has a great camera and battery life.',
        'The shoes are comfortable and stylish.',
        'The laptop is fast and reliable, perfect for gaming.',
        'This t-shirt shrinks after one wash, very disappointed.',
        'The food processor is powerful and easy to clean.'
    ],
    'category': [
        'electronics', 'clothing', 'electronics', 'clothing', 'appliances'
    ]
}

df = pd.DataFrame(data)

# Preprocess the text
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X_tfidf = tfidf_vectorizer.fit_transform(df['review'])
y = df['category']

# Train the Logistic Regression classifier
log_reg_classifier = LogisticRegression(max_iter=1000)
log_reg_classifier.fit(X_tfidf, y)

# Save the model and vectorizer
joblib.dump(log_reg_classifier, 'log_reg_classifier.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
