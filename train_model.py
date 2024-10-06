import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import joblib
import nltk
from nltk.corpus import stopwords
import string

# Ensure NLTK stopwords are downloaded
nltk.download('stopwords')

# Expanded dataset of product reviews with categories
data = {
    'review': [
        # Electronics
        "The battery life on this smartphone lasts all day, even with heavy use!",
        "This tablet is lightweight and perfect for reading books and browsing the web.",
        "The sound quality of these headphones is exceptional; I can hear every detail in my music.",
        "I love how fast this laptop boots up; it's perfect for my work needs.",
        "The smart TV has a user-friendly interface and offers a great selection of streaming apps.",
        
        # Clothing
        "These jeans fit perfectly and are incredibly comfortable for all-day wear.",
        "The dress is made from high-quality fabric and looks stunning at parties.",
        "I bought a pack of socks, and they are so soft and warm for winter!",
        "The jacket is stylish and keeps me warm without being too heavy.",
        "This hoodie has a great design, but it shrank a bit after the first wash.",
        
        # Appliances
        "The coffee maker brews the best coffee I've ever had; it’s a game changer in my mornings.",
        "This vacuum cleaner picks up pet hair effortlessly and is easy to maneuver.",
        "The blender is powerful and makes smoothies in seconds; I use it every day.",
        "I love the air fryer! It cooks food quickly and makes it crispy without oil.",
        "The washing machine is quiet and efficient, getting my clothes perfectly clean every time.",
        
        # Food
        "These protein bars are delicious and keep me full between meals.",
        "I tried the new pasta sauce, and it tastes just like homemade!",
        "The snacks in this variety pack are all amazing, and they cater to different tastes.",
        "I ordered the organic granola, and it’s packed with flavor and crunch!",
        "These frozen meals are a lifesaver; they taste great and are super convenient.",
        
        # Beauty and Personal Care
        "This moisturizer makes my skin feel so soft and hydrated.",
        "The lipstick has great pigmentation and lasts all day without fading.",
        "I love this shampoo; it leaves my hair shiny and manageable.",
        "This face mask is relaxing and makes my skin glow.",
        "The perfume has a lovely scent that isn't too overpowering.",
        
        # Books
        "This novel kept me hooked from start to finish; I couldn't put it down!",
        "The recipe book is well-organized and has easy-to-follow instructions.",
        "I learned so much from this self-help book; it's a must-read for everyone.",
        "The children's book is beautifully illustrated and captures their imagination.",
        "This biography provides an insightful look into the life of a remarkable person."
    ],
    'category': [
        # Corresponding categories
        'electronics', 'electronics', 'electronics', 'electronics', 'electronics',
        'clothing', 'clothing', 'clothing', 'clothing', 'clothing',
        'appliances', 'appliances', 'appliances', 'appliances', 'appliances',
        'food', 'food', 'food', 'food', 'food',
        'beauty and personal care', 'beauty and personal care', 'beauty and personal care', 'beauty and personal care', 'beauty and personal care',
        'books', 'books', 'books', 'books', 'books'
    ]
}

# Create a DataFrame from the initial data
df = pd.DataFrame(data)

# New appliance-related reviews to be added
new_reviews = [
    "This oven heats up quickly and bakes food evenly.",
    "The microwave has multiple settings and cooks food thoroughly.",
    "I love the precision of this food thermometer; it's essential for cooking.",
    "The blender has strong blades that make smoothies in seconds.",
    "This dishwasher is energy-efficient and cleans dishes thoroughly."
]

new_categories = ['appliances', 'appliances', 'appliances', 'appliances', 'appliances']

# Add the new data to the original dataset
df_new = pd.DataFrame({'review': new_reviews, 'category': new_categories})
df_updated = pd.concat([df, df_new], ignore_index=True)

# Preprocessing function to clean and lemmatize text
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    words = [word for word in text.split() if word not in stop_words]
    return " ".join(words)

# Apply preprocessing to all reviews
df_updated['cleaned_review'] = df_updated['review'].apply(preprocess_text)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_updated['cleaned_review'], df_updated['category'], test_size=0.2, random_state=42)

# Preprocess the updated text data using n-grams (unigrams + bigrams)
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.85, min_df=2)
X_tfidf_train = tfidf_vectorizer.fit_transform(X_train)
X_tfidf_test = tfidf_vectorizer.transform(X_test)

# Train SVM classifier with hyperparameter tuning
svm_classifier = SVC()

# Perform hyperparameter tuning using GridSearchCV
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': [1, 0.1, 0.01]
}

grid_search = GridSearchCV(svm_classifier, param_grid, cv=3)
grid_search.fit(X_tfidf_train, y_train)

# Get the best estimator after tuning
best_svm_classifier = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_svm_classifier.predict(X_tfidf_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)

# Save the retrained model and vectorizer
joblib.dump(best_svm_classifier, 'svm_classifier_updated.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer_updated.pkl')

# Print the accuracy of the model
print(f"Model retrained with SVM and saved with new appliance data! Accuracy on test set: {accuracy:.2f}")