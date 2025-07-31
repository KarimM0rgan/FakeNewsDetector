import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import *
from sklearn.metrics import *
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# dataset = https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

# Download files locally and Load datasets
fake_news = pd.read_csv("E:/Sewanee/Data Analysis/.py/Fake News Detector/Fake.csv")
legit_news = pd.read_csv("E:/Sewanee/Data Analysis/.py/Fake News Detector/True.csv")

# Label datasets. 1 for legit, and 0 for fake
fake_news['class'] = 0
legit_news['class'] = 1

# Combine & shuffle
df = pd.concat([legit_news, fake_news]).sample(frac=1)

# Clean text
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)       # Remove numbers
    text = text.lower()                   # Lowercase
    return text

# Combine 'title' and 'text' (body) columns into one
df['combined_text'] = df['title'] + ' ' + df['text']  # Store original combined text
df['cleaned_text'] = df['combined_text'].apply(clean_text)

# dataset train-test split to train the model
X_full = df[['cleaned_text', 'combined_text']]  # Keep both text versions
y = df['class']

X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_full,
    y,
    test_size=0.2,
    random_state=42
)

# Separate cleaned text for vectorization
X_train_clean = X_train_full['cleaned_text']
X_test_clean = X_test_full['cleaned_text']

# Initialize TF-IDF "translator of words into numbers for program to understand"
tfidf = TfidfVectorizer(
    max_features = 5000,
    stop_words=nltk.corpus.stopwords.words('english')
)

# Transform training text into Machine-readable numbers
X_train_tfidf = tfidf.fit_transform(X_train_clean)
X_test_tfidf = tfidf.transform(X_test_clean)

# Get the word-to-ID mapping
vocabulary = tfidf.vocabulary_

# Initialize the model, where we teach the program how to differentiate between fake and legit news
model = PassiveAggressiveClassifier(max_iter = 100)  # 100 learning passes

# Teach it using training data (numbers from TF-IDF + class)
model.fit(X_train_tfidf, y_train)  # X = word patterns, y = FAKE(0)/REAL(1)

# Test its knowledge and get predictions
y_pred = model.predict(X_test_tfidf)  # Predict on unseen data

# Create a DataFrame for test results
results_df = pd.DataFrame({
    'original_text': X_test_full['combined_text'].values,  # Use values from test split
    'true_label': y_test.values,
    'predicted_label': y_pred
})

# Add a column indicating if the prediction was correct
results_df['is_correct'] = (results_df['true_label'] == results_df['predicted_label'])

# Identify false negatives (actual fake news predicted as real) and false positives (actual real news predicted as fake)
false_negatives = results_df[
    (results_df['true_label'] == 0) & 
    (results_df['predicted_label'] == 1)
]

false_positives = results_df[
    (results_df['true_label'] == 1) & 
    (results_df['predicted_label'] == 0)
]

# Create a DataFrame specifically for misclassified fake news
misclassified_fake_news = false_negatives.copy()
misclassified_legit_news = false_positives.copy()

# Add the confusion matrix position information
misclassified_fake_news['error_type'] = 'False Negative (Fake predicted as Legit)'
misclassified_legit_news['error_type'] = 'False Positive (Legit predicted as Fake)'

# Save to CSV
all_errors = pd.concat([misclassified_legit_news, misclassified_fake_news])
all_errors.to_csv('misclassified_news.csv', index=False)

# Grading accuracy of model-generated results and store confusion matrix
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Visualization
plt.pie([len(false_positives), len(false_negatives)], labels=['False Positives', 'False Negatives'], autopct='%1.1f%%')
plt.title('Error Type Distribution')
plt.savefig('Misclassified News Chart.png', dpi=300)

# Output

### How to read Confusion Matrix: ###
# Predicted REAL  Predicted FAKE
# Actual REAL [[4687           20]   → 4687 correct (REAL), 20 false (FAKE)
# Actual FAKE  [23           4250]]  → 4250 correct (FAKE), 23 false (REAL)

print("Confusion Matrix:")
print(cm)
print(f"Accuracy: {accuracy*100:.2f}%")
print(f"Total misclassified news: {len(misclassified_fake_news) + len(misclassified_legit_news)}")
print("\nSample misclassified news:")
print(all_errors[['original_text', 'true_label', 'predicted_label']].head(3))