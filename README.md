# Testing
Testing

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the training data from an Excel file
excel_file_path = 'training_data.xlsx'  # Replace with your Excel file path
df = pd.read_excel(excel_file_path)

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(df['query'], df['response'], test_size=0.2, random_state=42)

# Create a text classification pipeline
text_clf = Pipeline([
    ('tfidf', TfidfVectorizer()),  # Convert text to TF-IDF features
    ('clf', MultinomialNB()),     # Use Multinomial Naive Bayes classifier
])

# Train the model on the training data
text_clf.fit(X_train, y_train)

# Function to predict responses with a default response
def predict_response_with_default(query):
    predicted_response = text_clf.predict([query])

    if len(predicted_response) > 0:
        return predicted_response[0]
    else:
        return "XYZ will check and get back to you"

# Predict responses on the test data
y_pred = [predict_response_with_default(query) for query in X_test]

# Evaluate the model's performance (optional)
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report_str)

# Save the trained model (optional)
import joblib
joblib.dump(text_clf, 'response_prediction_model.pkl')
