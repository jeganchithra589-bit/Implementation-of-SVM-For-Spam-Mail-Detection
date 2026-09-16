# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

# Algorithm: SVM for Spam Mail Detection
1. Start
2. Load the spam email dataset containing email messages and their labels.
3. Preprocess the email text by removing unnecessary characters and converting the text into a suitable format.
4. Split the dataset into training and testing data.
5. Use CountVectorizer or TF-IDF Vectorizer to convert email text into numerical feature vectors.
6. Create an SVM classifier (SVC).
7. Train the SVM model using the training email features and their corresponding labels.
8. Use the trained model to predict whether the test emails are Spam or Not Spam (Ham).
9. Evaluate the model using metrics such as accuracy, precision, recall, and F1-score.
10. Display the classification results.
11. Stop
    
## Program:

Program to implement the SVM For Spam Mail Detection..
# Developed by: Jegan P
# RegisterNumber:  212225240061


```
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin1")

# Select required columns
df = df[["v1", "v2"]].dropna()

# Input and output
X = df["v2"]
y = df["v1"]

# Convert text into numerical features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train SVM model
model = SVC(kernel="linear")
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

# Predict new message
new_message = input("\nEnter a new message: ")

# Convert new message using the same TF-IDF vectorizer
new_message_tfidf = vectorizer.transform([new_message])

# Prediction
prediction = model.predict(new_message_tfidf)

print("Prediction:", prediction[0])
```
## Output:

<img width="827" height="348" alt="image" src="https://github.com/user-attachments/assets/99aa2a1a-b65e-4c29-a21f-e1a96d8edc8f" />


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
