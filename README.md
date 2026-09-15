# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Program:

Program to implement the SVM For Spam Mail Detection..
# Developed by: Jegan P
# RegisterNumber:  212225240061

# program:

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
