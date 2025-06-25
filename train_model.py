import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os
import sys
sys.path.append(os.path.abspath('../training'))  # adjust path if needed

from data_preprocessing import preprocess_data


# Load the dataset
df = pd.read_csv('data/PS_20174392719_1491204439457_log.csv')

# Drop unnecessary columns
df.drop(['nameOrig', 'nameDest'], axis=1, inplace=True)

# Encode the 'type' column
df['type'] = df['type'].map({
    'CASH_OUT': 0,
    'TRANSFER': 1,
    'PAYMENT': 2,
    'DEBIT': 3,
    'CASH_IN': 4
})

# Split into features and label
X = df.drop('isFraud', axis=1)
y = df['isFraud']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Decision Tree Classifier
dtc = DecisionTreeClassifier(random_state=42)
dtc.fit(X_train, y_train)

# Evaluate model
y_pred = dtc.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("📊 Classification Report:\n", classification_report(y_test, y_pred))

# Save model to flask folder
model_path = 'flask/payments.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(dtc, f)

print(f"✅ Model saved to {model_path}")
