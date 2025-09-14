import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

# Load your fabricated dataset
df = pd.read_csv("fabricated_irrigation_data.csv")

# Features and target
feature_cols = [
    "soil_moisture", "soil_temp", "soil_ph", "tank_level",
    "ambient_humidity", "ambient_temp", "rain_next_48h"
]
X = df[feature_cols]
y = df["irrigate"]

# Split data for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model
with open("irrigation_model.pkl", "wb") as f:
    pickle.dump(clf, f)