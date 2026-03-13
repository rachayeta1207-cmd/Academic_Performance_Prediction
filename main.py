import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("dataset/students.csv")

# Encode performance
encoder = LabelEncoder()
data["Performance"] = encoder.fit_transform(data["Performance"])

# Split features and target
X = data.drop("Performance", axis=1)
y = data["Performance"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model trained and saved as model.pkl")