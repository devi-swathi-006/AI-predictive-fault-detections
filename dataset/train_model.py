import pandas as pd

# Load dataset
data = pd.read_csv("dataset/ai4i2020.csv")

# Show first 5 rows
print("First 5 rows of dataset:")
print(data.head())

# Show dataset information
print("\nDataset Information:")
print(data.info())
print("\nMachine Failure Count:")
print(data["Machine failure"].value_counts())
# Remove unnecessary columns
data = data.drop(["UDI", "Product ID"], axis=1)

print("\nColumns after removing ID columns:")
print(data.columns)
# Convert machine type into numeric columns
data = pd.get_dummies(data, columns=["Type"])

print("\nDataset after encoding Type column:")
print(data.head())
# Separate input features and target output
X = data.drop("Machine failure", axis=1)
y = data["Machine failure"]

print("\nFeature columns:")
print(X.columns)
from sklearn.model_selection import train_test_split

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining data size:", X_train.shape)
print("Testing data size:", X_test.shape)
from sklearn.ensemble import RandomForestClassifier
# Create the Random Forest model
model = RandomForestClassifier()

# Train the model
model.fit(X_train, y_train)

print("\nModel training completed!")
from sklearn.metrics import accuracy_score

# Predict using test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)
# Example new machine sensor data (simulation)
sample = X_test.iloc[0:1]

prediction = model.predict(sample)

print("\nPrediction for sample machine data:")

if prediction[0] == 1:
    print("⚠ Machine Failure Predicted")
else:
    print("✅ Machine Operating Normally")