import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("data.csv")

# Select input features and output
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X = data[features]
y = pd.to_numeric(data["Survived"], errors="coerce")

# Remove rows with missing output
valid_rows = y.notna()
X = X[valid_rows]
y = y[valid_rows].astype(int)

# Categorical and numerical features
categorical = ["Sex", "Embarked"]
numerical = ["Pclass", "Age", "SibSp", "Parch", "Fare"]

# Preprocessing
preprocessor = ColumnTransformer([
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical),
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numerical)
])

# Gaussian Naive Bayes model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", GaussianNB())
])

# Evaluate the model when sufficient data is available
if len(X) >= 10 and y.nunique() > 1:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if y.value_counts().min() >= 2 else None
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Accuracy:", round(
        accuracy_score(y_test, predictions) * 100, 2
    ), "%")
    print(classification_report(y_test, predictions, zero_division=0))
else:
    print("Small dataset: evaluation skipped.")

# Train on all available records
model.fit(X, y)
joblib.dump(model, "titanic_model.pkl")

print("Model trained and saved successfully!")
print("Training records:", len(X))
print("Classes:", list(model.classes_))