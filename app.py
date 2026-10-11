from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)

MODEL_PATH = "titanic_model.pkl"
DATA_PATH = "data.csv"

FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
model = None
accuracy = None
evaluation_message = "Accuracy is unavailable until enough data is provided."

def load_model():
    global model, accuracy, evaluation_message

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model not found. Run 'python model.py' first."
        )

    model = joblib.load(MODEL_PATH)

    if not os.path.exists(DATA_PATH):
        evaluation_message = "data.csv was not found; accuracy is unavailable."
        return

    data = pd.read_csv(DATA_PATH)

    if not all(column in data.columns for column in FEATURES + ["Survived"]):
        evaluation_message = "The dataset is missing required columns."
        return

    X = data[FEATURES]
    y = pd.to_numeric(data["Survived"], errors="coerce")

    valid = y.notna()
    X = X.loc[valid]
    y = y.loc[valid].astype(int)

    if len(X) < 10 or y.nunique() < 2 or y.value_counts().min() < 2:
        evaluation_message = (
            f"Holdout accuracy unavailable: only {len(X)} usable records "
            "or insufficient examples per class. Add more data."
        )
        return

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        evaluation_model = clone(model)
        evaluation_model.fit(X_train, y_train)
        predictions = evaluation_model.predict(X_test)

        accuracy = round(accuracy_score(y_test, predictions) * 100, 2)
        evaluation_message = (
            "Measured on a held-out test split (20% of usable records)."
        )
    except ValueError:
        accuracy = None
        evaluation_message = "Accuracy could not be calculated for this dataset."

try:
    load_model()
except (FileNotFoundError, ValueError) as error:
    print(error)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    survival_probability = None
    error = None
    form_data = {}

    if request.method == "POST":
        form_data = request.form.to_dict()

        try:
            passenger = pd.DataFrame([{
                "Pclass": int(request.form["Pclass"]),
                "Sex": request.form["Sex"].strip().lower(),
                "Age": float(request.form["Age"]),
                "SibSp": int(request.form["SibSp"]),
                "Parch": int(request.form["Parch"]),
                "Fare": float(request.form["Fare"]),
                "Embarked": request.form["Embarked"].strip().upper()
            }])

            if passenger["Pclass"].iloc[0] not in [1, 2, 3]:
                raise ValueError("Select passenger class 1, 2, or 3.")

            if passenger["Sex"].iloc[0] not in ["male", "female"]:
                raise ValueError("Please select a valid sex.")

            if passenger["Embarked"].iloc[0] not in ["C", "Q", "S"]:
                raise ValueError("Please select a valid embarkation port.")

            if passenger["Age"].iloc[0] < 0 or passenger["Fare"].iloc[0] < 0:
                raise ValueError("Age and fare cannot be negative.")

            if passenger["SibSp"].iloc[0] < 0 or passenger["Parch"].iloc[0] < 0:
                raise ValueError("Family counts cannot be negative.")

            if model is None:
                raise RuntimeError(
                    "The trained model is unavailable. Run 'python model.py' first."
                )

            result = int(model.predict(passenger)[0])
            prediction = result

            probabilities = model.predict_proba(passenger)[0]
            classes = list(model.classes_)

            if 1 in classes:
                survival_probability = round(
                    float(probabilities[classes.index(1)]) * 100, 2
                )

        except (ValueError, KeyError, TypeError) as exception:
            error = str(exception)
        except RuntimeError as exception:
            error = str(exception)

    return render_template(
        "index.html",
        prediction=prediction,
        survival_probability=survival_probability,
        accuracy=accuracy,
        evaluation_message=evaluation_message,
        error=error,
        form_data=form_data
    )


if __name__ == "__main__":
    app.run(debug=True)