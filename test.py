import pandas as pd
import joblib

model = joblib.load("titanic_model.pkl")

passenger = pd.DataFrame([{
    "Pclass": int(input("Passenger Class (1/2/3): ")),
    "Sex": input("Sex (male/female): ").strip().lower(),
    "Age": float(input("Age: ")),
    "SibSp": int(input("Number of Siblings/Spouses Aboard: ")),
    "Parch": int(input("Number of Parents/Children Aboard: ")),
    "Fare": float(input("Fare: ")),
    "Embarked": input("Port (C/Q/S): ").strip().upper()
}])

prediction = model.predict(passenger)[0]
probabilities = model.predict_proba(passenger)[0]

print("\nPredicted Survival:", prediction)

if prediction == 1:
    print("The passenger is predicted to have survived.")
else:
    print("The passenger is predicted not to have survived.")

print("Survival probability:", round(probabilities[list(model.classes_).index(1)] * 100, 2), "%")