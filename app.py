import os
import joblib
import pandas as pd

from flask import Flask, request, jsonify

app = Flask(__name__)

# ---------------------------------------
# Load saved model
# ---------------------------------------

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "titanic_model.joblib"
)

model = joblib.load(MODEL_PATH)

print("Machine Learning model loaded successfully")


# ---------------------------------------
# Home endpoint
# ---------------------------------------

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "message": "Titanic Survival Prediction API",
        "status": "running",
        "endpoints": [
            "/health",
            "/predict"
        ]
    })


# ---------------------------------------
# Health check endpoint
# ---------------------------------------

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })


# ---------------------------------------
# Prediction endpoint
# ---------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "status": "error",
                "message": "Please provide a valid JSON object."
            }), 400

        required_fields = [
            "Pclass",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "Embarked"
        ]

        # Check missing fields
        missing_fields = [
            field for field in required_fields
            if field not in data
        ]

        if missing_fields:
            return jsonify({
                "status": "error",
                "message": "Missing required fields.",
                "missing_fields": missing_fields
            }), 400

        # Convert input into DataFrame
        input_df = pd.DataFrame(
            [[data[field] for field in required_fields]],
            columns=required_fields
        )

        # Convert numerical columns
        numeric_columns = [
            "Pclass",
            "Age",
            "SibSp",
            "Parch",
            "Fare"
        ]

        for column in numeric_columns:
            input_df[column] = pd.to_numeric(
                input_df[column],
                errors="raise"
            )

        # Validate numerical values
        if input_df[numeric_columns].isna().any().any():
            raise ValueError("Numerical fields cannot be empty.")

        if not all(
            input_df[numeric_columns].map(
                lambda value: pd.notna(value) and abs(value) != float("inf")
            ).iloc[0]
        ):
            raise ValueError("Numerical values must be finite.")

        if input_df.loc[0, "Pclass"] not in [1, 2, 3]:
            raise ValueError("Pclass must be 1, 2, or 3.")

        if (
            input_df.loc[0, ["Age", "SibSp", "Parch", "Fare"]] < 0
        ).any():
            raise ValueError("Age, SibSp, Parch and Fare cannot be negative.")

        # Validate categorical values
        input_df["Sex"] = input_df["Sex"].astype(str).str.lower()
        input_df["Embarked"] = input_df["Embarked"].astype(str).str.upper()

        if input_df.loc[0, "Sex"] not in ["male", "female"]:
            raise ValueError("Sex must be male or female.")

        if input_df.loc[0, "Embarked"] not in ["S", "C", "Q"]:
            raise ValueError("Embarked must be S, C, or Q.")

        # Generate prediction
        prediction = int(model.predict(input_df)[0])

        probability = float(
            model.predict_proba(input_df)[0][1]
        )

        result = (
            "Survived" if prediction == 1
            else "Not Survived"
        )

        return jsonify({
            "status": "success",
            "prediction": prediction,
            "result": result,
            "survival_probability": round(probability, 4)
        })

    except (ValueError, TypeError) as error:

        return jsonify({
            "status": "error",
            "message": str(error)
        }), 400

    except Exception:

        app.logger.exception("Prediction failed")

        return jsonify({
            "status": "error",
            "message": "An internal prediction error occurred."
        }), 500


# ---------------------------------------
# Run Flask application
# ---------------------------------------

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )