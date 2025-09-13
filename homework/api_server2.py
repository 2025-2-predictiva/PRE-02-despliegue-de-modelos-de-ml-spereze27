"""API server example"""

#
# Usage from command line:
# curl http://127.0.0.1:5000 -X POST -H "Content-Type: application/json" \
# -d '{"bathrooms": "2", "bedrooms": "3", "sqft_living": "1800", \
# "sqft_lot": "2200", "floors": "1", "waterfront": "1", "condition": "3"}'
#

# Windows:
# curl http://127.0.0.1:5000 -X POST -H "Content-Type: application/json" -d "{\"bathrooms\": \"2\", \"bedrooms\": \"3\", \"sqft_living\": \"1800\", \"sqft_lot\": \"2200\", \"floors\": \"1\", \"waterfront\": \"1\", \"condition\": \"3\"}"

import pickle
import os
from flask import Flask, request, jsonify  # Import jsonify for better responses
import pandas as pd  # type: ignore

# --- Get the absolute path to the directory where this script is located ---
# This makes the script runnable from anywhere
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "house_predictor.pkl")


app = Flask(__name__)
app.config["SECRET_KEY"] = "you-will-never-guess"


FEATURES = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "condition",
]


@app.route("/", methods=["POST"])
def index():
    """API function"""
    try:
        # --- ESTA ES LA LÍNEA CLAVE ---
        # Usamos request.json para leer los datos JSON que envía el cliente.
        # Esto soluciona el error "KeyError" que ocurre al usar request.form.
        args = request.json
        
        if not args:
            return jsonify({"error": "Request body must be JSON."}), 400

        # Validate that all required features are present
        if not all(key in args for key in FEATURES):
            missing_keys = [key for key in FEATURES if key not in args]
            return jsonify({"error": f"Missing required keys: {missing_keys}"}), 400

        filt_args = {key: [int(args[key])] for key in FEATURES}
        df = pd.DataFrame.from_dict(filt_args)

        # Use the robust path to load the model
        with open(MODEL_PATH, "rb") as file:
            loaded_model = pickle.load(file)

        prediction = loaded_model.predict(df)
        
        # It's better practice to return JSON
        response = {"predicted_price": prediction[0][0]}

        return jsonify(response)

    except FileNotFoundError:
        return jsonify({"error": "Model file not found. Ensure 'house_predictor.pkl' is in the same directory as the server script."}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- THIS IS THE CRUCIAL PART THAT WAS MISSING ---
# This block starts the server when you run `python api_server.py`
if __name__ == "__main__":
    app.run(debug=True, port=5000)
