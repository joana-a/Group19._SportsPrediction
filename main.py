# app.py
import os
#import xgboost
from re import template
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load("C:\\Users\\ewura\\Documents\\Sports_Prediction\\venv\\xgmodel.pkl")
scaler = joblib.load("C:\\Users\\ewura\\Documents\\Sports_Prediction\\venv\\scaler.pkl")
# Define a route for the index page
@app.route("/")
def index():
    return render_template("index.html") 

# Define a route for prediction
@app.route("/predict", methods=["POST"])
def predict():
     if request.method == "POST":
        # Define a list of feature names
        feature_names = [
            "potential", "value_eur", "wage_eur", "age", 
            "international_reputation", "release_clause_eur", 
            "shooting", "passing", "dribbling", "physic", 
            "goalkeeping_speed", "power_combined", "mentality_combined", 
            "skill_combined", "attacking_combined", "movement_combined", 
            "defending_combined"
        ]

        # Create an empty dictionary to store input features
        input_features_dict = {}

        # Iterate through the feature names and get user input from the form
        for feature in feature_names:
            input_value = float(request.form[feature])
            input_features_dict[feature] = input_value
        # Scale the input features using the loaded scaler
        input_dataframe = pd.DataFrame([input_features_dict])
        scaled_features = scaler.transform(input_dataframe)
        print(scaled_features)

        print(model)
        # Make a prediction using the loaded model
        overall_rating = model.predict(scaled_features)
       

        return render_template("index.html", prediction="The player rating is" + overall_rating)

if __name__ == "__main__":
    app.run(debug=True)
