from flask import Flask, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("stacking_ensemble_model.pkl")
scaler = joblib.load("scaler.pkl")
selected_features = joblib.load("selected_features.pkl")

max_glu_serum_map = {"None": 0, "Norm": 1, ">200": 2}
A1Cresult_map = {"None": 0, "Norm": 1, ">7": 2}

@app.route('/')
def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Diabetes Readmission Risk Prediction</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Poppins', sans-serif;
                background: linear-gradient(120deg, #f8f9fc, #e9ecf4); 
                color: #2c3e50;
                display: flex;
                justify-content: center;
                align-items: flex-start; 
                min-height: 100vh;
                padding: 20px; 
                overflow-y: auto; 
            }
            
            .container {
                max-width: 700px;
                width: 100%;
                background: white;
                padding: 30px;
                border-radius: 16px;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
                animation: fadeIn 1s ease-out;
            }
            
            @keyframes fadeIn {
                from {
                    opacity: 0;
                    transform: translateY(20px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            h1 {
                text-align: center;
                font-size: 2.5rem;
                margin-bottom: 20px;
                color: #34495e;
                letter-spacing: 1px;
                position: relative;
            }
            
            h1::after {
                content: '';
                display: block;
                width: 60px;
                height: 4px;
                background: #3498db;
                margin: 10px auto 0;
                border-radius: 2px;
            }
            
            form {
                display: flex;
                flex-direction: column;
                gap: 20px;
            }
            
            label {
                font-size: 1.1rem;
                font-weight: 600;
                color: #2c3e50;
            }
            
            input {
                padding: 15px;
                border: 1px solid #dcdfe3;
                border-radius: 12px;
                font-size: 1rem;
                outline: none;
                transition: all 0.3s ease;
                background: #f9fafc;
                box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.05);
            }
            
            input:focus {
                border-color: #3498db;
                box-shadow: 0 0 8px rgba(52, 152, 219, 0.3);
                background: white;
            }
            
            button {
                padding: 15px;
                border: none;
                border-radius: 12px;
                background: linear-gradient(90deg, #6a11cb, #2575fc);
                color: white;
                font-size: 1.1rem;
                font-weight: bold;
                cursor: pointer;
                transition: transform 0.2s ease, background-color 0.3s ease;
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
            }
            
            button:hover {
                transform: scale(1.05);
                box-shadow: 0 6px 14px rgba(0, 0, 0, 0.15);
            }
            
            button:active {
                transform: scale(0.98);
            }
            
            .footer {
                text-align: center;
                margin-top: 20px;
                font-size: 0.9rem;
                color: #95a5a6;
            }
            
            @media (max-width: 768px) {
                h1 {
                    font-size: 2rem;
                }
            
                input, button {
                    font-size: 0.9rem;
                    padding: 12px;
                }
            }
            
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Diabetes Readmission Risk</h1>
            <form action="/predict" method="post">
                <label>Have you experienced a change? (1 for yes, 0 for no):</label>
                <input type="text" name="change" required>

                <label>Are you taking diabetes medication? (1 for yes, 0 for no):</label>
                <input type="text" name="diabetesMed" required>

                <label>Were you readmitted in the past? (1 for yes, 0 for no):</label>
                <input type="text" name="readmitted" required>

                <label>Maximum glucose serum level (None/Norm/>200):</label>
                <input type="text" name="max_glu_serum" required>

                <label>A1C result (None/Norm/>7):</label>
                <input type="text" name="A1Cresult" required>

                <label>Age group (0 for 0-10, 1 for 10-20, etc.):</label>
                <input type="text" name="age_t" required>

                <label>Time in hospital (days):</label>
                <input type="text" name="time_in_hospital" required>

                <label>Number of lab procedures:</label>
                <input type="text" name="num_lab_procedures" required>

                <label>Number of procedures:</label>
                <input type="text" name="num_procedures" required>

                <label>Number of medications:</label>
                <input type="text" name="num_medications" required>

                <label>Number of outpatient visits:</label>
                <input type="text" name="number_outpatient" required>

                <label>Number of inpatient visits:</label>
                <input type="text" name="number_inpatient" required>

                <label>Number of emergency visits:</label>
                <input type="text" name="number_emergency" required>

                <button type="submit">Predict</button>
            </form>
            <div class="footer">Made By: Justin Nguyen</div>
        </div>
    </body>
    </html>
    """
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form

        inputs = {
            "change": int(data["change"]),
            "diabetesMed": int(data["diabetesMed"]),
            "readmitted": int(data["readmitted"]),
            "max_glu_serum_ind": max_glu_serum_map[data["max_glu_serum"]],
            "A1Cresult_ind": A1Cresult_map[data["A1Cresult"]],
            "age_t": int(data["age_t"]),
            "time_in_hospital": np.log1p(int(data["time_in_hospital"])),
            "num_lab_procedures": np.log1p(int(data["num_lab_procedures"])),
            "num_procedures": np.log1p(int(data["num_procedures"])),
            "num_medications": np.log1p(int(data["num_medications"])),
            "number_outpatient": np.log1p(int(data["number_outpatient"])),
            "number_inpatient": np.log1p(int(data["number_inpatient"])),
            "number_emergency": np.log1p(int(data["number_emergency"])),
        }

        for feature in selected_features:
            if feature not in inputs:
                inputs[feature] = 0

        input_vector = np.array([inputs[feature] for feature in selected_features])
        input_vector_scaled = scaler.transform([input_vector])

        prediction = model.predict(input_vector_scaled)[0]
        result = "at risk of readmission" if prediction == 1 else "not at risk of readmission"

        return f'<h1 style="font-family: Arial, sans-serif; color: #4CAF50; font-size: 36px; text-align: center; background-color: #f4f4f4; padding: 20px; border-radius: 10px;">Prediction: You are {result}.</h1>'


    except Exception as e:
        return f'<center><h1 style="font-family: Arial, sans-serif; color: #ff4c4c; font-size: 36px; text-align: center; background-color: #f4f4f4; padding: 20px; border-radius: 10px;">Error: {str(e)}</h1></center>'

app.run(host="0.0.0.0", port=8080, debug=True)