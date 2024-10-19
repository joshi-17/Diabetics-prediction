from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('diabetes_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_features = [data['Pregnancies'], data['Glucose'], data['BloodPressure'],
                     data['SkinThickness'], data['Insulin'], data['BMI'],
                     data['DiabetesPedigreeFunction'], data['Age']]
    prediction = model.predict([input_features])
    return jsonify({'Diabetic': bool(prediction[0])})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)