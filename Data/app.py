from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('Vaccine_pred.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [
        request.form['h1n1_concern'],
        request.form['h1n1_knowledge'],
        request.form['behavioral_antiviral_meds'],
        request.form['behavioral_avoidance'],
        request.form['behavioral_face_mask'],
        request.form['behavioral_wash_hands'],
        request.form['behavioral_large_gatherings'],
        request.form['behavioral_outside_home'],
        request.form['behavioral_touch_face'],
        request.form['doctor_recc_h1n1'],
        request.form['doctor_recc_seasonal'],
        request.form['chronic_med_condition'],
        request.form['child_under_6_months'],
        request.form['health_worker'],
        request.form['health_insurance'],
        request.form['opinion_h1n1_vacc_effective'],
        request.form['opinion_h1n1_risk'],
        request.form['opinion_h1n1_sick_from_vacc'],
        request.form['opinion_seas_vacc_effective'],
        request.form['opinion_seas_risk'],
        request.form['opinion_seas_sick_from_vacc'],
        request.form['age_group'],
        request.form['education'],
        request.form['race'],
        request.form['sex'],
        request.form['income_poverty'],
        request.form['marital_status'],
        request.form['rent_or_own'],
        request.form['employment_status'],
        request.form['hhs_geo_region'],
        request.form['census_msa'],
        request.form['household_adults'],
        request.form['household_children']
    ]
    # Convert categorical features to appropriate encoding as needed
    # Example: age_group, education, etc.
    # You need to preprocess these features to match the model's training

    # Convert features to numpy array and reshape for prediction
    final_features = np.array(features).reshape(1, -1)
    prediction = model.predict(final_features)
    
    return render_template('index.html', prediction=prediction[0])

if __name__ == "__main__":
    app.run(debug=True)
