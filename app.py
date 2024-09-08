from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import pickle
import google.generativeai as genai
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Configure the Gemma API
genai.configure(api_key="AIzaSyD1FPKl0lENNaIw8JGtMBzPXopVDIqcab8")
model = genai.GenerativeModel("gemini-1.5-flash")

# Load the saved model, scaler, and feature names
with open('ensemble_model.pkl', 'rb') as model_file:
    ensemble_model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('feature_names.pkl', 'rb') as feature_names_file:
    feature_names = pickle.load(feature_names_file)

# Function to predict lung cancer and provide risk percentage and disease type based on input data
def predict_lung_cancer(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_df = pd.DataFrame([input_data_as_numpy_array], columns=feature_names)  # Create DataFrame with feature names
    input_data_scaled = scaler.transform(input_data_df)  # Scale input data
    
    # Predict using the ensemble model
    prediction = ensemble_model.predict(input_data_scaled)[0]
    
    # Get the prediction probabilities
    prediction_probabilities = ensemble_model.predict_proba(input_data_scaled)[0]
    
    # Determine lung cancer status and risk percentage
    if prediction == 1:
        risk = "The person is in the 1st stage of having lung cancer"
        risk_percentage = prediction_probabilities[1] * 100
        disease_type = "Lung Cancer Stage 1"
    elif prediction == 2:
        risk = "You are having severe lung cancer, please visit the doctor."
        risk_percentage = prediction_probabilities[2] * 100
        disease_type = "Severe Lung Cancer"
    else:
        risk = "The person is not having lung cancer"
        risk_percentage = None
        disease_type = "No Lung Cancer"
    
    return risk, risk_percentage, disease_type

def generate_prevention_report(risk, disease, age):
    prompt = f"""
    Provide a general wellness report with the following sections:

    1. **Introduction**
        -Purpose of the Report: Clearly state why this report is being generated, including its relevance to the individual’s health.
        -Overview of Health & Wellness: Briefly describe the importance of understanding and managing health risks, with a focus on proactive wellness and disease prevention.
        -Personalized Context: Include the user's specific details such as age, gender, and any relevant medical history that can be linked to the risk factor and disease.
    
    2. **Risk Description**
        -Detailed Explanation of Risk: Describe the identified risk factor in detail, including how it impacts the body and its potential consequences if left unaddressed.
        -Associated Conditions: Mention any other health conditions commonly associated with this risk factor.
        -Prevalence and Statistics: Provide some general statistics or prevalence rates to contextualize the risk (e.g., how common it is in the general population or specific age groups).
    
    3. **Stage of Risk**
        -Risk Level Analysis: Provide a more granular breakdown of the risk stages (e.g., low, medium, high), explaining what each stage means in terms of potential health outcomes.
        -Progression: Discuss how the risk may progress over time if not managed, and what signs to watch for that indicate worsening or improvement.
    
    4. **Risk Assessment**
        -Impact on Health: Explore how this specific risk factor might affect various aspects of health (e.g., cardiovascular, metabolic, etc.).
        -Modifiable vs. Non-Modifiable Risks: Distinguish between risks that can be changed (e.g., lifestyle factors) and those that cannot (e.g., genetic predisposition).
        -Comparative Risk: Compare the individual's risk to average levels in the general population or among peers.
        
    5. **Findings**
        -In-Depth Health Observations: Summarize the key findings from the assessment, explaining any critical areas of concern.
        -Diagnostic Insights: Provide insights into how the disease was identified, including the symptoms, biomarkers, or other diagnostic criteria used.
        -Data Interpretation: Offer a more detailed interpretation of the user's health data, explaining what specific values or results indicate.
    
    6. **Recommendations**
        -Personalized Action Plan: Suggest specific, actionable steps the individual can take to mitigate the risk or manage the disease (e.g., dietary changes, exercise plans, medical treatments).
        -Lifestyle Modifications: Tailor suggestions to the individual’s lifestyle, providing practical tips for integrating these changes.
        -Monitoring and Follow-up: Recommend how the user should monitor their health and when to seek follow-up care.
        
    7. **Way Forward**
        -Next Steps: Provide a clear path forward, including short-term and long-term goals for managing the identified risk or disease.
        -Preventive Measures: Highlight preventive strategies to avoid worsening the condition or preventing its recurrence.
        -Health Resources: Suggest additional resources, such as apps, websites, or support groups, that could help the individual manage their health.
        
    8. **Conclusion**
        -Summary of Key Points: Recap the most important points from the report, focusing on what the individual should remember and prioritize.
        -Encouragement: Offer positive reinforcement and encouragement for taking proactive steps toward better health.
    
    9. **Contact Information**
        -Professional Guidance: Include information on how to get in touch with healthcare providers for more personalized advice or follow-up.
        -Support Services: List any available support services, such as nutritionists, fitness coaches, or mental health professionals, that could assist in managing the risk.
    
    10. **References**
        -Scientific Sources: Provide references to the scientific literature or authoritative health guidelines that support the information and recommendations given in the report.
        -Further Reading: Suggest articles, books, or other educational materials for the individual to learn more about their condition and how to manage it.

    **Details:**
    Risk: {risk}
    Disease: {disease}
    Age: {age}

    Note: This information is for general wellness purposes. For specific health concerns, consult a healthcare professional.
    """

    try:
        response = model.generate_content(prompt)
        return response.text if response and hasattr(response, 'text') else "No content generated."
    except Exception as e:
        print(f"An error occurred during text generation: {e}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_data = [
        data.get('Age'),
        data.get('Gender'),
        data.get('Air Pollution'),
        data.get('Alcohol use'),
        data.get('Dust Allergy'),
        data.get('OccuPational Hazards'),
        data.get('Genetic Risk'),
        data.get('chronic Lung Disease'),
        data.get('Balanced Diet'),
        data.get('Obesity'),
        data.get('Smoking'),
        data.get('Passive Smoker'),
        data.get('Chest Pain'),
        data.get('Coughing of Blood'),
        data.get('Fatigue'),
        data.get('Weight Loss'),
        data.get('Shortness of Breath'),
        data.get('Wheezing'),
        data.get('Swallowing Difficulty'),
        data.get('Clubbing of Finger Nails'),
        data.get('Frequent Cold'),
        data.get('Dry Cough'),
        data.get('Snoring')
    ]

    risk, risk_percentage, disease_type = predict_lung_cancer(input_data)
    
    response = {
        'risk': risk,
        'risk_percentage': risk_percentage,
        'disease_type': disease_type
    }

    if disease_type != "No Lung Cancer":
        prevention_report = generate_prevention_report(risk, disease_type, data.get('age'))
        response['prevention_report'] = prevention_report

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
