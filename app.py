import streamlit as st
import numpy as np
import joblib

from tensorflow.keras.models import load_model

# Load model and scaler
@st.cache_resource
def load_ml_components():
    model = load_model("fetal_model.h5")  # Updated path
    scaler = joblib.load("scaler.pkl")    # Updated path
    return model, scaler

model, scaler = load_ml_components()

# Feature list and class mapping
features = [
    'baseline value', 'accelerations', 'fetal_movement', 'uterine_contractions',
    'light_decelerations', 'severe_decelerations', 'prolongued_decelerations',
    'abnormal_short_term_variability', 'mean_value_of_short_term_variability',
    'percentage_of_time_with_abnormal_long_term_variability', 'mean_value_of_long_term_variability',
    'histogram_width', 'histogram_min', 'histogram_max', 'histogram_number_of_peaks',
    'histogram_number_of_zeroes', 'histogram_mode', 'histogram_mean', 'histogram_median',
    'histogram_variance', 'histogram_tendency'
]

class_mapping = {
    0: "🟢 Normal",
    1: "🟡 Suspicious",
    2: "🔴 Pathological"
}

# App title and description
st.title("Fetal Health Classification System")
st.markdown("""
This application predicts fetal health status based on cardiotocography (CTG) features. 
Please enter the required parameters below and click 'Predict' to get the classification.
""")

# Create input form
with st.form("input_form"):
    st.header("Input Parameters")
    
    # Organize inputs into columns for better layout
    col1, col2, col3 = st.columns(3)
    
    inputs = []
    for i, feature in enumerate(features):
        # Distribute features across columns
        if i % 3 == 0:
            current_col = col1
        elif i % 3 == 1:
            current_col = col2
        else:
            current_col = col3
            
        with current_col:
            value = st.number_input(
                label=feature.replace("_", " ").title(),
                min_value=0.0,
                max_value=1000.0 if "percentage" not in feature else 100.0,
                value=0.0,
                step=0.01,
                format="%.2f",
                key=feature
            )
            inputs.append(value)
    
    # Predict button
    submitted = st.form_submit_button("Predict Fetal Health")

# Prediction logic
if submitted:
    try:
        data = np.array(inputs).reshape(1, -1)
        data_scaled = scaler.transform(data)
        prediction = model.predict(data_scaled)
        predicted_class = np.argmax(prediction, axis=1)[0]
        result = class_mapping.get(predicted_class, "Unknown")
        
        # Display results with appropriate styling
        st.subheader("Prediction Result")
        
        if predicted_class == 0:
            st.success(f"**Predicted Fetal Health:** {result}")
        elif predicted_class == 1:
            st.warning(f"**Predicted Fetal Health:** {result}")
        else:
            st.error(f"**Predicted Fetal Health:** {result}")
            
        # Add some explanation
        st.markdown("""
        **Interpretation:**
        - 🟢 **Normal**: Indicates a healthy fetus with normal CTG readings
        - 🟡 **Suspicious**: Suggests potential issues that may require monitoring
        - 🔴 **Pathological**: Indicates significant abnormalities requiring immediate attention
        """)
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")

# Add footer
st.markdown("---")
st.markdown("""
*Note: This tool is intended for professional use and should not replace clinical judgment.*  
*Always consult with a healthcare provider for medical decisions.*
""")