import gradio as gr
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Prevent GUI rendering issues
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import CoxPHFitter
import shap
import joblib
import os

# Set aesthetic styling for plots
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams.update({'font.size': 11, 'figure.figsize': (10, 5)})

# 1. Load packaged models and metadata
app_data_path = '../outputs/lung_cancer_app_data.pkl'
if not os.path.exists(app_data_path):
    # fallback for local testing in Cwd
    app_data_path = 'outputs/lung_cancer_app_data.pkl'
if not os.path.exists(app_data_path):
    # fallback for Hugging Face Space root directory
    app_data_path = 'lung_cancer_app_data.pkl'

app_data = joblib.load(app_data_path)
rf_model = app_data['rf_model']
cph_model = app_data['cph_model']
expected_columns = app_data['expected_columns']
cox_columns = app_data.get('cox_columns', ['Age', 'Cancer_Stage', 'Tumor_Size_cm'])
defaults = app_data['defaults']

# Initialize SHAP explainer once to save startup time
explainer = shap.TreeExplainer(rf_model)

# Helper function to dynamically show/hide smoking inputs in Gradio
def update_smoking_visibility(status):
    if status == "Never Smoked":
        return gr.update(visible=False, value=0), gr.update(visible=False, value=0)
    else:
        return gr.update(visible=True), gr.update(visible=True)

# 2. Main prediction and explanation function
def predict_survival(age, bmi, stage, tumor_size, metastasis, treatment, smoking_status, cigarettes, years_smoking):
    # Map Cancer Stage to numeric
    stage_numeric = {'Stage I': 1, 'Stage II': 2, 'Stage III': 3, 'Stage IV': 4}[stage]
    
    # Map Metastasis to binary
    metastasis_val = 1 if metastasis == 'Yes' else 0
    
    # 2.1 Assemble patient dictionary starting with overall defaults
    patient_dict = defaults.copy()
    patient_dict['Age'] = float(age)
    patient_dict['BMI'] = float(bmi)
    patient_dict['Cancer_Stage'] = stage_numeric
    patient_dict['Tumor_Size_cm'] = float(tumor_size)
    patient_dict['Metastasis'] = metastasis
    patient_dict['Treatment'] = treatment
    patient_dict['Smoking_Status'] = smoking_status
    patient_dict['Cigarettes_Per_Day'] = float(cigarettes)
    patient_dict['Years_Smoking'] = float(years_smoking)
    
    # 2.2 Encode patient features into expected format (initialize as float to avoid Pandas dtype warnings)
    patient_encoded = pd.DataFrame(0.0, index=[0], columns=expected_columns, dtype=float)
    
    for col, val in patient_dict.items():
        if col in expected_columns:
            # Numerical column
            patient_encoded.loc[0, col] = float(val)
        else:
            # Categorical or Binary column -> search for matching dummy column
            dummy_col = f"{col}_{val}"
            if dummy_col in expected_columns:
                patient_encoded.loc[0, dummy_col] = 1
                
    # 2.3 classification prediction (Random Forest)
    prob_survive = float(rf_model.predict_proba(patient_encoded)[0, 1])
    pred_class = "Survive (High Chance)" if prob_survive >= 0.50 else "High Risk of Mortality (Low Chance)"
    
    # 2.4 Cox Proportional Hazards prediction
    patient_cox = pd.DataFrame(0.0, index=[0], columns=cox_columns, dtype=float)
    patient_cox.loc[0, 'Age'] = float(age)
    patient_cox.loc[0, 'Cancer_Stage'] = float(stage_numeric)
    patient_cox.loc[0, 'Tumor_Size_cm'] = float(tumor_size)
    patient_cox.loc[0, 'Cigarettes_Per_Day'] = float(cigarettes)
    patient_cox.loc[0, 'Years_Smoking'] = float(years_smoking)
    
    if metastasis == 'Yes' and 'Metastasis_Yes' in cox_columns:
        patient_cox.loc[0, 'Metastasis_Yes'] = 1.0
        
    dummy_smoke = f"Smoking_Status_{smoking_status}"
    if dummy_smoke in cox_columns:
        patient_cox.loc[0, dummy_smoke] = 1.0
        
    dummy_treatment = f"Treatment_{treatment}"
    if dummy_treatment in cox_columns:
        patient_cox.loc[0, dummy_treatment] = 1.0
    
    # Predict personalized survival function (curves)
    surv_func = cph_model.predict_survival_function(patient_cox)
    months = surv_func.index
    probabilities = surv_func.values.flatten()
    
    # Find Median Survival Time (where probability drops below 50%)
    median_survival = "Survival probability remains above 50% past 72 months."
    if probabilities[0] <= 0.50:
        median_survival = "Under 7 Months (Critical High Risk)"
    else:
        for idx, prob in enumerate(probabilities):
            if prob <= 0.50:
                median_survival = f"{int(months[idx])} Months"
                break
            
    # 2.5 Plot the personalized survival curve
    fig_curve, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(months, probabilities * 100, color='#2a9d8f', linewidth=3, label='Patient Survival Curve')
    ax.axhline(y=50, color='#e63946', linestyle='--', label='50% Survival Threshold')
    ax.set_title(f'Predicted Personalized Survival Probability Over Time\n(Median Survival: {median_survival})', fontweight='bold', fontsize=12)
    ax.set_xlabel('Months after Diagnosis')
    ax.set_ylabel('Probability of Survival (%)')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left')
    plt.tight_layout()
    curve_plot_path = 'temp_km_curve.png'
    plt.savefig(curve_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2.6 Generate SHAP waterfall plot for local explainability (Filtered to UI features)
    shap_values = explainer(patient_encoded)[..., 1]
    
    # Filter the explanation object to ONLY show clinical features present in the UI
    ui_keywords = ['age', 'bmi', 'cancer stage', 'tumor size', 'metastasis', 'treatment', 'smoking status', 'cigarettes', 'years smoking']
    
    def clean_feature_name(name):
        return name.replace('_', ' ').replace('Yes', '').strip()
        
    ui_indices = []
    for idx, name in enumerate(expected_columns):
        clean_name = clean_feature_name(name)
        if any(kw in clean_name.lower() for kw in ui_keywords):
            # Clinical presence filter: for binary dummy categories, only keep if the patient actually has it (value == 1.0)
            if '_' in name:
                if patient_encoded.iloc[0, idx] == 1.0:
                    ui_indices.append(idx)
            else:
                ui_indices.append(idx)
                
    # Filter the SHAP explanation object and clean its feature labels for clinical presentation
    shap_values_filtered = shap_values[0, ui_indices]
    shap_values_filtered.feature_names = [clean_feature_name(expected_columns[i]) for i in ui_indices]
    
    fig_shap, ax_shap = plt.subplots(figsize=(8, 4.5))
    shap.plots.waterfall(shap_values_filtered, show=False)
    plt.title('Personalized Clinical Drivers (SHAP Explanation)', fontweight='bold', fontsize=12)
    plt.tight_layout()
    shap_plot_path = 'temp_shap_waterfall.png'
    plt.savefig(shap_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2.7 Construct Plain-English clinical reasoning
    feat_names = patient_encoded.columns
    shap_impacts = shap_values.values[0]
    patient_values = patient_encoded.iloc[0].values
    
    # Map raw dummy column names back to cleaner UI terms
    def clean_feature_name(name):
        return name.replace('_', ' ').replace('Yes', '').strip()
        
    impact_df = pd.DataFrame({
        'Raw_Feature': feat_names,
        'Feature': [clean_feature_name(n) for n in feat_names],
        'Value': patient_values,
        'Impact': shap_impacts
    })
    
    # Filter 1: Only explain features the user actually interacts with in the UI
    ui_keywords = ['age', 'bmi', 'cancer stage', 'tumor size', 'metastasis', 'treatment', 'smoking status', 'cigarettes', 'years smoking']
    impact_df = impact_df[impact_df['Feature'].apply(lambda x: any(kw in x.lower() for kw in ui_keywords))]
    
    # Filter 2: CRITICAL CLINICAL RULE! For binary/one-hot dummy features (which contain '_'),
    # we ONLY explain them if they are active (Value == 1.0) for this patient.
    # It is clinical nonsense to list a treatment or symptom the patient DOES NOT HAVE as a driver!
    def keep_by_clinical_presence(row):
        if '_' in row['Raw_Feature']:
            return row['Value'] == 1.0
        return True # Keep continuous features like Age, BMI, Tumor Size
        
    impact_df = impact_df[impact_df.apply(keep_by_clinical_presence, axis=1)]
    
    # Filter 3: Raise the significance threshold to 0.02 (2% probability shift) to filter out minor statistical noise (like Age having a 1% positive SHAP due to data quirks)
    impact_df = impact_df.sort_values(by='Impact', ascending=False)
    
    # Separate positive drivers and negative risk factors
    positives = impact_df[impact_df['Impact'] > 0.02].head(2)
    negatives = impact_df[impact_df['Impact'] < -0.02].tail(2)
    
    # Format reasoning strings (using HTML span tags with explicit colors to bypass Gradio CSS overrides)
    pos_text = ", ".join([f"<span style='font-weight: 700; color: #1e293b;'>{row['Feature']}</span>" for _, row in positives.iterrows()]) if not positives.empty else "None"
    neg_text = ", ".join([f"<span style='font-weight: 700; color: #1e293b;'>{row['Feature']}</span>" for _, row in negatives.iterrows()]) if not negatives.empty else "None"
    
    reasoning_html = f"""
    <div style="background-color: #f8f9fa; color: #1e293b; padding: 18px; border-left: 5px solid #2a9d8f; border-radius: 6px; margin-top: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
        <h4 style="margin: 0 0 12px 0; color: #2a9d8f; font-size: 1.25em; font-weight: 700;">🩺 Automated Clinical Summary</h4>
        <p style="margin: 0 0 8px 0; color: #2b2d42; font-size: 1.05em; line-height: 1.5;"><span style="font-weight: 700; color: #1e293b;">Predictive Verdict:</span> The patient has a <span style="font-weight: 700; color: #2a9d8f;">{prob_survive:.1%} chance</span> of 5-year survival, classified as <span style="font-weight: 700; color: #1e293b;">{pred_class}</span>.</p>
        <p style="margin: 0 0 8px 0; color: #2b2d42; font-size: 1.05em; line-height: 1.5;"><span style="font-weight: 700; color: #1e293b;">Predicted Median Survival:</span> <span style="font-weight: 700; color: #457b9d;">{median_survival}</span>.</p>
        <p style="margin: 0 0 8px 0; color: #2b2d42; font-size: 1.05em; line-height: 1.5;"><span style="color: #2a9d8f;">🟢</span> <span style="font-weight: 700; color: #1e293b;">Key Protective Drivers:</span> Features that significantly <span style="font-style: italic; color: #2b2d42;">increased</span> their survival chances: {pos_text}.</p>
        <p style="margin: 0; color: #2b2d42; font-size: 1.05em; line-height: 1.5;"><span style="color: #e63946;">🔴</span> <span style="font-weight: 700; color: #1e293b;">Key Risk Factors:</span> Features that significantly <span style="font-style: italic; color: #2b2d42;">decreased</span> their survival chances: {neg_text}.</p>
    </div>
    """
    
    return reasoning_html, curve_plot_path, shap_plot_path

# 3. Create the Gradio interface
with gr.Blocks(title="Lung Cancer 5-Year Survival Predictor & Survival Analyzer") as demo:
    gr.HTML("""
    <div style="text-align: center; margin-bottom: 20px; background: linear-gradient(135deg, #1d3557, #457b9d); color: white; padding: 25px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <h1 style="margin: 0; font-size: 2.2em; font-weight: bold; letter-spacing: 0.5px;">🩺 Clinical Lung Cancer Survival Predictor</h1>
        <p style="margin: 10px 0 0 0; font-size: 1.1em; opacity: 0.9;">End-to-End Decision Support System utilizing Random Forest & Cox Proportional Hazards</p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📋 Patient Demographics & Vitals")
            age = gr.Slider(minimum=18, maximum=90, value=60, step=1, label="Age")
            bmi = gr.Slider(minimum=15.0, maximum=40.0, value=24.5, step=0.1, label="BMI (Body Mass Index)")
            
            gr.Markdown("### 🚬 Lifestyle & Habits")
            smoking_status = gr.Dropdown(choices=["Never Smoked", "Former Smoker", "Current Smoker"], value="Never Smoked", label="Smoking Status")
            cigarettes = gr.Slider(minimum=0, maximum=50, value=0, step=1, label="Cigarettes Per Day", visible=False)
            years_smoking = gr.Slider(minimum=0, maximum=60, value=0, step=1, label="Years Smoking", visible=False)
            
            gr.Markdown("### 🔬 Clinical & Staging Details")
            stage = gr.Dropdown(choices=["Stage I", "Stage II", "Stage III", "Stage IV"], value="Stage I", label="Cancer Stage")
            tumor_size = gr.Slider(minimum=0.5, maximum=10.0, value=2.0, step=0.1, label="Tumor Size (cm)")
            metastasis = gr.Dropdown(choices=["No", "Yes"], value="No", label="Metastasis (Spread to other organs)")
            treatment = gr.Dropdown(choices=["Surgery", "Chemotherapy", "Radiation", "Targeted Therapy", "Immunotherapy"], value="Surgery", label="Primary Treatment Plan")
            
            predict_btn = gr.Button("🔮 Run Prognostic AI Prediction", variant="primary")
            
        with gr.Column(scale=1):
            gr.Markdown("### 🔬 Diagnostic & Prognostic Insights")
            output_summary = gr.HTML(label="Automated Clinical Summary")
            
            with gr.Row():
                with gr.Tab("📈 Survival Probability Curve"):
                    output_curve = gr.Image(type="filepath", label="Personalized Survival Curve")
                with gr.Tab("📊 Personalized Clinical Drivers"):
                    output_shap = gr.Image(type="filepath", label="SHAP Feature Explanations")
                    
    # Connect smoking visibility toggle
    smoking_status.change(
        fn=update_smoking_visibility,
        inputs=[smoking_status],
        outputs=[cigarettes, years_smoking]
    )
    
    predict_btn.click(
        fn=predict_survival,
        inputs=[age, bmi, stage, tumor_size, metastasis, treatment, smoking_status, cigarettes, years_smoking],
        outputs=[output_summary, output_curve, output_shap]
    )

if __name__ == "__main__":
    demo.launch()
