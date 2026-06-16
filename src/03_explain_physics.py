# -*- coding: utf-8 -*-
import joblib
import shap
import matplotlib.pyplot as plt
import os

def generate_shap_visuals():

    # --- Get correct paths relative to project root ---
    script_dir   = os.path.dirname(os.path.abspath(__file__))   # src/
    project_root = os.path.dirname(script_dir)                   # CSTR project/

    model_path = os.path.join(project_root, 'models', 'best_model.joblib')
    data_path  = os.path.join(project_root, 'models', 'train_data.joblib')
    output_dir = os.path.join(project_root, 'visuals')

    print(f"Loading model from : {model_path}")
    print(f"Loading data from  : {data_path}")
    print(f"Saving visuals to  : {output_dir}")

    # 1. Load model and training data
    model = joblib.load(model_path)
    data  = joblib.load(data_path)

    X_train        = data['X_train']
    X_train_scaled = data['X_train_scaled']
    features       = ['Temperature', 'Pressure', 'Catalyst_Loading', 'Residence_Time']

    os.makedirs(output_dir, exist_ok=True)
    print("\nGenerating SHAP physics explanations...")

    # 2. Build SHAP explainer (TreeExplainer for XGBoost, fallback for others)
    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train_scaled)
        print("Using TreeExplainer (XGBoost detected)")
    except Exception:
        explainer   = shap.LinearExplainer(model, X_train_scaled)
        shap_values = explainer.shap_values(X_train_scaled)
        print("Using LinearExplainer (fallback)")

    # 3. Summary Plot — which features matter most overall
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_train, feature_names=features, show=False)
    plt.title("SHAP Summary — Feature Impact on Yield")
    summary_path = os.path.join(output_dir, 'shap_summary.png')
    plt.savefig(summary_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"   Saved: visuals/shap_summary.png")

    # 4. Dependence Plot — how Temperature affects yield
    plt.figure(figsize=(8, 6))
    shap.dependence_plot("Temperature", shap_values, X_train, feature_names=features, show=False)
    plt.title("SHAP Dependence — Temperature vs Yield Impact")
    temp_path = os.path.join(output_dir, 'shap_temp_dependence.png')
    plt.savefig(temp_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"   Saved: visuals/shap_temp_dependence.png")

    # 5. Bar Plot — simple feature importance ranking
    plt.figure(figsize=(8, 5))
    shap.summary_plot(shap_values, X_train, feature_names=features, plot_type='bar', show=False)
    plt.title("Feature Importance Ranking")
    bar_path = os.path.join(output_dir, 'feature_importance_bar.png')
    plt.savefig(bar_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"   Saved: visuals/feature_importance_bar.png")

    print(f"\n[SUCCESS] All SHAP visuals saved to visuals/ folder")

if __name__ == "__main__":
    generate_shap_visuals()