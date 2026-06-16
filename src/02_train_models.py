# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

def train_and_evaluate_models(data_path='data/reaction_data.csv', model_dir='models/'):
    
    # --- Get the project root folder (one level up from src/) ---
    script_dir = os.path.dirname(os.path.abspath(__file__))       # src/
    project_root = os.path.dirname(script_dir)                     # CSTR project/
    
    data_path  = os.path.join(project_root, 'data', 'reaction_data.csv')
    model_dir  = os.path.join(project_root, 'models')

    print(f"Loading data from : {data_path}")
    print(f"Models will save to: {model_dir}")

    # 1. Load Data
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows.")

    # Inject Messiness & Clean
    np.random.seed(42)
    missing_indices = np.random.choice(df.index, size=int(len(df) * 0.05), replace=False)
    df.loc[missing_indices, 'Temperature'] = np.nan
    df_clean = df[df['Yield'] <= 100].copy()

    features = ['Temperature', 'Pressure', 'Catalyst_Loading', 'Residence_Time']
    X = df_clean[features]
    y = df_clean['Yield']

    # Split, Impute, and Scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp  = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled  = scaler.transform(X_test_imp)

    # 2. Compare Models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest":     RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        "XGBoost":           xgb.XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=5, random_state=42)
    }

    best_model = None
    best_name  = ""
    best_rmse  = float('inf')

    print(f"\n{'Model':<20} | {'RMSE (%)':<10} | {'R-squared':<10}")
    print("-" * 45)

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
        r2     = r2_score(y_test, y_pred)

        print(f"{name:<20} | {rmse:<10.2f} | {r2:<10.4f}")
        if rmse < best_rmse:
            best_rmse  = rmse
            best_model = model
            best_name  = name

    print("-" * 45)
    print(f"[WINNER] {best_name} with RMSE of {best_rmse:.2f}%")

    # 3. Save Artifacts to models/ folder
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(best_model,  os.path.join(model_dir, 'best_model.joblib'))
    joblib.dump(scaler,      os.path.join(model_dir, 'scaler.joblib'))
    joblib.dump(imputer,     os.path.join(model_dir, 'imputer.joblib'))
    joblib.dump(
        {'X_train': X_train, 'X_train_scaled': X_train_scaled},
        os.path.join(model_dir, 'train_data.joblib')
    )

    print(f"\n[SUCCESS] Files saved inside models/ folder:")
    for f in os.listdir(model_dir):
        print(f"   models/{f}")

if __name__ == "__main__":
    train_and_evaluate_models()