# -*- coding: utf-8 -*-
import joblib
import warnings
import os
import json
from scipy.optimize import minimize

warnings.filterwarnings('ignore')

PRODUCT_VALUE  = 500
TEMP_COST      = 2.5
CATALYST_COST  = 150

def run_economic_optimization():

    # --- Get correct paths relative to project root ---
    script_dir   = os.path.dirname(os.path.abspath(__file__))   # src/
    project_root = os.path.dirname(script_dir)                   # CSTR project/

    model_path  = os.path.join(project_root, 'models', 'best_model.joblib')
    scaler_path = os.path.join(project_root, 'models', 'scaler.joblib')
    output_path = os.path.join(project_root, 'models', 'optimal_conditions.json')

    print(f"Loading model from : {model_path}")
    print(f"Loading scaler from: {scaler_path}")

    # 1. Load model and scaler
    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # 2. Define economic objective (we minimize negative profit)
    def economic_objective(input_vars):
        temp, pressure, catalyst, time = input_vars
        scaled_inputs    = scaler.transform([[temp, pressure, catalyst, time]])
        predicted_yield  = model.predict(scaled_inputs)[0]
        revenue          = predicted_yield * PRODUCT_VALUE
        expenses         = (temp * TEMP_COST) + (catalyst * CATALYST_COST)
        profit           = revenue - expenses
        return -profit  # minimize negative profit = maximize profit

    # 3. Run optimization
    bounds        = ((120, 220), (10, 50), (0.1, 2.5), (0.5, 6.0))
    initial_guess = [160, 30, 1.2, 3.0]

    print("\nRunning L-BFGS-B Plant Optimization...")
    result = minimize(economic_objective, initial_guess, method='L-BFGS-B', bounds=bounds)

    # 4. Extract results
    opt         = result.x
    final_yield = model.predict(scaler.transform([[opt[0], opt[1], opt[2], opt[3]]]))[0]
    max_profit  = -result.fun

    # 5. Print results
    print("\n--- Optimal Plant Operating Conditions ---")
    print(f"Maximized Profit:   ${max_profit:,.2f} / cycle")
    print(f"Chemical Yield:     {final_yield:.2f}%\n")
    print(f"Temperature:        {opt[0]:.1f} degC")
    print(f"Pressure:           {opt[1]:.1f} bar")
    print(f"Catalyst Loading:   {opt[2]:.2f} mol%")
    print(f"Residence Time:     {opt[3]:.1f} hours")

    # 6. Save results to models/optimal_conditions.json
    results = {
        "max_profit_per_cycle": round(max_profit, 2),
        "predicted_yield_pct":  round(float(final_yield), 2),
        "optimal_conditions": {
            "Temperature_C":      round(float(opt[0]), 1),
            "Pressure_bar":       round(float(opt[1]), 1),
            "Catalyst_Loading_mol_pct": round(float(opt[2]), 2),
            "Residence_Time_hrs": round(float(opt[3]), 1)
        }
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\n[SUCCESS] Optimal conditions saved to models/optimal_conditions.json")

if __name__ == "__main__":
    run_economic_optimization()