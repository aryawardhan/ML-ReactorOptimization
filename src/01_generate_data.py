# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate 1,500 synthetic experimental runs
n_samples = 1500

# 1. Generate independent process variables within typical operational ranges
temperature = np.random.uniform(120, 220, n_samples)      # in °C
pressure = np.random.uniform(10, 50, n_samples)            # in bar
catalyst_loading = np.random.uniform(0.1, 2.5, n_samples)  # in mol%
residence_time = np.random.uniform(0.5, 6.0, n_samples)    # in hours

# 2. Define a physics-grounded underlying function for Yield (%)
T_k = temperature + 273.15
Ea_R = 4000  # Activation energy term
k = np.exp(-Ea_R / T_k) * 150  # Rate constant behavior

# Calculate yield with interacting non-linear terms
yield_ideal = (
    (k * residence_time * 15) +
    (pressure * 0.4) +
    (catalyst_loading * 12) -
    (0.003 * (temperature - 180)**2) -  # Thermal degradation peak at 180°C
    (2.5 * (catalyst_loading - 1.5)**2)  # Catalyst saturation/degradation peak at 1.5 mol%
)

# Normalize and add realistic experimental noise
yield_actual = yield_ideal + np.random.normal(0, 3, n_samples)
yield_actual = np.clip(yield_actual, 0, 100)

# 3. Assemble and save
df_synthetic = pd.DataFrame({
    'Temperature': temperature,
    'Pressure': pressure,
    'Catalyst_Loading': catalyst_loading,
    'Residence_Time': residence_time,
    'Yield': yield_actual
})

import os
os.makedirs('data', exist_ok=True)
df_synthetic.to_csv('data/reaction_data.csv', index=False)
print("Step 1 Complete: Successfully generated 'reaction_data.csv'")