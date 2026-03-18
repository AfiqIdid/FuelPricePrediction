import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# 1. Load the new 2000-2022 data
df = pd.read_csv('D:/UIA/LAB 1 Python Intsllation with VSCODE-20250306/Machine Learning/Fuel/fuelconsumption2.csv')

# 2. Select Features and Target
features = ['ENGINE SIZE', 'CYLINDERS', 'YEAR', 'FUEL', 'VEHICLE CLASS']
target = 'COMB (L/100 km)'
ml_df = df[features + [target]].dropna()

# 3. Encoding
le_fuel = LabelEncoder()
ml_df['FUEL'] = le_fuel.fit_transform(ml_df['FUEL'])
le_class = LabelEncoder()
ml_df['VEHICLE CLASS'] = le_class.fit_transform(ml_df['VEHICLE CLASS'])

X = ml_df[features]
y = ml_df[target]

# 4. Split: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train the Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. CALCULATE ACCURACY METRICS
y_pred = model.predict(X_test)

# R-Squared: How well the model fits the data (1.0 is perfect)
r2 = r2_score(y_test, y_pred)

# MAE: On average, how many liters are we off by?
mae = mean_absolute_error(y_test, y_pred)

# RMSE: Similar to MAE, but penalizes large errors more
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# CUSTOM ACCURACY: % of predictions that are within 10% of the actual value
# This is a common way to explain regression "accuracy" to non-technical users
percentage_error = np.abs((y_test - y_pred) / y_test)
accuracy_within_10 = (percentage_error <= 0.10).mean() * 100

print("-" * 30)
print(f"📊 MODEL EVALUATION RESULTS")
print("-" * 30)
print(f"✅ R2 Score: {r2:.4f}")
print(f"📍 Mean Absolute Error: {mae:.2f} L/100km")
print(f"🎯 Accuracy (within 10% tolerance): {accuracy_within_10:.2f}%")
print("-" * 30)

# 7. SAVE THE BRAIN
joblib.dump(model, 'fuel_model.pkl')
joblib.dump(le_fuel, 'fuel_encoder.pkl')
joblib.dump(le_class, 'class_encoder.pkl')