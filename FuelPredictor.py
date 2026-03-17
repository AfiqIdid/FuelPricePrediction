import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib # Used to save the model

# 1. Load the data
df = pd.read_csv('D:/UIA/LAB 1 Python Intsllation with VSCODE-20250306/Machine Learning/fuel.csv')

# 2. Select the "Features" (X) and the "Target" (y)
features = ['engine_displacement', 'engine_cylinders', 'year', 'fuel_type', 'class']
target = 'combined_mpg_ft1'

# Clean data: Remove rows with missing values
ml_df = df[features + [target]].dropna()

# 3. Encoding (Convert Text to Numbers)
le_fuel = LabelEncoder()
ml_df['fuel_type'] = le_fuel.fit_transform(ml_df['fuel_type'])

le_class = LabelEncoder()
ml_df['class'] = le_class.fit_transform(ml_df['class'])

X = ml_df[features]
y = ml_df[target]

# 4. Split: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train the "Random Forest"
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# 6. Check the score
y_pred = model.predict(X_test)
print(f"Model R2 Score: {r2_score(y_test, y_pred):.4f}")

# 7. SAVE THE BRAIN (Crucial step!)
joblib.dump(model, 'fuel_model.pkl')
joblib.dump(le_fuel, 'fuel_encoder.pkl')
joblib.dump(le_class, 'class_encoder.pkl')

print("Model and Encoders saved successfully!")