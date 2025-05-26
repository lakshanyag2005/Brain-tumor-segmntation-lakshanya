import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from geopy.distance import geodesic
import joblib

# Load data
df = pd.read_csv("amazon_delivery.csv")

# Clean data
df = df.dropna(subset=['Delivery_Time'])
df = df.drop_duplicates()

# Calculate distance
def calculate_distance(row):
    store_coords = (row['Store_Latitude'], row['Store_Longitude'])
    drop_coords = (row['Drop_Latitude'], row['Drop_Longitude'])
    return geodesic(store_coords, drop_coords).km

df['Distance'] = df.apply(calculate_distance, axis=1)

print("Sample Distances vs Delivery Times:")
print(df[['Distance', 'Delivery_Time']].sort_values(by='Distance', ascending=False).head(5))
# Remove unrealistic delivery times
df = df[df['Delivery_Time'] < 30]

# Define features and target
X = df[['Distance', 'Weather', 'Traffic']]
y = df['Delivery_Time']

# Preprocessing: encode categorical features
categorical_cols = ['Weather', 'Traffic']
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
], remainder='passthrough')  # keep Distance

# Define model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {mse:.2f}")

# Save model
joblib.dump(model, 'best_model.pkl')