import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = 'C:/Users/2273581/Downloads/Energy_dataset.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(df.head())

# Define the features (X) and target (y)
X = df[['energy_consumption_kwh', 'peak_hours_usage', 'off_peak_usage', 'renewable_energy_pct', 'household_size', 'temperature_avg']]
y = df['billing_amount']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeRegressor(random_state=42)
}

# Train and evaluate each model
results = {}
for model_name, model in models.items():
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Store the results
    results[model_name] = {
        "Mean Squared Error": mse,
        "R-squared": r2
    }

# Display the results
print(results)

# Display the coefficients of the Linear Regression model
if "Linear Regression" in models:
    lr_model = models["Linear Regression"]
    coefficients = pd.DataFrame(lr_model.coef_, X.columns, columns=['Coefficient'])
    print(coefficients)

# Determine the best and worst model based on R-squared value
best_model = max(results, key=lambda x: results[x]['R-squared'])
worst_model = min(results, key=lambda x: results[x]['R-squared'])

print(f"The best model is: {best_model} with R-squared value of {results[best_model]['R-squared']}")
print(f"The worst model is: {worst_model} with R-squared value of {results[worst_model]['R-squared']}")
