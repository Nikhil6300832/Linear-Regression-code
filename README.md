# 🏡 House Price Prediction using Linear Regression

## 📌 Step 1: Import Required Libraries


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    

## 📌 Step 2: Load the Dataset

file_path = r"C:\Users\Lenovo\Downloads\stock_price_data.csv"
df = pd.read_csv(file_path)
df.head()

## 📌 Step 3: Data Cleaning & Exploration


# Check for missing values
print(df.isnull().sum())

# Display basic dataset info
df.info()

# Display statistical summary
df.describe()
    

## 📌 Step 4: Feature Selection & Data Preprocessing


# Define Features (X) and Target (y)
X = df[['Open', 'High', 'Low', 'Volume', 'Sentiment']]
y = df['Close']

# Split into 80% Train, 20% Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Show the shapes of the training and testing data
X_train.shape, X_test.shape, y_train.shape, y_test.shape
    

## 📌 Step 5: Train the Linear Regression Model


# Create and Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Print Intercept & Coefficients
print("Intercept:", model.intercept_)
print("Coefficients:", dict(zip(X.columns, model.coef_)))
    

## 📌 Step 6: Make Predictions


# Predict on Test Data
y_pred = model.predict(X_test)

# Compare actual vs predicted values
results = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})
results.head()
    

## 📌 Step 7: Evaluate the Model


# Calculate Performance Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")
    

## 📌 Step 8: Visualize Predictions


plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Stock Prices')
plt.show()
    

## 📌 Step 9: Save the Model for Future Use


# Save the trained model to a file
joblib.dump(model, "house_price_model.pkl")

print("Model saved successfully as 'house_price_model.pkl'")
    

## 📌 Step 10: Load & Use the Model for New Predictions


# Load the saved model
loaded_model = joblib.load("house_price_model.pkl")

# Example new data for prediction
#new_house = np.array([[2000, 3, 2, 1, 10, 2]])  # Example: 2500 sqft, 3 beds, 2 baths, garage, 10 yrs old, suburban
# Load the new CSV file
file_path = r"C:\Users\Lenovo\Downloads\new_house_data.csv"
new_data = pd.read_csv(file_path)
new_data.head()

# Select the relevant features (ensure they match the columns used during training)
new_data_features = new_data[['Size_sqft', 'Bedrooms', 'Bathrooms', 'Garage', 'Age']]

# Ensure the new data is in the same format (numeric values for model predictions)
# If needed, convert categorical columns to numeric using label encoding or one-hot encoding.

# Use the loaded model to make predictions on the new data
predicted_prices = loaded_model.predict(new_data_features)

# Add the predictions as a new column in the original dataset
new_data['Predicted_Price'] = predicted_prices

# Display the new data with predicted prices
new_data.head()

# Save the results to a new CSV file
new_data.to_csv(r"C:\Users\Lenovo\Downloads\predicted_house_prices.csv", index=False)
