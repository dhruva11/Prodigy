import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the training dataset
train_df = pd.read_csv('train.csv')

# Select features and target for training
train_df['TotalBath'] = train_df['FullBath'] + 0.5 * train_df['HalfBath']
features = ['GrLivArea', 'BedroomAbvGr', 'TotalBath']
X_train = train_df[features]
y_train = train_df['SalePrice']

# Handle missing values in training data
X_train = X_train.fillna(X_train.mean())
y_train = y_train.fillna(y_train.mean())

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Load the test dataset
test_df = pd.read_csv('test.csv')

# Preprocess test data
test_df['TotalBath'] = test_df['FullBath'] + 0.5 * test_df['HalfBath']
X_test = test_df[features]

# Handle missing values in test data
X_test = X_test.fillna(X_test.mean())

# Make predictions
predictions = model.predict(X_test)

# Create submission DataFrame
submission = pd.DataFrame({
    'Id': test_df['Id'],
    'SalePrice': predictions
})

# Save predictions to CSV
submission.to_csv('submission.csv', index=False)
print("Predictions saved to submission.csv")

# Print model coefficients for reference
print("\nModel Coefficients:")
for feature, coef in zip(features, model.coef_):
    print(f"{feature}: {coef:.2f}")
print(f"Intercept: {model.intercept_:.2f}")

# Visualize distribution of predicted prices
plt.figure(figsize=(10, 6))
plt.hist(predictions, bins=30, edgecolor='black')
plt.xlabel('Predicted Sale Price ($)')
plt.ylabel('Frequency')
plt.title('Distribution of Predicted House Prices')
plt.tight_layout()
plt.show()