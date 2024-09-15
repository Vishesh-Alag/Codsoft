import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score


# Load the data
data = pd.read_csv("advertising.csv")

# EDA
print("First 5 Rows of Data:\n", data.head())
print("\nDatatypes of Data:\n", data.dtypes)
print("\nShape of Data:\n", data.shape)
print("\nCheck if Null Values are Present in Data or Not:\n", data.isnull().sum())
print("\nCheck if any Duplicate Values are Present in Data or Not:\n", data.duplicated().sum())
print(data.info())
print("\nSummary Statistics:\n", data.describe().T)

# Visualizing Distributions of the Features
plt.figure(figsize=(14, 8))
plt.subplot(2, 2, 1)
sns.histplot(data['TV'], kde=True)
plt.title('TV Advertising Distribution')

plt.subplot(2, 2, 2)
sns.histplot(data['Radio'], kde=True)
plt.title('Radio Advertising Distribution')

plt.subplot(2, 2, 3)
sns.histplot(data['Newspaper'], kde=True)
plt.title('Newspaper Advertising Distribution')

plt.subplot(2, 2, 4)
sns.histplot(data['Sales'], kde=True)
plt.title('Sales Distribution')

plt.tight_layout()
plt.show()

# Visualizing Relationships
sns.pairplot(data)
plt.show()

# Correlation Matrix
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Checking for Outliers using Boxplots
plt.figure(figsize=(14, 8))
plt.subplot(2, 2, 1)
sns.boxplot(x=data['TV'])
plt.title('TV Advertising')

plt.subplot(2, 2, 2)
sns.boxplot(x=data['Radio'])
plt.title('Radio Advertising')

plt.subplot(2, 2, 3)
sns.boxplot(x=data['Newspaper'])
plt.title('Newspaper Advertising')

plt.subplot(2, 2, 4)
sns.boxplot(x=data['Sales'])
plt.title('Sales')

plt.tight_layout()
plt.show()

# through EDA it is clearly visible that there is no considerable amount of outliers present.
# also there is a High positive correlation between TV and Sales , A moderate positive correlation between Radio and Sales
# and slightly negative correlation between Newspaper and Sales, suggesting that more newspaper advertising might not significantly boost sales.

# ML Model Building 
# We Choose TV as Independent Variable and Sales as Target Variable


# Prepare the data
X = data[['TV']]
y = data['Sales']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1),
    "Decision Tree Regression": DecisionTreeRegressor(random_state=42),
    "Random Forest Regression": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting Regression": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# Fit models and make predictions
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Print model performance
    print(f"\n{name}:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R^2 Score: {r2:.4f}")

    # Create a DataFrame with actual values, predictions, and residuals
    comparison_df = pd.DataFrame({
        'TV': X_test.values.flatten(),  # Convert to numpy array and flatten
        'Actual': y_test.values,        # Convert to numpy array
        'Predicted': y_pred,
        'Residual': y_test.values - y_pred
    })
    
    # Print all 40 testing data values
    print(f"\nComparison of Sales-Actual vs Sales-Predicted for {name}:")
    print(comparison_df)  

    # Plot Actual vs Predicted values
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Sales')
    plt.ylabel('Predicted Sales')
    plt.title(f'{name}: Actual vs. Predicted Sales')
    plt.show()















