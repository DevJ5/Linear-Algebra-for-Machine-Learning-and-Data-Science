import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Original data
X = np.array([[1, 5, 2], [2, 4, 2], [3, 3, 2], [4, 2, 2], [5, 1, 2]])
Y = np.array([10, 20, 30, 40, 50])

# Standardize the data
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_standardized)

# PCA components
components = pca.components_
print("PCA Components:\n", components)

# Use principal components in regression
regression = LinearRegression()
regression.fit(X_pca, Y)

# Coefficients of the regression model
coefficients = regression.coef_
print("Regression Coefficients:", coefficients)

# Calculate the contribution of each original feature to the target variable
feature_importance = np.abs(coefficients @ components)
feature_names = ["X1", "X2", "X3"]

# Create a DataFrame for better readability
importance_df = pd.DataFrame(
    {"Feature": feature_names, "Importance": feature_importance}
)
print("Feature Importance:\n", importance_df)
