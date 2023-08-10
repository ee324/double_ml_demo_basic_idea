from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load Diabetes Data
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = diabetes.target

# Consider 'bmi' as treatment
T = X['bmi']
X.drop('bmi', axis=1, inplace=True)

# Split the data into training and test datasets
X_train, X_test, y_train, y_test, T_train, T_test = train_test_split(X, y, T, test_size=0.2, random_state=0)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# First part of Double ML: train models to predict the treatment and the outcome
# Train model to predict treatment given covariates
treatment_model = LinearRegression().fit(X_train, T_train)

# Train model to predict outcome given covariates
outcome_model = LinearRegression().fit(X_train, y_train)

# Obtain residuals
T_resid_train = T_train - treatment_model.predict(X_train)
y_resid_train = y_train - outcome_model.predict(X_train)

# Second part of Double ML: estimate the effect of treatment on outcome using the residuals
effect_model = LinearRegression().fit(T_resid_train.values.reshape(-1, 1), y_resid_train)

# Now effect_model.coef_ gives the estimated causal effect of treatment on outcome
print("Estimated causal effect of treatment on outcome: ", effect_model.coef_[0])
