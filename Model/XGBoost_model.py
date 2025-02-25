import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
import joblib
import random
import os

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

data = pd.read_excel(r'Original_data')

feature_names = data.columns[2:17]

# data standardization
scaler = MinMaxScaler(feature_range=(-1, 1))
data.loc[:, data.columns[2:17]] = scaler.fit_transform(data.loc[:, data.columns[2:17]])
scaler_filename = 'scaler.pkl'
joblib.dump(scaler, scaler_filename)

# input and output
X = data.iloc[:, 2:17].values
y = data.iloc[:, 17].values

# data splitting based on Isotherm
unique_isotherms = data['Isotherm'].unique()
train_indices = []
test_indices = []

num_isotherms_test = int(0.1 * len(unique_isotherms))  # 10% for test-set

test_isotherms = np.random.choice(unique_isotherms, size=num_isotherms_test, replace=False)
remaining_isotherms = np.setdiff1d(unique_isotherms, test_isotherms)

for isotherm in unique_isotherms:
    indices = data[data['Isotherm'] == isotherm].index

    if isotherm in test_isotherms:
        test_indices.extend(indices)
    else:
        train_indices.extend(indices)

X_test = X[test_indices]
y_test = y[test_indices]
X_train = X[train_indices]
y_train = y[train_indices]

# model construction
model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, objective='reg:squarederror')

kf = KFold(n_splits=5, shuffle=True)
y_pred = cross_val_predict(model, X_train, y_train, cv=kf)

# model evaluation
rmse = np.sqrt(mean_squared_error(y_train, y_pred))
mae = mean_absolute_error(y_train, y_pred)
r2 = r2_score(y_train, y_pred)

print("XGBoost training results:")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R2 Score: {r2:.4f}")

model.fit(X_train, y_train)

y_pred_test = model.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print("XGBoost testing results:")
print(f"RMSE: {rmse_test:.4f}")
print(f"MAE: {mae_test:.4f}")
print(f"R2 Score: {r2_test:.4f}")

# Save the model
model_filename = 'PWM_XGBoost.joblib'
joblib.dump(model, model_filename)

# LogKd prediction based on Standard AC

pollutant_properties_df = pd.read_excel(r'Original data', sheet_name='Prediction_Stadard_AC')

loaded_scaler = joblib.load(scaler_filename)
pollutant_properties_df.loc[:, pollutant_properties_df.columns[1:17]] = loaded_scaler.transform(pollutant_properties_df.loc[:, pollutant_properties_df.columns[1:17]])

X_pollutants = pollutant_properties_df.iloc[:, 1:17].values

loaded_model = joblib.load(model_filename)

target_predictions = loaded_model.predict(X_pollutants)

pollutant_names = pollutant_properties_df.iloc[:, 0].values

predictions_df = pd.DataFrame({
    'Pollutant': pollutant_names,
    'XGBoost Predicted logKd': target_predictions
})

output_folder = r"XGBoost_logKd_prediction"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

predictions_df.to_excel(os.path.join(output_folder, 'XGBoost_Standard_AC_predictions.xlsx'), index=False)

