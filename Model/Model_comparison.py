import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
import os

data = pd.read_excel(r'Original_data')

feature_names = data.columns[2:17]

# data standardization
scaler = MinMaxScaler(feature_range=(-1, 1))
data.loc[:, data.columns[2:17]] = scaler.fit_transform(data.loc[:, data.columns[2:17]])

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

# Performance results output
output_dir = r'Output'
os.makedirs(output_dir, exist_ok=True)
excel_file_path = os.path.join(output_dir, 'MLad_model_performance_resluts.xlsx')

performance_df = pd.DataFrame()

# Define the hyper-parameters search space
param_grids = {
    'RandomForest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [50, 100, 200],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5]
    },
    'ExtraTrees': {
        'n_estimators': [50, 100, 200],
        'max_depth': [50, 100, 200],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5]
    },
    'SVM': {
        'C': [0.1, 1, 10],
        'epsilon': [0.1, 1],
        'gamma': ['scale', 'auto'],
        'kernel': ['rbf', 'poly', 'sigmoid']
    },
    'ANN': {
        'hidden_layer_sizes': [(50,), (75,), (100,)],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate_init': [0.0001, 0.001, 0.01],
        'batch_size': [16, 32, 64],
        'activation': ['relu', 'tanh', 'logistic'],
        'solver': ['adam']
    },
    'XGBoost': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.2]
    }
}
                           
models = {
    'RandomForest': RandomForestRegressor(),
    'ExtraTrees': ExtraTreesRegressor(),
    'SVM': SVR(),
    'ANN': MLPRegressor(max_iter=10000),
    'XGBoost': XGBRegressor(objective='reg:squarederror')
}

# Perform 20 times training and hyper-parameter optimization
num_iterations = 20
for model_name, model in models.items():
    model_r2_scores = []
    model_rmse_scores = []
    model_mae_scores = []
    for _ in range(num_iterations):
        train_isotherms = unique_isotherms[train_indices]
        test_isotherms = unique_isotherms[test_indices]

        train_data = data[data['Isotherm'].isin(train_isotherms)]
        test_data = data[data['Isotherm'].isin(test_isotherms)]

        X_train = train_data.iloc[:, 1:17].values
        y_train = train_data.iloc[:, 17].values
        X_test = test_data.iloc[:, 1:17].values
        y_test = test_data.iloc[:, 17].values

        # Grid-search
        grid_search = GridSearchCV(model, param_grids[model_name], cv=5, scoring='r2', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Train the model with the best parameters and predict
        best_model = grid_search.best_estimator_
        y_pred_test = best_model.predict(X_test)
        r2 = r2_score(y_test, y_pred_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        mae = mean_absolute_error(y_test, y_pred_test)

        model_r2_scores.append(r2)
        model_rmse_scores.append(rmse)
        model_mae_scores.append(mae)

    performance_df[model_name + '_R2'] = model_r2_scores
    performance_df[model_name + '_RMSE'] = model_rmse_scores
    performance_df[model_name + '_MAE'] = model_mae_scores

performance_df.to_excel(excel_file_path, index=False, sheet_name='Optimized_Performance')
