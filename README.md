# first_ml_proj

Overview
This project involves building a machine learning model to predict the sale price of machinery using historical data. The model is built using the XGBoost algorithm and utilizes grid search for hyperparameter tuning.

Requirements
Python 3.6 or higher
Libraries:
pandas
numpy
scikit-learn
xgboost
You can install the required libraries using pip:

bash
Copy code
pip install pandas numpy scikit-learn xgboost
Steps
Load the Data

***The data is loaded from a CSV. make sure to add the path to the csv file before running***

python
Copy code
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

df = pd.read_csv(r"your_data_path.csv", low_memory=False)
Remove Invalid Year Entries and Sample Data

We remove invalid entries and sample the data to reduce processing time.

python
Copy code
df = df[df['YearMade'] != 1000]
df = df.sample(20000)
Define the Preprocessing Function

The function handles date conversion, calculates the age of the machinery, and fills missing values.

python
Copy code
def pre_procces_data(df):
    df['saledate'] = pd.to_datetime(df['saledate'])
    df['saleYear'] = df['saledate'].dt.year
    df['age'] = df['saleYear'] - df['YearMade']
    mean_hours_per_age = df.groupby('age')['MachineHoursCurrentMeter'].transform('mean')
    df['MachineHoursCurrentMeter'] = df['MachineHoursCurrentMeter'].fillna(mean_hours_per_age)
    overall_mean_hours = df['MachineHoursCurrentMeter'].mean()
    df['MachineHoursCurrentMeter'].fillna(overall_mean_hours, inplace=True)
    df['MachineHoursCurrentMeter'] = df['MachineHoursCurrentMeter'].replace(0, overall_mean_hours)
    df['saleCount'] = df.groupby('MachineID').cumcount() + 1

    df = df.drop(['datasource', 'fiModelDesc', 'fiBaseModel', 'auctioneerID', 'saledate', 'fiSecondaryDesc',
                  'fiModelSeries', 'fiModelDescriptor', 'ProductGroupDesc',
                  'Drive_System', 'Enclosure', 'UsageBand', 'Forks', 'state', 'Pad_Type', 'Ride_Control',
                  'Stick', 'Transmission', 'Turbocharged', 'Blade_Extension',
                  'Blade_Width', 'Enclosure_Type', 'Hydraulics',
                  'Pushblock', 'Ripper', 'Scarifier', 'Tip_Control', 'Tire_Size',
                  'Coupler', 'Coupler_System', 'Grouser_Tracks', 'Hydraulics_Flow', 'Track_Type', 'Undercarriage_Pad_Width', 'Stick_Length', 'Thumb',
                  'Pattern_Changer', 'Grouser_Type', 'Backhoe_Mounting', 'Blade_Type',
                  'Travel_Controls', 'Steering_Controls', 'Differential_Type'], axis=1)

    # Handle categorical variables
    df = pd.get_dummies(df, columns=['ProductGroup', 'ProductSize', 'Engine_Horsepower', 'fiProductClassDesc'])

    df = df.fillna(0)
    return df

df = pre_procces_data(df)
Define Features and Target

We separate the features (X) and the target variable (y).

python
Copy code
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']
Split the Data

The data is split into training and testing sets.

python
Copy code
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Hyperparameter Tuning

We use GridSearchCV to find the best hyperparameters for the XGBoost model.

python
Copy code
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.1]
}

model = XGBRegressor()
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

best_model = grid_search.best_estimator_
Evaluate the Model

We evaluate the model using RMSE on the training and testing sets.

python
Copy code
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"Training RMSE: {train_rmse}")
print(f"Test RMSE: {test_rmse}")
Feature Importance

We print the importance of each feature.

python
Copy code
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("Feature importances:")
print(importance_df)
Running the Script
To run the script, execute the following command:

bash
Copy code
python val.py
Make sure that the Train.csv file is located in the correct directory and that you have installed all the necessary libraries.

This project is a simple example of how to build a machine learning model with basic preprocessing and hyperparameter tuning. It is not optimized for production use.