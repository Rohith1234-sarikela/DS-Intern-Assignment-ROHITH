# DS-Intern-Assignment-ROHITH

Energy Consumption Prediction Project
Overview
This project aims to predict equipment energy consumption using a dataset containing environmental and operational features such as temperature, humidity, and atmospheric conditions across multiple zones. The model employs a RandomForestRegressor with preprocessing and feature selection to achieve accurate predictions.
Dataset
The dataset is sourced from a Google Drive URL and contains 16,857 rows and 29 columns, including:

Target Variable: equipment_energy_consumption (numeric, representing energy usage).
Features: Includes zone-specific temperature and humidity, outdoor conditions, and time-based features (hour, day of week, month, weekend indicator).
Challenges: Missing values, non-numeric entries ('error', 'unknown', '???'), and potential outliers.

Requirements
To run the project, install the required Python libraries:
pip install pandas numpy matplotlib seaborn scikit-learn

Project Structure

Data Loading: Loads the dataset from a Google Drive URL using pandas.
Preprocessing:
Converts timestamp to datetime and extracts features (hour, dayofweek, month, is_weekend).
Drops irrelevant columns (random_variable1, random_variable2).
Handles missing values and non-numeric entries by replacing 'error', 'unknown', and '???' with NaN.
Converts all features to numeric and fills missing values with column means.


Feature Engineering:
Splits data into features (X) and target (y).
Applies StandardScaler for scaling and SelectKBest for feature selection (selecting top 15 features based on f_regression).


Modeling:
Uses a RandomForestRegressor within a Pipeline.
Performs GridSearchCV to tune hyperparameters (n_estimators, max_depth, min_samples_split).


Evaluation:
Metrics: R² Score, Mean Squared Error (MSE), Mean Absolute Error (MAE), and Cross-Validation R² Score.
Visualizations: Actual vs. Predicted plot and Feature Importance bar plot.


Output:
Saves visualizations as actual_vs_predicted.png and feature_importances.png.



How to Run

Open in Colab: Copy the provided Jupyter notebook code into a Google Colab environment.
Install Dependencies: Run the first cell to install required libraries if not already installed.
Execute Cells: Run each cell sequentially to load data, preprocess, train the model, and evaluate results.
View Results: Check the console for performance metrics and view generated plots.

Results

Best Parameters: 
n_estimators: 100
max_depth: 10
min_samples_split: 2


Performance Metrics:
R² Score: ~0.048
Mean Squared Error: ~31,766
Mean Absolute Error: ~72.76
Cross-Validation R² Score: ~0.049


Key Features: The most important features include hour, zone6_humidity, zone3_temperature, and zone9_humidity.

Visualizations

Actual vs. Predicted: A scatter plot comparing actual and predicted energy consumption values.
Feature Importances: A bar plot showing the relative importance of selected features.

Notes

The low R² score (~0.048) suggests the model struggles to capture variance in the target variable. Potential improvements include:
Exploring additional features or feature interactions.
Trying other algorithms (e.g., Gradient Boosting, Neural Networks).
Addressing potential outliers or noise in the data.


The dataset contains missing values and non-numeric entries, which are handled through imputation and coercion to numeric types.
The project assumes access to a stable internet connection for loading the dataset from Google Drive.

Future Improvements

Feature Engineering: Add interaction terms or polynomial features.
Model Tuning: Expand the hyperparameter grid for GridSearchCV.
Data Quality: Investigate outliers and data inconsistencies.
Alternative Models: Test XGBoost, LightGBM, or deep learning approaches.


