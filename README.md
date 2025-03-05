# Fare Prediction

## Project Overview
# **Uber/Lyft Fare Prediction**

### This project predicts ride fares for Uber and Lyft rides based on historical ride data. The objective is to provide accurate predictions for fare prices, enabling insights into the factors influencing ride costs.

## Kaggle Link: 
https://www.kaggle.com/datasets/brllrb/uber-and-lyft-dataset-boston-ma 

## Dataset Overview-
General Information
Rows: 693,071

Columns: 57


## Key Features
### Target/Predictor Variable:
**price**: Continuous variable representing the fare amount. Missing values to be handled during preprocessing.

### Input Features:
Temporal: hour, day, month, datetime (can extract weekday).
Categorical: cab_type, source, destination, name, timezone.
Numerical: distance, surge_multiplier, temperature, humidity, windSpeed.
Weather-Related: precipIntensity, cloudCover, uvIndex, visibility.
Other Notable Features:
Geospatial: latitude, longitude for pickup and dropoff.
Redundant: visibility.1 (duplicated column), id (identifier).

## The dataset includes the following key columns:

## **1. Ride Information**  
Describes the ride details such as ID, source, destination, and ride type.

- **id**: Unique identifier for each ride.  
- **source**: Starting location of the ride.  
- **destination**: End location of the ride.  
- **cab_type**: Type of service (e.g., Lyft or Uber).  
- **product_id**: Ride product identifier.  
- **name**: Ride product name (e.g., Shared, Lux).  
- **price**: Cost of the ride.  
- **distance**: Trip distance in miles or kilometers.  
- **surge_multiplier**: Surge pricing multiplier.  

---

## **2. Temporal Data**  
Includes time-related information about the ride and relevant timestamps.

- **timestamp**: Unix timestamp of the ride request.  
- **datetime**: Human-readable format of the timestamp.  
- **hour, day, month**: Decomposed time features for easier analysis.  
- **timezone**: Timezone for the ride.  
- **temperatureHighTime, temperatureLowTime**: Times for high and low temperatures.  
- **sunriseTime, sunsetTime**: Sunrise and sunset times.  
- **windGustTime, uvIndexTime**: Times for maximum wind gust and UV index.  

---

## **3. Weather Conditions**  
Provides data on weather during the ride.

- **temperature**: Actual temperature during the ride.  
- **apparentTemperature**: Feels-like temperature.  
- **short_summary**: Brief weather description (e.g., "Partly Cloudy").  
- **long_summary**: Detailed weather description.  
- **precipIntensity, precipProbability**: Intensity and likelihood of precipitation.  
- **humidity**: Relative humidity as a percentage.  
- **visibility**: Visibility distance.  
- **icon**: Weather condition icon (e.g., "rain").  
- **dewPoint**: Temperature at which dew forms.  
- **cloudCover**: Cloud coverage percentage.  
- **uvIndex**: UV index level.  
- **ozone**: Ozone level in the atmosphere.  
- **pressure**: Atmospheric pressure.  

---

## **4. Temperature Extremes**  
Records daily high and low temperatures, along with their "feels-like" counterparts.

- **temperatureHigh, temperatureLow**: Daily maximum and minimum temperatures.  
- **apparentTemperatureHigh, apparentTemperatureLow**: Feels-like max and min temperatures.  
- **temperatureMin, temperatureMax**: Minimum and maximum temperatures of the day.  
- **apparentTemperatureMin, apparentTemperatureMax**: Feels-like min and max temperatures.  

---

## **5. Wind Conditions**  
Provides wind-related data during the ride.

- **windSpeed**: Wind speed during the ride.  
- **windGust**: Maximum observed wind speed.  
- **windBearing**: Direction of the wind in degrees.  

---

## **6. Additional Context**  
Provides additional atmospheric and temporal context.

- **moonPhase**: Phase of the moon.  
- **precipIntensityMax**: Maximum intensity of precipitation.  
- **visibility**: Distance at which objects can be clearly seen.  

## Data Lineage
The dataset is sourced from Kaggle and includes ride data for Uber and Lyft with fields for timestamps, source/destination locations, and ride details such as surge multipliers and distances.


#Checking for Outliers using Box-Plots

## Data Cleaning
- Converting the price column to a double (floating-point number) to ensure all values in this column are numeric and suitable for calculations
- Replacing missing values in price with the mean i.e. Imputation is done
- Removing the row with the maximum price as part of outlier removal
- Dropping rows with null values in specific columns


## Data Transformation
- Converting datetime to a Spark Timestamp for consistentcy
- Extracted day from the datetime column for better insights
- Converted categorical variables to one-hot encoding for modeling
- Adding a binary column of is_weekend to help identify differences in behavior between weekdays and weekends


### Reasons for Dropping Unused Columns

1. **Redundancy with `datetime`:**
   - Columns: `timestamp`, `hour`, `day`, `month`
   - Reason: These are already encapsulated in the `datetime` column, making them redundant for analysis.

2. **Specific Identifiers/Metadata:**
   - Columns: `id`, `timezone`, `source`, `destination`, `product_id`, `name`
   - Reason: These columns provide granular details (e.g., unique identifiers, source/destination specifics) that are unnecessary when focusing on higher-level attributes like `distance` or `cab_type`.

3. **Irrelevant Weather Details:**
   - Columns: `precipIntensity`, `precipProbability`, `windGust`, `windGustTime`, `temperatureHigh`, `temperatureHighTime`, `temperatureLow`, `temperatureLowTime`, `apparentTemperatureHigh`, `apparentTemperatureHighTime`, `apparentTemperatureLow`, `apparentTemperatureLowTime`, `icon`, `dewPoint`, `pressure`, `windBearing`, `cloudCover`, `uvIndex`, `ozone`, `sunriseTime`, `sunsetTime`, `moonPhase`, `precipIntensityMax`, `uvIndexTime`, `temperatureMin`, `temperatureMinTime`, `temperatureMax`, `temperatureMaxTime`, `apparentTemperatureMin`, `apparentTemperatureMinTime`, `apparentTemperatureMax`, `apparentTemperatureMaxTime`
   - Reason: These columns detail fine-grained or extreme weather conditions that may not meaningfully affect high-level cab pricing and distance analysis.

4. **Duplicate/Unnecessary Metrics:**
   - Columns: `visibility.1`
   - Reason: This duplicates the `visibility` column and offers no added value.


## Data Preprocessing
- Converting categorical columns to numericals using StringIndexer (a PySpark transformation that maps string values to numerical indices)


## Feature Engineering/Preparing Data for Model training using VectorAssembler
- The dataset is split into training and test sets, and a VectorAssembler is used to combine input feature columns into a single features vector. A Pipeline is initialized with the assembler as its base stage for streamlined transformations.

## Pipeline based Regression Model Implementation using Linear Regression, Random Forest and GBT
- Initialize Models: Three regression models — Linear Regression, Random Forest Regressor, and Gradient Boosted Trees Regressor — are initialized with the label column set to "price" and the features column set to "features."
- Define Pipelines: Separate pipelines are created for each model by adding the respective regression model to the base pipeline, enabling streamlined processing for each modeling approach.

## Multi-model Hyperparameter Tuning Using ParamGridBuilder for Model Optimization
- Build Parameter Grids: Parameter grids are created for each regression model (Linear Regression, Random Forest, and Gradient Boosted Trees) to test various hyperparameter combinations such as regularization, number of iterations, number of trees, and tree depth.
- Merge Parameter Grids: The individual parameter grids for each model are combined into one comprehensive grid, allowing cross-validation to explore all hyperparameter options across the models.

## Cross Validator to find best performing model using pipeline
- Cross-Validation Setup: A CrossValidator is initialized to perform hyperparameter tuning using a specified pipeline, parameter grid, and evaluation metric (R²). 
- R² (Coefficient of Determination) measures how well the model's predictions match the actual prices. The data is split into 3 folds for cross-validation to ensure that the model is evaluated on multiple subsets of the training data, helping to prevent overfitting.
- Model Fitting: The CrossValidator is applied to the training data to find the best model by fitting it across all parameter combinations defined in the parameter grid.
- The best model from cross-validation (cvModel.bestModel) is selected, and its hyperparameters are displayed using the extractParamMap() method.

## Best model selection & evaluation using R2, RMSE & MAE
- Best Model Selection: The Gradient Boosted Model (GBM) is selected as the best model based on cross-validation.
- Model Testing and Evaluation: The best model is tested on the test data, and its R², RMSE and MAE score is calculated using a RegressionEvaluator. 

# Prediction Results
- The model predicts ride fares with an R² of 0.95 on the test dataset, indicating a strong relationship between the features and the target variable. 
- Additionally, the model has a Root Mean Squared Error (RMSE) of 1.92 and a Mean Absolute Error (MAE) of 1.28, suggesting that, on average, the model's predictions deviate from the actual fares by approximately $1.92 (RMSE) and $1.28 (MAE). These values indicate that the model performs with good accuracy and precision.


# Plot of Actual vs Predicted Prices

## Summary
The **Gradient Boosted Model (GBM)** achieved an **R² of 0.95**, demonstrating strong performance in predicting Uber/Lyft Fare prices.

### - Graph Analysis:
The graph represents the relationship between Actual Prices (x-axis) and Predicted Prices (y-axis) for ride fares using the Gradient Boosted model. The red dashed line indicates a perfect 1:1 correlation (ideal predictions), while the blue dots represent the individual data points.

The model is performing well, as indicated by the centered and symmetric residuals with a narrow spread.

### - Key Observations:
The majority of the points are tightly clustered around the red line, indicating high prediction accuracy.
There are slight deviations for higher price values, where the model occasionally underestimates or overestimates, but these deviations are minimal.

### - Metrics Interpretation:
- **R-squared (R²)**: 0.95
The model explains 95% of the variance in Uber prices, showing strong predictive ability.

- **Root Mean Squared Error (RMSE)**: 1.92
On average, the predictions deviate from actual prices by about $1.92, indicating that the model’s predictions have some variance but are still relatively close to the actual values.

- **Mean Absolute Error (MAE)**: 1.28
The average absolute prediction error is $1.28, showing that, on average, the model’s predictions are off by about $1.28, which is a reasonable level of error for price prediction.

Overall, the Gradient Boosted model effectively predicts ride fares with minimal errors and significant accuracy, making it a reliable choice for fare estimation.

### 1. Extraordinary visualizations using Tableau
- Tableau dashboard is published to below link - https://public.tableau.com/views/FarePredictionProject/DataAnalysisDashboard?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link 


- **Geographic Analysis of Rides**: A map visualization shows ride locations categorized by price ranges (e.g., $0-$10, $10-$20, etc.).
-  **Ride Trends Over Time**: A line chart displays the number of trips for Uber and Lyft across the days of the week.
- **Price vs. Distance**: A histogram illustrates the relationship between ride price and distance traveled.
- **Peak Ride Hours**: A line chart highlights the busiest ride hours during the day based on the number of trips.
  
This dashboard enables quick comparisons of ride patterns and pricing for Uber and Lyft.

#### 1. **Data Preprocessing (with PySpark)**:
- **Categorical Data Handling**: 
   - The columns `source`, `destination`, `cab_type`, and `name` are categorical, so we use **`StringIndexer`** to convert them into numerical values.
- **Outlier Removal**: 
   - Outliers in the `price` column are removed based on the **1st and 99th percentiles** using `approxQuantile`.
- **Feature Engineering**: 
   - Various features like `hour`, `month`, and others are combined into a single vector using **`VectorAssembler`**.
   - The target variable (`price`) is **log-transformed** using **log** to reduce skewness and make the data more suitable for modeling.

#### 2. **Model Definition (PyTorch)**:
- The model is defined as a **feed-forward neural network (FFNN)** with three layers:
   - **Input layer**: Size based on the number of features (input dimension).
   - **Hidden layer 1**: 32 neurons.
   - **Hidden layer 2**: 16 neurons.
   - **Output layer**: 1 neuron for regression output.
- **ReLU activation** is used in the hidden layers to introduce non-linearity and help the model learn complex patterns.

#### 3. **Training**:
- **Mean Squared Error (MSE)** is used as the loss function since this is a regression problem.
- The model is trained using the **Adam optimizer**, which is an adaptive learning rate optimizer.
- **Early stopping** is implemented to halt training if the loss does not improve for a specified number of epochs.

#### 4. **Evaluation**:
- After training, the model is evaluated on the test data.
- The performance metrics used are:
   - **RMSE (Root Mean Squared Error)**: This is the average error between the predicted and actual values.
   - **R2 Score**: This indicates how well the model explains the variance in the target variable. An R2 score close to 1 means the model is a good fit.

### Results:

The loss consistently decreased with each epoch, showing that the model was learning and improving over time. This gradual reduction in loss indicates effective optimization of the model's parameters.

After training, the model's performance is:
- **RMSE**: 0.1600 (average error in predictions).
- **R2 Score**: 0.9145 (indicating that the model explains 91.45% of the variance in the target variable).

The model's high **R2 score** and low **RMSE** suggest that it is a strong predictor of the target variable, and its performance can be considered quite satisfactory for this regression task.



### 2. Create ML app from PySpark

### Flask for Machine Learning Applications

**Flask** is a lightweight web framework in Python used to build APIs and web applications. For machine learning, Flask allows to deploy trained models as RESTful APIs, enabling applications to send data for predictions and receive results in real-time.

---

##### Steps:

- **Load Pre-trained Model**:
  - A `PipelineModel` is loaded from the path `file:/databricks/driver/rideshare_model` to handle both data transformation and prediction.

- **Set Up Flask Application**:
  - A Flask app is initialized to serve as the API framework.

- **Define `/predict` Endpoint**:
  - Accepts **POST** requests with input JSON data.
  - Converts the input JSON into a PySpark DataFrame.
  - Applies the loaded model to generate predictions.
  - Returns predictions in JSON format or error messages for failed requests.

- **Error Handling**:
  - Captures and returns errors as JSON responses with an HTTP 500 status code.

- **Start Flask Server**:
  - Runs the API on `host="0.0.0.0"` (external access) and `port=5000`.
  - Disables Flask’s automatic reloader for Databricks compatibility.

- **API Workflow**:
  - **Input**: Accepts feature data in JSON format.
  - **Output**: Returns predicted values as a JSON response.

This setup integrates **PySpark** for scalable model predictions and **Flask** for API deployment, making it suitable for production environments like Databricks.


### Accessing the API Endpoint and Sending a Prediction Request

On executing above Flask code, it starts a server and generates a URL (e.g., `http://10.172.163.225:5000/predict`) where the model API can be accessed. This URL allows you to send data for predictions.

---

#### Method 1: Using the API with `curl` in Databricks

To test the API, you can use the following `curl` command in a new Databricks notebook cell:

```bash
%sh
curl -X POST http://10.172.163.225:5000/predict \
-H "Content-Type: application/json" \
-d '{
    "hour": 10,
    "month": 5,
    "source_index": 1,
    "destination_index": 3,
    "cab_type_index": 0,
    "name_index": 2,
    "distance": 4.5,
    "surge_multiplier": 1.2,
    "day": 15,
    "is_weekend": 0
}'
```


#### Expected Output:
When the above command is executed, the API will return a JSON response with the predicted value. For example:
{
    "prediction": 15.67
}





#### Method 2: Using Python `requests` Library in Databricks

Alternatively, we can test the API programmatically using the Python `requests` library within a Databricks notebook. Below is the code to interact with the API:

```python
import requests
import json

# Define the API endpoint
url = "http://10.172.163.225:5000/predict"

# Define the test input data
data = {
    "hour": 10,
    "month": 5,
    "source_index": 1,
    "destination_index": 3,
    "cab_type_index": 0,
    "name_index": 2,
    "distance": 4.5,
    "surge_multiplier": 1.2,
    "day": 15,
    "is_weekend": 0
}

# Make the POST request
response = requests.post(url, json=data)

# Print the response
print(response.json())
````

#### Expected Output:
On executing the code, the API will return a JSON response similar to:
{
    "prediction": 15.67
}



### **2. Docker (GCP)**
The following document has the screenshots showing the results from running the project in GCP using GKE

https://docs.google.com/document/d/1vNjiwCT2coxZ1FLp14fde4uebVbvDETzWO6qeHozjTY/edit?usp=sharing


GitHub Link: https://github.com/arjunghosh4/spark/tree/main/k8spark-1

## Installation & Setup
This project runs on **Databricks**. To set up and run the notebook, follow these steps:

1. Upload the notebook to your Databricks workspace.
2. Ensure you have access to the necessary data sources (e.g., ride fare datasets).
3. Install dependencies using:
   ```python
   %pip install pandas scikit-learn numpy matplotlib seaborn
   ```
4. Run the cells sequentially to preprocess data, train models, and evaluate performance.

## Features
- Data preprocessing for Uber/Lyft ride fare datasets.
- Feature engineering to extract key ride attributes.
- Machine learning models for fare price prediction.
- Data visualization and model evaluation.

## Usage
1. Load the dataset into the workspace.
2. Execute the notebook step by step.
3. Evaluate model performance and analyze predictions.

## Dependencies
- Python 3.x
- Databricks environment
- Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn

## Author
- Arjun Ghosh

## License
This project is intended for educational and research purposes.

