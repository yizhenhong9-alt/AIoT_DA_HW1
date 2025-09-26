# **Project Report: Simple Linear Regression with CRISP-DM**

## **Author**

*   **Email:** yizhenhong9@gmail.com

## **1. Business Understanding**

The primary objective of this project is to develop a simple linear regression model to analyze and visualize the linear relationship between an independent variable `x` and a dependent variable `y`. The relationship is defined by the equation `y = ax + b`.

The project will be developed as an interactive tool, allowing users to modify key parameters:
*   **Slope (`a`) and Y-intercept (`b`):** To understand their effect on the regression line.
*   **Noise:** To simulate real-world data imperfections.
*   **Number of data points:** To see how sample size influences the model's accuracy.

This interactive application will serve as an educational tool for understanding the fundamentals of linear regression. The project will adhere to the Cross-Industry Standard Process for Data Mining (CRISP-DM) framework.

## **2. Data Understanding**

The data for this project will be synthetically generated, providing a controlled environment for experimentation. This allows us to know the "true" underlying relationship and evaluate the model's ability to uncover it.

### **Data Generation Process**

The data is generated using the following steps:
1.  Generate `x` values within a specified range.
2.  Calculate the "true" `y` values using `y = ax + b`.
3.  Add random noise to the `y` values to create the observed data.

Here is a Python snippet demonstrating the data generation:

```python
import numpy as np

def generate_data(n_points, a, b, noise_level):
    """Generates synthetic data for linear regression."""
    x = np.linspace(0, 10, n_points)
    true_y = a * x + b
    noise = np.random.normal(0, noise_level, n_points)
    observed_y = true_y + noise
    return x, observed_y
```

## **3. Data Preparation**

Before feeding the data to the model, it needs to be prepared. This involves structuring the data and splitting it into training and testing sets.

*   **Structuring Data:** We will use pandas DataFrames to manage the data.
*   **Train-Test Split:** The dataset will be split into a training set (to train the model) and a testing set (to evaluate its performance). A common split ratio is 80% for training and 20% for testing.

```python
from sklearn.model_selection import train_test_split
import pandas as pd

# Assuming 'x' and 'observed_y' are from the previous step
X = x.reshape(-1, 1) # scikit-learn expects a 2D array for features
y = observed_y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## **4. Modeling**

A simple linear regression model will be implemented using Python's `scikit-learn` library. This library provides an efficient and well-documented implementation of the algorithm.

### **Model Training**

The model is trained on the `X_train` and `y_train` datasets. The goal of the training process is to find the optimal values for the model's coefficients (slope and intercept) that minimize the difference between the predicted and actual `y` values.

```python
from sklearn.linear_model import LinearRegression

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Get the learned coefficients
learned_a = model.coef_[0]
learned_b = model.intercept_
```

## **5. Evaluation**

The performance of the trained model is evaluated on the unseen test data (`X_test` and `y_test`). This helps to assess how well the model generalizes to new data.

### **Evaluation Metrics**

*   **R-squared (R²):** The coefficient of determination. It represents the proportion of the variance in the dependent variable that is predictable from the independent variable. An R² of 1 indicates a perfect fit.
*   **Mean Squared Error (MSE):** The average of the squared differences between the predicted and actual values. A lower MSE indicates a better fit.

```python
from sklearn.metrics import mean_squared_error, r2_score

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
```

## **6. Deployment**

The final step is to deploy the model as an interactive web application. This will allow users to easily interact with the model without needing to run the code themselves.

### **Deployment Options**

*   **Streamlit:** A Python library that makes it easy to create and share custom web apps for machine learning and data science. It's known for its simplicity and fast development cycle.
*   **Flask:** A micro web framework for Python. It is more flexible than Streamlit but requires more boilerplate code for setting up the application.

The web application will feature a user interface with sliders or input boxes for adjusting the data generation parameters (`a`, `b`, noise, number of points). The results, including the regression plot and evaluation metrics, will be displayed dynamically.

## **7. How to Run**

1.  Install the required libraries:
    ```bash
    pip install numpy pandas scikit-learn streamlit
    ```
2.  Run the Streamlit application:
    ```bash
    streamlit run main.py
    ```
