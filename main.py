
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

def generate_data(n_points, a, b, noise_level):
    """Generates synthetic data for linear regression."""
    x = np.linspace(0, 10, n_points)
    true_y = a * x + b
    noise = np.random.normal(0, noise_level, n_points)
    observed_y = true_y + noise
    return x, observed_y

def run():
    st.title("Simple Linear Regression with CRISP-DM")

    st.sidebar.header("Parameters")
    a = st.sidebar.slider("Slope (a)", -10.0, 10.0, 2.0, 0.1)
    b = st.sidebar.slider("Y-intercept (b)", -10.0, 10.0, 1.0, 0.1)
    noise_level = st.sidebar.slider("Noise Level", 0.0, 10.0, 1.0, 0.1)
    n_points = st.sidebar.slider("Number of data points", 10, 1000, 100, 10)

    x, observed_y = generate_data(n_points, a, b, noise_level)

    X = x.reshape(-1, 1)
    y = observed_y

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    learned_a = model.coef_[0]
    learned_b = model.intercept_

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.header("Data Visualization")
    st.line_chart(pd.DataFrame({
        'Observed Data': observed_y,
        'True Line': a * x + b,
        'Regression Line': model.predict(X)
    }, index=x))

    st.header("Model Evaluation")
    st.write(f"Original slope (a): {a}")
    st.write(f"Learned slope (a): {learned_a:.4f}")
    st.write(f"Original y-intercept (b): {b}")
    st.write(f"Learned y-intercept (b): {learned_b:.4f}")
    st.write(f"Mean Squared Error: {mse:.4f}")
    st.write(f"R-squared: {r2:.4f}")

if __name__ == '__main__':
    run()
