import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Streamlit App Title
st.title("Shop Sale Analysis and Prediction")

# File Uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    # Load the dataset
    df = pd.read_csv(uploaded_file)
    
    # Check if required columns exist
    if "Daily_Customer_Count" in df.columns and "Store_Sales" in df.columns:
        
        st.write("### Dataset Preview")
        st.dataframe(df.head())

        # Handling Missing Data
        if df.isnull().sum().sum() > 0:
            st.warning("Dataset contains missing values. Consider cleaning it before analysis.")
            df = df.dropna()  # Drop missing values for simplicity
        
        # Summary Statistics
        st.write("### Summary Statistics")
        st.write(df.describe())

        # Visualization: Pairplot
        st.write("### Pairplot of Features")
        try:
            fig = sns.pairplot(df)
            st.pyplot(fig.figure)
        except Exception as e:
            st.error(f"Pairplot could not be generated: {e}")

        # Visualization: Store Sales Analysis
        st.write("### Store Sales Analysis")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.scatterplot(x=df["Daily_Customer_Count"], y=df["Store_Sales"], ax=ax)
        plt.xlabel("Daily Customer Count")
        plt.ylabel("Store Sales")
        st.pyplot(fig)

        # **Sales Prediction Model**
        st.write("## Train Sales Prediction Model")
        
        # Prepare data
        X = df[["Daily_Customer_Count"]]  # Feature
        y = df["Store_Sales"]  # Target
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Model Evaluation
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5

        st.write(f"**Model Performance:**")
        st.write(f"- Mean Absolute Error (MAE): {mae:.2f}")
        st.write(f"- Root Mean Squared Error (RMSE): {rmse:.2f}")
        
        # **Prediction Input**
        st.write("## Predict Sales")
        user_input = st.number_input("Enter Daily Customer Count", min_value=0, value=50, step=1)
        
        if st.button("Predict Sales"):
            predicted_sales = model.predict([[user_input]])[0]
            st.success(f"Predicted Store Sales: ${predicted_sales:.2f}")

    else:
        st.error("Required columns 'Daily_Customer_Count' and 'Store_Sales' not found in the dataset.")
