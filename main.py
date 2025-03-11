import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load the trained K-Means model
with open("customer_segmentation_model.pkl", "rb") as file:
    kmeans = pickle.load(file)

# Load dataset for visualization
data = pd.read_csv("Mall_Customers.csv")

# Convert Gender to numeric
data["Gender"] = data["Gender"].map({"Male": 1, "Female": 0})

# Selecting relevant features
features = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Define segment names
segment_names = {
    0: "Budget-Conscious Shoppers",
    1: "Luxury Shoppers",
    2: "Careful Investors",
    3: "Impulse Buyers",
    4: "Balanced Buyers"
}

# Streamlit UI
st.title("🛍 Customer Segmentation App")
st.write("Enter customer details to predict their shopping behavior segment.")

# User Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", min_value=18, max_value=70, value=30)
annual_income = st.number_input("Annual Income (k$)", min_value=0, value=50)
spending_score = st.slider("Spending Score (1-100)", min_value=1, max_value=100, value=50)

# Convert categorical inputs
gender_numeric = 1 if gender == "Male" else 0

# Prepare input data for prediction
input_data = np.array([age, annual_income, spending_score]).reshape(1, -1)
scaled_input = scaler.transform(input_data)

# Predict customer segment
if st.button("Predict Segment"):
    cluster = kmeans.predict(scaled_input)[0]
    segment = segment_names.get(cluster, "Unknown Segment")
    st.success(f"The customer belongs to **{segment}**")

# Cluster Visualization
st.subheader("📊 Customer Segmentation Visualization")

plt.figure(figsize=(8, 6))
sns.scatterplot(x=data["Annual Income (k$)"], y=data["Spending Score (1-100)"], hue=kmeans.labels_, palette="viridis", s=100)
plt.scatter(annual_income, spending_score, color="red", marker="X", s=200, label="New Customer")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segmentation Clusters")
plt.legend(title="Segment")
st.pyplot(plt)
