import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df = pd.read_csv("dataset.csv")

brand_encoder = LabelEncoder()
condition_encoder = LabelEncoder()
os_encoder = LabelEncoder()

df["Brand"] = brand_encoder.fit_transform(df["Brand"])
df["Condition"] = condition_encoder.fit_transform(df["Condition"])
df["OS"] = os_encoder.fit_transform(df["OS"])

X = df.drop("Resale_Price_INR", axis=1)
y = df["Resale_Price_INR"]


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = r2_score(y_test, y_pred)

st.set_page_config(
    page_title="Used Laptop Resale Price Predictor",
    page_icon="💻"
)

st.title("💻 Used Laptop Resale Price Predictor")

st.markdown(
    """
    Predict the resale value of your laptop.
    """
)
st.write(f"### Model R² Score: {accuracy:.2f}")
st.write(f"Model Accuracy: {accuracy*100:.2f}%")
st.divider()

brand = st.selectbox(
    "Laptop Brand",
    brand_encoder.classes_
)
ram = st.selectbox(
    "RAM (GB)",
    [4, 8, 16, 32, 64]
)
ssd = st.selectbox(
    "SSD Storage (GB)",
    [128, 256, 512, 1024, 2048]
)
rating = st.slider("Laptop Rating", 3.0,5.0,4.0)
display_size = st.selectbox(
    "Display Size (Inches)",
    [13.3, 14.0, 15.6, 16.0]
)
age = st.slider(
    "Laptop Age (Years)",
    0,
    10,
    2
)
condition = st.selectbox(
    "Laptop Condition",
    condition_encoder.classes_
)
os_name = st.selectbox(
    "Operating System",
    os_encoder.classes_
)

if st.button("Predict Resale Price"):
    brand_encoded = brand_encoder.transform([brand])[0]
    condition_encoded = condition_encoder.transform([condition])[0]
    os_encoded = os_encoder.transform([os_name])[0]
    input_df = pd.DataFrame(
        [[
            brand_encoded,
            ram,
            ssd,
            rating,
            display_size,
            age,
            condition_encoded,
            os_encoded
        ]],
        columns=X.columns
    )
    prediction = model.predict(input_df)
    st.success(
        f"Estimated Resale Price: ₹ {prediction[0]:,.0f}"
    )
    st.balloons()
