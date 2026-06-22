import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import google.generativeai as genai
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

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

# Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)
lr_pred = model.predict(X_test)
lr_score = r2_score(y_test, lr_pred)

# Random Forest
rf_model = RandomForestRegressor( n_estimators=100,random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_score = r2_score(y_test, rf_pred)

def generate_description(brand,ram,ssd,age,condition,os_name,price):
    prompt = f"""
    Create a professional resale advertisement.
    Brand: {brand}
    RAM: {ram} GB
    SSD: {ssd} GB
    Age: {age} years
    Condition: {condition}
    Operating System: {os_name}

    Expected Price: ₹{price:.0f}

    Keep it under 100 words.
    """
    response = gemini_model.generate_content(prompt)
    return response.text

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
st.subheader("Model Comparison")

st.write(
    f"Linear Regression Accuracy: {lr_score*100:.2f}%"
)

st.write(
    f"Random Forest Accuracy: {rf_score*100:.2f}%"
)

best_model_name = (
    "Random Forest"
    if rf_score > lr_score
    else "Linear Regression"
)

st.success(
    f"Best Model: {best_model_name}"
)
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
    lr_price = model.predict(input_df)[0]

    rf_price = rf_model.predict(input_df)[0]

    st.write(f"Linear Regression Prediction: ₹{lr_price:,.0f}")
    st.write(f"Random Forest Prediction: ₹{rf_price:,.0f}")

    if rf_score > lr_score:
        final_price = rf_price
    else:
        final_price = lr_price

    st.success(
        f"Estimated Resale Price: ₹ {final_price:,.0f}"
    )
    st.balloons()
    st.info(
        f"Expected Price Range: ₹ {min(lr_price, rf_price):,.0f} - ₹ {max(lr_price, rf_price):,.0f}"
    )
    try:
        description = generate_description(
        brand,
        ram,
        ssd,
        age,
        condition,
        os_name,
        final_price
        )

        st.subheader("AI Generated Resale Description")
        st.write(description)

     except Exception as e:
         st.warning("Gemini description could not be generated.")

    
    
