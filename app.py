
import streamlit as st
import pandas as pd
from prophet import Prophet
from deep_translator import GoogleTranslator
import matplotlib.pyplot as plt
from datetime import datetime
from langchain.llms import OpenAI
from langchain.chains import ConversationChain

# Supported languages
supported_languages = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr",
    "Spanish": "es",
    "French": "fr"
}

def forecast_expenses(df):
    df['Date'] = pd.to_datetime(df['Date'])
    monthly_expense = df.groupby(pd.Grouper(key='Date', freq='M')).sum().reset_index()
    monthly_expense.rename(columns={'Date': 'ds', 'Amount': 'y'}, inplace=True)
    
    model = Prophet()
    model.fit(monthly_expense)
    future = model.make_future_dataframe(periods=1, freq='M')
    forecast = model.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

def multilingual_agent(user_message, target_lang):
    # Translate user input to English
    input_en = GoogleTranslator(source=target_lang, target='en').translate(user_message)
    
    # Mock AI agent response (replace with actual API call in production)
    response_en = "Based on your spending trends, consider reducing dining out expenses by 20% next month."
    
    # Translate back to target language
    response_translated = GoogleTranslator(source='en', target=target_lang).translate(response_en)
    return response_translated

st.title("🌍 Personal Finance Forecast & Advisor")

language = st.selectbox("Select your language:", list(supported_languages.keys()))
target_lang = supported_languages[language]

uploaded_file = st.file_uploader("Upload your bank statement CSV (Date, Amount)", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("📊 Uploaded Data Sample:", data.head())
    
    forecast = forecast_expenses(data)
    
    st.write("📈 Expense Forecast (Next Month):")
    st.line_chart(forecast.set_index('ds')['yhat'])
    
    user_question = st.text_input("Ask finance-related question:")
    if st.button("Get Advice"):
        reply = multilingual_agent(user_question, target_lang)
        st.write("🤖 AI Agent Response:", reply)
