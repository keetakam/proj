import json
import streamlit as st
import pandas as pd

# Load data from the JSON file
with open('data/pico_companies.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(data)

# Display the data in Streamlit
st.title("ข้อมูลบริษัท พิโกไฟแนนซ์ และ พิโกพลัส")
st.dataframe(df)