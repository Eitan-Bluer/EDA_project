import pandas as pd
import streamlit as st

# Function to read data file from local path
def read_data_file(path: str, file_name: str, file_type: str):
    file_path = f"{path}/{file_name}.{file_type}"
    if file_type == 'csv':
        df = pd.read_csv(file_path)
    elif file_type in ['xls', 'xlsx']:
        df = pd.read_excel(file_path)
    else:
        st.error("Unsupported file type")
        return None
    return df

# Function for uploaded file
def uploaded_file_function(uploaded_file):
    df = None
    if uploaded_file is not None:
        if uploaded_file.name.endswith("csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith("xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file type")
    return df

# Function to read data from URL
def read_data_file_from_url(url: str):
    if url.endswith('.csv'):
        df = pd.read_csv(url)
    elif url.endswith('.xls') or url.endswith('.xlsx'):
        df = pd.read_excel(url)
    else:
        st.error("Unsupported file type. Please provide a URL for a .csv or .xls file.")
        return None
    return df
