
# the files for import data frame to analyze
import openpyxl
import pandas as pd
import streamlit as st
# Function to read the data file from local file
def read_data_file(path: str, file_name: str, file_type: str):
    """
     Reads a data file (CSV or Excel) from a local path.

     Args:
         path (str): The directory path where the file is located.
         file_name (str): The name of the file (without the extension).
         file_type (str): The type of the file, either 'csv', 'xls', or 'xlsx'.

     Returns:
         pd.DataFrame: A pandas DataFrame containing the data from the file, or None if there is an error.

     Raises:
         ValueError: If the file type is not supported (not 'csv', 'xls', or 'xlsx').
     """
    file_path = f"{path}/{file_name}.{file_type}"

    # Read the CSV or XLSX file based on the user input
    if file_type == 'csv':
        df = pd.read_csv(file_path)
    elif file_type == 'xls' or file_type == 'xlsx':
        df = pd.read_excel(file_path)
    else:
        st.error("Unsupported file type")
        return None

    return df

# Function to read the data file from URL
def read_data_file_from_url(url: str):
    """
       Reads a data file (CSV or Excel) from a provided URL.

       Args:
           url (str): The URL where the file is located. The URL must point directly to a CSV or Excel file.

       Returns:
           pd.DataFrame: A pandas DataFrame containing the data from the file, or None if there is an error.

       Raises:
           ValueError: If the URL does not point to a CSV or Excel file.
       """
    # Check if the URL ends with .csv or .xls
    if url.endswith('.csv'):
        df = pd.read_csv(url)
    elif url.endswith('.xls') or url.endswith('.xlsx'):
        df = pd.read_excel(url)
    else:
        st.error("Unsupported file type. Please provide a URL for a .csv or .xls file.")
        return None

    return df
