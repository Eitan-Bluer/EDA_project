import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
def numeric_category(df: pd.DataFrame):
    ## Variable types (numeric and categorical)
    numeric_vars = []
    categorical_vars = []

    # Check each column to determine if it is numeric or categorical
    for column in df.columns:
        if (df[column].dtype in ['float', 'int', 'int64', 'float64']) and (len(df[column].unique()) > 10):
            numeric_vars.append(column)
        else:
            categorical_vars.append(column)
    return numeric_vars,categorical_vars

def overview(df: pd.DataFrame):
    """
    Provides an overview of the DataFrame, including variable types,
    number of variables, observations, missing values, and duplicates.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.

    Returns:
        dict: A dictionary containing the overview statistics.
    """
    # Number of variables (columns) and observations (rows)
    num_variables = df.shape[1]
    num_observations = df.shape[0]

    ## Variable types (numeric and categorical)

    numeric_vars,categorical_vars=numeric_category(df)

    num_numeric_vars = len(numeric_vars)
    num_categorical_vars = len(categorical_vars)

    # Missing values
    missing_cells = df.isnull().sum().sum()
    missing_cells_percentage = (missing_cells / (num_observations * num_variables)) * 100

    # Duplicate rows
    duplicate_rows = df.duplicated().sum()
    duplicate_rows_percentage = (duplicate_rows / num_observations) * 100

    # Overview dictionary
    overview_stats = {
        "Number of Variables": num_variables,
        "Number of Observations": num_observations,
        "Number of Numeric Variables": num_numeric_vars,
        "Number of Categorical Variables": num_categorical_vars,
        "Missing Cells": missing_cells,
        "Missing Cells (%)": missing_cells_percentage,
        "Duplicate Rows": duplicate_rows,
        "Duplicate Rows (%)": duplicate_rows_percentage
    }

    return overview_stats
def write_overview_to_st(overview_stats):
    """
    Display the overview statistics in Streamlit.

    Args:
        overview_stats (dict): A dictionary containing the overview statistics.
    """
    # Display the overview in Streamlit
    st.markdown("### DataFrame Overview")
    st.write(f"**Number of Variables:** {overview_stats['Number of Variables']}")
    st.write(f"**Number of Observations:** {overview_stats['Number of Observations']}")
    st.write(f"**Number of Numeric Variables:** {overview_stats['Number of Numeric Variables']}")
    st.write(f"**Number of Categorical Variables:** {overview_stats['Number of Categorical Variables']}")
    st.write(f"**Missing Cells:** {overview_stats['Missing Cells']}")
    st.write(f"**Missing Cells (%):** {overview_stats['Missing Cells (%)']:.2f}%")
    st.write(f"**Duplicate Rows:** {overview_stats['Duplicate Rows']}")
    st.write(f"**Duplicate Rows (%):** {overview_stats['Duplicate Rows (%)']:.2f}%")




def describe_columns(df: pd.DataFrame, selected_column=None):
    """
    Provides detailed statistics for each column in the DataFrame.
    For numeric columns: min, max, mean, std, median, percentiles.
    For categorical columns: mode, unique values, and missing values.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        selected_column (str): The name of the column to display its description. If None, display all columns' descriptions.

    Returns:
        dict: A dictionary containing column statistics.
    """
    column_descriptions = {}

    for column in df.columns:
        column_desc = {}

        # Check if the column is numeric or categorical
        if (df[column].dtype in ['float', 'int', 'int64', 'float64']) and (len(df[column].unique()) > 10):
            column_desc["Type"] = "Numeric"
        else:
            column_desc["Type"] = "Categorical"

        column_desc["Unique Values"] = df[column].nunique()
        column_desc["Missing Values"] = df[column].isnull().sum()
        column_desc["Missing Values (%)"] = (df[column].isnull().sum() / df.shape[0]) * 100
        column_desc["Sample Data"] = df[column].head(5).to_list()

        if column_desc["Type"] == "Numeric":
            column_desc["Min"] = df[column].min()
            column_desc["Max"] = df[column].max()
            column_desc["Mean"] = df[column].mean()
            column_desc["Std"] = df[column].std()
            column_desc["Median"] = df[column].median()
            column_desc["25th Percentile"] = df[column].quantile(0.25)
            column_desc["75th Percentile"] = df[column].quantile(0.75)
        else:
            column_desc["Mode"] = df[column].mode()[0] if not df[column].mode().empty else None

        # Ensure we always add a description for each column
        column_descriptions[column] = column_desc

    if selected_column:
        # If a specific column is selected, return the description for that column
        if selected_column in column_descriptions:
            return column_descriptions[selected_column]
        else:
            return None  # If the selected column doesn't exist in the dataframe
    else:
        # If no column is selected, return the full dictionary of column descriptions
        return column_descriptions


def create_histogram(df: pd.DataFrame, column: str, bins: int = 30):
    """
    Create a histogram or bar plot for the selected column in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        column (str): The name of the column to plot the histogram for.
        bins (int, optional): The number of bins for the histogram. Default is 10.
    """
    # Check if the column is numeric and has more than 10 unique values
    if df[column].dtype in ['float', 'int', 'int64', 'float64'] and len(df[column].unique()) > 10:
        plt.figure(figsize=(10, 6))
        # If bins is None, use the default number of bins
        if bins is None:
            bins = 10  # Set to a default value if None
        plt.hist(df[column].dropna(), bins=bins, edgecolor='black', color='skyblue')
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        st.pyplot(plt)
    else:
        # Handle categorical columns or columns with few unique values
        plt.figure(figsize=(10, 6))
        sns.countplot(x=df[column], palette='Blues')
        plt.title(f'Bar plot of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        st.pyplot(plt)