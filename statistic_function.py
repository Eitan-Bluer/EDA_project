import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

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

    # Variable types (numeric and categorical)
    numeric_vars = df.select_dtypes(include=['number']).columns
    categorical_vars = df.select_dtypes(include=['object', 'category']).columns
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


def histogram_numeric(data:str,col:str):
    plt.figure(figsize=(15, 10))
    sns.displot(
        data=data,
        x=col,
        height=5,  # Sets the overall height of the plot
        aspect=1.5,  # Adjusts the aspect ratio for better visibility
        kde=True,  # Adds a Kernel Density Estimate (smooth curve)
        bins=30,  # Adjusts the number of bins for histograms
        color="skyblue",  # Custom color for the plot
        # kde_kws={"color": "lightgreen", "linewidth": 2}, # Customize KDE curve
    )

    # Add titles and labels for clarity
    plt.title(f'Distribution of {col}', fontsize=14)
    plt.xlabel(f'{col}', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)

    # Improve tick formatting
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    # Add grid for better visual appeal
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Show the plot
    plt.show()