import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
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
    st.write(f"""
    **Number of Variables:** {overview_stats['Number of Variables']}  
    **Number of Observations:** {overview_stats['Number of Observations']}  
    **Number of Numeric Variables:** {overview_stats['Number of Numeric Variables']}  
    **Number of Categorical Variables:** {overview_stats['Number of Categorical Variables']}  
    **Missing Cells:** {overview_stats['Missing Cells']}  
    **Missing Cells (%):** {overview_stats['Missing Cells (%)']:.2f}%  
    **Duplicate Rows:** {overview_stats['Duplicate Rows']}  
    **Duplicate Rows (%):** {overview_stats['Duplicate Rows (%)']:.2f}%  
    """)




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
        column_desc["DataType"] = df[column].dtype

        if column_desc["Type"] == "Numeric":
            column_desc["Min"] = df[column].min()
            column_desc["Max"] = df[column].max()
            column_desc["Mean"] = df[column].mean()
            column_desc["Std"] = df[column].std()
            column_desc["Median"] = df[column].median()
            column_desc["25th Percentile"] = df[column].quantile(0.25)
            column_desc["75th Percentile"] = df[column].quantile(0.75)
            # Basic Quantile Statistics
            column_desc["5th Percentile"] = df[column].quantile(0.05)
            column_desc["Q1"] = df[column].quantile(0.25)
            column_desc["Q3"] = df[column].quantile(0.75)
            column_desc["95th Percentile"] = df[column].quantile(0.95)
            column_desc["Range"] = df[column].max() - df[column].min()
            # Descriptive Statistics
            column_desc["Sum"] = df[column].sum()
            column_desc["Variance"] = df[column].var()
            column_desc["Kurtosis"] = df[column].kurt()
        else:
            value_counts = df[column].value_counts().reset_index()
            value_counts.columns = [column, 'Count']

            column_desc["Value Counts"] = value_counts.sort_values(column).set_index(column)

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


def create_histogram_(df: pd.DataFrame, column: str, bins: int = 30):
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
        sns.countplot(x=df[column],hue=df[column], palette='Blues',legend=False)
        plt.title(f'Bar plot of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        st.pyplot(plt)

def create_histogram__(df: pd.DataFrame, column: str, bins: int = 30):
    """
    Create a histogram or bar plot for the selected column in the DataFrame.
    If weight is provided, the histogram will be weighted.
    """
    # Check if the column is numeric and has more than 10 unique values
    if df[column].dtype in ['float', 'int', 'int64', 'float64'] and len(df[column].unique()) > 10:
        plt.figure(figsize=(10, 6))

        plt.hist(df[column].dropna(), bins=bins, edgecolor='black', color='skyblue')
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        st.pyplot(plt)
    else:
        # Handle categorical columns or columns with few unique values
        plt.figure(figsize=(10, 6))



        category_counts = df[column].value_counts().reset_index(name='Count')
        category_counts.columns = [column, 'Count']
        sns.barplot(x=category_counts[column], y=category_counts['Count'], color='skyblue')
        plt.title(f'Bar Plot of {column}')

        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        st.pyplot(plt)

def create_histogram(df: pd.DataFrame, column: str, bins: int = 30):
    try:
        if pd.api.types.is_numeric_dtype(df[column]):
            plt.figure(figsize=(10, 6))
            plt.hist(df[column].dropna(), bins=bins, edgecolor='black', color='skyblue')
            plt.title(f'Histogram of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
        else:
            # Handle categorical columns or mixed data
            category_counts = df[column].value_counts().reset_index(name='Count')
            category_counts.columns = [column, 'Count']
            sns.barplot(x=category_counts[column], y=category_counts['Count'], color='skyblue')
            plt.title(f'Bar Plot of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.xticks(rotation=45)
        st.pyplot(plt)
    except Exception as e:
        st.error(f"Error creating plot for {column}: {e}")


def plot_histogram(data, numeric_var, category_var=None, bins=30, kde=True):
    """
    Plots a histogram for a numeric variable, optionally grouped by a categorical variable.

    Parameters:
    - data (pd.DataFrame): The dataset containing the variables.
    - numeric_var (str): The name of the numeric variable to plot.
    - category_var (str, optional): The name of the categorical variable to group by. Default is None.
    - bins (int): Number of bins for the histogram. Default is 30.
    - kde (bool): Whether to overlay a density plot on the histogram. Default is True.

    Returns:
    - None: Displays the plot.
    """
    plt.figure(figsize=(10, 6))

    if category_var:
        sns.histplot(
            data=data,
            x=numeric_var,
            hue=category_var,
            bins=bins,
            kde=kde,
            palette="Set2",
            alpha=0.6
        )
    else:
        sns.histplot(
            data=data,
            x=numeric_var,
            bins=bins,
            kde=kde,
            color="blue",
            alpha=0.6
        )

    plt.title(f"Histogram of {numeric_var}" + (f" by {category_var}" if category_var else ""))
    plt.xlabel(numeric_var)
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    return plt



def calculate_correlation(df: pd.DataFrame):
    """
    Calculates the correlation matrix for the numeric columns in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing numeric columns.

    Returns:
        pd.DataFrame: A correlation matrix for numeric columns.
    """
    # Calculate the correlation matrix for numeric columns
    corr_matrix = df.corr()
    return corr_matrix


def plot_heatmap(corr_matrix: pd.DataFrame):
    """
    Plots a heatmap of the correlation matrix.

    Args:
        corr_matrix (pd.DataFrame): The correlation matrix to visualize.
    """
    plt.figure(figsize=(10, 8))

    # Create the heatmap using seaborn
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

    # Add title
    plt.title("Correlation Heatmap", fontsize=16)

    return plt


def box_plot_calculation(df: pd.DataFrame, column: str,category: str = None):
    # Create a Matplotlib figure
    fig, ax = plt.subplots()
    sns.boxplot(data=df, y=column,x=category, ax=ax)
    if category:
        ax.set_title(f"Box Plot of {column} by {category}")
        ax.set_xlabel(category)
    else:
        ax.set_title(f"Box Plot of {column}")
    return fig
def bar_plot_category(df: pd.DataFrame, category_column: str):
    """
    Generates and displays a bar plot for a categorical column using Streamlit.

    Parameters:
    - df: pd.DataFrame: The DataFrame containing the data.
    - category_column: str: The categorical column to plot.
    """
    # Calculate category counts
    category_counts = df[category_column].value_counts()

    # Create a Matplotlib figure
    fig, ax = plt.subplots()
    sns.barplot(x=category_counts.index, y=category_counts.values, ax=ax)
    ax.set_title(f"Bar Plot of {category_column}")
    ax.set_xlabel(category_column)
    ax.set_ylabel("Count")
    plt.xticks(rotation=45)  # Rotate x-axis labels if needed
    return fig


def scatterplot_with_regression(df, y_col, x_col):
    """
    Creates a scatterplot with regression line, and displays R-squared and beta.

    Parameters:
    - data (pd.DataFrame): The dataset containing the variables.

    Returns:
    - None: Displays the scatterplot and regression results in Streamlit.
    """

    if x_col and y_col:
        # Prepare data
        df_clean = df[[x_col, y_col]].dropna()
        x = df_clean[x_col].values.reshape(-1, 1)  # Reshape for sklearn
        y = df_clean[y_col].values

        # Fit linear regression
        model = LinearRegression()
        model.fit(x, y)
        y_predict = model.predict(x)

        # Calculate beta and R-squared
        beta = model.coef_[0]
        r_squared = r2_score(y, y_predict)

        # Display regression metrics


        # Plot scatterplot with regression line
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(x, y, label="Data", alpha=0.7)
        ax.plot(x, y_predict, color="red", label="Regression Line")
        ax.set_title(f"Scatterplot of {y_col} vs {x_col}")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig, beta, r_squared

