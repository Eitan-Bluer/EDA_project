import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
def describe_columns_weight(df: pd.DataFrame, selected_column=None, weight=None):
    """
    Provides detailed statistics for each column in the DataFrame.
    For numeric columns: min, max, mean, std, median, percentiles.
    For categorical columns: mode, unique values, and missing values.
    If weight is provided, calculations are adjusted by the weight.
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
            if weight is not None:
                # Weighted calculations
                column_desc["Min"] = df[column].min()
                column_desc["Max"] = df[column].max()
                column_desc["Mean"] = weighted_stat(df[column], df[weight], stat_type='mean')
                column_desc["Std"] = weighted_stat(df[column], df[weight], stat_type='std')
                column_desc["Median"] = weighted_stat(df[column], df[weight], stat_type='median')
                column_desc["25th Percentile"] = weighted_stat(df[column], df[weight], stat_type='percentile', percentile=25)
                column_desc["75th Percentile"] = weighted_stat(df[column], df[weight], stat_type='percentile', percentile=75)
                column_desc["5th Percentile"] = weighted_stat(df[column], df[weight], stat_type='percentile', percentile=5)
                column_desc["95th Percentile"] = weighted_stat(df[column], df[weight], stat_type='percentile', percentile=95)
                column_desc["Range"] = column_desc["Max"] - column_desc["Min"]
                column_desc["Sum"] = (df[column] * df[weight]).sum()
                column_desc["Variance"] = weighted_stat(df[column], df[weight], stat_type='std') ** 2
                column_desc["Kurtosis"] = df[column].kurt()
            else:
                # Standard calculations
                column_desc["Min"] = df[column].min()
                column_desc["Max"] = df[column].max()
                column_desc["Mean"] = df[column].mean()
                column_desc["Std"] = df[column].std()
                column_desc["Median"] = df[column].median()
                column_desc["25th Percentile"] = df[column].quantile(0.25)
                column_desc["75th Percentile"] = df[column].quantile(0.75)
                column_desc["5th Percentile"] = df[column].quantile(0.05)
                column_desc["95th Percentile"] = df[column].quantile(0.95)
                column_desc["Range"] = df[column].max() - df[column].min()
                column_desc["Sum"] = df[column].sum()
                column_desc["Variance"] = df[column].var()
                column_desc["Kurtosis"] = df[column].kurt()
        else:
            # Categorical column handling
            if weight is not None:
                weighted_value_counts = df.groupby(column).apply(lambda x: x[weight].sum()).reset_index()
                weighted_value_counts.columns = [column, 'Weighted Count']
                column_desc["Value Counts"] = weighted_value_counts.sort_values(column).set_index(column)
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

def describe_columns_weight_(df: pd.DataFrame, selected_column=None, weight=None):
    """
    Provides detailed statistics for each column in the DataFrame.
    For numeric columns: min, max, mean, std, median, percentiles.
    For categorical columns: mode, unique values, and missing values.
    If weight is provided, calculations are adjusted by the weight.
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
            if weight is not None:
                # Weighted calculations
                column_desc["Min"] = (df[column]).min()
                column_desc["Max"] = (df[column]).max()
                #column_desc["Mean"] = df[column].mean()
                column_desc['Mean'] = weighted_stat(df[column], df[weight], stat_type='mean')
                column_desc["Std"] = weighted_stat(df[column], df[weight], stat_type='std')
                column_desc["Median"] = weighted_stat(df[column], df[weight], stat_type='median')
                column_desc["25th Percentile"] = weighted_stat(df[column], df[weight], stat_type='percentile', percentile=25)
                column_desc["75th Percentile"] = weighted_stat(df[column], df[weight], stat_type='percentile', percentile=75)
            else:
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
            # Convert value counts to a DataFrame and display it as a table
            if weight is not None:
                weighted_value_counts = df.groupby(column).apply(lambda x: (x[weight]).sum()).reset_index()
                weighted_value_counts.columns = [column, 'Weighted Count']
                column_desc["Weighted Value Counts"] = weighted_value_counts.sort_values(column).set_index(column)
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

def create_histogram_weighted(df: pd.DataFrame, column: str, bins: int = 30, weight=None):
    """
    Create a histogram or bar plot for the selected column in the DataFrame.
    If weight is provided, the histogram will be weighted.
    """
    # Check if the column is numeric and has more than 10 unique values
    if df[column].dtype in ['float', 'int', 'int64', 'float64'] and len(df[column].unique()) > 10:
        plt.figure(figsize=(10, 6))
        if weight is not None:
            # Weighted histogram
            plt.hist(df[column].dropna(), bins=bins, weights=df[weight].dropna(), edgecolor='black', color='skyblue')
        else:
            plt.hist(df[column].dropna(), bins=bins, edgecolor='black', color='skyblue')
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        st.pyplot(plt)
    else:
        # Handle categorical columns or columns with few unique values
        plt.figure(figsize=(10, 6))

        if weight is not None:
            # Compute weighted counts manually using groupby and sum
            weighted_counts = df.groupby(column).apply(lambda x: (x[weight].dropna()).sum()).reset_index(name='Count')
            sns.barplot(x=weighted_counts[column], y=weighted_counts['Count'], color='skyblue')
            plt.title(f'Weighted Bar Plot of {column}')
        else:
            # If no weight is provided, calculate the count of each category and use it for plotting
            category_counts = df[column].value_counts().reset_index(name='Count')
            category_counts.columns = [column, 'Count']
            sns.barplot(x=category_counts[column], y=category_counts['Count'], color='skyblue')
            plt.title(f'Bar Plot of {column}')

        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        st.pyplot(plt)


def weighted_stat(values, weights, stat_type='mean', percentile=None):
    """
    Calculate weighted mean, weighted standard deviation, weighted median, or weighted percentiles.

    Parameters:
    values (array-like): The data values.
    weights (array-like): The weights corresponding to the values.
    stat_type (str): The type of statistic to calculate ('mean', 'std', 'median', or 'percentile').
    percentile (float, optional): The desired percentile (0-100) for weighted percentiles. Used only if stat_type is 'percentile'.

    Returns:
    float: The computed weighted statistic (mean, std, median, or percentile).
    """
    # Convert values and weights to numpy arrays (in case they are not)
    values = np.array(values)
    weights = np.array(weights)

    # Weighted Mean
    if stat_type == 'mean':
        return np.average(values, weights=weights)

    # Weighted Standard Deviation
    elif stat_type == 'std':
        weighted_mean = np.average(values, weights=weights)
        weighted_variance = np.average((values - weighted_mean) ** 2, weights=weights)
        return np.sqrt(weighted_variance)

    # Weighted Median
    elif stat_type == 'median':
        sorted_indices = np.argsort(values)
        sorted_values = values[sorted_indices]
        sorted_weights = weights[sorted_indices]

        cumulative_weights = np.cumsum(sorted_weights)
        total_weight = np.sum(sorted_weights)
        median_weight = total_weight / 2
        median_index = np.searchsorted(cumulative_weights, median_weight)
        return sorted_values[median_index]

    # Weighted Percentiles
    elif stat_type == 'percentile' and percentile is not None:
        sorted_indices = np.argsort(values)
        sorted_values = values[sorted_indices]
        sorted_weights = weights[sorted_indices]

        cumulative_weights = np.cumsum(sorted_weights)
        total_weight = np.sum(sorted_weights)
        percentile_weight = percentile / 100 * total_weight
        percentile_index = np.searchsorted(cumulative_weights, percentile_weight)
        return sorted_values[percentile_index]

    else:
        raise ValueError("Invalid stat_type or missing percentile for 'percentile' calculation.")


def scatterplot_with_regression(df, y_col, x_col, weights_col=None):
    """
    Creates a scatterplot with regression line, and displays R-squared and beta.
    Supports weighted regression if a weights column is provided.

    Parameters:
    - df (pd.DataFrame): The dataset containing the variables.
    - y_col (str): The dependent variable.
    - x_col (str): The independent variable.
    - weights_col (str, optional): The column containing weights. Default is None.

    Returns:
    - fig: The matplotlib figure object.
    - beta: The regression coefficient (slope).
    - r_squared: The R-squared value.
    """

    if x_col and y_col:
        # Prepare data
        cols = [x_col, y_col] + ([weights_col] if weights_col else [])
        df_clean = df[cols].dropna()

        x = df_clean[x_col].values
        y = df_clean[y_col].values
        weights = df_clean[weights_col].values if weights_col else None

        # Add intercept for statsmodels
        x = sm.add_constant(x)

        # Fit weighted or ordinary least squares regression
        if weights_col:
            model = sm.WLS(y, x, weights=weights).fit()
        else:
            model = sm.OLS(y, x).fit()

        y_predict = model.predict(x)

        # Extract beta (slope) and intercept
        intercept, beta = model.params
        r_squared = model.rsquared

        # Plot scatterplot with regression line
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(df_clean[x_col], y, label="Data", alpha=0.7, c='blue')
        ax.plot(df_clean[x_col], y_predict, color="red", label="Regression Line")
        ax.set_title(f"Scatterplot of {y_col} vs {x_col}")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig, beta, r_squared
    else:
        raise ValueError("Both x_col and y_col must be specified.")

def bar_plot_category(df: pd.DataFrame, category_column: str, weights_column: str = None):
    """
    Generates and displays a weighted bar plot for a categorical column using Streamlit.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - category_column (str): The categorical column to plot.
    - weights_column (str, optional): The column containing weights. Default is None.

    Returns:
    - fig: The Matplotlib figure object containing the bar plot.
    """
    # Calculate weighted or unweighted counts
    if weights_column:
        category_counts = df.groupby(category_column)[weights_column].sum()
    else:
        category_counts = df[category_column].value_counts()

    # Create a Matplotlib figure
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=category_counts.index, y=category_counts.values, ax=ax)
    ax.set_title(f"Bar Plot of {category_column}")
    ax.set_xlabel(category_column)
    ax.set_ylabel("Weighted Count" if weights_column else "Count")
    plt.xticks(rotation=45)  # Rotate x-axis labels if needed
    return fig


def box_plot_calculation(df: pd.DataFrame, column: str, category: str = None, weights: str = None):
    """
    Generates a box plot for a numeric column, optionally grouped by a categorical column,
    and adjusts for weights if provided.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - column (str): The numeric column for the box plot.
    - category (str, optional): The categorical column to group by. Default is None.
    - weights (str, optional): The column containing weights. Default is None.

    Returns:
    - fig: The Matplotlib figure object containing the box plot.
    """
    if weights:
        # Adjust data based on weights
        df = df.loc[df.index.repeat(df[weights].round().astype(int))]

    # Create a Matplotlib figure
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=df, y=column, x=category, ax=ax)

    # Customize the plot
    if category:
        ax.set_title(f"Box Plot of {column} by {category}")
        ax.set_xlabel(category)
    else:
        ax.set_title(f"Box Plot of {column}")

    ax.set_ylabel(column)
    return fig
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def weighted_corr(x, y, weights):
    """
    Computes the weighted Pearson correlation coefficient between two variables.

    Args:
        x (pd.Series): First variable.
        y (pd.Series): Second variable.
        weights (pd.Series): Weights for the observations.

    Returns:
        float: Weighted correlation coefficient.
    """
    # Ensure the weights are normalized
    weights = weights / weights.sum()

    # Compute weighted means
    x_mean = (x * weights).sum()
    y_mean = (y * weights).sum()

    # Compute weighted covariance and variances
    covariance = ((x - x_mean) * (y - y_mean) * weights).sum()
    x_variance = ((x - x_mean) ** 2 * weights).sum()
    y_variance = ((y - y_mean) ** 2 * weights).sum()

    # Compute the weighted correlation coefficient
    return covariance / np.sqrt(x_variance * y_variance)


def weighted_corr_matrix(df: pd.DataFrame, weights: pd.Series):
    """
    Computes the weighted correlation matrix for a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the numeric variables.
        weights (pd.Series): Weights for the observations.

    Returns:
        pd.DataFrame: Weighted correlation matrix.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = pd.DataFrame(index=numeric_cols, columns=numeric_cols)

    for col1 in numeric_cols:
        for col2 in numeric_cols:
            corr_matrix.loc[col1, col2] = weighted_corr(df[col1], df[col2], weights)

    return corr_matrix.astype(float)


def plot_heatmap(corr_matrix: pd.DataFrame):
    """
    Plots a heatmap of the correlation matrix.

    Args:
        corr_matrix (pd.DataFrame): The correlation matrix to visualize.

    Returns:
        plt.Figure: The heatmap plot.
    """
    plt.figure(figsize=(10, 8))

    # Create the heatmap using seaborn
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

    # Add title
    plt.title("Correlation Heatmap", fontsize=16)

    return plt
def plot_histogram(data, numeric_var, category_var=None, bins=30, kde=True, weights_var=None):
    """
    Plots a histogram for a numeric variable, optionally grouped by a categorical variable,
    and adjusts for weights if provided.

    Parameters:
    - data (pd.DataFrame): The dataset containing the variables.
    - numeric_var (str): The name of the numeric variable to plot.
    - category_var (str, optional): The name of the categorical variable to group by. Default is None.
    - bins (int): Number of bins for the histogram. Default is 30.
    - kde (bool): Whether to overlay a density plot on the histogram. Default is True.
    - weights_var (str, optional): The name of the column containing weights. Default is None.

    Returns:
    - plt: The Matplotlib plot object.
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
            alpha=0.6,
            weights=data[weights_var] if weights_var else None
        )
    else:
        sns.histplot(
            data=data,
            x=numeric_var,
            bins=bins,
            kde=kde,
            color="blue",
            alpha=0.6,
            weights=data[weights_var] if weights_var else None
        )

    plt.title(f"Histogram of {numeric_var}" + (f" by {category_var}" if category_var else ""))
    plt.xlabel(numeric_var)
    plt.ylabel("Weighted Count" if weights_var else "Count")
    plt.grid(True, alpha=0.3)
    return plt
