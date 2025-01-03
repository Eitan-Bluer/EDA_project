import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


def describe_column(df: pd.DataFrame, column: str):
    """
    Provides detailed statistics for a single column in the DataFrame.
    For numeric columns: min, max, mean, std, median, percentiles.
    For categorical columns: mode, unique values, and missing values.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        column (str): The name of the column to display its description.
    """
    column_desc = {}

    # Check if the column is numeric or categorical
    if df[column].dtype in ['int64', 'float64']:
        column_desc["Type"] = "Numeric"
    else:
        column_desc["Type"] = "Categorical"

    column_desc["Unique Values"] = df[column].nunique()
    column_desc["Missing Values"] = df[column].isnull().sum()
    column_desc["Missing Values (%)"] = (df[column].isnull().sum() / df.shape[0]) * 100
    column_desc["Sample Data"] = df[column].head(5).to_list()

    # For numeric columns, add descriptive statistics
    if column_desc["Type"] == "Numeric":
        column_desc["Min"] = df[column].min()
        column_desc["Max"] = df[column].max()
        column_desc["Mean"] = df[column].mean()
        column_desc["Std"] = df[column].std()
        column_desc["Median"] = df[column].median()
        column_desc["25th Percentile"] = df[column].quantile(0.25)
        column_desc["75th Percentile"] = df[column].quantile(0.75)

    # For categorical columns, add the mode
    else:
        column_desc["Mode"] = df[column].mode()[0] if not df[column].mode().empty else None

    return column_desc


def create_histogram(df: pd.DataFrame, column: str):
    """
    Create a histogram for the selected numeric column.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        column (str): The name of the column to plot the histogram for.
    """
    if df[column].dtype in ['int64', 'float64']:
        plt.figure(figsize=(10, 6))
        plt.hist(df[column].dropna(), bins=30, edgecolor='black', color='skyblue')
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        st.pyplot(plt)
    else:
        st.warning(f"The column '{column}' is not numeric, so a histogram cannot be created.")


def display_in_streamlit():
    """
    Handles the Streamlit UI process, including file upload, column selection,
    and calling the description and histogram functions.
    """
    st.title('Streamlit DataFrame Column Descriptions and Histogram')

    # File uploader for CSV or Excel files
    uploaded_file = st.file_uploader("Upload your file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        # Load the file into a DataFrame
        if uploaded_file.name.endswith("csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith("xlsx"):
            df = pd.read_excel(uploaded_file)

        # Get the list of numeric columns for histogram
        numeric_columns = df.select_dtypes(include=['number']).columns

        # Create columns side by side for description and histogram
        for column in numeric_columns:
            # Use a two-column layout for each variable's description and histogram
            col1, col2 = st.columns(2)

            # Description in col1
            with col1:
                column_desc = describe_column(df, column)
                st.subheader(f"**{column}**")
                st.write(f"**Type:** {column_desc['Type']}")
                st.write(f"**Unique Values:** {column_desc['Unique Values']}")
                st.write(
                    f"**Missing Values:** {column_desc['Missing Values']} ({column_desc['Missing Values (%)']:.2f}%)")
                st.write(f"**Sample Data:** {column_desc['Sample Data']}")

                if column_desc["Type"] == "Numeric":
                    st.write(f"**Min:** {column_desc['Min']}")
                    st.write(f"**Max:** {column_desc['Max']}")
                    st.write(f"**Mean:** {column_desc['Mean']:.2f}")
                    st.write(f"**Std Dev:** {column_desc['Std']:.2f}")
                    st.write(f"**Median:** {column_desc['Median']}")
                    st.write(f"**25th Percentile:** {column_desc['25th Percentile']}")
                    st.write(f"**75th Percentile:** {column_desc['75th Percentile']}")
                else:
                    st.write(f"**Mode:** {column_desc['Mode']}")

            # Histogram in col2
            with col2:
                create_histogram(df, column)


def main():
    """
    Main function that initializes the Streamlit app by calling the UI and description functions.
    """
    display_in_streamlit()


if __name__ == "__main__":
    main()
