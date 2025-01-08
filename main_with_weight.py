import streamlit as st
import pandas as pd

import statistic_function as sf
import merge_weight_file as mwf
import weight_statistic_fuction as wsf
def streamlit_description_base(column,column_desc):
    st.subheader(f"**{column}**")
    st.write(f"""
    **Type:** {column_desc['Type']} (data type: {column_desc["DataType"]})  
    **Unique Values:** {column_desc['Unique Values']}  
    **Missing Values:** {column_desc['Missing Values']} ({column_desc['Missing Values (%)']:.2f}%)""")
    if column_desc["Type"] == "Numeric":
        st.write(f"""
        **Min:** {column_desc['Min']}  
        **Max:** {column_desc['Max']}  
        **Mean:** {column_desc['Mean']:.2f}  
        **Std Dev:** {column_desc['Std']:.2f}  
        **Median:** {column_desc['Median']}  
        **25th Percentile:** {column_desc['25th Percentile']}  
        **75th Percentile:** {column_desc['75th Percentile']}  
        """)
    else:
        st.write(f"**Value Counts:**")
        if len(column_desc["Value Counts"])>15:
           st.write("The number of categories exceeds 15. The categories can be seen in the focused analysis of the variable. ")
        else:
            st.table(column_desc["Value Counts"])

def streamlit_description_statistics(column,column_desc):
    st.subheader(f"**{column}**")
    st.write(f"""
    **Type:** {column_desc['Type']} (data type: {column_desc["DataType"]})  
    **Unique Values:** {column_desc['Unique Values']}  
    **Missing Values:** {column_desc['Missing Values']} ({column_desc['Missing Values (%)']:.2f}%)""")
    if column_desc["Type"] == "Numeric":
        st.write(f"""
        **Min:** {column_desc['Min']}  
        **Max:** {column_desc['Max']}  
        **Sum:** {column_desc['Sum']:.2f}  
        **Mean:** {column_desc['Mean']:.2f}  
        **Variance:** {column_desc['Variance']:.2f}  
        **Std Dev:** {column_desc['Std']:.2f}  
        **Median:** {column_desc['Median']}  
        **25th Percentile:** {column_desc['25th Percentile']}  
        **75th Percentile:** {column_desc['75th Percentile']}  
        **95th Percentile:** {column_desc['95th Percentile']}  
        **Range:** {column_desc['Range']}  
        """)
    else:
        st.write(f"**Value Counts: see tab table**")
        #st.table(column_desc["Value Counts"])


def show_correlation_table(corr_matrix: pd.DataFrame):
    """
    Displays the correlation matrix as a table in Streamlit.

    Args:
        corr_matrix (pd.DataFrame): The correlation matrix to display.
    """
    st.write("### Correlation Matrix")
    st.dataframe(corr_matrix)




from weight_statistic_fuction import weighted_corr_matrix, weighted_stat

def highlight_columns(df: pd.DataFrame, numeric: list, weight_column: str = None,
                      corr_threshold: float = 0.8, missing_threshold: float = 20,
                      std_threshold: float = 0.95):
    """
    Analyzes the DataFrame for highlights based on weighted correlation, missing values, and weighted standard deviation.

    Args:
        df (pd.DataFrame): The input DataFrame.
        numeric (list): List of numeric columns to consider.
        weight_column (str): The column name to use as weights (optional).
        corr_threshold (float): The threshold for high correlation.
        missing_threshold (float): The percentage threshold for missing values.
        std_threshold (float): The threshold for high weighted standard deviation.

    Returns:
        dict: A dictionary containing highlights for each column.
    """
    highlights = {}

    # Calculate correlation matrix
    if weight_column and weight_column in df.columns:
        weights = df[weight_column]
        corr_matrix = wsf.weighted_corr_matrix(df[numeric], weights)
    else:
        corr_matrix = df[numeric].corr()

    # Iterate over columns
    for col in df.columns:
        column_highlights = {}

        # Check for high correlation
        if col in corr_matrix.columns:
            correlated_cols = corr_matrix[col][
                (corr_matrix[col] > corr_threshold) & (corr_matrix[col] < 1.0)].index.tolist()
            if correlated_cols:
                column_highlights['Highly Correlated Columns'] = correlated_cols

        # Check for missing values
        missing_percentage = df[col].isnull().mean() * 100
        if missing_percentage > missing_threshold:
            column_highlights['Missing Percentage'] = f"{missing_percentage:.2f}%"

        # Check for high standard deviation
        if pd.api.types.is_numeric_dtype(df[col]):
            if weight_column and weight_column in df.columns:
                std_dev = wsf.weighted_stat(df[col], df[weight_column], stat_type='std')
            else:
                std_dev = df[col].std()

            if std_dev > std_threshold:
                column_highlights['Weighted Standard Deviation' if weight_column else 'Standard Deviation'] = std_dev

        # Add to highlights if any conditions are met
        if column_highlights:
            highlights[col] = column_highlights

    return highlights



def main_weight(df: pd.DataFrame):
    """
        Performs an interactive exploratory data analysis (EDA) on a dataset using a specified weight column.

        Args:
            df (pd.DataFrame): The dataset to analyze. Must be a Pandas DataFrame.

        Features:
        1. Weight Column Selection:
           - Allows users to select a weight column from the dataset for weighted calculations.

        2. Dataset Overview:
           - Generates and displays summary statistics such as column data types, missing values, and duplicates.

        3. Column Highlights:
           - Detects and displays key column properties, including high correlations, missing values, and weighted standard deviations.

        4. Detailed Column Analysis:
           - Allows users to view detailed statistics (e.g., weighted mean, weighted median, weighted percentiles) for selected or all columns.
           - Includes visualizations such as histograms, box plots, scatterplots, and bar plots with weights.

        5. Correlation Analysis:
           - Calculates and displays the weighted correlation matrix for numeric columns.
           - Provides both tabular and heatmap visualizations.

        6. Data Display:
           - Displays the first 5 rows of the dataset for preview.

        Outputs:
            Interactive insights and visualizations rendered in the Streamlit app.
    """
    weight_column=None

    """weight_location=st.radio("Do the weight column is part of your dataset or you need to import"
             " it from external source?", ["internal", "external"])
    if weight_location=="external":
        merged_df = mwf.upload_and_merge(df)
        df = merged_df.copy()
        if merged_df is not None:
            st.write("data merge")
         #   weight_column = st.selectbox("Select the weight column:", [None] + merged_df.columns.tolist())
          #  if weight_column is not None:
           #     st.write(f"weight column selected {weight_column}.")
            #    df = merged_df.copy()
            #else:
             #   st.write("No weight column selected.")
        else:
            st.write("data didn't merge")
        #weight_column = st.selectbox("Select the weight column:", [None] + df.columns.tolist())
    else:
        pass"""

    weight_column = st.selectbox("Select the weight column:", [None] + df.columns.tolist(), key="weight_column")



    if  weight_column:

        if df is not None:
            # Numeric and Categorical Columns
            numeric_columns, categorical_columns = sf.numeric_category(df)
            col1, col2 = st.columns(2)
            with col1:
                # Generate the overview statistics
                overview_stats = sf.overview(df)
                # Display the overview in Streamlit
                sf.write_overview_to_st(overview_stats)
            with col2:
                # Detect highlights
                st.write("### Column Highlights")
                highlights = highlight_columns(df,numeric_columns)
                # Toggle highlights visibility
                st.session_state.show_highlights = st.session_state.get("show_highlights", False)
                if st.button("Show highlights" if not st.session_state.show_highlights else "Hide highlights"):
                    st.session_state.show_highlights = not st.session_state.show_highlights
                if st.session_state.show_highlights:
                    if  highlights:
                        for col, details in highlights.items():
                            st.write(f"**Column: {col}**")
                            for key, value in details.items():
                                st.write(f"- {key}: {value}")
                            st.write("---")
                    else:
                        st.write("No highlights found based on the given thresholds.")

            # Numeric and Categorical Columns
            #numeric_columns, categorical_columns = sf.numeric_category(df)

            st.write("---------------------------------------------\n"
                     " ## Variables")
            st.write("#### Statistics ")
            # Get the list of columns for selection
            column_to_describe = st.selectbox("Select a Variable", [None] + [col for col in df.columns if (col != weight_column) ])

            # Display details for numeric columns
            # if column_to_describe is None
            if column_to_describe is None:
                if (numeric_columns is not None) or (categorical_columns is not None):
                    for column in df.columns:
                        col1, col2 = st.columns(2)

                        with col1:
                            column_desc = wsf.describe_columns_weight(df, column,weight_column)

                            if column_desc is not None:
                                streamlit_description_base(column, column_desc)
                            else:
                                st.error(f"Column {column} description could not be generated.")

                        with col2:
                            bins = 30
                            wsf.create_histogram_weighted(df, column,bins,weight_column)
                else:
                    st.error(f"No Numeric or categorical column are in the data you provided")
            else:
                column=column_to_describe
                bins=30

                col1, col2 = st.columns(2)

                with col1:
                    column_desc = wsf.describe_columns_weight(df, column,weight_column)

                    if column_desc is not None:
                        streamlit_description_statistics(column, column_desc)

                    else:
                        st.error(f"Column {column} description could not be generated.")

                with col2:
                    #sf.create_histogram(df, column,bins,weight_column)
                    if column in numeric_columns:
                        tab1, tab2, tab3 = st.tabs(["Histogram", "Box plot","Scater_plot"])
                    elif column in categorical_columns:
                        tab1, tab2 = st.tabs(["Bar Plot", "table"])
                        tab3 = None

                    with tab1:
                        if column in numeric_columns:
                            bins = st.select_slider('Select number of bins', options=[10, 20, 30, 40, 50],value=10)
                            category_var=st.selectbox("Select a Category Variable", [None] + categorical_columns)


                            plt=wsf.plot_histogram(df, column,category_var, bins,True, weight_column)
                            st.pyplot(plt)
                        elif column in categorical_columns:
                            fig=wsf.bar_plot_category(df,column,weight_column)
                            #fig=sf.bar_plot_category(df,column)
                            st.pyplot(fig)
                    with tab2:
                        if column in numeric_columns:
                            category_var_bp = st.selectbox("Select a Category Variable to box plot", [None] + categorical_columns)
                            fig=wsf.box_plot_calculation(df, column,category_var_bp,weight_column)
                            #fig=sf.box_plot_calculation(df, column,category_var_bp)
                            st.pyplot(fig)
                        elif column in categorical_columns:
                            st.table(column_desc["Value Counts"])
                    if tab3:
                        with tab3:
                            st.write("### Scatterplot with Regression")
                            # Dropdowns to select x and y columns
                            x_col = st.selectbox("Select X-axis column", options=[col for col in numeric_columns if ((col != column) & (col != weight_column)) ])
                            fig, beta, r_squared =wsf.scatterplot_with_regression(df, column, x_col,weight_column)
                            #fig, beta, r_squared = sf.scatterplot_with_regression(df, column, x_col)
                            # Display the plot in Streamlit
                            # Display regression metrics
                            st.write(f"**Beta (Slope):** {beta:.4f} **R-squared:** {r_squared:.4f}")

                            st.pyplot(fig)




            # Calculate the correlation matrix
            st.write("---------------------------------------------\n"
                     " ## Correlation")
            corr_matrix =wsf.weighted_corr_matrix(df[numeric_columns],df[weight_column])
            #corr_matrix = sf.calculate_correlation(df[numeric_columns])

            # Create tabs for the correlation table and heatmap
            tab1, tab2 = st.tabs(["Correlation Table", "Heatmap"])

            with tab1:
                show_correlation_table(corr_matrix)

            with tab2:
                plt=wsf.plot_heatmap(corr_matrix)
                #plt=sf.plot_heatmap(corr_matrix)
                st.pyplot(plt)

            st.write("#### Sample ")
            st.dataframe(df.head(5))




#if __name__ == "__main__":
 #   main()
