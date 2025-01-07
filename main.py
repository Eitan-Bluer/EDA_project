import streamlit as st
import pandas as pd
import import_data_scripts as ids
import statistic_function as sf

def main():
    st.title('Exploratory Data Analysis (EDA)')

    # Choose the data source (Local or URL)
    data_source = st.radio("Choose the data source:", ('Local File', 'URL'))
    # Add the weight option

    # Now you can use `weight_column` in your statistics or other logic.

    df = None

    if data_source == 'Local File':
        uploaded_file = st.file_uploader("Upload your file", type=["csv", "xlsx"])
        df = ids.uploaded_file_function(uploaded_file)

    elif data_source == 'URL':
        url = st.text_input('Enter the URL of the data file please make sure you provide "raw" file URL')
        if url:
            df = ids.read_data_file_from_url(url)
    # If 'Yes' is selected, ask for the weight column name

    weight_option = st.radio("Do you want to use a weight column?", ("No", "Yes"))

    weight_column = None
    if weight_option == "Yes":
        weight_column = st.selectbox("If you want to use a weight calculation, Please enter the name of the weight column:",[None] + list(df.columns))

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
        column_to_describe = st.selectbox("Select a Variable", [None] + list(df.columns))

        # Display details for numeric columns
        # if column_to_describe is None
        if column_to_describe is None:
            if (numeric_columns is not None) or (categorical_columns is not None):
                for column in df.columns:
                    col1, col2 = st.columns(2)

                    with col1:
                        column_desc = sf.describe_columns(df, column)

                        if column_desc is not None:
                            streamlit_description_base(column, column_desc)
                        else:
                            st.error(f"Column {column} description could not be generated.")

                    with col2:
                        sf.create_histogram(df, column)
            else:
                st.error(f"No Numeric or categorical column are in the data you provided")
        else:
            column=column_to_describe
            bins=30

            col1, col2 = st.columns(2)

            with col1:
                column_desc = sf.describe_columns(df, column)

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

                        plt=sf.plot_histogram(df, column,category_var, bins)
                        st.pyplot(plt)
                    elif column in categorical_columns:
                        fig=sf.bar_plot_category(df,column)
                        st.pyplot(fig)
                with tab2:
                    if column in numeric_columns:
                        category_var_bp = st.selectbox("Select a Category Variable to box plot", [None] + categorical_columns)
                        fig=sf.box_plot_calculation(df, column,category_var_bp)
                        st.pyplot(fig)
                    elif column in categorical_columns:
                        st.table(column_desc["Value Counts"])
                if tab3:
                    with tab3:
                        st.write("### Scatterplot with Regression")
                        # Dropdowns to select x and y columns
                        x_col = st.selectbox("Select X-axis column", options=[col for col in numeric_columns if col != column])
                        fig, beta, r_squared = sf.scatterplot_with_regression(df, column, x_col)
                        # Display the plot in Streamlit
                        # Display regression metrics
                        st.write(f"**Beta (Slope):** {beta:.4f} **R-squared:** {r_squared:.4f}")

                        st.pyplot(fig)




        # Calculate the correlation matrix
        st.write("---------------------------------------------\n"
                 " ## Correlation")
        corr_matrix = sf.calculate_correlation(df[numeric_columns])

        # Create tabs for the correlation table and heatmap
        tab1, tab2 = st.tabs(["Correlation Table", "Heatmap"])

        with tab1:
            show_correlation_table(corr_matrix)

        with tab2:
            plt=sf.plot_heatmap(corr_matrix)
            st.pyplot(plt)

        st.write("#### Sample ")
        st.dataframe(df.head(5))




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


import pandas as pd


def highlight_columns(df: pd.DataFrame,numeric, corr_threshold: float = 0.8, missing_threshold: float = 20,
                      std_threshold: float = 0.95):
    """
    Analyzes the DataFrame for highlights based on correlation, missing values, and standard deviation.

    Args:
        df (pd.DataFrame): The input DataFrame.
        corr_threshold (float): The threshold for high correlation.
        missing_threshold (float): The percentage threshold for missing values.
        std_threshold (float): The threshold for high standard deviation.

    Returns:
        dict: A dictionary containing highlights for each column.
    """
    highlights = {}
    corr_matrix= None
    # Calculate correlation matrix
    if numeric is not None:
        corr_matrix = df[numeric].corr()

    # Iterate over columns
    for col in df.columns:
        column_highlights = {}

        # Check for high correlation
        if corr_matrix is not None:
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
            std_dev = df[col].std()
            if std_dev > std_threshold:
                column_highlights['Standard Deviation'] = std_dev

        # Add to highlights if any conditions are met
        if column_highlights:
            highlights[col] = column_highlights

    return highlights


if __name__ == "__main__":
    main()
