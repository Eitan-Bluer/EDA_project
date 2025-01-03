import streamlit as st
import import_data_scripts as ids
import statistic_function as sf

def main():
    st.title('Data Import, Statistics and Visualization')

    # Choose the data source (Local or URL)
    data_source = st.radio("Choose the data source:", ('Local File', 'URL'))

    df = None

    if data_source == 'Local File':
        uploaded_file = st.file_uploader("Upload your file", type=["csv", "xlsx"])
        df = ids.uploaded_file_function(uploaded_file)

    elif data_source == 'URL':
        url = st.text_input('Enter the URL of the data file please make sure you provide "raw" file URL')
        if url:
            df = ids.read_data_file_from_url(url)


    if df is not None:
        # Generate the overview statistics
        overview_stats = sf.overview(df)
        # Display the overview in Streamlit
        sf.write_overview_to_st(overview_stats)

        # Numeric and Categorical Columns
        numeric_columns, categorical_columns = sf.numeric_category(df)
        # Get the list of columns for selection
        column_to_describe = st.selectbox("Select a column to see its description", [None] + list(df.columns))

        # Display details for numeric columns
        # if column_to_describe is None
        if column_to_describe is None:
            if (numeric_columns is not None) or (categorical_columns is not None):
                for column in df.columns:
                    col1, col2 = st.columns(2)

                    with col1:
                        column_desc = sf.describe_columns(df, column)

                        if column_desc is not None:
                            streamlit_description(column, column_desc)
                        else:
                            st.error(f"Column {column} description could not be generated.")

                    with col2:
                        sf.create_histogram(df, column)
            else:
                st.error(f"No Numeric or categorical column are in the data you provided")
        else:
            column=column_to_describe
            bins=30

            if column in numeric_columns:
                bins = st.slider('Select number of bins', min_value=5, max_value=50, value=10, step=1)
            col1, col2 = st.columns(2)

            with col1:
                column_desc = sf.describe_columns(df, column)

                if column_desc is not None:
                    streamlit_description(column, column_desc)

                else:
                    st.error(f"Column {column} description could not be generated.")

            with col2:
                sf.create_histogram(df, column,bins)



def streamlit_description(column,column_desc):
    st.subheader(f"**{column}**")
    st.write(f"**Type:** {column_desc['Type']}")
    st.write(f"**Unique Values:** {column_desc['Unique Values']}")
    st.write(
        f"**Missing Values:** {column_desc['Missing Values']} ({column_desc['Missing Values (%)']:.2f}%)")
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


if __name__ == "__main__":
    main()
