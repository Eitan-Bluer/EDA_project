#import pandas as pd
import streamlit as st
#import openpyxl
import import_data_scripts as ids
import statistic_function as sf  # Assuming you have this module for additional stats functions

def main():
    st.title('File Upload and Data Reading Example (Local or "raw" file URL)')

    # Choose the data source (Local or URL)
    data_source = st.radio("Choose the data source:", ('Local File', 'URL'))

    # Initialize df as None
    df = None

    # If Local File is selected
    if data_source == 'Local File':
        path = st.text_input('Enter the file path')
        file_name = st.text_input('Enter the file name')
        file_type = st.selectbox('Select file type', ['csv', 'xls', 'xlsx'])

        # Button to load the local file
        if st.button('Load Local Data'):
            if path and file_name:
                # Read the file using your existing function
                df = ids.read_data_file(path, file_name, file_type)
                if df is not None:
                    st.write(f"Data from {file_name}.{file_type}:")
                    st.dataframe(df.head(5))  # Show first 5 rows
                else:
                    st.error("Failed to load data")
            else:
                st.error("Please provide both file path and file name")

    # If URL is selected
    elif data_source == 'URL':
        url = st.text_input('Enter the "raw" URL of the data file')

        # Button to load the data from URL
        if st.button('Load Data from URL'):
            if url:
                # Read the data from the raw URL using your existing function
                df = ids.read_data_file_from_url(url)
                if df is not None:
                    st.write(f"Data from the file at {url}:")
                    st.dataframe(df.head(5))  # Show first 5 rows
                else:
                    st.error("Failed to load data")
            else:
                st.error("Please provide a valid URL")

    # After loading the data, generate and display the overview
    if df is not None:
        # Generate the overview statistics
        overview_stats = sf.overview(df)

        # Display the overview in Streamlit
        sf.write_overview_to_st(overview_stats)
if __name__ == "__main__":
    main()


