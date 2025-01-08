
import streamlit as st

import import_data_scripts as ids

import main_with_weight as mww
import main_unweighted as mu
def main():
    st.title('Exploratory Data Analysis (EDA)')
    data_source = st.radio("Choose the data source:", ('Local File', 'URL'))
    df = None

    if data_source == 'Local File':
        uploaded_file2 = st.file_uploader("Upload your file", type=["csv", "xlsx"])
        df = ids.uploaded_file_function(uploaded_file2)

    elif data_source == 'URL':
        url = st.text_input('Enter the URL of the data file please make sure you provide "raw" file URL')
        if url:
            df = ids.read_data_file_from_url(url)
    weight_option = st.radio("Do you want to select a weight column?", ["No", "Yes"])
    if df is not None:
        if weight_option == "Yes":
            mww.main_weight(df)
        else:
            mu.main_unweighted(df)
    else:
        st.write("upload data")


if __name__ == "__main__":
    main()
