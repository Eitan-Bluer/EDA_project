import streamlit as st
import pandas as pd
import import_data_scripts as ids


def upload_and_merge(existing_df):
    """
    Upload a new file, filter it, and merge it with the existing DataFrame.

    Args:
        existing_df (pd.DataFrame): The existing DataFrame from main.py.

    Returns:
        pd.DataFrame: The merged DataFrame.
    """
    st.subheader("Upload File to Merge")
    uploaded_file = st.file_uploader("Upload your file", type=["csv", "xlsx"], key="merge_file_uploader")

    if uploaded_file:
        # Load the new file
        new_df = ids.uploaded_file_function(uploaded_file)
        filtered_df = None
        if new_df is not None:
            st.success(f"Successfully loaded {uploaded_file.name}")
            st.write("New File Preview:")
            st.dataframe(new_df.head())

            # Filtering Section
            st.subheader("Filter the Uploaded File")
            columns_to_filter = st.multiselect("Select columns you want to keep from weight data", new_df.columns.tolist(),
                                               key="filter_columns")

            if columns_to_filter:
                filtered_df = new_df[columns_to_filter].copy()

            else:
                st.write("you need to select columns to merge and weight column") # No filtering applied

            # Select columns for merging
            st.subheader("Merge Settings")

            if filtered_df is not None:
                existing_columns = existing_df.columns.tolist()
                new_columns = filtered_df.columns.tolist()

                left_on = st.multiselect("Select column from existing data to merge on", existing_columns, key="left_on")
                right_on = st.multiselect("Select column from uploaded file to merge on", new_columns, key="right_on")
                merge_type = st.radio("Select Merge Type", ["inner", "outer", "left", "right"], key="merge_type")

                if st.button("Merge Files", key="merge_button") :
                    try:
                        merged_df = pd.merge(existing_df, filtered_df, left_on=left_on, right_on=right_on, how=merge_type)
                        st.success("Files merged successfully!")
                        st.write("Merged DataFrame Preview:")
                        st.dataframe(merged_df.head())
                        return merged_df
                    except Exception as e:
                        st.error(f"Error during merge: {e}")
        else:
            st.error("Failed to load the uploaded file.")
    return existing_df
