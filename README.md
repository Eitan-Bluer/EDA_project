
# README

## Overview
This project is designed to help users analyze and visualize data using Streamlit. It includes functionalities for data import, statistical calculations, and data visualization. The main features are divided into three Python scripts:

1. **`import_data_scripts.py`**: Handles data importing.
2. **`statistic_function.py`**: Contains statistical and visualization functions.
3. **`main.py`**: The main Streamlit application.

---

## File Descriptions

### 1. `import_data_scripts.py`
This script provides functions to read data files from different sources.

#### Functions:
- **`read_data_file(path, file_name, file_type)`**
  - Reads a data file from a local path.
  - Supports CSV and Excel files.

- **`uploaded_file_function(uploaded_file)`**
  - Reads a file uploaded via Streamlit.

- **`read_data_file_from_url(url)`**
  - Reads a data file from a provided URL.

### 2. `statistic_function.py`
This script includes functions for analyzing and visualizing data.

#### Functions:
- **`numeric_category(df)`**
  - Classifies columns into numeric and categorical types.

- **`overview(df)`**
  - Generates a summary of the DataFrame, including missing values, duplicates, and column types.

- **`write_overview_to_st(overview_stats)`**
  - Displays the DataFrame overview in the Streamlit app.

- **`describe_columns(df, selected_column=None, weight=None)`**
  - Provides detailed statistics for each column.

- **`create_histogram(df, column, bins=30, weight=None)`**
  - Creates a histogram for numeric data or a bar plot for categorical data.

- **`calculate_correlation(df)`**
  - Computes the correlation matrix for numeric columns.

- **`plot_heatmap(corr_matrix)`**
  - Visualizes the correlation matrix using a heatmap.

- **`box_plot_calculation(df, column)`**
  - Generates a box plot for a numeric column.

- **`bar_plot_category(df, category_column)`**
  - Generates a bar plot for a categorical column.

### 3. `main.py`
This is the entry point of the Streamlit application. It integrates the functions from the other scripts to provide an interactive interface for users.

#### Features:
- Data Import:
  - Upload files via Streamlit.
  - Import files from a URL.

- Data Overview:
  - Display statistics like missing values, numeric/categorical column counts, and duplicate rows.

- Column Highlights:
  - Identify columns with high correlation, missing values, or high standard deviation.

- Data Visualization:
  - Histograms and bar plots for selected columns.
  - Correlation matrix and heatmap.

- Integration with `statistic_function.py` for detailed column analysis and visualization.

---

## Installation

This project uses Poetry for dependency management.

1. Clone this repository:
   ```bash
   git clone <repository_url>
   ```

2. Install Poetry if not already installed:
   ```bash
   pip install poetry
   ```

3. Install the project dependencies:
   ```bash
   poetry install
   ```

4. Run the Streamlit app:
   ```bash
   poetry run streamlit run main.py
   ```

---

## Usage

1. Open the app in your browser (Streamlit provides the local URL).
2. Upload a file or provide a URL to load your dataset.
3. Explore the dataset through the following features:
   - View an overview of the data.
   - Analyze specific columns for missing values, correlation, and statistical details.
   - Generate visualizations like histograms, bar plots, box plots, and heatmaps.

---

## Example

### File Upload Example:
1. Select "Local File" as the data source.
2. Upload a CSV or Excel file.
3. View statistics and generate visualizations for the uploaded dataset.

### URL Example:
1. Select "URL" as the data source.
2. Paste the URL of a CSV or Excel file.
3. Explore the dataset interactively.

---

## Highlights Detection
The `highlight_columns` function in `main.py` identifies columns with:
- **High Correlation**: Correlation > 0.8 with other columns.
- **Missing Values**: More than 20% missing data.
- **High Standard Deviation**: Standard deviation > 0.95.

---

## Dependencies
- Python 3.12
- Streamlit 1.41.1
- Pandas 2.2.3
- Seaborn 0.13.2
- OpenPyXL 3.1.5

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

## Contributing
Feel free to submit issues and pull requests for improvements or new features.
