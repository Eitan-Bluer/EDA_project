
# EDA Project

## Overview

The EDA Project is a Streamlit-based application that enables users to perform exploratory data analysis (EDA) with ease. It offers capabilities for data import, statistical analysis, and data visualization, making it an ideal tool for exploring datasets interactively.

---

## File Structure and Descriptions

### 1. `main.py`
The main script of the Streamlit application, integrating functionalities from other modules for an interactive EDA workflow.

#### Key Features:
- **Data Import:**
  - Upload files via Streamlit or provide a URL to load datasets (CSV/Excel).
- **Data Overview:**
  - Displays key dataset statistics such as missing values, duplicates, and column types.
- **Highlights Detection:**
  - Identifies:
    - Highly correlated columns (correlation > 0.8).
    - Columns with high missing values (>20%).
    - Columns with high standard deviation (>0.95).
- **Data Visualization:**
  - Histograms, bar plots, box plots, scatter plots with regression, and correlation heatmaps.
- **Column Descriptions:**
  - Generates detailed statistics for selected columns, including mean, median, range, and value counts.

---

### 2. `statistic_function.py`
Provides core statistical analysis and visualization functions.

#### Functions:
- **Data Classification:**
  - `numeric_category(df)` separates columns into numeric and categorical types.
- **Overview:**
  - `overview(df)` generates a summary of the dataset, including variables, observations, missing values, and duplicates.
  - `write_overview_to_st(overview_stats)` displays the overview in Streamlit.
- **Column Descriptions:**
  - `describe_columns(df, selected_column=None)` provides detailed statistics for columns.
- **Visualization:**
  - `create_histogram(df, column, bins=30)` creates histograms for numeric data or bar plots for categorical data.
  - `box_plot_calculation(df, column, category=None)` generates box plots for numeric columns, optionally grouped by categories.
  - `bar_plot_category(df, category_column)` creates bar plots for categorical columns.
  - `scatterplot_with_regression(df, y_col, x_col)` produces scatter plots with regression lines and calculates R² and slope.
  - `plot_heatmap(corr_matrix)` visualizes the correlation matrix as a heatmap.
- **Correlation:**
  - `calculate_correlation(df)` computes the correlation matrix for numeric columns.

---

### 3. `import_data_scripts.py`
Handles data importing from local files, URLs, or file uploads.

#### Functions:
- **File Import:**
  - `read_data_file(path, file_name, file_type)` reads a file from a specified local path (CSV/Excel).
  - `uploaded_file_function(uploaded_file)` handles file uploads via Streamlit.
  - `read_data_file_from_url(url)` imports data from a URL (supports CSV/Excel).

---

## Installation

This project uses Poetry for dependency management.

1. Clone this repository:
   ```bash
   https://github.com/Eitan-Bluer/EDA_project.git
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

### Prerequisites
- Python 3.12 or higher

### Steps
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd eda-project
   ```

2. Install dependencies using Poetry:
   ```bash
   pip install poetry
   poetry install
   ```

3. Run the application:
   ```bash
   poetry run streamlit run main.py
   ```

---

## Usage

1. Launch the Streamlit app.
2. Choose a data source:
   - **Local File**: Upload a CSV or Excel file.
   - **URL**: Provide a direct link to a dataset (CSV/Excel).
3. Analyze the dataset:
   - View a statistical overview of the dataset.
   - Explore individual columns for missing values, descriptive statistics, and visualizations.
4. Visualize correlations and relationships:
   - Generate heatmaps, scatter plots, and other visualizations to uncover patterns in the data.

---

## Example Workflow

### Scenario 1: File Upload
1. Select "Local File" as the data source.
2. Upload your dataset (CSV/Excel).
3. View an overview of the dataset, including missing values and column types.
4. Analyze specific columns or generate plots.

### Scenario 2: URL Data Source
1. Select "URL" as the data source.
2. Enter a direct URL to a dataset.
3. Perform the same analysis and visualizations as with uploaded files.

---

## Dependencies

This project relies on the following Python packages:

- **Streamlit**: Interactive web application framework (`^1.41.1`)
- **Pandas**: Data manipulation and analysis (`^2.2.3`)
- **Seaborn**: Statistical data visualization (`^0.13.2`)
- **Matplotlib**: General-purpose plotting library (`^3.10.0`)
- **Numpy**: Numerical computations (`^2.2.1`)
- **Scikit-learn**: Machine learning utilities (`^1.6.0`)
- **Statsmodels**: Advanced statistical modeling (`^0.14.4`)
- **OpenPyXL**: Excel file handling (`^3.1.5`)

All dependencies are listed in `pyproject.toml` for seamless installation via Poetry.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

