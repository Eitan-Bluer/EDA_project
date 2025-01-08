
# EDA Project

## Overview

The EDA Project is a Streamlit-based application designed for interactive exploratory data analysis (EDA). It allows users to analyze datasets with or without weights, providing capabilities for data import, statistical analysis, and visualization.

# streamlit link
https://edaproject-nw83a9ftjfworft9amo6jr.streamlit.app/
---

## File Structure and Descriptions

### 1. `main.py`
The main script that integrates functionalities from other modules to perform EDA.

#### Key Features:
- **Data Import:**
  - Upload files via Streamlit or provide a URL to load datasets (CSV/Excel).
- **Unweighted Analysis:**
  - Provides an overview of dataset statistics, column highlights, and visualizations without considering weights.
- **Weighted Analysis:**
  - Enables users to specify a weight column and adjust all calculations accordingly.
- **Data Visualization:**
  - Includes histograms, bar plots, box plots, scatter plots with regression, and correlation heatmaps.
- **Column Highlights:**
  - Detects:
    - Highly correlated columns (correlation > 0.8).
    - Columns with missing values (>20%).
    - Columns with high (weighted/unweighted) standard deviation.

---

### 2. `statistic_function.py`
Provides core statistical analysis and visualization functions for unweighted analysis.

#### Functions:
- **Data Classification:**
  - `numeric_category(df)` identifies numeric and categorical columns.
- **Overview:**
  - `overview(df)` generates a dataset summary, including variables, observations, missing values, and duplicates.
  - `write_overview_to_st(overview_stats)` displays the summary in Streamlit.
- **Column Descriptions:**
  - `describe_columns(df, selected_column=None)` provides detailed statistics for columns.
- **Visualization:**
  - `create_histogram(df, column, bins=30)` creates histograms or bar plots.
  - `box_plot_calculation(df, column, category=None)` generates box plots for numeric data, grouped by categories if specified.
  - `bar_plot_category(df, category_column)` creates bar plots for categorical data.
  - `scatterplot_with_regression(df, y_col, x_col)` creates scatter plots with regression lines.
  - `plot_heatmap(corr_matrix)` visualizes the correlation matrix.

---

### 3. `weight_statistic_function.py`
Enhances statistical analysis and visualization for weighted datasets.

#### Functions:
- **Weighted Statistics:**
  - `describe_columns_weight(df, selected_column=None, weight=None)` generates detailed statistics considering weights.
  - `weighted_stat(values, weights, stat_type, percentile=None)` calculates weighted mean, standard deviation, median, and percentiles.
- **Weighted Visualization:**
  - `create_histogram_weighted(df, column, bins=30, weight=None)` generates weighted histograms.
  - `bar_plot_category(df, category_column, weights_column)` creates weighted bar plots.
  - `box_plot_calculation(df, column, category=None, weights=None)` generates weighted box plots.
- **Correlation Analysis:**
  - `weighted_corr_matrix(df, weights)` computes a weighted correlation matrix.

---

### 4. `import_data_scripts.py`
Handles importing datasets from local files, URLs, or file uploads.

#### Functions:
- **File Import:**
  - `read_data_file(path, file_name, file_type)` reads files from a specified local path.
  - `uploaded_file_function(uploaded_file)` processes file uploads via Streamlit.
  - `read_data_file_from_url(url)` fetches data from a URL.

---

### 5. `main_unweighted.py`
Focuses on unweighted exploratory data analysis.

#### Features:
- Provides detailed statistics, column highlights, and visualizations for datasets without weights.
- Includes support for histograms, box plots, scatter plots, and correlation heatmaps.

---

### 6. `main_with_weight.py`
Handles weighted exploratory data analysis.

#### Features:
- Enables selection of a weight column from the dataset or an external source.
- Adjusts all statistics and visualizations to account for weights.

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
   - **Local File**: Upload a CSV or Excel file. in the folder you can find the  pps_sample_csv.csv  and vvs_1_work_excel
   - **URL**: Provide a direct link to a dataset (CSV/Excel) raw datasets. such as https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
3. Analyze the dataset:
   - **Unweighted Analysis**: Explore general dataset characteristics and relationships.
   - **Weighted Analysis**: Specify a weight column and perform weighted statistical analysis.
4. Visualize correlations and relationships:
   - Generate heatmaps, scatter plots, and other visualizations to uncover patterns.

---

## Dependencies

The project relies on the following Python packages:
- **Streamlit**: Interactive web application framework (`^1.41.1`)
- **Pandas**: Data manipulation and analysis (`^2.2.3`)
- **Seaborn**: Statistical data visualization (`^0.13.2`)
- **Matplotlib**: General-purpose plotting library (`^3.10.0`)
- **Numpy**: Numerical computations (`^2.2.1`)
- **Scikit-learn**: Machine learning utilities (`^1.6.0`)
- **Statsmodels**: Advanced statistical modeling (`^0.14.4`)
- **OpenPyXL**: Excel file handling (`^3.1.5`)

All dependencies are listed in `pyproject.toml` for seamless installation via Poetry.



