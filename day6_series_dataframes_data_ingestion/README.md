# Day 6 - Pandas Series, DataFrames & Data Ingestion

## Objective

Learn Pandas Series, DataFrames, and data ingestion from multiple file formats.

## Tasks Completed

### Task 2 - Pandas Series

Created Series using:
- List
- Dictionary
- NumPy array
- Custom index

### Task 3 - Pandas DataFrames

Created DataFrames using:
- Dictionary of lists
- List of dictionaries
- NumPy 2D array

### Task 4 - Data Ingestion

Created a realistic e-commerce dataset with 150 rows and 7 columns.

Generated:
- CSV
- Excel
- JSON

### Task 5 - Dataset Inspection

Inspected datasets using:
- `head()`
- `tail()`
- `info()`
- `describe()`
- `dtypes`
- `columns`
- `shape`

### Task 6 - Practical DataFrame Functions

Implemented:
- Dataset loading
- DataFrame summary
- Column selection
- `.loc`
- `.iloc`
- Row slicing

### Task 7 - Testing

Created tests for:
- CSV loading
- Excel loading
- JSON loading
- Column validation
- Missing files
- Unsupported formats
- Null-value validation

Result:

7 tests passed.

## Dataset

The canonical e-commerce dataset contains:

- Order_ID
- Customer
- Product
- Category
- Quantity
- Unit_Price
- Order_Date

Dataset size: 150 rows x 7 columns.

## Project Structure

```text
day6_series_dataframes_data_ingestion/
|-- data/
|   |-- ecommerce_orders.csv
|   |-- ecommerce_orders.xlsx
|   `-- ecommerce_orders.json
|-- day6_series_dataframes.py
|-- task3_dataframes.py
|-- task4_data_ingestion.py
|-- task5_dataset_inspection.py
|-- task6_practical_functions.py
|-- test_day6.py
`-- README.md