# 05 - Tabular Preprocessing & Exploratory Data Analysis (EDA)

Before feeding tabular data into PyTorch neural networks, data preprocessing, exploratory analysis, and cleaning are essential steps. This module demonstrates data preparation pipelines using Pandas and Matplotlib on an industrial maintenance dataset.

---

## 📁 Module Structure

```text
05_pandas_for_ml/
├── dataframe_inspection.py                           # Loading CSV, inspecting schema, shape, head, data types, summary statistics
├── exploratory_data_analysis.py                      # Distribution histograms, summary metrics, correlation matrix
├── filtering.py                                      # Handling missing values (NaNs), imputation strategies, duplicate detection
├── feature_engineering.py                            # Feature transformations, normalization, and encoding foundations
├── industrial_maintenance_with_missing_values.csv    # Industrial equipment dataset with sensor readings and missing values
└── README.md                                         # Complete preprocessing & EDA guide
```

---

## 🧠 Core Preprocessing Steps

### 1. DataFrame Inspection (`dataframe_inspection.py`)
Understanding the data distribution, column types, and volume:
- `df.shape`: Dimensionality of the tabular dataset `(rows, columns)`.
- `df.columns`: Column names representing operational sensor telemetry.
- `df.info()`: Memory usage, column non-null counts, and data types.
- `df.describe()`: Five-number summary (mean, std, min, 25%, 50%, 75%, max) for numeric features.

---

### 2. Exploratory Data Analysis (`exploratory_data_analysis.py`)
- **Distribution Analysis**: Plotting feature histograms (`operating_hours.hist()`) to examine skewness, outliers, and operational ranges.
- **Feature Correlation**: Calculating Pearson correlation coefficients (`df.corr(numeric_only=True)`) to assess multicollinearity and dependencies among sensor measurements.

---

### 3. Missing Value Handling & Imputation (`filtering.py`)
- **Missing Value Detection**: `df.isnull().sum()` identifies columns containing `NaN` or unrecorded telemetry.
- **Statistical Imputation**: Replacing missing values with feature statistics (e.g. mean imputation: `df['operating_hours'].fillna(mean)`).
- **Duplicate Verification**: `df.duplicated().sum()` identifies and tracks duplicate records.

---

## 📊 Dataset Overview: Industrial Predictive Maintenance

The accompanying dataset (`industrial_maintenance_with_missing_values.csv`) contains industrial sensor measurements:
- **`operating_hours`**: Cumulative operational runtime of the equipment.
- **`rotation_speed_rpm`**: Angular rotational speed of the motor.
- **`torque_nm`**: Measured shaft torque in Newton-meters.
- **`power_kw`**: Electrical power drawn by the machine.
- **`ambient_temp_c`**: Ambient operating temperature.
- **`failure_mode_type`**: Classification target indicator.

---

## 🚀 How to Run

```bash
python 05_pandas_for_ml/dataframe_inspection.py
python 05_pandas_for_ml/exploratory_data_analysis.py
python 05_pandas_for_ml/filtering.py
```
