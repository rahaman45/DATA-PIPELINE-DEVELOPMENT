## DATA PIPELINE DEVELOPMENT 


**NAME:**  KS ABDUL RAHAMAN 

**INTERN ID:**  CT04DG2064

**DOMAIN:**  DATA SCIENCE 

 **DURATION:** 4 WEEKS

  *MENTOR: *  NEELA SANTOSH


The goal of this project is to create a robust and modular pipeline that processes raw data into a clean and model-ready format. This pipeline is particularly useful in real-world machine learning projects where data cleaning, transformation, and preprocessing are routine but crucial tasks.

# We use:

Pandas for data handling

Scikit-learn for building preprocessing pipelines (Pipeline, ColumnTransformer)

Optional: integration with joblib, pickle, or model training libraries for extended functionality

📦 Project Requirements
🧰 Python Libraries:
Install necessary packages via pip:


pip install pandas scikit-learn
✅ Required Libraries in Code:

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
🛠️ Step-by-Step Code and Explanation
1. Load Raw Dataset
python
Copy code
# Sample raw dataset with missing values and mixed types
data = {
    'age': [25, 30, None, 40, 22],
    'gender': ['Male', 'Female', 'Female', None, 'Male'],
    'salary': [50000, 60000, 70000, None, 40000],
    'purchased': ['No', 'Yes', 'Yes', 'No', 'Yes']
}
df = pd.DataFrame(data)
We simulate a real-world scenario with:

Missing values in numerical and categorical columns

Mixed data types (numeric, categorical, binary)

2. Separate Features and Target
python
Copy code
X = df.drop('purchased', axis=1)
y = df['purchased']
X: Feature matrix (inputs to the model)

y: Target column (purchased)

3. Define Column Types
python
Copy code
numeric_features = ['age', 'salary']
categorical_features = ['gender']
This is important because different types of columns require different kinds of preprocessing.

4. Create Pipelines for Preprocessing
4.1. Numeric Pipeline
python
Copy code
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),        # Fill missing values with mean
    ('scaler', StandardScaler())                        # Standardize numerical data
])
4.2. Categorical Pipeline
python
Copy code
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing categorical values with mode
    ('encoder', OneHotEncoder(handle_unknown='ignore'))    # One-hot encode categorical variables
])
Each pipeline:

Uses an imputer to fill missing values

Applies encoding or scaling relevant to data type

5. Combine Pipelines Using ColumnTransformer
python
Copy code
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])
The ColumnTransformer applies different preprocessing pipelines to different subsets of columns — crucial for efficient and clean data transformation.

6. Integrate into a Full Pipeline
python
Copy code
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor)
])
The full_pipeline becomes a reusable, single-step object that transforms raw data into a model-ready format.

7. Apply Pipeline to Data
python
Copy code
X_processed = full_pipeline.fit_transform(X)
fit_transform() both learns from the data (mean, mode) and transforms it

The result is a NumPy array with processed features, ready for modeling

To convert this into a readable DataFrame (optional):

python
Copy code
from sklearn.compose import make_column_selector as selector

# Get feature names
encoded_columns = full_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(categorical_features)
all_columns = list(encoded_columns) + numeric_features

# Convert to DataFrame
processed_df = pd.DataFrame(X_processed.toarray(), columns=all_columns)
8. Save Final Processed Data
python
Copy code
processed_df.to_csv('processed_data.csv', index=False)
print("Data saved to processed_data.csv")
Saving the final output allows integration with model training pipelines or further analytics.

📊 Why Use This Advanced Pipeline?
✅ Benefits:
Modularity: You can swap out components (e.g., imputer or scaler) easily

Reusability: Use the same pipeline for training and test datasets

Compatibility: Directly integrates with GridSearchCV and modeling APIs

Maintainability: Clean separation of logic for numeric and categorical data

Efficiency: Prevents data leakage by isolating fit and transform operations

🧠 Use Cases
Building ML models with Scikit-learn

Preprocessing datasets before feeding into deep learning models

Automating pipelines for MLOps workflows

Teaching or demoing best practices in data science

✅ Final Notes
This project sets up a reliable and scalable data preprocessing pipeline using industry-standard tools. It prepares you for building machine learning workflows in a production or real-world setting, and ensures that your data transformation logic is both repeatable and explainable.

If you're building a larger project, consider integrating this with:

Model training pipelines

Joblib for saving pipelines

MLflow or DVC for tracking experiments

Let me know if you want this entire project as a ready-to-run Python script or Jupyter notebook.

## OUTPUT:1
![Image](https://github.com/user-attachments/assets/2c1a69a6-4a34-4384-8d00-9680f433c298)

## OUTPUT:2

![Image](https://github.com/user-attachments/assets/3f8050e2-13d5-4c34-b36b-71bc4f39fbf7)




