# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Step 1: Create a sample dataset
data = {
    'age': [25, 30, 35, 40, None],               # Some missing age values
    'gender': ['Male', 'Female', None, 'Male', None],  # Some missing gender values
    'salary': [50000, 60000, 70000, None, 40000],      # Some missing salary values
    'purchased': ['No', 'Yes', 'Yes', 'No', 'Yes']     # Purchase status
}

# Convert the dictionary to a pandas DataFrame
df = pd.DataFrame(data)

# Step 2: Handle missing values
# Fill missing age values with the mean of the 'age' column
df['age'].fillna(df['age'].mean(), inplace=True)

# Fill missing gender values with the most frequent value (mode)
df['gender'].fillna(df['gender'].mode()[0], inplace=True)

# Fill missing salary values with the mean of the 'salary' column
df['salary'].fillna(df['salary'].mean(), inplace=True)

# Step 3: Encode categorical variables using Label Encoding
# Encode 'gender': Male will become 1, Female will become 0
le_gender = LabelEncoder()
df['gender'] = le_gender.fit_transform(df['gender'])

# Encode 'purchased': Yes will become 1, No will become 0
le_purchased = LabelEncoder()
df['purchased'] = le_purchased.fit_transform(df['purchased'])

# Step 4: Scale numerical features (age and salary) using StandardScaler
scaler = StandardScaler()

# Apply scaling and replace the original values with the scaled ones
df[['age', 'salary']] = scaler.fit_transform(df[['age', 'salary']])

# Step 5: Rename the scaled columns to indicate they are scaled
df.rename(columns={'age': 'age_scaled', 'salary': 'salary_scaled'}, inplace=True)

# Step 6: Save the cleaned and processed DataFrame to a CSV file
df.to_csv("complete_output.csv", index=False)

# Final confirmation message
print("Saved: complete_output.csv")
