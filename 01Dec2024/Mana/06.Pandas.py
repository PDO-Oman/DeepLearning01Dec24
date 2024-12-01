import pandas as pd
import numpy as np

# 1. **Creating Pandas DataFrames**

# Create a DataFrame from a dictionary
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Edward'],
    'Age': [24, 27, 22, 32, 29],
    'Salary': [50000, 60000, 55000, 80000, 65000],
    'Department': ['HR', 'Finance', 'IT', 'IT', 'HR']
}

df = pd.DataFrame(data)
print("1. DataFrame Created from Dictionary:\n", df)

# Create a DataFrame from a NumPy array
data_array = np.array([
    ['Alice', 24, 50000, 'HR'],
    ['Bob', 27, 60000, 'Finance'],
    ['Charlie', 22, 55000, 'IT'],
    ['David', 32, 80000, 'IT'],
    ['Edward', 29, 65000, 'HR']
])

df_from_array = pd.DataFrame(data_array, columns=['Name', 'Age', 'Salary', 'Department'])
print("\n2. DataFrame Created from NumPy Array:\n", df_from_array)


# 2. **Data Selection and Indexing**

# Select a specific column
age_column = df['Age']
print("\n3. Select 'Age' Column:\n", age_column)

# Select multiple columns
salary_department = df[['Salary', 'Department']]
print("\n4. Select 'Salary' and 'Department' Columns:\n", salary_department)

# Select rows by index (e.g., first 3 rows)
top_rows = df.head(3)
print("\n5. First 3 Rows of DataFrame:\n", top_rows)

# Select rows by condition (e.g., employees with salary > 60000)
high_salary = df[df['Salary'] > 60000]
print("\n6. Employees with Salary > 60000:\n", high_salary)
#

# 3. **Data Cleaning**

# Add a new column with missing values
df['Bonus'] = [5000, np.nan, 3000, 10000, np.nan]
print("\n7. DataFrame with Missing Values:\n", df)

# Fill missing values with a specific value (e.g., filling NaN with 0)
df_filled = df.fillna(0)
print("\n8. DataFrame After Filling Missing Values:\n", df_filled)

# Drop rows with missing values
df_dropped = df.dropna()
print("\n9. DataFrame After Dropping Rows with Missing Values:\n", df_dropped)


# 4. **Data Transformation**

# Apply a function to a column (e.g., doubling the salary)
df['Double Salary'] = df['Salary'].apply(lambda x: x * 2)
print("\n10. Salary Doubled:\n", df[['Name', 'Salary', 'Double Salary']])

# Normalize a column (e.g., normalizing the 'Salary' column)
salary_normalized = (df['Salary'] - df['Salary'].min()) / (df['Salary'].max() - df['Salary'].min())
df['Normalized Salary'] = salary_normalized
print("\n11. Normalized Salary:\n", df[['Name', 'Salary', 'Normalized Salary']])


# 5. **Grouping and Aggregation**

# Group by 'Department' and calculate the mean salary
grouped_by_dept = df.groupby('Department')['Salary'].mean()
print("\n12. Mean Salary by Department:\n", grouped_by_dept)

# Group by 'Department' and get multiple aggregations
aggregated_data = df.groupby('Department').agg({
    'Salary': ['mean', 'max', 'min'],
    'Age': ['mean', 'std']
})
print("\n13. Aggregated Data (Mean, Max, Min, etc.):\n", aggregated_data)


# 6. **Merging and Concatenation**

# Concatenate two DataFrames (vertical concatenation)
df1 = pd.DataFrame({
    'Name': ['Alice', 'Bob'],
    'Age': [24, 27],
    'Salary': [50000, 60000]
})
df2 = pd.DataFrame({
    'Name': ['Charlie', 'David'],
    'Age': [22, 32],
    'Salary': [55000, 80000]
})
df_concatenated = pd.concat([df1, df2], ignore_index=True)
print("\n14. Concatenated DataFrames:\n", df_concatenated)

# Merge two DataFrames based on a common column
df_left = pd.DataFrame({'Name': ['Alice', 'Bob'], 'Department': ['HR', 'Finance']})
df_right = pd.DataFrame({'Department': ['HR', 'Finance'], 'Budget': [50000, 60000]})
merged_df = pd.merge(df_left, df_right, on='Department')
print("\n15. Merged DataFrames:\n", merged_df)


# 7. **Handling Categorical Data**

# Convert 'Department' to a categorical variable
df['Department'] = df['Department'].astype('category')
print("\n16. Department as Categorical:\n", df.dtypes)

# Get the categories of the 'Department' column
categories = df['Department'].cat.categories
print("\n17. Categories in 'Department':\n", categories)


# 8. **Data Pivoting**

# Pivot the data (for example, pivoting by department)
pivoted_df = df.pivot_table(values='Salary', index='Department', aggfunc='mean')
print("\n18. Pivoted DataFrame (Mean Salary by Department):\n", pivoted_df)

