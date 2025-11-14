# -----------------------------------------
# 1. Import Libraries
# -----------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------
# 2. Create Sample Dataset (since you have no file)
# -----------------------------------------
data = {
    "Name": ["Ram", "Sam", "Priya", "Anu", None, "Vijay"],
    "Age": [23, 25, None, 22, 21, 29],
    "City": ["Chennai", "Madurai", "Chennai", None, "Trichy", "Chennai"],
    "Salary": [25000, 30000, 28000, None, 26000, 500000]  # 500000 = outlier
}

df = pd.DataFrame(data)

print("🔹 Original Dataset:")
print(df)
print("\n")


# -----------------------------------------
# 3. Check Missing Values
# -----------------------------------------
print("🔹 Missing Values:")
print(df.isnull().sum())
print("\n")


# -----------------------------------------
# 4. Handle Missing Values
# -----------------------------------------
df['Name'] = df['Name'].fillna("NoName")
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['City'] = df['City'].fillna("Unknown")
df['Salary'] = df['Salary'].fillna(df['Salary'].median())

print("🔹 After Handling Missing Values:")
print(df)
print("\n")


# -----------------------------------------
# 5. Remove Duplicate Rows
# -----------------------------------------
df = df.drop_duplicates()

print("🔹 After Removing Duplicates:")
print(df)
print("\n")


# -----------------------------------------
# 6. Detect & Remove Outliers
# (Salary > 95th percentile removed)
# -----------------------------------------
q95 = df['Salary'].quantile(0.95)
df = df[df['Salary'] < q95]

print("🔹 After Removing Outliers:")
print(df)
print("\n")


# -----------------------------------------
# 7. Encode Categorical Data (One-Hot Encoding)
# -----------------------------------------
df = pd.get_dummies(df, columns=['City'])

print("🔹 After Encoding Categorical Data:")
print(df)
print("\n")


# -----------------------------------------
# 8. Feature Scaling (Standardization)
# -----------------------------------------
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[['Age', 'Salary']] = scaler.fit_transform(df[['Age', 'Salary']])

print("🔹 After Scaling Numerical Values:")
print(df)
print("\n")


# -----------------------------------------
# 9. Save Cleaned Dataset
# -----------------------------------------
df.to_csv("cleaned_data.csv", index=False)

print("✅ Cleaned data saved as 'cleaned_data.csv'")
