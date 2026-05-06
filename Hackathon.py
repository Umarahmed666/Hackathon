### 1. Data Collection

import pandas as pd
df = pd.read_csv("diabetes.csv")
print(df.head)
print(df.shape)

### 2. Data Preprocessing for Diabetes Prediction Project

# Step 1 — Import Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Step 2 — Load Dataset

df = pd.read_csv("diabetes.csv")
print(df.head())

#Step 3 — Check Missing Values

print(df.isnull().sum())

# Step 4 — Replace Invalid Zeros with NaN

cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

df[cols] = df[cols].replace(0, np.nan)

print(df.isnull().sum())

# Step 5 — Handle Missing Values

#Replace missing values using median:

df.fillna(df.median(), inplace=True)

print(df.isnull().sum())

# Step 6 — Separate Features and Target

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Step 7 — Feature Scaling / Normalization

#Using StandardScaler:

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print(X_scaled)

# Step 8 — Remove Outliers

#Using IQR Method:

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)

IQR = Q3 - Q1

df_clean = df[~((df < (Q1 - 1.5 * IQR)) | 
                (df > (Q3 + 1.5 * IQR))).any(axis=1)]

print(df.shape)
print(df_clean.shape)

# Step 9 — Final Clean Dataset

print(df_clean.head())

#Train/Test Split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42
)

### 3. Exploratory Data Analysis (EDA) — Diabetes Prediction Project

# Step 1 — Import Libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2 — Load Dataset

df = pd.read_csv("diabetes.csv")

print(df.head())

# Step 3 — Basic Dataset Information

print(df.info())

print(df.describe())

# 1. Age vs Diabetes :this shows how diabetes varies with age.

plt.figure(figsize=(8,5))

sns.boxplot(x='Outcome', y='Age', data=df)

plt.title("Age vs Diabetes")
plt.xlabel("Diabetes Outcome")
plt.ylabel("Age")

plt.show()

# 2. Glucose Distribution : Glucose is one of the most important features.

plt.figure(figsize=(8,5))

sns.histplot(df['Glucose'], kde=True)

plt.title("Glucose Distribution")
plt.xlabel("Glucose Level")

plt.show()

# 3. Correlation Heatmap : Shows relationships between features.

plt.figure(figsize=(10,8))

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

plt.title("Correlation Heatmap")

plt.show()

# 4. BMI Analysis

plt.figure(figsize=(8,5))

sns.boxplot(x='Outcome', y='BMI', data=df)

plt.title("BMI vs Diabetes")
plt.xlabel("Diabetes Outcome")
plt.ylabel("BMI")

plt.show()

### Optional Extra Visualizations :

# Diabetes Count Plot : Shows number of diabetic vs non-diabetic patients.

sns.countplot(x='Outcome', data=df)

plt.title("Diabetes Count")

plt.show()

# Pairplot : Useful for understanding feature relationships.

sns.pairplot(df, hue='Outcome')

plt.show()

# Pregnancies vs Diabetes

sns.barplot(x='Outcome', y='Pregnancies', data=df)

plt.title("Pregnancies vs Diabetes")

plt.show()

### 4. Model Training — Diabetes Prediction Project

# Step 1 — Import Libraries

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Step 2 — Prepare Features and Target

X = df.drop("Outcome", axis=1)

y = df["Outcome"]

# Step 3 — Split Data : 80% for training  20% for testing

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Step 4 — Check Shapes

print(X_train.shape)
print(X_test.shape)

# Step 5 — Create Model : Using Random Forest Classifier:

model = RandomForestClassifier()

# Step 6 — Train the Model

model.fit(X_train, y_train)

# Step 7 — Make Predictions

y_pred = model.predict(X_test)

print(y_pred)

# Step 8 — Check Accuracy

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

### 5. Model Evaluation — Diabetes Prediction Project

# Step 1 — Import Evaluation Libraries

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Step 2 — Accuracy : Measures overall correctness.

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# Step 3 — Precision : Controls false positives. Important in healthcare to avoid wrong positive predictions.

precision = precision_score(y_test, y_pred)

print("Precision:", precision)

# Step 4 — Recall : Measures how many actual diabetic patients were correctly identified. Very important in medical systems.

recall = recall_score(y_test, y_pred)

print("Recall:", recall)

# Step 5 — F1-Score Balanced score combining precision and recall.

f1 = f1_score(y_test, y_pred)

print("F1-Score:", f1)

# Step 6 — Confusion Matrix : Shows prediction breakdown.

cm = confusion_matrix(y_test, y_pred)

print(cm)

# Step 7 — Visualize Confusion Matrix

import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# Step 8 — Classification Report : Complete evaluation summary.

print(classification_report(y_test, y_pred))