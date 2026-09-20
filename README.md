# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:

To write a Python program to implement the Decision Tree Classifier model for predicting employee churn.

## EQUIPMENTS REQUIRED:

1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter Notebook
3. Employee.csv

## ALGORITHM

1. Import the required Python libraries such as Pandas and Scikit-learn.
2. Load the `Employee.csv` dataset using Pandas.
3. Select the relevant employee attributes as input features and `left` as the target variable.
4. Split the dataset into training and testing data.
5. Create a Decision Tree Classifier model.
6. Train the Decision Tree model using the training dataset.
7. Predict the employee churn status for the testing dataset.
8. Calculate the accuracy of the Decision Tree Classifier.
9. Display the confusion matrix and classification report.
10. Predict the churn status of a new employee using the trained model.

## PROGRAM:

```python
# Program to implement the Decision Tree Classifier
# Model for Predicting Employee Churn.
#
# Developed by: NISHALI SHREE R
# RegisterNumber: 212225080036

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Load the dataset
data = pd.read_csv("Employee.csv")

# Display the first five records
print("First five records of the dataset:")
print(data.head())

# Select input features
X = data[
    [
        "satisfaction_level",
        "last_evaluation",
        "number_project",
        "average_montly_hours",
        "time_spend_company",
        "Work_accident",
        "promotion_last_5years"
    ]
]

# Select target variable
y = data["left"]

# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create Decision Tree Classifier
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Predict employee churn
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Display actual and predicted values
print("\nActual Churn Status:")
print(y_test.values[:20])

print("\nPredicted Churn Status:")
print(y_pred[:20])

# Display accuracy
print("\nAccuracy:", accuracy)

print("\nAccuracy Percentage:", accuracy * 100, "%")

# Display confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Predict churn for a new employee
new_employee = pd.DataFrame({
    "satisfaction_level": [0.35],
    "last_evaluation": [0.80],
    "number_project": [5],
    "average_montly_hours": [250],
    "time_spend_company": [5],
    "Work_accident": [0],
    "promotion_last_5years": [0]
})

# Predict employee churn
prediction = model.predict(new_employee)

print("\nNew Employee Details:")
print("Satisfaction Level: 0.35")
print("Last Evaluation: 0.80")
print("Number of Projects: 5")
print("Average Monthly Hours: 250")
print("Time Spent in Company: 5 years")
print("Work Accident: 0")
print("Promotion in Last 5 Years: 0")

# Display prediction
if prediction[0] == 1:
    print("\nPredicted Churn Status: Employee will leave")
else:
    print("\nPredicted Churn Status: Employee will stay")
```

## Output:

<img width="659" height="810" alt="image" src="https://github.com/user-attachments/assets/e8c7e707-e51b-4794-9cdd-0bc1fe4b1769" />


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
