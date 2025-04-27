import pandas as pd
print("DATA COLLECTING AND LOADING")
df = pd.read_excel('student_dropout_data.xlsx')
print("DATA LOADED")


print("DATA PREPROCESSING")
print(df.head())
print(df.info())
print(df.isnull().sum())
print("DATA IS INSPECTED")
df.dropna(inplace=True)
print("MISSING VALUES HANDLED")


print("EDA")
print("Basic statistics")
print(df.describe())
# Check for missing values
print("Missing Values:\n", df.isnull().sum())
print("Final DataFrame Info:")
print(df.info())
print("Final DataFrame Head:")
print(df.head())

# Creating a "Dropout" column based on predefined conditions
df["Dropout"] = ((df["Attendance"] < 50) | 
                 (df["Marks_Obtained"] < 40) | 
                 (df["Exam_Performance"] == "Poor")).astype(int)

# Display the first few rows to verify the new column
df[["Attendance", "Marks_Obtained", "Exam_Performance", "Dropout"]].tail()
print(df.info())

print("LOGISTIC REGRESSION")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Feature selection
features = ["Age", "Gender", "Location", "Distance_to_School", "Family_Income",
            "Family_Size", "Family_Stability", "Parents_Education_Level", "Marks_Obtained",
            "Attendance", "Exam_Performance", "Disciplinary_Records", "Learning_Habits",
            "School_Infrastructure", "Quality_of_Teaching", "Health_Issues"]

# Categorical columns that need encoding
categorical_columns = ["Gender", "Location", "Family_Stability", "Parents_Education_Level",
                        "Exam_Performance", "Disciplinary_Records", "Learning_Habits",
                        "School_Infrastructure", "Quality_of_Teaching", "Health_Issues"]

# Encoding categorical variables
label_encoders = {}
for col in categorical_columns:
    if col in df.columns and df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le  # Store encoders for future use

# Define features (X) and target variable (y)
X = df[features]
y = df["Dropout"]

# Split the dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print evaluation metrics
print(f"Accuracy of Logistic Regression: {accuracy * 100:.2f}")
print(f"Precision of Logistic Regression: {precision * 100:.2f}")
print(f"Recall of Logistic Regression: {recall * 100:.2f}")
print(f"F1-Score of Logistic Regression: {f1 * 100:.2f}")


print("NAVIE BAYS")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset (assuming 'df' is already loaded)
# Feature selection
features = ["Age", "Gender", "Location", "Distance_to_School", "Family_Income",
            "Family_Size", "Family_Stability", "Parents_Education_Level", "Marks_Obtained",
            "Attendance", "Exam_Performance", "Disciplinary_Records", "Learning_Habits",
            "School_Infrastructure", "Quality_of_Teaching", "Health_Issues"]

# Categorical columns that need encoding
categorical_columns = ["Gender", "Location", "Family_Stability", "Parents_Education_Level",
                        "Exam_Performance", "Disciplinary_Records", "Learning_Habits",
                        "School_Infrastructure", "Quality_of_Teaching", "Health_Issues"]

# Encoding categorical variables
label_encoders = {}
for col in categorical_columns:
    if col in df.columns and df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le  # Store encoders for future use

# Define features (X) and target variable (y)
X = df[features]
y = df["Dropout"]

# Split the dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features (Naive Bayes does not require scaling, but it may help in some cases)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Naive Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Make predictions
y_pred_nb = nb_model.predict(X_test)

# Evaluate the model
accuracy_nb = accuracy_score(y_test, y_pred_nb)
precision_nb = precision_score(y_test, y_pred_nb)
recall_nb = recall_score(y_test, y_pred_nb)
f1_nb = f1_score(y_test, y_pred_nb)

# Print evaluation metrics
print(f"Naive Bayes Accuracy: {accuracy_nb * 100:.2f}")
print(f"Naive Bayes Precision: {precision_nb * 100:.2f}")
print(f"Naive Bayes Recall: {recall_nb * 100:.2f}")
print(f"Naive Bayes F1-Score: {f1_nb * 100:.2f}")


print("MODEL EVALUATION")

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming `y_test` is the actual labels and `y_pred` is the predicted labels from your model
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Dropout", "Dropout"], yticklabels=["Not Dropout", "Dropout"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()



print("ACCURACY COMAPRISION")


import matplotlib.pyplot as plt

# Accuracy values
accuracy_lr = 86.83  # Logistic Regression
accuracy_nb = 96.67  # Naive Bayes

# Model names and corresponding accuracy
models = ["Logistic Regression", "Naive Bayes"]
accuracies = [accuracy_lr, accuracy_nb]

# Creating a pie chart
plt.figure(figsize=(6, 6))
colors = ['skyblue', 'lightgreen']
explode = (0.1, 0)  # Slightly separate the first slice

plt.pie(accuracies, labels=models, autopct='%1.1f%%', startangle=140, colors=colors, explode=explode, shadow=True)

# Adding title
plt.title("Comparison of Logistic Regression and Naive Bayes Accuracy")
plt.show()



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Select only the required features
selected_features = ["Exam_Performance", "Family_Stability", "Health_Issues", 
                     "School_Infrastructure", "Quality_of_Teaching", "Location"]

# Get the coefficients of these features
feature_importance = pd.DataFrame({
    "Feature": selected_features,
    "Coefficient": model.coef_[0][[X.columns.get_loc(f) for f in selected_features]]
})

# Take absolute values to show impact strength
feature_importance["Abs_Coefficient"] = feature_importance["Coefficient"].abs()

# Sort by impact
feature_importance = feature_importance.sort_values(by="Abs_Coefficient", ascending=False)

# Print the most important reasons
print("Top reasons affecting dropout:")
print(feature_importance)

# Plot the feature importance
plt.figure(figsize=(8, 5))
plt.barh(feature_importance["Feature"], feature_importance["Abs_Coefficient"], color='blue')
plt.xlabel("Impact on Dropout")
plt.ylabel("Factors")
plt.title("Key Factors Influencing Dropout")
plt.gca().invert_yaxis()
plt.show()









