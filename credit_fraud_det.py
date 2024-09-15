
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, classification_report

# Load dataset
data = pd.read_csv("creditcard.csv")

# 1. Data Overview
print("Shape of the dataset:", data.shape)
print("\nColumns in the dataset:", data.columns)
print("\nMissing values in each column:\n", data.isnull().sum())
print("\nBasic statistics of the dataset:\n", data.describe())

# 2. Class Distribution
fraud_count = data['Class'].value_counts()
print("\nFraudulent vs Non-Fraudulent transactions:\n", fraud_count)
fraud_percentage = data['Class'].value_counts(normalize=True) * 100
print("\nPercentage of fraudulent transactions:\n", fraud_percentage)

#  Bar plot for class distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Class', data=data)
plt.title('Class Distribution (Fraud vs Non-Fraud)')  
plt.show()

# 3. Statistical Summary of Time and Amount Columns
print("\nStatistics of 'Time' column:\n", data['Time'].describe())
print("\nStatistics of 'Amount' column:\n", data['Amount'].describe())

# 4. Correlation Matrix & Heatmap
plt.figure(figsize=(14,12))  # Increase figure size
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f", annot_kws={"size": 8})  # Adjust annotation size
plt.title('Correlation Matrix of Features')

# Rotate tick labels for better readability
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.show()

# 5. Distribution of 'Amount' and 'Time'
plt.figure(figsize=(8,6))
sns.histplot(data['Amount'], bins=50, kde=True)
plt.title('Distribution of Transaction Amount')
plt.show()

plt.figure(figsize=(8,6))
sns.histplot(data['Time'], bins=50, kde=True)
plt.title('Distribution of Transaction Time')
plt.show()


# 6. Undersampling the majority class
# Separate majority and minority classes
data_class_0 = data[data['Class'] == 0]
data_class_1 = data[data['Class'] == 1]

# Downsample majority class (Class 0) to match minority class (Class 1)
data_class_0_downsampled = resample(data_class_0, 
                                  replace=False,    # sample without replacement
                                  n_samples=492,    # match minority class count
                                  random_state=42)  # reproducible results

# Combine the downsampled majority class with the minority class
data_balanced = pd.concat([data_class_0_downsampled, data_class_1])

# Verify the new class distribution
print("\nClass distribution after balancing:")
print(data_balanced['Class'].value_counts())

# 7. Save the balanced dataset to a CSV file
data_balanced.to_csv("creditcard_balanced.csv", index=False)

# 8. Bar plot for balanced class distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Class', data=data_balanced)
plt.title('Balanced Class Distribution (Fraud vs Non-Fraud)')
plt.show()

# ---------------------- ML MODELLING -------------------------------------------------------------

# Load the balanced dataset
df = pd.read_csv("creditcard_balanced.csv")

# Features and Target variable
X = df.drop('Class', axis=1)
y = df['Class']

# Columns to scale
columns_to_scale = ['Amount', 'Time']

# Initialize StandardScaler
scaler = StandardScaler()

# Scale only the specified columns
X[columns_to_scale] = scaler.fit_transform(X[columns_to_scale])

# Split the data into train and test sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(probability=True),  # Enable probability estimates for ROC curve
    'Naive Bayes': GaussianNB(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# Store evaluation metrics and ROC curve data
model_evaluations = {}
roc_curves = {}

# Train and evaluate each model
for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)
    
    # Predict the test set
    y_pred = model.predict(X_test)
    
    # Get probabilities for ROC curve
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    # Save evaluation metrics
    model_evaluations[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC AUC': roc_auc
    }
    
    # ROC curve data
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_curves[model_name] = (fpr, tpr, roc_auc)
    
    # Print classification report
    print(f"\nClassification Report for {model_name}:\n")
    print(classification_report(y_test, y_pred))

# Display evaluation metrics
eval_df = pd.DataFrame(model_evaluations).T
print("\nModel Evaluation Metrics:\n")
print(eval_df)

# Plot ROC curves for all models
plt.figure(figsize=(10, 8))
for model_name, (fpr, tpr, roc_auc) in roc_curves.items():
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

# Plot the diagonal line for random guessing
plt.plot([0, 1], [0, 1], 'k--', label="Random Guessing (AUC = 0.50)")

# Customize the plot
plt.title('ROC Curves for Different Models')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()