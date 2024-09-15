import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import warnings

# Ignore all warnings
warnings.filterwarnings('ignore')

data = pd.read_csv("titanic_dataset.csv")
#print(data)
#print(data.info())
#print(data.isnull().sum())

data.drop(['Cabin'], axis=1,inplace=True)
data['Age'].fillna(data['Age'].mean(), inplace = True)
data['Age'] = data['Age'].round(0)
#print(data.isnull().sum())
print(data.duplicated().sum())
mode_embarked = data['Embarked'].mode()[0]
#print(mode_embarked)
data['Embarked'].fillna(mode_embarked, inplace=True)
#print(data.isnull().sum())
#print(data.info())
encoder= LabelEncoder()
# male =1 , female =0
#embarked = s=2, c=0, q=1
data['Sex']=encoder.fit_transform(data['Sex'])
data['Embarked']=encoder.fit_transform(data['Embarked'])
#print(data.info())
#print(data.corr(numeric_only=True))

# Count the number of survivors and non-survivors
survival_counts = data['Survived'].value_counts()
print("Number of survivors and non-survivors:")
print(survival_counts)

# Count the number of males and females among survivors and non-survivors
gender_survival_counts = data.groupby(['Survived', 'Sex']).size().unstack(fill_value=0)
print("\nGender distribution among survivors and non-survivors:")
print(gender_survival_counts)


#Univariate Analysis
# Visualize the target variable 'Survived'
sns.countplot(data=data, x='Survived')
plt.title('Survival Distribution')
plt.show()

# Distribution of Age
plt.figure(figsize=(8,6))
sns.histplot(data['Age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.show()

# Distribution of Fare
plt.figure(figsize=(8,6))
sns.histplot(data['Fare'], bins=30, kde=True)
plt.title('Fare Distribution')
plt.show()

# Count plot for categorical features
sns.countplot(data=data, x='Pclass')
plt.title('Passenger Class Distribution')
plt.show()

sns.countplot(data=data, x='Sex')
plt.title('Gender Distribution')
plt.show()

sns.countplot(data=data, x='Embarked')
plt.title('Embarked Port Distribution')
plt.show()

#Bivariate Analysis
# Survival rate by gender
sns.barplot(x='Sex', y='Survived', data=data)
plt.title('Survival Rate by Gender')
plt.show()

# Survival rate by passenger class
sns.barplot(x='Pclass', y='Survived', data=data)
plt.title('Survival Rate by Passenger Class')
plt.show()

# Survival rate based on Age distribution
plt.figure(figsize=(8,6))
sns.histplot(data=data, x='Age', hue='Survived', bins=30, kde=True)
plt.title('Age Distribution by Survival')
plt.show()

#Multivariate Analysis
# Pairplot of numerical features with hue as 'Survived'
sns.pairplot(data[['Survived', 'Pclass', 'Age', 'Fare']], hue='Survived', diag_kind='kde')
plt.show()

#correlation heatmap
# Selecting numeric columns only
numeric_data = data.select_dtypes(include=['float64', 'int64'])

# Correlation matrix for numeric features
correlation_matrix = numeric_data.corr()

# Plotting the correlation heatmap

plt.figure(figsize=(10,6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Numeric Feature Correlation Heatmap')
plt.show()
#data frame which used in Ml model
df=data.drop(['Name','Ticket','PassengerId'],axis=1)
#print(df.corr())

# Define features and target
X = df.drop('Survived', axis=1)
y = df['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize models (including Naive Bayes)
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Support Vector Classifier': SVC(),
    'Random Forest': RandomForestClassifier(),
    'Naive Bayes': GaussianNB()  # Adding Naive Bayes
}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()




#So, selected Logistic Regression as it gets the highest accuracy among all other models
#approach - all the data treated as training data and user input data will treat as testing data

# Initialize and train Logistic Regression model on the entire dataset
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X, y)

# Function to preprocess user input and make predictions
def predict_survival(user_input):
    # Convert user input to DataFrame
    user_df = pd.DataFrame([user_input], columns=X.columns)
    
    # Predict the survival probability
    prob_survival = logistic_model.predict_proba(user_df)[0][1]
    pred_survival = logistic_model.predict(user_df)[0]
    
    return pred_survival, prob_survival

# Collect user input
print("Please enter the following details:")

pclass = int(input("Pclass (1, 2, or 3): "))
sex = int(input("Sex (0 for female, 1 for male): "))
age = float(input("Age: "))
sibsp = int(input("SibSp (Number of siblings/spouses aboard): "))
parch = int(input("Parch (Number of parents/children aboard): "))
fare = float(input("Fare: "))
embarked = int(input("Embarked (0 for C - Cherbourg, 1 for Q - Queenstown, 2 for S - Southampton): "))

# Ensure the user input is in the same format as the training data
user_input = {
    'Pclass': pclass,
    'Sex': sex,
    'Age': age,
    'SibSp': sibsp,
    'Parch': parch,
    'Fare': fare,
    'Embarked': embarked
}

# Predict survival and probability for the user input
survival_prediction, survival_probability = predict_survival(user_input)

# Display results
print(f"\nSurvival Prediction: {'Survived' if survival_prediction == 1 else 'Not Survived'}")
print(f"Chance of survival: {survival_probability * 100:.2f}%")




