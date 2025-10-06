import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import seaborn as sns
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OrdinalEncoder, StandardScaler
# Testing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
# Accuracy metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV, train_test_split

import warnings


warnings.filterwarnings('ignore')

# data Exploration
data = "C:/Users/anluo/OneDrive/Desktop/Projects/Project 2/term-deposit-marketing-2020.csv"
df = pd.read_csv(data)
print(df.head())
print(df.info())

# Pre-processing
df.isnull().sum()

# Outliers
numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
# Apply Winsorization to each numerical column (5% from each tail by default)
for col in numerical_columns:
    # Winsorize with limits=0.05 (5% from lower and upper tails)
    df[col] = winsorize(df[col], limits=[0.05, 0.05])
    print(f"Winsorized column: {col}")
print("Winsorization applied. Updated dataframe info:")
print(df.info())
print(df.describe())

# Value Count
for i in range(len(df.columns)):
    if df[df.columns[i]].dtypes != 'int64':
        print(df[df.columns[i]].value_counts(), '\n')

# Encoding
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical columns to encode: {categorical_columns}")

# Initialize OrdinalEncoder
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# Fit and transform categorical columns
df_encoded = df.copy()
df_encoded[categorical_columns] = ordinal_encoder.fit_transform(df[categorical_columns])

# Display the encoded dataframe info and head
print("Encoded dataframe info:")
print(df_encoded.info())
print("\nEncoded dataframe head:")
print(df_encoded.head())

# Imbalanced data

# Showing imbalance within data
print(df_encoded['y'].value_counts(normalize=True).multiply(100))

smote = SMOTE(random_state=42)
dfx = df_encoded.drop(['y'], axis=1)
dfy = df_encoded['y']
dfx_smote, dfy_smote = smote.fit_resample(dfx, dfy)

print(dfy_smote.value_counts(normalize=True).multiply(100))

# Feature selection through data correlation
data_correlation = pd.concat([dfx_smote, dfy_smote], axis=1)
corr = data_correlation.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True)
plt.show()

remove = corr.loc[:, abs(corr.loc['y']) < abs(0.1)].columns
dfx_selection = dfx_smote.drop(remove, axis=1)
print("Feature removed:\n" + str(remove))
# feature removed: 'age', 'job', 'marital', 'education', 'default', 'balance', 'loan', 'day', 'month'

# Creating new data set
X = dfx_selection
y = dfy_smote

df_final = pd.concat([X, y], axis=1)
# WIP: df_final.to_csv('C:/Users/anluo/OneDrive/Desktop/Projects/Project 2', index = False)

# Model

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model selection and hyperparameter tuning with GridSearchCV
models = ['LogisticRegression', 'DecisionTreeClassifier', 'RandomForestClassifier', 'XGBClassifier', 'KNeighborsClassifier', 'SVC']
param_grid = dict().fromkeys(models)

param_grid['LogisticRegression'] = {'penalty': ['l1', 'l2', 'elasticnet'],
                                    'C': [1/0.001, 1/0.01, 1/0.1] # Inverse of regularization
                                    }
param_grid['DecisionTreeClassifier'] = {'criterion': ['entropy', 'gini'], # measure of impurity
                                        'max_depth': np.arange(1, len(X_test.columns), 1), # Level of tree
                                        'min_samples_split': [3, 4, 5], # samples to split
                                        'ccp_alpha': np.arange(0, 0.040, 0.005) # cost-complexity pruning
                                       }

param_grid['RandomForestClassifier'] = {'n_estimators': [50,100, 150], # number of trees
                                        'max_depth': np.arange(1, len(X_test.columns), 1), # level of tree
                                        'min_samples_split': [3, 4, 5], # samples to split
                                       }

param_grid['XGBClassifier'] = {'n_estimators': [50, 150, 200], # number of trees
                               'reg_lambda': [0.001, 0.01, 0.1], # regularization term
                               'booster': ['gbtree', 'gblinear', 'dart']
                            }

param_grid['KNeighborsClassifier'] = {'n_neighbors' : np.arange(5, 35, 5)} # groups of neighbors

param_grid['SVC'] = {'C': [0.001, 0.01, 0.1], # regularization term
                     'kernel': ['rbf', 'linear', 'poly'] # kernel type
                     }


def load_model(model):
    if model == 'LogisticRegression':
        return LogisticRegression()
    if model == "DecisionTreeClassifier":
        return DecisionTreeClassifier(random_state=42)
    if model == 'RandomForestClassifier':
        return RandomForestClassifier(random_state=42)
    if model == 'XGBClassifier':
        return XGBClassifier(random_state=42)
    if model == 'KNeighborsClassifier':
        return KNeighborsClassifier()
    if model == 'SVC':
        return SVC(random_state=42)

dict_models = {}
for model in models:
    estimator = load_model(model)
    gs = GridSearchCV(estimator, param_grid = param_grid[model])
    gs.fit(X_test, y_test)
    dict_models[model] = gs.best_estimator_
    y_pred = dict_models[model].predict(X_test)
    print("\n Model Report:\n ", dict_models[model],"\n Test Set \n",
        classification_report(y_test, y_pred), "\n")

