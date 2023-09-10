#!pip install h5py
#!pip install typing-extensions
#!pip install wheel

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import os
import joblib
from matplotlib import pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#!pip install -U hopsworks --quiet

import pandas as pd
import numpy as np

df = pd.read_excel('https://github.com/sunilcristo/hopswork-test/tree/main/Data/Consumer creditworthiness train data.xlsx')

df.head()

## Treat Null values
#Even though it is float type. It has only 2 values.
df['Credit_History'].fillna(df['Credit_History'].mode()[0],inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0],inplace=True)
#Mean value is a float value which cannot be true for Loan Term. If we check mode value, it is 360.
#We can either use mode value to replace null values or median value as it is a float datatype.
df['Loan_Amount_Term'].fillna(np.nanmedian(df['Loan_Amount_Term']),inplace=True)
#categorical feature. Using mode value to replace null values
df['Dependents'].fillna(df['Dependents'].mode()[0],inplace=True)
df['Gender'].fillna(df['Gender'].mode()[0],inplace=True)
df['Married'].fillna(df['Married'].mode()[0],inplace=True)

#After treating null values for each column
df.isna().sum().sort_values(ascending=False)

df['Loan_Status'] = np.where(df['Loan_Status']=="Y",1,0)

#Split Independent and Dependent(Target) features into different dataframe
Y = df[['Loan_Status']]
X = df.drop(['Loan_Status'],axis=1)

num = X.select_dtypes(include='number')
char = X.select_dtypes(include='object')

#Total income
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']

#EMI
#Function to calculate EMI
#Assumption : 9% interest rate
def emi(p, r, t):
    # for one month interest
    r = r/(12*100)
    t = t
    emi = (p*r*pow(1+r,t))/(pow(1+r,t)-1)
    return emi
df['EMI'] = emi(df['LoanAmount'], 9,df['Loan_Amount_Term'] )

#Balance Income
df['BalanceIncome'] = df['TotalIncome'] - df['EMI']

#Salaried
df['salaried'] = np.where((df['Education'] == "Graduate") & (df['Self_Employed'] == "No"),"Yes","No")

df.columns

df.drop('salaried',axis=1,inplace=True)
df.drop('Loan_Status',axis=1,inplace=True) #It is a target variable
df.drop('ApplicantIncome',axis=1,inplace=True) #As we already have total income now
df.drop('CoapplicantIncome',axis=1,inplace=True) #As we already have total income now

df.columns

df.drop(char,axis=1,inplace=True)

df.columns

char.drop(['Gender','Dependents','Self_Employed'],axis=1, inplace=True)

encoded_df = pd.get_dummies(char,drop_first=True)
encoded_df

concat_encoded_df = pd.concat([df,encoded_df,Y],axis=1)
concat_encoded_df

import hopsworks


project = hopsworks.login()
fs = project.get_feature_store()

loan_fg = fs.get_or_create_feature_group(
    name="loan_approval",
    version=2,
    description="Loan approval FG2",
    primary_key=["Loan_ID"],
    event_time= None,
    online_enabled=True
)

x = list(concat_encoded_df.columns)
x = [i.lower() for i in x]

concat_encoded_df.columns = x

concat_encoded_df.rename(columns={'education_not graduate':'education_not_graduate'},inplace=True)

#concat_encoded_df.columns

concat_encoded_df.head()

loan_fg.insert(concat_encoded_df)

feature_descriptions = [
    {"name": "loan_id", "description": "Loan id"},
    {"name": "gender", "description": "Gender"},
    {"name": "married", "description": "Person is married or single"},
    {"name": "dependents", "description": "No. of dependents a person have"},
    {"name": "education", "description": "Person is graduate or not"},
    {"name": "self_employed", "description": "Self employed or salaried"},
    {"name": "applicantincome", "description": "Applicant Income"},
    {"name": "coapplicantincome", "description": "Co Applicant Income"},
    {"name": "loanamount", "description": "Amount of loan taken"},
    {"name": "loan_amount_term", "description": "Term of loan amount"},
    {"name": "credit_history", "description": "Person have credit history or not"},
    {"name": "property_area", "description": "Property is located in Urban/Semi-urban/Rural area"},
    ]

for desc in feature_descriptions:
    loan_fg.update_feature_description(desc["name"], desc["description"])

