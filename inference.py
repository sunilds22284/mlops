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
import re
import hopsworks
import requests

project = hopsworks.login()

import pandas as pd
import numpy as np

df_orig = pd.read_excel('https://github.com/sunilds22284/mlops/raw/main/Data/Consumer_creditworthiness_test_data.xlsx')
list_orig_df = []
list_orig_df = df_orig.values.tolist()

list_of_columns = []
list_of_columns = list(df_orig.columns)
list_of_columns.append("Loan_Status")

df = df_orig.copy()

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

X = df

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

df.drop('ApplicantIncome',axis=1,inplace=True) #As we already have total income now
df.drop('CoapplicantIncome',axis=1,inplace=True) #As we already have total income now
df.drop('Loan_ID',axis=1,inplace=True) #As it is not needed

df.drop(char,axis=1,inplace=True)

char.drop(['Gender','Dependents','Self_Employed'],axis=1, inplace=True)

encoded_df = pd.get_dummies(char,drop_first=True)

concat_encoded_df = pd.concat([df,encoded_df],axis=1)

concat_encoded_df.rename(columns={'education_not graduate':'education_not_graduate'},inplace=True)

# prepate input to make predictions
input = concat_encoded_df.values.tolist()

import hsml

ms = project.get_model_serving()

# get deployment object
deployment = ms.get_deployment("ladeployment")
state = deployment.get_state()
state.describe()
# get Hopsworks Model Registry
mr = project.get_model_registry()

# get model
model = mr.get_model(deployment.model_name, deployment.model_version)

# make predictions
list_of_prediction = []
for i in input:
    prediction = deployment.predict({"instances":i})
    pred = re.findall(r'\d+', str(prediction))
    list_of_prediction.append(pred)

for j in range(0,len(list_orig_df)):
        list_orig_df[j].extend(list_of_prediction[j])

df_final = pd.DataFrame(data=list_orig_df,columns=list_of_columns)
df_final.to_csv("results.csv")

# Get the working directory
workspace = os.environ['GITHUB_WORKSPACE']

# Define the path to the CSV file
csv_file_path = os.path.join(workspace, 'results.csv')

github_token = os.environ.get('GH_TOKEN')

# Set your GitHub access token and repository information
access_token = github_token
username = "sunilds22284"
repository = "mlops"
file_path = csv_file_path
f_path = "Results/results.csv"
commit_message = "Add results.csv"  # Commit message

# Encode the file content as base64
import base64

with open(file_path, "rb") as file:
    file_content = base64.b64encode(file.read()).decode()

# Define the API endpoint for creating a new file
api_url = f"https://api.github.com/repos/{username}/{repository}/contents/{f_path}"

# Prepare the request payload
data = {
    "message": commit_message,
    "content": file_content,
}

# Headers with the access token
headers = {
    "Authorization": f"token {access_token}",
}

# Send a PUT request to create the file
response = requests.put(api_url, headers=headers, json=data)

if response.status_code == 201:
    print(f"{file_path} uploaded successfully.")
else:
    print(f"Failed to upload {file_path}. Status code: {response.status_code}")
    print(response.text)

