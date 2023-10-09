import hopsworks
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
import seaborn as sns
import numpy as np
import pandas as pd
import hsfs
from hsml.schema import Schema
from hsml.model_schema import ModelSchema
from matplotlib import pyplot
import matplotlib.pyplot as plt

project = hopsworks.login ()
fs = project.get_feature_store()

loan_fg = fs.get_or_create_feature_group(
    name="loan_approval",
    version=2)

query = loan_fg.select(['loanamount', 'loan_amount_term', 'credit_history',
       'totalincome', 'emi', 'balanceincome', 'married_yes',
       'education_not_graduate', 'property_area_semiurban',
       'property_area_urban', 'loan_status']) #Except loan_id, take everything

feature_view = fs.get_or_create_feature_view(
    name='loan_approval_feature_view',
    version=3,
    query=query,
    labels=["loan_status"],
)

TEST_SIZE = 0.2

td_version, td_job = feature_view.create_train_test_split(
    description='loan approval training dataset',
    data_format='csv',
    test_size=TEST_SIZE,
    write_options={'wait_for_job': True}
)

X_train, X_test, y_train, y_test = feature_view.get_train_test_split(1)

#X_train

#regularization is penalty = 'l1' to improve performance metrics.
model_LR2 = LogisticRegression(penalty='l1',C=5,solver='liblinear',max_iter=1000)

model_LR2.fit(X_train, y_train)

pred_y2 = model_LR2.predict(X_test)

# Compute f1 score
metrics = {"f1_score": f1_score(y_test, pred_y2, average='macro')}
#metrics

#print('f1_score:', f1_score(y_test,pred_y2))
#print('precision_score:',precision_score(y_test,pred_y2))
#print('recall:',recall_score(y_test,pred_y2))



input_schema = Schema(X_train)
output_schema = Schema(y_train)
model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)

model_schema.to_dict()

model_dir="loan_approval_model_v3"
if os.path.isdir(model_dir) == False:
    os.mkdir(model_dir)

joblib.dump(model_LR2, model_dir + '/logistic_regression_model.pkl')

plt.savefig(model_dir + "/confusion_matrix.png")

mr = project.get_model_registry()

loan_approval_model = mr.python.create_model(
    name="loan_approval_v3",
    metrics=metrics,
    model_schema=model_schema,
    input_example=X_test.sample().to_numpy(),
    description="Loan approval predictor")

loan_approval_model.save(model_dir)

