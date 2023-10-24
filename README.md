# Repo containing artifacts for MLops pipeline
It uses Hopsworks to build MLOps pipeline.

It make use of ML model based on logistic regression for classification problem :
a) Loan should be approved
b) Loan should not be approved

Objective Achieved:
1) Run feature pipeline on arrival of training data in Data folder of this repo.
2) On completion of feature pipeline, training pipeline should be autotriggered to train the model with this data.
3) Trained Model should be saved in model registry and deployment should be created using this newly trained model.
4) Test data on which we need prediction should be uploaded in Data folder of this repo.
5) Initiate inference pipeline to get prediction on this Test data. Results of prediction will be saved as a csv file in Results folder of this repo.
