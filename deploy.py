import hopsworks
import os
import numpy as np
import hsfs
import joblib

project = hopsworks.login()

# get Hopsworks Model Registry handle
mr = project.get_model_registry()

my_model = mr.get_model("loan_approval_v3", version=1)

# get Hopsworks Model Serving handle
ms = project.get_model_serving()

#delete existing deployments
list_of_deployments = ms.get_deployments()
for i in list_of_deployments:
	i.delete(force=True)

dataset_api = project.get_dataset_api()

uploaded_file_path = dataset_api.upload("./predictor.py", "Models", overwrite=True)
predictor_script_path = os.path.join("/Projects", project.name, uploaded_file_path)

deployment_name = "ladeployment"
my_deployment = my_model.deploy(name=deployment_name,
                                          description="Loan Approval",
                                          script_file=predictor_script_path,
                                          resources={"num_instances": 0})
my_deployment.start()





