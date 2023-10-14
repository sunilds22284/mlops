import hopsworks
import os
import numpy as np
import hsfs
import joblib

project = hopsworks.login()

# get Hopsworks Model Registry handle
mr = project.get_model_registry()

la_model = mr.get_model("loan_approval_v3", version=1)

# get Hopsworks Model Serving handle
ms = project.get_model_serving()

dataset_api = project.get_dataset_api()

uploaded_file_path = dataset_api.upload("predictor.py", "Models", overwrite=True)
predictor_script_path = os.path.join("/Projects", project.name, uploaded_file_path)

deployment_name = "loan_approval_v3"
my_deployment = la_model.deploy(name=deployment_name,
                                description="Deployment which can do loan approval prediction",
                                script_file=predictor_script_path,
                                resources={"num_instances": 0})
my_deployment.start()





