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

dataset_api = project.get_dataset_api()

uploaded_file_path = dataset_api.upload("predictor.py", "Models", overwrite=True)
predictor_script_path = os.path.join("/Projects", project.name, uploaded_file_path)

my_predictor = ms.create_predictor(my_model,
                                   # optional
                                   model_server="PYTHON",
                                   serving_tool="KSERVE",
                                   script_file=predictor_script_path
                                   )

my_deployment = my_predictor.deploy()



