import dagshub
dagshub.init(repo_owner='MitadruMridha05', repo_name='House_Price_predictor', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)