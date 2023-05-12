#! /bin/bash
echo $MLFLOW_S3_ENDPOINT_URL
export AWS_ACCESS_KEY_ID=''
export AWS_SECRET_ACCESS_KEY='chnageme'
export AWS_DEFAULT_REGION=us-east-1
export MLFLOW_TRACKING_URI='http://ec2-54-173-241-251.compute-1.amazonaws.com:5000/'
export MLFLOW_EXPERIMENT_NAME="projet_mlops"


mlflow run ./mlflow-example -P remote_server_uri=$MLFLOW_TRACKING_URI -P experiment_name=$MLFLOW_EXPERIMENT_NAME \
-P data_url=https://minio.lab.sspcloud.fr/pengfei/mlflow-demo/pokemon-partial.csv \
-P n_estimator=50 -P max_depth=30 -P min_samples_split=2
