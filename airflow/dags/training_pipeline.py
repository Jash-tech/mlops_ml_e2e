from __future__ import annotations
import json
from textwrap import dedent
import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator
from src.pipelines.training import Training

training_pipeline=Training()

with DAG(
    "gemstone_training_pipeline",
    default_args={"retries":2},
    description="Its my training pipeline",
    schedule="@weekly",
    catchup=False,
    tags=["ML","Classification"],
    start_date=pendulum.datetime(2024,6,13,tz="UTC")
) as dag:
    dag.doc_md=__doc__

    def data_ingestion(**kwargs):
        ti=kwargs["ti"]
        train_data_path,test_data_path=training_pipeline.start_data_ingestion()
        ti.xcom_push("data_ingestion_artifact",{"train_data_path":train_data_path,"test_data_path":test_data_path})

    def data_transformation(**kwargs):
        ti=kwargs["ti"]
        data_ingestion_artifact=ti.xcom_pull(task_ids="data_ingestion",key="data_ingestion_artifact")
        train_arr,test_arr=training_pipeline.start_data_transformation(data_ingestion_artifact)
        train_arr=train_arr.tolist()
        test_arr=test_arr.tolist()
        ti.xcom_push("data_transformation_artifact",{"train_arr":train_arr,"test_arr":test_arr})