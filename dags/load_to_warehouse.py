from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    'start_date': datetime(2025, 6, 1),
    'catchup': False
}

with DAG(
    "load_to_warehouse",
    default_args=default_args,
    schedule='@daily',
    description="Loads data from staging to warehouse",
    tags=["warehouse", "load"]
) as dag:

    load_to_warehouse = BashOperator(
        task_id="load_to_pg",
        bash_command="python /usr/local/airflow/spark_jobs/load_to_warehouse.py"
    )

    load_to_warehouse
