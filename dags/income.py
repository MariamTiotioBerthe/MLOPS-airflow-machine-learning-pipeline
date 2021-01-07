from airflow import DAG
#from airflow.operators import PythonOperator
from datetime import datetime
import sys
from airflow.operators.python_operator import PythonOperator

sys.path.append("..")
sys.path.append("/home/dkaki/airflow/dags/python/")

from python.part1 import get_train_data1,get_train_data2,get_train_data3

from python.cleaning import preprocessing
from python.merge import merging
from python.visualisation import visualisation
from python.Selection import selection
from python.Selection import modeling_LR,modeling_RF,modeling_NB,modeling_SVM
from python.Selection import save_model

from airflow.operators.email_operator import EmailOperator
from airflow.operators.bash_operator import BashOperator
from airflow.operators.http_operator import SimpleHttpOperator
from airflow.operators.sensors import HttpSensor
from datetime import timedelta
# The DAG object; we'll need this to instantiate a DAG
from airflow import DAG
import os
from sklearn.model_selection import train_test_split
from airflow.operators.bash_operator import BashOperator
from airflow.utils.dates import days_ago

default_args = {
    'owner': 'Berthe-Dkaki',
    'depends_on_past': False,
    'email': ['e-dkaki.a@ucd.ma'],
    'email_on_failure': False,
    'email_on_retry': False,
    'email_on_success': False,
    #'retries': 1,
    #'retry_delay': timedelta(minutes=5), 
    # 'end_date': datetime(2020, 1, 30),
    # 'on_failure_callback': some_function,
    # 'on_success_callback': some_other_function,
    # 'on_retry_callback': another_function,
    
}
dag = DAG(
    dag_id = 'MLOPS',
    start_date = datetime(2020,1,1),
    default_args = default_args,
    description='MLOPS :automatisation des pipelines machines learnings à l aide de airflow '

    )


    
    
get_data1 = PythonOperator(
    task_id = 'get_train_data1',
    python_callable = get_train_data1,
    xcom_push=True,
    provide_context=True,
    dag = dag)
    
get_data2 = PythonOperator(
    task_id = 'get_train_data2',
    python_callable = get_train_data2,
    xcom_push=True,
    provide_context=True,
    dag = dag) 
    
get_data3 = PythonOperator(
    task_id = 'get_train_data3',
    python_callable = get_train_data3,
    xcom_push=True,
    provide_context=True,
    dag = dag)    
    
data_merging = PythonOperator(
    task_id = 'data_merging',
    python_callable = merging,
    xcom_push=True,
    provide_context=True,
    dag = dag)
    
data_preprocessing = PythonOperator(
    task_id = 'data_preprocessing',
    python_callable = preprocessing,
    provide_context=True,
    xcom_push=True,
    dag = dag)
    
data_visualisation = PythonOperator(
    task_id = 'data_visualisation',
    python_callable = visualisation,
    provide_context=True,
    dag = dag)

feauture_engineering = PythonOperator(
    task_id = 'data_feautures',
    python_callable = selection,
    provide_context=True,
    xcom_push=True,
    dag = dag) 
    
LR_modeling = PythonOperator(
    task_id = 'modele_LR',
    python_callable = modeling_LR,
    provide_context=True,
    dag = dag)

RF_modeling = PythonOperator(
    task_id = 'modele_RF',
    python_callable = modeling_LR,
    provide_context=True,
    dag = dag)

SVM_modeling = PythonOperator(
    task_id = 'modele_SVM',
    python_callable = modeling_SVM,
    provide_context=True,
    dag = dag)

NB_modeling = PythonOperator(
    task_id = 'modele_NB',
    python_callable = modeling_NB,
    provide_context=True,
    dag = dag)

saving_modele = PythonOperator(
    task_id = 'save_model',
    python_callable = save_model,
    provide_context=True,
    dag = dag)    




#data_merging.set_upstream(get_data1)
#data_merging.set_upstream(get_data2)
#data_merging.set_upstream(get_data3)
#data_preprocessing.set_upstream(data_merging)
#data_visualisation.set_upstream(data_preprocessing)
#feauture_engineering.set_upstream(data_visualisation)

get_data1 >> data_merging
get_data2 >> data_merging 
get_data3 >> data_merging 
data_merging >> data_preprocessing 
data_preprocessing >> data_visualisation 
data_visualisation >> feauture_engineering
feauture_engineering >> LR_modeling
feauture_engineering >> RF_modeling
feauture_engineering >> SVM_modeling
feauture_engineering >> NB_modeling


LR_modeling >> saving_modele
RF_modeling >> saving_modele
SVM_modeling >> saving_modele
NB_modeling >> saving_modele


