import pandas as pd
def get_train_data1(**kwargs):
    dataset1= pd.read_csv('/home/dkaki/airflow/dags/datasets/adult1.csv') 
   
    kwargs['ti'].xcom_push(key='dataset1', value=dataset1)
    dataset1.head()
    dataset1.info()
def get_train_data2(**kwargs):
    dataset2= pd.read_csv('/home/dkaki/airflow/dags/datasets/adult2.csv') 
    
    kwargs['ti'].xcom_push(key='dataset2', value=dataset2)
    dataset2.head()
    dataset2.info() 

def get_train_data3(**kwargs):
    dataset3= pd.read_csv('/home/dkaki/airflow/dags/datasets/adult3.csv') 
    
    kwargs['ti'].xcom_push(key='dataset3', value=dataset3)
    dataset3.head()
    dataset3.info()    
