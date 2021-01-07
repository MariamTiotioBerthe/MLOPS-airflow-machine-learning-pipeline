import sys 
import pandas as pd
sys.path.append("..")
sys.path.append("/home/dkaki/airflow/dags/python/")
from part1 import get_train_data1,get_train_data2,get_train_data3
#from part2 import get_train_data2
#from part3 import get_train_data3



def merging(**kwargs):
    df1 =kwargs['ti'].xcom_pull(task_ids='get_train_data1',key='dataset1')
    df2 =kwargs['ti'].xcom_pull(task_ids='get_train_data2',key='dataset2')
    df3 =kwargs['ti'].xcom_pull(task_ids='get_train_data3',key='dataset3')

    df = df1.merge(df2,on='id').merge(df3,on='id')
    print(df.head(30))
     
    kwargs['ti'].xcom_push(key='df', value=df) 
    
    
    