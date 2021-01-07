from merge import merging
import pandas 
import numpy

def preprocessing(**kwargs):
    df =kwargs['ti'].xcom_pull(task_ids='data_merging',key='df')
    
    print(df.info())
    print(df.describe().T)
    print(round((df.isnull().sum() / df.shape[0]) * 100, 2).astype(str) + ' %')
    print(round((df.isin(['?']).sum() / df.shape[0])
      * 100, 2).astype(str) + ' %')
      
    col_names = df.columns 
  
    for c in col_names: 
      df = df.replace("?", numpy.NaN) 
    df = df.apply(lambda x:x.fillna(x.value_counts().index[0]))   
      
    
    kwargs['ti'].xcom_push(key='df', value=df) 