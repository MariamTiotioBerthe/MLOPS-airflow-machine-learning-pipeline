from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import pickle5 as pickle
 
def selection(**kwargs):
    df =kwargs['ti'].xcom_pull(task_ids='data_preprocessing',key='df')
    
    for col in df.columns: #Label Encoding
        if df[col].dtypes == 'object':
           encoder = LabelEncoder()
           df[col] = encoder.fit_transform(df[col])
       
    X = df.drop('income', axis = 1) 
    Y = df['income']   
    
    selector = ExtraTreesClassifier(random_state = 42)
    selector.fit(X, Y)
    feature_imp = selector.feature_importances_
    for index, val in enumerate(feature_imp):
        print(index, round((val * 100), 2))
        
    X = X.drop(['workclass', 'education', 'race', 'gender', 'capital-loss', 'native-country'], axis = 1)
   
    for col in X.columns:     
      scaler = StandardScaler()     
      X[col] = scaler.fit_transform(X[col].values.reshape(-1, 1)) 
    round(Y.value_counts(normalize=True) * 100, 2).astype('str') + ' %'

    ros = RandomOverSampler(random_state=42)
    ros.fit(X, Y)    
    X_resampled, Y_resampled = ros.fit_resample(X, Y)
    print(round(Y_resampled.value_counts(normalize=True) * 100, 2).astype('str') + ' %')

    X_train, X_test, Y_train, Y_test = train_test_split(X_resampled, Y_resampled, test_size = 0.2, random_state = 42)
    
    print("X_train shape:", X_train.shape) 
    print("X_test shape:", X_test.shape) 
    print("Y_train shape:", Y_train.shape) 
    print("Y_test shape:", Y_test.shape)
    
    kwargs['ti'].xcom_push(key='X_train',value=X_train)
    kwargs['ti'].xcom_push(key='X_test',value=X_test)
    kwargs['ti'].xcom_push(key='Y_train',value=Y_train)
    kwargs['ti'].xcom_push(key='Y_test',value=Y_test)
    
def modeling_LR(**kwargs):


    X_train =kwargs['ti'].xcom_pull(task_ids='data_feautures',key='X_train')
    X_test =kwargs['ti'].xcom_pull(task_ids='data_feautures',key='X_test')
    Y_train =kwargs['ti'].xcom_pull(task_ids='data_feautures',key='Y_train')
    Y_test =kwargs['ti'].xcom_pull(task_ids='data_feautures',key='Y_test')
   
    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(X_train, Y_train)
    
    Y_pred_log_reg = log_reg.predict(X_test)
    
    print('Logistic Regression:')
    print('Accuracy score:', round(accuracy_score(Y_test, Y_pred_log_reg) * 100, 2))
    print('F1 score:', round(f1_score(Y_test, Y_pred_log_reg) * 100, 2))
    
def modeling_RF(**kwargs):


    X_train =kwargs['ti'].xcom_pull(task_ids='data_feautures',key='X_train')
    X_test =kwargs['ti'].xcom_pull(task_ids='data_feautures',key='X_test')
    Y_train =kwargs['ti'].xcom_pull(task_ids='data_feautures',key='Y_train')
    Y_test =kwargs['ti'].xcom_pull(task_ids='data_feautures',key='Y_test')
   
    ran_for = RandomForestClassifier(random_state=42)

    ran_for.fit(X_train, Y_train)
    Y_pred_ran_for = ran_for.predict(X_test)
    print('Random Forest Classifier:')
    print('Accuracy score:', round(accuracy_score(Y_test, Y_pred_ran_for) * 100, 2))
    print('F1 score:', round(f1_score(Y_test, Y_pred_ran_for) * 100, 2)) 
    
    kwargs['ti'].xcom_push(key='ran_for', value=ran_for) 

def modeling_SVM(**kwargs):


    X_train =kwargs['ti'].xcom_pull(task_ids='data_feautures',key='X_train')
    X_test =kwargs['ti'].xcom_pull(task_ids='data_feautures',key='X_test')
    Y_train =kwargs['ti'].xcom_pull(task_ids='data_feautures',key='Y_train')
    Y_test =kwargs['ti'].xcom_pull(task_ids='data_feautures',key='Y_test')
   
    svc = SVC(random_state=42)
    svc.fit(X_train, Y_train)
    Y_pred_svc = svc.predict(X_test)
    print('Support Vector Classifier:')
    print('Accuracy score:', round(accuracy_score(Y_test, Y_pred_svc) * 100, 2))
    print('F1 score:', round(f1_score(Y_test, Y_pred_svc) * 100, 2))
    
def modeling_NB(**kwargs):


    X_train =kwargs['ti'].xcom_pull(task_ids='data_feautures',key='X_train')
    X_test =kwargs['ti'].xcom_pull(task_ids='data_feautures',key='X_test')
    Y_train =kwargs['ti'].xcom_pull(task_ids='data_feautures',key='Y_train')
    Y_test =kwargs['ti'].xcom_pull(task_ids='data_feautures',key='Y_test')
   
    nb = GaussianNB()
    nb.fit(X_train, Y_train)
    Y_pred_nb = nb.predict(X_test)
    
    print('Naive Bayes Classifier:')
    print('Accuracy score:', round(accuracy_score(Y_test, Y_pred_nb) * 100, 2))
    print('F1 score:', round(f1_score(Y_test, Y_pred_nb) * 100, 2)) 

def save_model(**kwargs):


    ran_for =kwargs['ti'].xcom_pull(task_ids='modele_RF',key='ran_for')
    



    pickle.dump(ran_for, open('/home/dkaki/airflow/dags/Model.pkl','wb'))
    

        
        
    
    