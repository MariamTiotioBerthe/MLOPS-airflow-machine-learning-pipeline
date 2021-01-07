import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def visualisation(**kwargs):
    df =kwargs['ti'].xcom_pull(task_ids='data_preprocessing',key='df')
    
    income = df['income'].value_counts()

    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(7, 5))
    sns.barplot(income.index, income.values, palette='bright')
    plt.title('Distribution of Income', fontdict={
          'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
    plt.xlabel('Income', fontdict={'fontname': 'Monospace', 'fontsize': 15})
    plt.ylabel('Number of people', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
    plt.tick_params(labelsize=10)
    
    plt.savefig("/home/dkaki/airflow/dags/chart/distribution-income.pdf", bbox_inches='tight')
    
    age = df['age'].value_counts()

    plt.figure(figsize=(10, 5))
    plt.style.use('fivethirtyeight')
    sns.distplot(df['age'], bins=20)
    plt.title('Distribution of Age', fontdict={
          'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
    plt.xlabel('Age', fontdict={'fontname': 'Monospace', 'fontsize': 15})
    plt.ylabel('Number of people', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
    plt.tick_params(labelsize=10)
    plt.savefig("/home/dkaki/airflow/dags/chart/Distribution-Age.pdf", bbox_inches='tight')
    
    
    
    edu = df['education'].value_counts()

    plt.style.use('seaborn')
    plt.figure(figsize=(10, 5))
    sns.barplot(edu.values, edu.index, palette='Paired')
    plt.title('Distribution of Education', fontdict={
          'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
    plt.xlabel('Number of people', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
    plt.ylabel('Education', fontdict={'fontname': 'Monospace', 'fontsize': 15})
    plt.tick_params(labelsize=12)

    plt.savefig("/home/dkaki/airflow/dags/chart/Distribution-Education.pdf", bbox_inches='tight')
    
    
    hours = df['hours-per-week'].value_counts().head(10)

    plt.style.use('bmh')
    plt.figure(figsize=(15, 7))
    sns.barplot(hours.index, hours.values, palette='colorblind')
    plt.title('Distribution of Hours of work per week', fontdict={
          'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
    plt.xlabel('Hours of work', fontdict={'fontname': 'Monospace', 'fontsize': 15})
    plt.ylabel('Number of people', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
    plt.tick_params(labelsize=12)
    plt.show()
    
    plt.savefig("/home/dkaki/airflow/dags/chart/Hours-work-per-week.pdf", bbox_inches='tight')

    
    
    

