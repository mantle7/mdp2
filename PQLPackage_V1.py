#!/usr/bin/env python
# coding: utf-8

# # Importing Packages

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date,timedelta

import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve

import shap

import pickle
import json

import streamlit as st


# # Variable Initializiation

# In[2]:


historical_data_file='LTV_with target variable.csv'
# target='LTV_spender'

live_data_file='LTV_predict target variable.csv'

detailed_output_file='ScoreSheetDetailed.csv'
concise_output_file='ScoreSheetConcise.csv'


# # Method Definitions 

# In[3]:


def create_historical_dataframe(historical_data_file):
    df=pd.read_csv(historical_data_file)
    return df


# In[4]:


def null_handler(df,user_id_provided):
    nullPercentages=pd.DataFrame(df.isnull().sum(),columns=['Number of Null Values'])
    nullPercentages['Percentage of Null Values']=np.round(df.isnull().sum()/df.shape[0],5)*100
    nullPercentages=nullPercentages.sort_values(['Number of Null Values'],ascending=False)
    
    nullcols=[]
    for i in range(nullPercentages.shape[0]):
#     for i in nullPercentages['Percentage of Null Values']:
        if nullPercentages.iloc[i,1] >= 75:
#         if i>=75:
            nullcols.append(nullPercentages.index[i])
            
    df=df.drop(nullcols,axis='columns')
    
    if user_id_provided==True:
        df=df.drop([df.columns[0]],axis='columns')
    df=df.dropna()
    # st.write("null_handler")
    return df,nullcols      #########################


# In[5]:


def pre_processing(df):
    numeric_columns=[x for x in df.select_dtypes(include=np.number).columns]
    cat_columns=[x for x in df.select_dtypes(include=np.object).columns]
    
    cols_to_scale=[]
    for col in numeric_columns:
        if len(df[col].unique()) > 2:
            cols_to_scale.append(col)
            
    df2=df.copy()
    
    sc=StandardScaler()
    df2[cols_to_scale]=sc.fit_transform(df2[cols_to_scale])
    # st.write("pre_processing")
    
    return df2,cat_columns


# In[6]:


def train_test(df2,target,train_size,random_state):
    Y=df2[target]
    X=df2.drop([target],axis='columns')
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=train_size,random_state=random_state)
    # st.write('train_test')
    return X,Y,X_train,X_test,Y_train,Y_test
    


# In[7]:


# def model_build(X_train,X_test,Y_train,Y_test,cat_columns):
#     cbc=CatBoostClassifier(iterations=100)
#     cbc.fit(X_train,Y_train,cat_features=cat_columns,plot=False,eval_set=(X_test,Y_test),silent=True)
#     Y_predicted=cbc.predict(X_test)
    
#     return cbc


# In[8]:


def feature_importances(cbc,X,X_test):
    importance=cbc.get_feature_importance()
    feature_scores={}
    for score, name in sorted(zip(importance,X.columns), reverse=True):
        feature_scores.update({name:score})
    
    explainer = shap.Explainer(cbc)
    shap_values = explainer(X_test)
    shap.plots.bar(shap_values, max_display=X_test.shape[0])
    # st.write("feature_importances")
    # fig=shap.plots.bar(shap_values, max_display=X_test.shape[0])
    # st.plotly_chart(fig)  #####################################################################


# In[9]:


def optimal_score(cbc,X,X_train,X_test,Y_train,Y_test,null_cols):
    probs_train=cbc.predict_proba(X_train)
    probs_train=[x[1] for x in probs_train]

    probs_test=cbc.predict_proba(X_test)
    probs_test=[x[1] for x in probs_test]

    probs_total=cbc.predict_proba(X)
    probs_total=[x[1] for x in probs_total]
    
    result_metrics = pd.DataFrame(columns=['cutoff','train_acc','train_sen','train_spec','train_prec',
                                      'test_acc','test_sen','test_spec','test_prec'])
    
    fpr, tpr, thresholds = roc_curve(Y_test,probs_test, drop_intermediate = False )
    # fpr, tpr, thresholds = roc_curve(Y_test,Y_predicted, drop_intermediate = False )
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    return optimal_threshold
    
    


# In[10]:


# def check_optimal_score(cutoff_df,test_score=0.5):
#     cutoff_df.plot.line(x='Probability', y=['Accuracy','Sensitivity','Specificity'],figsize=(10,8))
#     plt.vlines(x=test_score, ymax=1, ymin=0, colors="g", linestyles="--")
#     plt.show()


# In[11]:


def model_final_fit(cbc,X,Y,cat_columns):
    cbc.fit(X,Y,cat_features=cat_columns,silent=True)


# In[12]:


# def create_live_dataframe(live_data_file):
#     test_df=pd.read_csv(live_data_file)
#     st.write("4")
#     return test_df
    
    


# In[21]:


def final_output(cbc,test_df,nullcols,optimal_score,user_id_provided=True):
    final_output1=test_df.copy() #optional

    test_df=test_df.drop(nullcols,axis='columns')
    
    final_output2=test_df.copy()

    test_df=test_df.dropna()

    final_output3=test_df.copy()   #for non null user ids
    
    if user_id_provided==True:
        test_df=test_df.drop(test_df.columns[0],axis='columns')
        
    numeric_columns=[x for x in test_df.select_dtypes(include=np.number).columns]
    cat_columns=[x for x in test_df.select_dtypes(include=np.object).columns]
    cols_to_scale=[]
    for col in numeric_columns:
        if len(test_df[col].unique()) > 2:
            cols_to_scale.append(col)
        
    test_df2=test_df.copy()

    from sklearn.preprocessing import StandardScaler
    sc=StandardScaler()
    test_df2[cols_to_scale]=sc.fit_transform(test_df2[cols_to_scale])

    probs_final=cbc.predict_proba(test_df2)
    probs_final=[x[1] for x in probs_final]

    predicted_values_final=[]
    for i in probs_final:
        if(i>=optimal_score):
            predicted_values_final.append(1)
        else:
            predicted_values_final.append(0)

#     test_df['Predicted']=predicted_values_final
    # print(final_output3.columns[0])
    # user_column=final_output3.pop(final_output3.columns[0])
    # print(user_column)

    if user_id_provided==True:
        # test_df['USERID']=final_output3['USERID']
        test_df.insert(0,column='USERID',value=final_output3[final_output3.columns[0]])


    test_df["Lead Scores"]=probs_final
    test_df["Lead Scores"]=test_df["Lead Scores"]*100

    test_df=test_df.reset_index(drop=True)
        
    # test_df.to_csv(detailed_output_file)
    
    if user_id_provided==True:    
        score_sheet=pd.DataFrame(columns=['USERID','Conversion_Scores'])
        score_sheet['USERID']=test_df['USERID']
        score_sheet['Conversion_Scores']=test_df['Lead Scores']
    else:
        score_sheet=pd.DataFrame(columns=['Conversion_Scores'])
        score_sheet['Conversion_Scores']=test_df['Lead Scores']
#     print("Results stored to {} and {}".format(detailed_output_file,concise_output_file))
#     score_sheet.to_csv(concise_output_file)
    score_sheet=score_sheet.reset_index(drop=True)
    return test_df


# # Handling Historical Data

# In[22]:


def lead_score_generator(df,test_df):
    # st.info("Cleaning Data")
    df,nullcols=null_handler(df,user_id_provided=True)

    # st.info("Preprocessing Data")
    df2,cat_cols=pre_processing(df)

    # st.info("Splitting Data")
    X,Y,X_train,X_test,Y_train,Y_test=train_test(df2,target=df.columns[df2.shape[1]-1],train_size=0.8,random_state=100)

    # st.info("Building Model")
    cbc=CatBoostClassifier(iterations=40,
                       early_stopping_rounds=35)
    cbc.fit(X_train,Y_train,cat_features=cat_cols,plot=False,eval_set=(X_test,Y_test),silent=True)

    # st.info("Generating Feature Importances")
    feature_importances(cbc,X,X_test)

    # st.info("Generating Optimal Lead Score")
    cut_off_score=optimal_score(cbc,X,X_train,X_test,Y_train,Y_test,nullcols)

    # st.info("Model Finalizing")
    model_final_fit(cbc,X,Y,cat_cols)

    # st.info("Getting Output Ready")
    result_df=final_output(cbc,test_df,nullcols,cut_off_score,user_id_provided=True)
    return result_df,cut_off_score*100
    


