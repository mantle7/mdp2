import streamlit as st
import numpy as np
import plotly_express as px
import pandas as pd
import time
import base64
import s3fs
import os
import boto3
import requests
import json
from datetime import date
import datetime
from mixpanel_utils import MixpanelUtils
import pickle
from pathlib import Path
import streamlit_authenticator as stauth
from PIL import Image
import extra_streamlit_components as stx
from io import StringIO
import json

from generate_keys import names,usernames
from PQLPackage_V1 import lead_score_generator
from secrets_aws import my_access_key,my_secret_access_key

global df,test_df

st.set_page_config(
	page_title="LeadEngine | Home",
	layout='wide',
	page_icon="🤖")

st.set_option('deprecation.showfileUploaderEncoding',False)

hide_streamlit_style = """
            <style>
            
            footer {visibility: hidden;}
            </style>
            """
#MainMenu {visibility: hidden;}
st.markdown(hide_streamlit_style, unsafe_allow_html=True)




if "my_input" not in st.session_state:
	st.session_state["my_input"]="xyz"

if "my_input2" not in st.session_state:
	st.session_state["my_input2"]="xyz"

if "my_input3" not in st.session_state:
	st.session_state["my_input3"]="xyz"

if "df_input" not in st.session_state:
	st.session_state['df_input']='xyz'


# AUTHENTICATION #########################################

file_path=Path(__file__).parent/'hashed_pw.pkl'
with file_path.open('rb') as file:
	hashed_passwords=pickle.load(file)

authenticator=stauth.Authenticate(names,usernames,hashed_passwords,'pql_dashboard','abcdef',cookie_expiry_days=0)
name,authentication_status,username=authenticator.login("Login","main")

@st.cache
def upload_data(hist_file,live_file):
	df=pd.read_csv(hist_file)
	test_df=pd.read_csv(live_file)
	return df,test_df

@st.cache
def store_data_aws(df,test_df):
	s3=boto3.resource(service_name='s3',
						region_name='us-east-1',
						aws_access_key_id=my_access_key,
						aws_secret_access_key=my_secret_access_key)
					

	csv_buf=StringIO()
	df.to_csv(csv_buf)
	csv_buf.seek(0)
	s3.Bucket('lead.engine').put_object(Key='{}/train.csv'.format(username), Body=csv_buf.getvalue())

	csv_buf2=StringIO()
	test_df.to_csv(csv_buf2)
	csv_buf2.seek(0)
	s3.Bucket('lead.engine').put_object(Key='{}/test.csv'.format(username), Body=csv_buf2.getvalue())


if authentication_status==False:
	st.error("Incorrect Credentials")
if authentication_status==None:
	st.warning("Enter Credentials")
if authentication_status==True:


	# DATA INTEGRATION #########################################

	authenticator.logout("Logout","sidebar")
	with open('saved_users.json','r') as f:
		json_object=json.loads(f.read())


	if username in json_object['saved_users']:
		client=boto3.client('s3',
					aws_access_key_id=my_access_key,
					aws_secret_access_key=my_secret_access_key)

		obj1=client.get_object(Bucket='lead.engine',Key='{}/train.csv'.format(username))
		obj2=client.get_object(Bucket='lead.engine',Key='{}/test.csv'.format(username))

		df=pd.read_csv(obj1['Body'],index_col=[0])
		test_df=pd.read_csv(obj2['Body'],index_col=[0])
		score_sheet_detailed,cut_off_score=lead_score_generator(df,test_df)
		st.session_state["my_input"]=score_sheet_detailed
		st.session_state["my_input3"]=cut_off_score

		st.info("Data source already connected, go to page 2 from the sidebar")

	else:
		st.title("Connect with Data Source")

		source=st.selectbox("Enter Data Source",['AWS Bucket','Mixpanel','Manual Upload','Demo Data'])

		if(source=='AWS Bucket'):
			st.info("All credentials provided here will not be shared")
			access_key=st.text_input("Enter AWS S3 Bucket Access Key")
			secret_access_key=st.text_input("Enter AWS S3 Bucket Secret Access Key",type='password')
			bucket_link=st.text_input("Enter Bucket Name")
			train_file_name=st.text_input("Enter Historical Data File Name(.csv)")
			test_file_name=st.text_input("Enter Live Data File Name (.csv)")

			with st.expander("Click to know more about data integration..."):
				image = Image.open('demo_data.png')
				st.image(image,caption="An overview of the data to be uploaded...")

			button=st.button("Click after providing information")
			st.session_state["my_input2"]=button

			if(button):
				try:
					client=boto3.client('s3',
					aws_access_key_id=access_key,
					aws_secret_access_key=secret_access_key)

					obj1=client.get_object(Bucket=bucket_link,Key=train_file_name)
					obj2=client.get_object(Bucket=bucket_link,Key=test_file_name)

					df=pd.read_csv(obj1['Body'],index_col=[0])
					test_df=pd.read_csv(obj2['Body'],index_col=[0])

					json_object['saved_users'].append(username)
					with open('saved_users.json','w') as f:
						json.dump(json_object,f)
					store_data_aws(df,test_df)

				except Exception as e:
					print(e)
					print("Error in integration, please verify details and try again")

			

		elif(source=='Mixpanel'):
			st.info("All credentials provided here will not be shared")
			username=st.text_input("Enter your Service Account Username")
			secret=st.text_input("Enter your Service Account Secret Key",type='password')
			project_id=st.text_input("Enter your Project ID")
			token=st.text_input("Enter your Token")
			today_date=date.today()
			pivot_date=st.date_input("Enter Pivot Date (data after this date will be considered as your active leads) ",max_value=today_date)

			with st.expander("Click to know more about data integration..."):
				image = Image.open('demo_data.png')
				st.image(image,caption="An overview of the data to be uploaded...")

			button=st.button("Click after providing information")
			st.session_state["my_input2"]=button

			if(button):
				try:
					# url="https://data.mixpanel.com/api/2.0/export?from_date={}&to_date={}".format(from_date,to_date)
					mputils = MixpanelUtils(
						secret,
						service_account_username=username,
						project_id=project_id,
						token=token,
					)
					mputils.export_people('people_export.csv',format='csv')
					df=pd.read_csv('people_export.csv')
					# st.dataframe(df)
					# st.write(df.info())


					# api_secret_bytes=api_secret.encode("ascii")
					# base64_bytes = base64.b64encode(api_secret_bytes)
					# base64_string = base64_bytes.decode("ascii")
					# headers = {
					#     "accept": "text/plain",
					#     "authorization": "Basic "+base64_string
					# }
					# response=requests.get(url,headers=headers)

					# df=pd.read_json(response.text,lines=True)

					#### DATA SPLITTING ######################################
					date_col=[x for x in df.select_dtypes(include=np.datetime64).columns]
					date_df=df[date_col]
					
					test_idx=[]

					for i in range(df.shape[0]):
						if (date_df.iloc[i,0] >= pivot_date):
							test_idx.append(i)

					test_df=df.iloc[test_idx,:]
					df=df.drop(test_idx)

					json_object['saved_users'].append(username)
					with open('saved_users.json','w') as f:
						json.dump(json_object,f)

					store_data_aws(df,test_df)

				except Exception as e:
					print(e)
					print("Error in integration, please verify details and try again")



		elif(source=='Manual Upload'):
			hist_file=st.file_uploader(label="Upload Historical Data (with target as the last feature)",
				type=['csv'])
			live_file=st.file_uploader(label="Upload Live Data",
				type=['csv'])

			with st.expander("Click to know more about data integration..."):
				image=Image.open('demo_data.png')
				st.image(image,caption="An overview of the data to be uploaded...")

			button=st.button("Click after uploading data")
			st.session_state["my_input2"]=button

			if(button):
				try:
					df,test_df=upload_data(hist_file,live_file)

					json_object['saved_users'].append(username)
					with open('saved_users.json','w') as f:
						json.dump(json_object,f)
					store_data_aws(df,test_df)

				except Exception as e:
					print(e)
					print("Error in uploading, please verify and try again")

		elif(source=='Demo Data'):
			client=boto3.client('s3',
			aws_access_key_id=my_access_key,
			aws_secret_access_key=my_secret_access_key)

			button=st.button("Click here to continue with demo data")
			st.session_state["my_input2"]=button

			obj1=client.get_object(Bucket='sample0104',Key='train.csv') #pql.demodata     saas/train.csv
			obj2=client.get_object(Bucket='sample0104',Key='test.csv') #pql.demodata     saas/test.csv

			df=pd.read_csv(obj1['Body'])
			test_df=pd.read_csv(obj2['Body'])

		if(button):
			try:
				score_sheet_detailed,cut_off_score= lead_score_generator(df,test_df)
				st.success("Upload Done...")
				st.session_state["my_input"]=score_sheet_detailed
				st.session_state["my_input3"]=cut_off_score
				# st.dataframe(score_sheet)
			except Exception as e:
				print(e)
				st.sidebar.error("Error in uploading, please try again")
		# else:
		# 	try:
		# 		score_sheet_detailed,cut_off_score=lead_score_generator(df,test_df)
		# 		st.session_state["my_input"]=score_sheet_detailed
		# 		st.session_state["my_input3"]=cut_off_score

		# 		st.info("Demo Data Loaded...")
		# 	except Exception as e:
		# 		print(e)
		# 		st.sidebar.error("Error in uploading, please try again")


