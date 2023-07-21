import streamlit as st
import numpy as np
import plotly_express as px
import pandas as pd
import time
import plotly.graph_objects as go
# from st_aggrid import AgGrid

st.set_page_config(
	page_title="LeadEngine | Insights",
	layout='wide',
	page_icon="🤖")

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
            

st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

try:
	donut_test_df=st.session_state['my_input4']

	numeric_columns=[x for x in donut_test_df.select_dtypes(include=np.number).columns]

	cols_to_scale=[]
	for col in numeric_columns:
	    if len(donut_test_df[col].unique()) > 2:
	        cols_to_scale.append(col)



	cols_to_scale.pop()  #to remove lead scores columns

	metrics_df=pd.DataFrame()
	metrics_df['USERID']=donut_test_df['USERID']
	# st.write(donut_test_df)  #streamlit
	metrics_df['Lead Scores']=donut_test_df['Lead Scores']
	metrics_df['Win Funnel']=donut_test_df['Win Funnel']

	relative_metrics_df=pd.DataFrame()

	# GETTING AVG SCORES ##############################################################

	avg_vals_3=[]
	avg_vals_2=[]
	avg_vals_1=[]
	avg_vals_0=[]

	df_0=donut_test_df[donut_test_df['Win Funnel']=="Cold Lead"]
	df_1=donut_test_df[donut_test_df['Win Funnel']=="Warm Lead"]
	df_2=donut_test_df[donut_test_df['Win Funnel']=="Hot Lead"]

	# df_3=donut_test_df[donut_test_df['Win Funnel']=="Whales"]

	for i in cols_to_scale:
		# avg_vals_3.append(df_3[i].mean())
		avg_vals_2.append(df_2[i].median())
		avg_vals_1.append(df_1[i].median())
		avg_vals_0.append(df_0[i].median())


	# COMPARING EACH VALUE WITH AVG VALUE #########################################

	# diff_2=[]
	diff_1=[]
	diff_0=[]

	# df_2_2=df_2[cols_to_scale]
	df_1_2=df_1[cols_to_scale]
	df_0_2=df_0[cols_to_scale]

	# metrics_df_2=pd.DataFrame()
	metrics_df_1=pd.DataFrame()
	metrics_df_0=pd.DataFrame()
	
	for i,j in enumerate(cols_to_scale):
		# x=df_2_2[j].apply(lambda x:(((x-avg_vals_2[i])/avg_vals_3[i])* 100))
		# metrics_df_2=pd.concat([metrics_df_2,x],axis='columns')

		y=df_1_2[j].apply(lambda x:(((x-avg_vals_1[i])/avg_vals_2[i])* 100))
		metrics_df_1=pd.concat([metrics_df_1,y],axis='columns')

		z=df_0_2[j].apply(lambda x:(((x-avg_vals_0[i])/avg_vals_2[i])* 100))
		metrics_df_0=pd.concat([metrics_df_0,z],axis='columns')

	
	metrics_df_0=metrics_df_0.dropna(axis='columns')
	metrics_df_1=metrics_df_1.dropna(axis='columns')
	# metrics_df_2=metrics_df_2.dropna(axis='columns')

	# metrics_okay_df= metrics_okay_df.dropna(axis='columns')
	# metrics_no_df=metrics_no_df.dropna(axis='columns')

	metrics_df_0=metrics_df_0.round(2)
	metrics_df_1=metrics_df_1.round(2)
	# metrics_df_2=metrics_df_2.round(2)

	# metrics_okay_df=metrics_okay_df.round(2)
	# metrics_no_df=metrics_no_df.round(2)

	# metrics_values_2=metrics_df_2.copy()
	metrics_values_1=metrics_df_1.copy()
	metrics_values_0=metrics_df_0.copy()

	# metrics_values_okay=metrics_okay_df.copy()
	# metrics_values_no=metrics_no_df.copy()

	# info_2=metrics_df[metrics_df["Win Funnel"]=='Dolphins']
	# metrics_df_2=pd.concat([info_2,metrics_df_2],axis='columns')

	info_1=metrics_df[metrics_df["Win Funnel"]=='Warm Lead']
	metrics_df_1=pd.concat([info_1,metrics_df_1],axis='columns')

	info_0=metrics_df[metrics_df["Win Funnel"]=='Cold Lead']
	metrics_df_0=pd.concat([info_0,metrics_df_0],axis='columns')

	# okay_info=metrics_df[metrics_df["Win Funnel"]=='Still Exploring']
	# metrics_okay_df=pd.concat([okay_info,metrics_okay_df],axis='columns')

	# no_info=metrics_df[metrics_df["Win Funnel"]=='No Go Lead']
	# metrics_no_df=pd.concat([no_info,metrics_no_df],axis='columns')

	# insights_df_2=info_2.copy()
	insights_df_1=info_1.copy()
	insights_df_0=info_0.copy()

	# insights_okay_df=okay_info.copy()
	# insights_no_df=no_info.copy()

	# min_2_row=metrics_df_2.min(axis=1)
	# max_2_row=metrics_df_2.max(axis=1)
	# min_2_col=metrics_values_2.idxmin(axis=1)
	# max_2_col=metrics_values_2.idxmax(axis=1)

	min_1_row=metrics_df_1.min(axis=1)
	max_1_row=metrics_df_1.max(axis=1)
	min_1_col=metrics_values_1.idxmin(axis=1)
	max_1_col=metrics_values_1.idxmax(axis=1)

	min_0_row=metrics_df_0.min(axis=1)
	max_0_row=metrics_df_0.max(axis=1)
	min_0_col=metrics_values_0.idxmin(axis=1)
	max_0_col=metrics_values_0.idxmax(axis=1)



	# min_okay_row=metrics_okay_df.min(axis=1)
	# max_okay_row=metrics_okay_df.max(axis=1)

	# max_okay_col=metrics_values_okay.idxmax(axis=1)
	# min_okay_col=metrics_values_okay.idxmin(axis=1)

	# min_no_row=metrics_no_df.min(axis=1)
	# max_no_row=metrics_no_df.max(axis=1)

	# max_no_col=metrics_values_no.idxmax(axis=1)
	# min_no_col=metrics_values_no.idxmin(axis=1)

	# insights_df_2['Highest Deviation']=max_2_row
	# insights_df_2['Highest Deviator']=max_2_col
	# insights_df_2['Lowest Deviation']=min_2_row
	# insights_df_2['Lowest Deviator']=min_2_col

	insights_df_1['Highest Deviation']=max_1_row
	insights_df_1['Highest Deviator']=max_1_col
	insights_df_1['Lowest Deviation']=min_1_row
	insights_df_1['Lowest Deviator']=min_1_col

	insights_df_0['Highest Deviation']=max_0_row
	insights_df_0['Highest Deviator']=max_0_col
	insights_df_0['Lowest Deviation']=min_0_row
	insights_df_0['Lowest Deviator']=min_0_col

	# insights_okay_df['Highest Deviation']=max_okay_row
	# insights_okay_df['Highest Deviator']=max_okay_col
	# insights_okay_df['Lowest Deviation']=min_okay_row
	# insights_okay_df['Lowest Deviator']=min_okay_col

	# insights_no_df['Highest Deviation']=max_no_row
	# insights_no_df['Highest Deviator']=max_no_col
	# insights_no_df['Lowest Deviation']=min_no_row
	# insights_no_df['Lowest Deviator']=min_no_col

	# insights_df_2=insights_df_2.reset_index(drop=True)
	insights_df_1=insights_df_1.reset_index(drop=True)
	insights_df_0=insights_df_0.reset_index(drop=True)

	# insights_okay_df=insights_okay_df.reset_index(drop=True)
	# insights_no_df=insights_no_df.reset_index(drop=True)

	st.title("Lead Insights")

	win=st.selectbox("Filter results on",['Warm Lead','Cold Lead'])


	# GREEN AND RED INDICATIONS IN DATAFRAME #######################################
	def style_negative(v, props=''):
	    try: 
	        return props if v < 0 else None
	    except:
	        pass

	def style_positive(v, props=''):
		try: 
			return props if v > 0 else None
		except:
			pass

	st.subheader("Deviations from ideal values")
	# if win=='Dolphins':
	# 	insights_df_2=insights_df_2.reset_index(drop=True)
	# 	insights_df_2_scores=insights_df_2['Lead Scores']
	# 	insights_df_2=insights_df_2.drop(['Lead Scores'],axis='columns')
	# 	insights_df_2_styled=insights_df_2.style.hide_index().applymap(style_negative,props='color:red;').applymap(style_positive, props='color:green;')

	# 	st.dataframe(insights_df_2_styled)
	if win=='Warm Lead':
		insights_df_1=insights_df_1.reset_index(drop=True)
		insights_df_1_scores=insights_df_1['Lead Scores']
		insights_df_1=insights_df_1.drop(['Lead Scores'],axis='columns')
		insights_df_1_styled=insights_df_1.style.hide_index().applymap(style_negative,props='color:red;').applymap(style_positive, props='color:green;')

		st.dataframe(insights_df_1_styled)

		# insight_no_df=insights_no_df.reset_index(drop=True)
		# insights_no_df_scores=insights_no_df['Lead Scores']
		# insights_no_df=insights_no_df.drop(['Lead Scores'],axis='columns')
		# insights_no_df_styled=insights_no_df.style.hide_index().applymap(style_negative,props='color:red;').applymap(style_positive, props='color:green;')
		
		# st.dataframe(insights_no_df_styled)

		# # st.dataframe(insights_no_df.applymap(style_negative,props='color:red;').applymap(style_positive, props='color:green;'))

	else:
		insights_df_0=insights_df_0.reset_index(drop=True)
		insights_df_0_scores=insights_df_0['Lead Scores']
		insights_df_0=insights_df_0.drop(['Lead Scores'],axis='columns')
		insights_df_0_styled=insights_df_0.style.hide_index().applymap(style_negative,props='color:red;').applymap(style_positive, props='color:green;')

		st.dataframe(insights_df_0_styled)

	# st.write(metrics_df_2)

	metrics_df_full=pd.concat([metrics_df_1,metrics_df_0])

	user_id=st.text_input('Enter USERID')

	# attributes=donut_test_df.columns.tolist()
	attr=st.multiselect('Attributes',options=cols_to_scale,default=cols_to_scale)

	if (user_id==""):
		st.info("Enter USERID")
	else:

		metric_card_df=metrics_df_full[metrics_df_full['USERID']==user_id]
		metric_card_df2=donut_test_df[donut_test_df['USERID']==user_id]

		cols=st.columns(len(attr))

		for i in attr:
			st.metric(label=i,value=metric_card_df2[i],delta=metric_card_df[i].values[0])

		# metric_card_df=metrics_okay_df[metrics_okay_df['USERID']==user_id]
		# metric_card_df2=donut_test_df[donut_test_df['USERID']==user_id]
		# for i in attr:
		# 	st.metric(label=i,value=metric_card_df2[i],delta=metric_card_df[i].values[0])


	# with st.expander("Click to export data"):
	# 	@st.experimental_memo
	# 	def convert_df(df):
	# 	   return df.to_csv(index=False).encode('utf-8')

	# 	df=st.session_state["my_input"]
	# 	csv=convert_df(df)
	# 	st.download_button(
	# 	   "Download as CSV",
	# 	   csv,
	# 	   "file.csv",
	# 	   "text/csv",
	# 	   key='download-csv'
	# 	)

		




	# for i in attr:
	# 	st.metric(i,)
except Exception as e:
	print(e)
	st.error("Upload Data in Homepage")

