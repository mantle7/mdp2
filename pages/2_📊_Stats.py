import streamlit as st
import plotly_express as px
import pandas as pd
import time
import plotly.graph_objects as go
# from st_aggrid import AgGrid

global df,test_df

st.set_page_config(
	page_title="LeadEngine | Stats",
	layout='wide',
	page_icon="🤖")

st.set_option('deprecation.showfileUploaderEncoding',False)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
            

st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

hide_dataframe_row_index = """
            <style>
            .row_heading.level0 {display:none}
            .blank {display:none}
            </style>
            """

st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)

if "my_input4" not in st.session_state:
	st.session_state["my_input4"]="xyz"

if "my_input5" not in st.session_state:
	st.session_state["my_input5"]="xyz"



try:
	test_df=st.session_state["my_input"]
	optimal_score=st.session_state["my_input3"]
	test_df['Lead Scores']=test_df['Lead Scores'].astype(int)

	# st.write(optimal_score)
	

	#CUSTOMER SEGMENTATION ##############################################
	percentile=test_df.sort_values(by=['Lead Scores'],ascending=True)

	percentile=percentile.reset_index(drop=True)

	percentile_idx1=percentile.shape[0]*0.98
	percentile_idx1=int(percentile_idx1)
	percentile_score1=percentile.iloc[percentile_idx1,percentile.shape[1]-1]

	# percentile_idx2=percentile.shape[0]*0.95
	# percentile_idx2=int(percentile_idx2)
	# percentile_score2=percentile.iloc[percentile_idx2,percentile.shape[1]-1]

	# x=percentile.iloc[percentile_idx-1,:]
	# percentile_idx_score=percentile.iloc[percentile_idx,test_df.shape[1]-1]

	#

	df_0=test_df[test_df['Lead Scores']<optimal_score]


	df_1=test_df[test_df['Lead Scores']>=optimal_score]
	df_1=df_1[df_1['Lead Scores']<percentile_score1]

	# df_2=test_df[test_df['Lead Scores']>=percentile_score2]
	# df_2=df_2[df_2['Lead Scores']<percentile_score1]

	df_2=test_df[test_df['Lead Scores']>=percentile_score1]

	donut_test_df=test_df.copy()
	win_funnel=[]

	for i in range(donut_test_df.shape[0]):
		# if donut_test_df.iloc[i,donut_test_df.shape[1]-1]==0:
		# 	win_funnel.append('Zero Spenders')
		if donut_test_df.iloc[i,donut_test_df.shape[1]-1]>=percentile_score1:
			win_funnel.append('Hot Lead')
		elif donut_test_df.iloc[i,donut_test_df.shape[1]-1]>=optimal_score and donut_test_df.iloc[i,donut_test_df.shape[1]-1]<percentile_score1:
			win_funnel.append('Warm Lead')
		else:
			win_funnel.append('Cold Lead')

	donut_test_df['Win Funnel']=win_funnel
	value0=df_0.shape[0]
	value1=df_1.shape[0]
	value2=df_2.shape[0]

	# value3=df_3.shape[0]

	st.session_state['my_input4']=donut_test_df

	labels=["Cold Lead","Warm Lead","Hot Lead"]
	values=[value0,value1,value2]

	#DONUT FOR SEGMENTATION ##############################################
	fig1=px.pie(donut_test_df,values=values,names=labels,hole=0.55)

	#  TOP HOT LEADS #####################################################
	top_hot=df_2.sort_values(by=['Lead Scores'],ascending=False)
	top_hot=top_hot.reset_index(drop=True)

	# TOP WARM LEADS ########################################################
	top_warm=df_1.sort_values(by=['Lead Scores'],ascending=False)
	top_warm=top_warm.reset_index(drop=True)

	#TOP 10 ##############################################

	# top10=test_df.sort_values(by='Lead Scores',ascending=False)
	# top10=top10.iloc[0:10,test_df.shape[1]-1]
	# top10_idx=top10.index

	# x=test_df.iloc[0:10,0]
	# top10=top10.reset_index(drop=True)

	# top10=pd.concat([x,top10],axis='columns')
	# top10=top10.sort_values(by='Lead Scores',ascending=False)

	# fig2=px.bar(top10,x='USERID',y='Lead Scores',orientation='v')


	##############  ROW1 ###########################################
	col1,col2,col3=st.columns((1,1,4))

	with col1:

		col1.metric("Hot Leads (last month)",df_2.shape[0])
		col1.write("---")
		col1.metric("Cold Leads (last month)",df_0.shape[0])
		# st.plotly_chart(fig2)

	with col2:

		col2.metric("Warm Leads (last month)",df_1.shape[0])
		col2.write("---")
		# col2.metric("Zero Spenders (last month)",df_0.shape[0])
		# st.plotly_chart(fig2)

	col3.subheader("Lead Segmentation")
	with col3:    
	    # col3.subheader = "Leads By Win Funnel"
	    st.plotly_chart(fig1)


	##############  ROW2 SIDEBAR  ###########################################
	st.write("---")

	col4,col5=st.columns((1,1))

	col4.subheader("Top Hot Leads")
	col5.subheader("Top Warm Leads")

	with col4:
		df_2=df_2.sort_values(by='Lead Scores',ascending=False)
		st.write(df_2)

	with col5:
		df_1=df_1.sort_values(by='Lead Scores',ascending=False)
		st.write(df_1)



	# user_id=st.text_input('Enter USERID')

	# # win=st.sidebar.multiselect("Win Funnel",
	# # 	options=donut_test_df['Win Funnel'].unique(),
	# # 	default=donut_test_df['Win Funnel'].unique())

	# win = st.selectbox("Filter results on", donut_test_df['Win Funnel'].unique().tolist())

	# attributes=test_df.columns.tolist()
	# attr=st.multiselect('Attributes',options=attributes,default=attributes)

	# global df_selection1,df_selection1_idx,df_selection2
	# if user_id=="":
	# 	df_selection1_idx=donut_test_df[donut_test_df['Win Funnel']==win].index
	# 	df_selection2=donut_test_df.loc[df_selection1_idx,attr]
	# 	# AgGrid(df_selection2)
	# 	st.dataframe(df_selection2)
	# else:
		
	# 	try:
	# 		df_selection1_idx=donut_test_df[donut_test_df['Win Funnel']==win].index
	# 		df_selection2=donut_test_df.loc[df_selection1_idx,attr]
	# 		df_selection3=df_selection2[df_selection2[donut_test_df.columns[0]]==user_id]
	# 		# AgGrid(df_selection3)
	# 		st.dataframe(df_selection3)
	# 	except Exception as e:
	# 		print(e)
	# 		st.error("Clear Filters and try again")


except Exception as e:
	print(e)
	st.error("Upload Data in Homepage")






