import streamlit as st
import plotly.express as px
import pandas as pd
import os
import warnings
import plotly.graph_objects as go
warnings.filterwarnings('ignore')

st.set_page_config(page_title='Lahori Wind Analysis!!!', page_icon=':bar_chart:',layout='wide')

st.title(" :bar_chart: Lahori Wind Analysis")
st.markdown('<style>div.block-container{padding-top:1rem;}</style',unsafe_allow_html=True)

fl = st.file_uploader(":file_folder: Upload a file", type=(['csv','txt','xlsx','xls']))
if fl is not None:
    dataframe = pd.read_excel(fl)
    st.write(dataframe)

st.header("MA% Analysis")

df = pd.read_excel('Consolidated.xlsx',sheet_name='DGR File')


col1,col2 = st.columns((2))
df['Log Date'] = pd.to_datetime(df['Log Date'])

#getting min & max date
startDate = pd.to_datetime(df['Log Date']).min()
endDate = pd.to_datetime(df['Log Date']).max()

#filter data with date
with col1:
    date1 = pd.to_datetime(st.date_input('Start Date',startDate))

with col2:
    date2 = pd.to_datetime(st.date_input("End Date",endDate))

df = df[(df['Log Date']>=date1) & (df['Log Date']<=date2)].copy()

#filter data based on other parameters:

st.sidebar.header("Choose your filter: ")
site_name = st.sidebar.multiselect("Pick your Site", df['site Name'].unique())

if not site_name:
    df2 = df.copy()
else:
    df2 = df[df['site Name'].isin(site_name)]

turbine_code = st.sidebar.multiselect('Pick the state',df['Turbine Code'].unique())

if not turbine_code:
    df3 = df2.copy()
else:
    df3 = df2[df2['Turbine Code'].isin(turbine_code)]

#filter the data based on region state & city
if not site_name and not turbine_code:
    filtered_df = df

elif site_name and turbine_code:
    filtered_df = df[(df['site Name'].isin(site_name)) & (df['Turbine Code'].isin(turbine_code))]
elif not site_name:
    filtered_df = df[df['Turbine Code'].isin(turbine_code)]
elif not turbine_code:
    filtered_df = df[df['site Name'].isin(site_name)]
else:
    filtered_df = df3[df3['site Name'].isin(site_name) & df3['Turbine Code'].isin(turbine_code)]


ma_df = filtered_df.groupby(['Turbine Code'])['MMA'].mean().sort_values().reset_index()



with col1:
    st.subheader('Turbine wise MA%')
    threshold = 99.5
    # Define a custom color map based on the threshold value
    color_map = {'High Performing WTGs': 'blue', 'Low Performing WTGs': 'red'}
    # Create a new column to categorize values based on the threshold
    ma_df['ColorCategory'] = ma_df['MMA'].apply(lambda x: 'Low Performing WTGs' if x <= threshold else 'High Performing WTGs')
    fig = px.bar(ma_df,x='Turbine Code',y='MMA',template = 'seaborn',color='ColorCategory',color_discrete_map=color_map)
    st.plotly_chart(fig,use_container_width=True,height = 200)

ma_df2 = filtered_df.groupby(['site Name'])['MMA'].mean().reset_index()

with col2:
    st.subheader('Customer wise MA%')
    # Define a custom color map based on the threshold value
    color_map = {'High Performing': 'blue', 'Low Performing': 'red'}
    # Create a new column to categorize values based on the threshold
    ma_df2['ColorCategory'] = ma_df2['MMA'].apply(lambda x: 'Low Performing' if x <= threshold else 'High Performing')
    fig = px.bar(ma_df2,x='site Name',y='MMA',template = 'seaborn',color='ColorCategory',color_discrete_map=color_map)
    fig.update_traces(text = ma_df2['site Name'], textposition = 'outside')
    st.plotly_chart(fig, use_container_width=True) 

st.divider()

st.header("Error Analysis")

cl1,cl2= st.columns((2))
error_df = pd.read_excel('Consolidated.xlsx',sheet_name='Error Log')
#getting min & max date
start_date = pd.to_datetime(error_df['Date']).min()
end_date = pd.to_datetime(error_df['Date']).max()

#filter data with date
with col1:
    date_1 = pd.to_datetime(st.date_input('Starting Date for Error Analysis',start_date))

with col2:
    date_2 = pd.to_datetime(st.date_input("Ending Date for Error Analysis",end_date))

error_df = error_df[(error_df['Date']>=date_1) & (error_df['Date']<=date_2)].copy()

cl_1,cl_2,cl_3 = st.columns((3))

with cl_1:
    customer = st.multiselect("Choose O&M Customer Here:",error_df['Customer'].unique())

with cl_2:
    wtg = st.multiselect("Choose WTG Name Here:",error_df['Location'].unique())

with cl_3:
    main_category = st.multiselect("Choose Error Category Name Here:",error_df['Main Category'].unique())

if main_category and wtg and customer:
    filtered_df = error_df[
        (error_df['Location'].isin(wtg)) &
        (error_df['Main Category'].isin(main_category))&(error_df['Customer'].isin(customer))]

elif customer and wtg:
    filtered_df = error_df[
        (error_df['Customer'].isin(customer)) &
        (error_df['Location'].isin(wtg))]

elif customer and main_category:
    filtered_df = error_df[
        (error_df['Customer'].isin(customer)) &
        (error_df['Main Category'].isin(main_category))]

elif main_category and wtg:
    filtered_df = error_df[
        (error_df['Location'].isin(wtg)) &
        (error_df['Main Category'].isin(main_category))]
    
elif customer:
    filtered_df = error_df[error_df['Customer'].isin(customer)]
elif wtg:
    filtered_df = error_df[error_df['Location'].isin(wtg)]

elif main_category:
    filtered_df = error_df[error_df['Main Category'].isin(main_category)]

else:
    filtered_df = error_df


col_3,col_4 = st.columns((2))

error_df2 = filtered_df.groupby(['Error Category'])['Duration'].sum().reset_index()

with col_3:
    st.subheader('Error Categories Based on Fault Duration in Hrs.')
    fig = px.pie(error_df2,values='Duration',names='Error Category',template='gridon')
    fig.update_traces(text = error_df2['Error Category'],textposition = 'inside')
    st.plotly_chart(fig,use_container_width=True)

   