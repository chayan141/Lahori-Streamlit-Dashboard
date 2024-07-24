import streamlit as st
import plotly.express as px
import pandas as pd
import os
import warnings
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error, mean_squared_error
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
with cl1:
    date_1 = pd.to_datetime(st.date_input('Starting Date for Error Analysis',start_date))

with cl2:
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

with col_4:
    st.subheader('Error Categories Based on Fault Count in Hrs.')
    # st.subheader(" ")
    error_df3 = filtered_df.groupby(['Error Category'])['Duration'].count().reset_index()
    error_df3.columns = ['Error Category','Error Count']
    fig = px.pie(error_df3,values='Error Count',names='Error Category',template='gridon')
    fig.update_traces(text = error_df3['Error Category'],textposition = 'inside')
    st.plotly_chart(fig,use_container_width=True)

with col_3:
    st.subheader('Turbine Wise MTBF Hrs.')

    max_date = filtered_df['Date'].max()
    min_date = filtered_df['Date'].min()

    # Calculate the number of days between the maximum and minimum dates
    days_difference = (max_date - min_date).days

    # Add the days_difference as a new column to the DataFrame
    filtered_df['days_difference'] = days_difference


    # Display the modified DataFrame
    # st.write(filtered_df)

    mtbf_df = filtered_df.groupby(['Location','Customer']).agg({
    'Duration': ['sum','count'],
    'days_difference':'mean'   
    }).reset_index()

    # Rename the columns for clarity (optional)
    mtbf_df.columns = ['Location','Customer','Total Duration (Hrs.)','Total Error Count','days_difference']
    mtbf_df['MTBF Hrs.'] = ((mtbf_df['days_difference']*24)-mtbf_df['Total Duration (Hrs.)'])/mtbf_df['Total Error Count']
    mtbf_df['Target MTBF Hrs.'] = 100
    mtbf_df['MTTR'] = mtbf_df['Total Duration (Hrs.)']/mtbf_df['Total Error Count']

    # st.write(mtbf_df)
    x_values = mtbf_df['Location']
    y_values_bar = mtbf_df['MTBF Hrs.']
    y_values_line1 = mtbf_df['Target MTBF Hrs.']
    y_values_line2 = mtbf_df['MTTR']
    

    # Create a bar chart trace
    trace_bar = go.Bar(x=x_values, y=y_values_bar, name='MTBF Hrs.')

    threshold = 100  # You can set your desired threshold value

    # Create traces for bars above and below the threshold
    trace_above_threshold = go.Bar(
        x=x_values[y_values_bar >= threshold],
        y=y_values_bar[y_values_bar >= threshold],
        name='Above Threshold',
        marker=dict(color='green')
    )

    trace_below_threshold = go.Bar(
        x=x_values[y_values_bar < threshold],
        y=y_values_bar[y_values_bar < threshold],
        name='Below Threshold',
        marker=dict(color='red')
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Create a line chart trace
    fig.add_trace(go.Scatter(x=x_values, y=y_values_line1, mode='lines+markers', name='Target MTBF Hrs.',line=dict(color='blue')), secondary_y=False)
    fig.add_trace(go.Scatter(x=x_values, y=y_values_line2, mode='lines+markers', name='MTTR',line=dict(color='yellow')), secondary_y=True)

    # Add bar chart trace to the secondary y-axis
    fig.add_trace(trace_above_threshold, secondary_y=False)
    fig.add_trace(trace_below_threshold, secondary_y=False)

    # Update the layout
    fig.update_layout(xaxis=dict(title='turbine Name',showticklabels=True,tickmode='array', dtick=1),
                    legend=dict(x=0.5, y=1.1),
                    )

    # Update the y-axis labels
    fig.update_yaxes(title_text="MTBF Hrs.", secondary_y=False)
    fig.update_yaxes(title_text="MTTR", secondary_y=True)

    st.plotly_chart(fig,use_container_width=True)




with col_4:
    st.subheader('Customer Wise MTBF Hrs')
    customer_mtbf = mtbf_df.groupby(['Customer']).agg({
    'MTBF Hrs.': 'mean',
    'Target MTBF Hrs.': 'mean',
    'MTTR':'mean'   
    }).reset_index()

    x_values = customer_mtbf['Customer']
    y_values_bar = customer_mtbf['MTBF Hrs.']
    y_values_line1 = customer_mtbf['Target MTBF Hrs.']
    y_values_line2 = customer_mtbf['MTTR']
    

    # Create a bar chart trace
    trace_bar = go.Bar(x=x_values, y=y_values_bar, name='MTBF Hrs.')

    threshold = 100  # You can set your desired threshold value

    # Create traces for bars above and below the threshold
    trace_above_threshold = go.Bar(
        x=x_values[y_values_bar >= threshold],
        y=y_values_bar[y_values_bar >= threshold],
        name='MTBF Above Threshold',
        marker=dict(color='green')
    )

    trace_below_threshold = go.Bar(
        x=x_values[y_values_bar < threshold],
        y=y_values_bar[y_values_bar < threshold],
        name='MTBF Below Threshold',
        marker=dict(color='red')
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Create a line chart trace
    fig.add_trace(go.Scatter(x=x_values, y=y_values_line1, mode='lines+markers', name='Target MTBF Hrs.',line=dict(color='blue')), secondary_y=False)
    fig.add_trace(go.Scatter(x=x_values, y=y_values_line2, mode='lines+markers', name='MTTR',line=dict(color='yellow')), secondary_y=True)

    # Add bar chart trace to the secondary y-axis
    fig.add_trace(trace_above_threshold, secondary_y=False)
    fig.add_trace(trace_below_threshold, secondary_y=False)

    # Update the layout
    fig.update_layout(xaxis=dict(title='turbine Name',showticklabels=True,tickmode='array', dtick=1),
                    legend=dict(x=0.5, y=1.1),
                    )

    # Update the y-axis labels
    fig.update_yaxes(title_text="MTBF Hrs.", secondary_y=False)
    fig.update_yaxes(title_text="MTTR", secondary_y=True)

    st.plotly_chart(fig,use_container_width=True)

st.divider()

st.header("Power Curve Analysis")

df_1 = pd.read_excel('Consolidated.xlsx',sheet_name = 'Power Curve')
df_2 = pd.read_excel('Consolidated.xlsx',sheet_name = 'Turbine Hub')
pc_df = pd.merge(df_1, df_2, on='WTG')

col_5,col_6 = st.columns((2))
with col_5:
    # st.write(df_1)
    # st.write(pc_df)
    wtg2 = st.multiselect("Choose WTG Name Here:",pc_df['WTG'].unique())
    

with col_6:
    plc_stats1 = st.multiselect("Choose PLC Status Here:",pc_df['PLC Status Minutes'].unique())

if wtg2 and plc_stats1:
    filtered_pc = pc_df[
        (pc_df['WTG'].isin(wtg2)) &
        (pc_df['PLC Status Minutes'].isin(plc_stats1))]
elif wtg2:
    filtered_pc = pc_df[
        pc_df['WTG'].isin(wtg2)]

elif plc_stats1:
    filtered_pc = pc_df[
        pc_df['PLC Status Minutes'].isin(plc_stats1)]
    
else:
    filtered_pc = pc_df


actual_wind_df = pd.read_excel('Consolidated.xlsx',sheet_name = 'Static Data')

# X = actual_wind_df[['Wind Speed']]
# y = actual_wind_df['Power']

# model = LinearRegression()
# model.fit(X, y)

# new_wind_speeds = filtered_pc['Wind speed (M/sec)']
# new_wind_speeds_2d = new_wind_speeds.values.reshape(-1, 1)

# filtered_pc['predicted_active_power'] = model.predict(new_wind_speeds_2d)

# st.write(filtered_pc)

filtered_pc['Air Pressure'] = 101325*pow(1-((0.0065*filtered_pc['Hub Height (mt)'])/(filtered_pc['Ambient temperature (cls.)']+273.15+(0.0065*filtered_pc['Hub Height (mt)']))),5.257)

filtered_pc['Air density'] = filtered_pc['Air Pressure'] /(287.05*(filtered_pc['Ambient temperature (cls.)']+273.15))

filtered_pc['Standard Power curve Wind speed (m/sec)'] = filtered_pc['Wind speed (M/sec)']*pow((filtered_pc['Air density']/1.225),0.33)

filtered_pc['Rounded Standard Wind Speed'] = filtered_pc['Standard Power curve Wind speed (m/sec)'].round()
filtered_pc['Nearest Below Wind Speed'] = np.floor(filtered_pc['Standard Power curve Wind speed (m/sec)']).astype(int)

filtered_pc1 = pd.merge(filtered_pc, actual_wind_df, left_on='Rounded Standard Wind Speed', right_on='Wind Speed', how='inner')

filtered_pc2 = pd.merge(filtered_pc1, actual_wind_df, left_on='Nearest Below Wind Speed', right_on='Wind Speed', how='inner')



columns_to_drop = ['Site Specific AD_x', 'Wind Speed_x','WTGs (Error Log)_x','TML_x','Actual WTGs_x','TML.1_x','O&M Customer_x',
                   'Site Specific AD_y', 'Wind Speed_y','WTGs (Error Log)_y','TML_y','Actual WTGs_y','TML.1_y','O&M Customer_y'
]

filtered_pc3 = filtered_pc2.drop(columns=columns_to_drop)

# Define the growth function
def growth_function(x, a, b):
    return a * np.exp(b * x)

# Fit the growth function to the data
params, covariance = curve_fit(growth_function, filtered_pc3['Nearest Below Wind Speed'], filtered_pc3['Power_y'])

# Extract the fitted parameters
a_fit, b_fit = params

# Predict values using the fitted growth model
predicted_values = growth_function(filtered_pc3['Standard Power curve Wind speed (m/sec)'], a_fit, b_fit)

# Add the predicted values to the DataFrame
filtered_pc3['Predicted Power'] = predicted_values

filtered_pc3['PC Loss'] = (filtered_pc3['Active Power (KW)'] - filtered_pc3['Predicted Power'])/filtered_pc3['Predicted Power']

filtered_pc3['Unit Loss in KWh'] = (filtered_pc3['Active Power (KW)'] - filtered_pc3['Predicted Power'])/6

filtered_pc3['Revenue Loss in INR'] =filtered_pc3['Unit Loss in KWh']*2.5

filtered_pc3['Curtailment Power in KW'] = filtered_pc3['Active Power (KW)'] - filtered_pc3['Predicted Power']

st.write(filtered_pc3)

# col_7,col_8 = st.columns((2))

# with col_7:
#     pc_dev_df = filtered_pc3.groupby(['WTG']).agg({
#     'PC Loss': 'mean',
#     'Unit Loss in KWh': 'sum',
#     'Revenue Loss in INR' : 'sum',
#     'Curtailment Power in KW' : 'sum'

#     }).reset_index()

#     pc_dev_df['PC Loss'] = pc_dev_df['PC Loss'] * 100

#     st.write(pc_dev_df)


    
