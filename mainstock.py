#import plotly
from Plotly import graph_objs as go
#requirement.txt

import streamlit as st
from datetime import date

#from prophet import Prophet
#from prophet.plot import plot_plotly
import csv
import pandas as pd



st.title("Stock Prediction App")

stocks = ("BRITANNIA", 'MARUTI', 'TATASTEEL', 'ASIANPAINTS', 'SBIBANK')
selected_stocks = st.selectbox("Select dataset for prediction", stocks)

n_years=st.slider("Years of prediction: ", 1, 5)
period=n_years*365

# Function to load data
@st.cache_data
def load_data(csv_file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)
    return df

# Define the file paths for each stock
file_paths = {
    "BRITANNIA": "britannia_data.csv",
    "MARUTI": "maruti_data.csv",
    "TATASTEEL": "tatasteel_data.csv",
    "ASIANPAINTS": "asian_paints_data.csv",
    "SBIBANK": "sbi_bank_data.csv"
}

# Load the data
data_load_state = st.text("Load Data...")
data = load_data(file_paths[selected_stocks])
data_load_state.text("Loading Data...Done!")

st.subheader('Initial Data')

# Display the data
st.write(data)


data = data.fillna(method='ffill')
data = data.fillna(method='bfill')

def plot_raw_data():
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Stock_Open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Stock_Close'))
    fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

df_train=data[['Date', 'Close']]
df_train=df_train.rename(columns={'Date': "ds", "Close": "y"})

m=Prophet()
m.fit(df_train)
future=m.make_future_dataframe(period)
forecast=m.predict(future)

st.subheader('Forecast Data')
st.write(forecast.tail())    

st.write('forecast data')
fig1=plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write('forecast components')
fig2=m.plot_components(forecast)
st.write(fig2)
