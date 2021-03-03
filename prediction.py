import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import webbrowser

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.naive_bayes import GaussianNB
st.markdown('<style>body{background-color: #E8E8E8;}</style>',unsafe_allow_html=True)

url = 'http://localhost/ovi/admin-dashboard.php'

st.title("Online Vehicle Identification System")
if st.button('Back'):
    webbrowser.open_new_tab(url)


@st.cache
def load_data(choise_ds):
   data = pd.read_csv('Traffic_Violations_ds.csv', low_memory=False)
   del data['Date Of Stop']
   del data['Time Of Stop']
   data.dropna(axis=0, subset=['Latitude'], inplace=True)
   data.dropna(axis=0, subset=['Longitude'], inplace=True)
   data.dropna(axis=0, subset=['Year'], inplace=True)
   data.dropna(axis=0, subset=['Article'], inplace=True)
   data.dropna(axis=0, subset=['Geolocation'], inplace=True)
   data['Description'].fillna('Other', inplace = True)
   data['Location'].fillna('Other', inplace = True)
   data['State'].fillna('Other', inplace = True)
   data['Make'].fillna('Other', inplace = True)
   data['Model'].fillna('Other', inplace = True)
   data['Color'].fillna('Other', inplace = True)
   data['Driver City'].fillna('Other', inplace = True)
   data['Driver State'].fillna('Other', inplace = True)
   data['DL State'].fillna('Other', inplace = True)
   return data

@st.cache
def load_data2(parking_data):
   data = pd.read_csv('Parking_Violations_Issued.csv', low_memory=False)
   del data['Summons Number']
   data['Violation In Front Of Or Opposite'].fillna('F', inplace = True)
   data.fillna('other', inplace = True)
   return data




st.header("Predict Class Variable Using Random Forest Classifier")

selected_ds = st.selectbox("Select Datasets",("Traffic Violation Datasets","Parking Violation Datasets"))

if(selected_ds=='Traffic Violation Datasets'):

    st.subheader("Traffic Violation:")
    # ->1. Traffic Violations Datasets 
    df = load_data(selected_ds)
    df_fs = df.copy()
    df_fs = df_fs.head(10000)
    dataset = df_fs.copy()

    varx = st.selectbox("Select Veriable To Predict:",("Gender","Race","Contributed To Accident","Alcohol"))
    year = st.selectbox("Select Year:",(2001,2006,2010))
    lat = st.slider('Latitude', min_value=-94.610988, max_value=40.111822)
    lon = st.slider('Longitude', min_value=-77.732495, max_value=41.543160)

    dataset = dataset[['Latitude', 'Longitude','Year', varx]]
    df_X = dataset.drop(varx, axis=1) 
    y = dataset[varx]

    X = pd.get_dummies(df_X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

    rf = RandomForestClassifier(random_state=10)
    rf.fit(X_train, y_train)
    Race = rf.predict([[lat,lon,year]])
    st.write('Year Of Incident : ',year)
    st.write('Latitude : ',lat)
    st.write('Longitude : ',lon)
    st.write(varx,' is : '+Race[0])

    fig = plt.figure()
    b=sns.countplot(x=varx, data = dataset)
    plt.title('Frequency of '+varx)
    st.pyplot(fig)

else:
    st.subheader("Parking Violation Issued:")

    # ->2. Parking Violations Datasets 
    data_parking = load_data2(selected_ds)
    dset =data_parking.head(10000)
    df_new = data_parking.copy()
    df_ds = data_parking.copy()

    var_y = st.selectbox("Select Variable To Predict:",("Plate Type","Plate ID","Vehicle Make","Vehicle Body Type"))
    vc = st.slider('Violation Code', min_value=0, max_value=99)
    Street1 = st.slider('Street Code 1', min_value=0, max_value=98020)
    Street2 = st.slider('Street Code 2', min_value=0, max_value=98260)
    Street3 = st.slider('Street Code 3', min_value=0, max_value=98260)

    dset = dset[['Street Code1','Street Code2','Street Code3','Violation Code',var_y]]

    df_X = dset.drop(var_y, axis=1) 
    y = dset[var_y]
    X = pd.get_dummies(df_X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(X_train, y_train)

    pred_val= knn.predict([[Street1,Street2,Street3,vc]])
    st.write('Violation Code : ',vc)
    st.write('Street 1 : ',Street1)
    st.write('Street 2 : ',Street2)
    st.write('Street 3 : ',Street3)
    st.write(var_y,' is : '+pred_val[0])

    fig = plt.figure()
    sns.distplot(dset["Violation Code"], bins=10)
    st.pyplot(fig)

