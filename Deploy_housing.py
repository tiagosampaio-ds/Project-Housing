#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import streamlit as st
import joblib


# In[2]:


dados = pd.read_csv('data_train.csv')

st.title('Machine learning project to predict median house values')

st.text('This is a machine learning app to predict the median house value with in a block.\n'  
        'The data that was trained pertains to the houses found in a given California\n'
        'district and some summary stats about them based on the 1990 census data.\n'
        'The location based on the latitude and longitude will appears on the map.\n' 
        'Feel free to set the values of the features and make your predictions.')


longitude = st.slider('Longitude: A measure of how far west a house is; a higher value is farther west', 
                      min_value=float(dados['longitude'].min()), 
                      max_value=float(dados['longitude'].max()), 
                      value=float(dados['longitude'].mean()))

latitude = st.slider('Latitude: A measure of how far north a house is; a higher value is farther north', 
                     min_value=float(dados['latitude'].min()),
                     max_value=float(dados['latitude'].max()), 
                     value=float(dados['latitude'].mean()))

housing_median_age = st.slider('Housing median age: Total number of rooms within a block',
                               min_value=int(dados['housing_median_age'].min()),
                               max_value=int(dados['housing_median_age'].max()), 
                               value=int(dados['housing_median_age'].mean()))

total_rooms = st.slider('Total rooms: Total number of rooms within a block', 
                        min_value=int(dados['total_rooms'].min()),
                        max_value=int(dados['total_rooms'].max()), 
                        value=int(dados['total_rooms'].mean()))

total_bedrooms = st.slider('Total bedrooms: Total number of bedrooms within a block', 
                           min_value=int(dados['total_bedrooms'].min()), 
                           max_value=int(dados['total_bedrooms'].max()), 
                           value=int(dados['total_bedrooms'].mean()))

population = st.slider('Population: Total number of people residing within a block', 
                       min_value=int(dados['population'].min()), 
                       max_value=int(dados['population'].max()), 
                       value=int(dados['population'].mean()))

households = st.slider('Households: Total number of households, a group of people residing within a home unit, for a block', 
                       min_value=int(dados['households'].min()), 
                       max_value=int(dados['households'].max()), 
                       value=int(dados['households'].mean()))

median_income = st.slider('Median income: Median income for households within a block of houses (measured in tens of thousands of US Dollars)', 
                          min_value=float(dados['median_income'].min()), 
                          max_value=float(dados['median_income'].max()), 
                          value=float(dados['median_income'].mean()), step = 0.0001)

ocean_proximity = st.selectbox('Ocean proximity: Location of the house w.r.t ocean', 
                               ('1H_OCEAN', 'INLAND', 'ISLAND', 'NEAR_BAY', 'NEAR_OCEAN'))

rooms_per_household = total_rooms/households

bedrooms_per_room = total_bedrooms/total_rooms

population_per_household = population/households

dictmap = {
            'longitude': longitude, 
            'latitude': latitude
}

maps = pd.DataFrame(dictmap, index=[0])

st.map(maps)
                               
features = {
            'longitude': longitude, 
            'latitude': latitude,
            'housing_median_age': housing_median_age,
            'total_rooms': total_rooms,
            'total_bedrooms': total_bedrooms,
            'population': population,
            'households': households,
            'median_income': median_income,
            'ocean_proximity': ocean_proximity,
            'rooms_per_household': rooms_per_household,
            'bedrooms_per_room': bedrooms_per_room,
            'population_per_household': population_per_household
}                               
    
botao = st.button('Predict median house value')

if botao:                         
    pipe = joblib.load('pipe.joblib')
    features = pd.DataFrame(features, index=[0])
    features_prepared = pipe.transform(features) #pd.DataFrame(pipe.transform(features), columns=pipe.get_feature_names_out())
    modelo = joblib.load('housing.joblib')
    Valor_medio_casa = modelo.predict(features_prepared)
    f'The median house value within this block is $ {Valor_medio_casa[0]:.2f}' # como a saida é uma array, 


# In[8]:


#!streamlit run Deploy_housing.py


# In[66]:


# dados = pd.read_csv('data_train.csv')

# longitude = -122.23

# latitude = 37.88

# housing_median_age = 41

# total_rooms = 880

# total_bedrooms = 129

# population = 322

# households = 126

# median_income = 8.3252

# ocean_proximity = 'NEAR_BAY'

# rooms_per_household = total_rooms/households

# bedrooms_per_room = total_bedrooms/total_rooms

# population_per_household = population/households
                               
# features = {
#             'longitude': longitude, 
#             'latitude': latitude,
#             'housing_median_age': housing_median_age,
#             'total_rooms': total_rooms,
#             'total_bedrooms': total_bedrooms,
#             'population': population,
#             'households': households,
#             'median_income': median_income,
#             'ocean_proximity': ocean_proximity,
#             'rooms_per_household': rooms_per_household,
#             'bedrooms_per_room': bedrooms_per_room,
#             'population_per_household': population_per_household
# }                               


                        
# pipe = joblib.load('pipe.joblib')
# features = pd.DataFrame(features, index=[0])
# features_prepared = pipe.transform(features) #pd.DataFrame(pipe.transform(features), columns=pipe.get_feature_names_out())
# modelo = joblib.load('housing.joblib')
# Valor_medio_casa = modelo.predict(features_prepared)
# f'O Valor médio da casa é $ {Valor_medio_casa[0]:.2f}' # como a saida é uma array, 
#                                 #selecionando 0 garante que vai aparecer apenas o valor que queremos

