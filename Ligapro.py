#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


# ### Leemos el fichero

# In[70]:


Ligapro = pd.read_excel('PFM.xlsx')


# In[71]:


Ligapro.head(140)


# #### Mostrar los nombres de todas las columnas
# In[23]:
st.title("**Machine Learning**")

from PIL import Image
image = Image.open('Negra.png')
st.sidebar.image(image)

st.sidebar.title("Data Analytics en fútbol")

# #### Mostrar los nombres de todas las columnas
option = st.selectbox(
     'Selecciona tu Jugador',
     (Ligapro))

st.write(' ')
st.write(' ')     
st.write('Jugador de mayor similitud:')

# In[72]:


print(Ligapro.columns.values) # The names of all the columns in the data.


# In[86]:


selected_player = Ligapro[Ligapro["Jugadores"] == "A. Burbano"].iloc[0]


# ### Seleccionar columnas numéricas para computar la distancia euclidea
# Choose only the numeric columns (we use these to compute euclidean distance)

# In[87]:


distance_columns = ['Goles ', 'Asistencias ', 'Tiros ', 'Tiros a portería ',
 'Pases ', 'Pases efectivos ', 'Pases de finalización ',
 'Pases de finalización efectivos ', 'Centros ', 'Centros efectivos ',
 'Balones recuperados ', 'Recuperaciones en campo rival ',
 'Goles esperados ', 'Expected assists ', 'Disputas ganadas ',
 'Disputas defensivas ganadas ', 'Disputas en ataque ganadas ',
 'Disputas por arriba ganadas ', 'Regates efectivos ', 'Entradas efectivas ',
 'Interceptaciones ', 'Rechaces ']


# In[27]:
sorted = ['Goles ', 'Asistencias ', 'Tiros ', 'Tiros a portería ',
 'Pases ', 'Pases efectivos ', 'Pases de finalización ',
 'Pases de finalización efectivos ', 'Centros ', 'Centros efectivos ',
 'Balones recuperados ', 'Recuperaciones en campo rival ',
 'Goles esperados ', 'Expected assists ', 'Disputas ganadas ',
 'Disputas defensivas ganadas ', 'Disputas en ataque ganadas ',
 'Disputas por arriba ganadas ', 'Regates efectivos ', 'Entradas efectivas ',
 'Interceptaciones ', 'Rechaces ']

hmap_params = st.sidebar.multiselect("Select parameters", sorted, sorted)

# ### Creamos una funcion para realizar una distancia 'casera'

# In[100]:


def euclidean_distance(row):
    """
    A simple euclidean distance function
    """
    inner_value = 0
    for k in distance_columns:
        inner_value += (row[k] - selected_player[k]) ** 2
    return math.sqrt(inner_value)


# ### Find the distance from each player in the dataset to lebron.

# In[101]:


lebron_distance = Ligapro.apply(euclidean_distance, axis=1)


# #### Select only the numeric columns from the NBA dataset
ligapro_numeric = Ligapro[hmap_params]
# In[102]:


ligapro_numeric = Ligapro[distance_columns]


# In[103]:


ligapro_normalized = (ligapro_numeric - ligapro_numeric.mean()) / ligapro_numeric.std()


# In[104]:


from scipy.spatial import distance


# In[105]:


ligapro_normalized.fillna(0, inplace=True)


# In[119]:


vector_normalized = ligapro_normalized[Ligapro["Jugadores"] == option]


# In[120]:


euclidean_distances = ligapro_normalized.apply(lambda row: distance.euclidean(row, vector_normalized), axis=1)


# In[121]:


distance_frame = pd.DataFrame(data={"dist": euclidean_distances, "idx": euclidean_distances.index})
distance_frame.sort_values("dist", inplace=True)


# In[122]:


second_smallest = distance_frame.iloc[1]["idx"]
most_similar_to_pro = Ligapro.loc[int(second_smallest)]["Jugadores"]

most_similar_to_pro

######

st.write(' ')
distance_frame_full = distance_frame.merge(Ligapro, right_index=True, left_on='idx')
st.write(' ')

# In[123]:


distance_frame_full = distance_frame.merge(Ligapro, right_index=True, left_on='idx')
distance_frame_full.head(10)
#####

tabla = distance_frame_full.drop(['idx'], axis=1)
tabla.head(10)

# In[125]:
st.write('Top 25 similarity')
st.dataframe(tabla.head(25))

def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv().encode('utf-8')

csv = convert_df(tabla.head(25))

st.download_button(
     label="Descargar tu listado",
     data=csv,
     file_name='large_df.csv',
     mime='text/csv',
 )




####




# In[ ]:




