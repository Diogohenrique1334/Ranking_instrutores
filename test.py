import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import streamlit as st
from streamlit_echarts import st_echarts


colunas = ['Instrutor','Turmas','Dias em treinamento','Treinamentos por dia','Treinados','horas em treinamento por dia',
           'Horas por turma','Horas em treinamento','Moderação']

df = pd.read_csv('dados_instrutores.csv',sep=';',usecols=colunas)

scaler = MinMaxScaler()

st.set_page_config(
    page_title="Notas dos instrutores",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!",
    }
)

df_normalizado = pd.DataFrame(df['Instrutor'])
for col in colunas[1:]:

    temp =  pd.DataFrame({col:[x for x in scaler.fit_transform(df[col].values.reshape(-1,1)).flatten()]})
    
    df_normalizado[col] = temp

pesos = {
    'Turmas':[1],
    'Dias em treinamento':[1.5],
    'Treinamentos por dia':[1.5],
    'Treinados':[1],
    'horas em treinamento por dia':[2],
    'Horas por turma':[1.5],
    'Horas em treinamento':[1.5],
    'Moderação':[1]
    }
    

df_pesos = pd.DataFrame( pesos
)


with st.form('Pesoso'):
    st.subheader('Faça ajustes nos pesos')
       
    turmas = st.number_input('Turmas',1.0,step=0.5)
    dias_em_treinamento = st.number_input('Dias em treinamento',1.0,step=0.5)
    treinamentos_por_dia = st.number_input ('Treinamentos por dia',1.0,step=0.5)
    treinados = st.number_input ('Treinados',1.0,step=0.5)
    horas_em_treinamento_por_dia = st.number_input ('horas em treinamento por dia',1.0,step=0.5)
    horas_em_treinamento = st.number_input ('Horas em treinamento',1.0,step=0.5)
    horas_por_turma = st.number_input ('Horas por turma',1.0,step=0.5)
    moderacao = st.number_input ('Moderação',1.0,step=0.5)

    
        

    submit_button = st.form_submit_button('Atualizar pesos')

    if submit_button:
        df_pesos['Turmas'] = [turmas]
        df_pesos['Dias em treinamento'] = [dias_em_treinamento]
        df_pesos['Treinamentos por dia'] = [treinamentos_por_dia]
        df_pesos['Treinados'] = [treinados]
        df_pesos['horas em treinamento por dia'] = [horas_em_treinamento_por_dia]
        df_pesos['Horas por turma'] = [horas_por_turma]
        df_pesos['Horas em treinamento'] = [horas_em_treinamento]
        df_pesos['Moderação'] = [moderacao]
        df_pesos['Total'] = df_pesos.apply(lambda x: x.sum(), axis=1 )

        st.write('Pesos atualizados:')
        st.write(df_pesos)


df_final = df_normalizado
for col in colunas[1:]:    
    df_final[col] = df_final[col] * df_pesos[col].values    
    


df_final['Total'] = df_final[df_final.columns[1:]].apply(lambda x: x.sum(), axis=1 )



st.dataframe(
    df_final.sort_values(by='Total',ascending=False).set_index('Instrutor').reset_index()
)


