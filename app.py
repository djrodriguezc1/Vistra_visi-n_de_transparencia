import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from pyspark.sql.functions import col
from pyspark.sql.types import StructType,StructField, StringType, IntegerType
import plotly.express as px
from PIL import Image
import streamlit as st
from pyspark.sql import Row
from pyspark.sql import SparkSession
from pyspark.ml.pipeline import PipelineModel
from pyspark.ml.classification import DecisionTreeClassificationModel
spark = SparkSession.builder.getOrCreate()


#Funcion para desplegar el modelo y el pipeline

def models():
    feature_pipe = PipelineModel.load('feature_pipe')
    model = DecisionTreeClassificationModel.load('model_dct')
    return feature_pipe,model
feature_pipe,model=models()

# FUnción para la predicción de los contratos que pueden o no superar el 50% del valor del contrato
def predict(df):
    pred = feature_pipe.transform(df)
    pred2=model.transform(pred)
    predic_1=pred2.select("prediction").collect()[0][0]
    if predic_1== 0:
        t1=st.success("Probabilidades bajas a moderadas de tener adiciones futuras en la cuantía del contrato mayores al 50%")
        return t1
    else:
        t2=st.error("Probabilidades altas de tener adiciones futuras en la cuantía del contrato mayores al 50%")
        return t2
# Archivo en csv para usarlo en el filtrado de municipios por Departamento y entidades por Municipio
entidades=pd.read_csv("entidades2.csv",sep='|')
def base():
    entidades=pd.read_csv("entidades2.csv",sep='|')
    return entidades
data_2=base()
# Dccionario para realizar la seleccione de opciones para cada variable
def dic():
    new_dict = np.load('dic.npy',allow_pickle='TRUE').item()
    return new_dict
dict=dic()
# FUnción principal para el despliegue de la applicación
def main():

    img = Image.open("vistra.jpg")

    # Encabezados y texto explicativo de la herramienta predictiva
    # width is used to set the width of an image
    st.image(img, width=200)
    st.title("Predicción Adiciones en la Cuantia de los Contratos de Contratación Pública")
    st.markdown('El objetivo del modelo predictivo  es realizar una predicción acerca de si un nuevo contrato superará el 50% del valor en adiciones con respecto a su valor inicial, incumpliendo el párrafo 2 del parágrafo único del artículo 40 de la ley 80 de 1993, para esto se tienen en cuenta las siguientes 12 características del contrato:')
    st.markdown('**Tipo de Proceso,Objeto a Contratar,Tipo de Contrato,Nombre de la Entidad,Nombre Grupo,Nombre Clase,Plazo de Ejecución del Contrato,Rango de Ejecución del Contrato,Cuantiá del Contrato,Origen de los Recursos,Municipio de la Entidad,Departamento de la Entidad**')
    st.markdown('Si el contrato tiene probabilidades bajas a moderadas de incumplir la norma el resultado es un letrero en letras verdes con el siguiente mensaje:')
    st.success("Probabilidades bajas a moderadas de tener adiciones futuras en la cuantiá del contrato mayores al 50%")
    st.markdown('Si el contrato tiene altas probabilidades de incumplir la norma el resultado es un letrero en letras rojas con el siguiente mensaje:')
    st.error('Probabilidades altas de tener adiciones futuras en la cuantía del contrato mayores al 50%')
    st.subheader('**A continuación use el cursor o escriba las 12 características solicitadas del contrato que quiere evaluar.**')

    # Despliegiçue de las variables
    tipo_proceso = st.selectbox(
    'Tipo de Proceso',dict["tipo_proceso"])
    tipo_proceso=str(tipo_proceso)
    st.write('Seleccionaste:', tipo_proceso)
    objeto_contratar = st.selectbox(
    'Objeto a Contratar',dict['objeto_contratar'])
    objeto_contratar=str(objeto_contratar)
    st.write('Seleccionaste:', objeto_contratar)
    tipo_contrato = st.selectbox(
    'Tipo de Contrato',dict['tipo_contrato'])
    tipo_contrato=str(tipo_contrato)
    st.write('Seleccionaste:', tipo_contrato)
    nombre_grupo = st.selectbox(
    'Nombre Grupo',dict['nombre_grupo'])
    nombre_grupo=str(nombre_grupo)
    st.write('Seleccionaste:', nombre_grupo)
    nombre_clase = st.selectbox(
    'Nombre Clase',dict['nombre_clase'])
    nombre_clase=str(nombre_clase)
    st.write('Seleccionaste:', nombre_clase)
    departamento = st.selectbox(
    'Departamento Entidad',dict['departamento'])
    departamento=str(departamento)
    st.write('Seleccionaste:', departamento)
    plazo = st.number_input('Plazo de Ejecución del Contrato(inserte valores enteros mayores o iguales a 1)',1,1200,1)
    st.write('El plazo es de:', plazo)

    rango = st.radio("Rango de Ejecución en meses(M),días(D),no sabe(N): ", ('M', 'D','N'))

    if (rango == 'M'):
        st.success("Meses")
    elif(rango == 'D'):
        st.success("Días")
    else:
        st.success("No sabe")
    st.write('Seleccionaste:', rango)
    cuantia = st.number_input('Cuantia del Contrato(inserte valores enteros mayores a 1 millón)',1000000,100000000000000,1000000)
    st.write('La cuantia del contrato en millones es:', cuantia/1000000)
    origen_recursos = st.selectbox(
    'Origen de los Recursos',dict['origen_recursos'])
    origen_recursos=str(origen_recursos)
    st.write('Seleccionaste:', origen_recursos)
    enti=entidades[entidades['Departamento Entidad']==departamento]
    l=enti['Municipio Entidad'].unique()
    municipio = st.selectbox(
    'Municipio Entidad',l)
    municipio=str(municipio)
    st.write('Seleccionaste:', municipio)
    mune=entidades[entidades['Municipio Entidad']==municipio]
    lent=mune['Nombre de la Entidad'].unique()
    nombre_entidad = st.selectbox(
    'Nombre de la Entidad',lent)
    nombre_entidad=str(nombre_entidad)
    st.write('Seleccionaste:', nombre_entidad)

    c = Row('Tipo de Proceso','Objeto a Contratar','Tipo de Contrato','Nombre de la Entidad','Nombre Grupo','Nombre Clase','Plazo de Ejec del Contrato','Rango de Ejec del Contrato','Cuantia Contrato','Origen de los Recursos','Municipio Entidad','Departamento Entidad')
    d = c(tipo_proceso,objeto_contratar,tipo_contrato,nombre_entidad,nombre_grupo,nombre_clase,plazo,rango,cuantia,origen_recursos,municipio,departamento)
    cd = [d]
    df2=spark.createDataFrame(cd)


    Adiciones_cuantia_contrato=''

    if st.button('Predicción Adiciones al contrato'):
        Adiciones_cuantia_contrato= predict(df2)
    print(Adiciones_cuantia_contrato)
if __name__=='__main__':
    main()
