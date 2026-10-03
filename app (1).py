import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("Predicción de Aprobación de Curso")
st.write("Esta aplicación procesa las variables de entrada y realiza predicciones utilizando un modelo de Bagging pre-entrenado.")

# Selector en la barra lateral para el tipo de entrada
opcion_entrada = st.sidebar.selectbox(
    "Selecciona el método de entrada:",
    ["Predicción Individual (Manual)", "Predicción Masiva (Subir Archivo Excel)"]
)

# Cargar transformadores y modelo de forma global para optimizar
try:
    one_hot_transformer = joblib.load('one_hot_columns.joblib')
    scaler = joblib.load('min_max_scaler.joblib')
    model = joblib.load('bagging_optimizado.joblib')
    
    if isinstance(one_hot_transformer, list):
        si_columnas_one_hot = [col for col in one_hot_transformer if 'Felder_' in col]
    else:
        si_columnas_one_hot = []
except Exception as e:
    st.error(f"Error al cargar los archivos del modelo o transformadores: {e}")
    st.stop()

def procesar_y_predecir(df_input):
    """Función para procesar los datos de entrada y generar predicciones."""
    df_procesado = df_input.copy()
    
    # 1. Eliminar variables innecesarias si están presentes
    columnas_a_eliminar = ['ID', 'Año - Semestre', 'Nota_final', 'Aprobo']
    df_procesado = df_procesado.drop(columns=[col for col in columnas_a_eliminar if col in df_procesado.columns], errors='ignore')
    
    # 2. Aplicar One-Hot Encoding para la variable 'Felder'
    if 'Felder' in df_procesado.columns:
        if isinstance(one_hot_transformer, list):
            for col_name in si_columnas_one_hot:
                valor_esperado = col_name.replace('Felder_', '')
                df_procesado[col_name] = (df_procesado['Felder'] == valor_esperado).astype(float)
        else:
            # Fallback en caso de que sea un transformador de sklearn u otro objeto
            df_encoded = pd.get_dummies(df_procesado[['Felder']])
            df_procesado = pd.concat([df_procesado, df_encoded], axis=1)
            
        df_procesado = df_procesado.drop(columns=['Felder'], errors='ignore')
    
    # Asegurar que existan todas las columnas que el modelo espera
    if isinstance(one_hot_transformer, list):
        for col in si_columnas_one_hot:
            if col not in df_procesado.columns:
                df_procesado[col] = 0.0
                
    # 3. Normalizar la variable 'Examen_admisión'
    columna_examen = 'Examen_admisión' if 'Examen_admisión' in df_procesado.columns else ('Examen_admision' if 'Examen_admision' in df_procesado.columns else None)
    if columna_examen:
        df_procesado['Examen_admision_scaled'] = scaler.transform(df_procesado[[columna_examen]])
        df_procesado = df_procesado.drop(columns=[columna_examen], errors='ignore')
    
    # Reordenar las columnas conforme lo espera el modelo
    columnas_ordenadas = si_columnas_one_hot + ['Examen_admision_scaled']
    df_procesado = df_procesado[columnas_ordenadas]
    
    # 4. Realizar predicción
    predicciones = model.predict(df_procesado)
    return df_procesado, predicciones

if opcion_entrada == "Predicción Individual (Manual)":
    st.header("Datos de Entrada Manual")
    
    # Opciones para la variable Felder
    opciones_felder = ['sensorial', 'activo', 'visual', 'equilibrio', 'secuencial', 'reflexivo', 'verbal', 'intuitivo']
    
    felder_input = st.selectbox("Selecciona el estilo de aprendizaje (Felder):", opciones_felder)
    examen_input = st.number_input("Examen de Admisión:", min_value=0.0, max_value=5.0, value=3.83, step=0.01)
    
    if st.button("Realizar Predicción"):
        try:
            df_input = pd.DataFrame({
                'Felder': [felder_input],
                'Examen_admisión': [examen_input]
            })
            
            df_proc, preds = procesar_y_predecir(df_input)
            
            st.subheader("Datos Procesados para el Modelo")
            st.dataframe(df_proc)
            
            st.success(f"La predicción del modelo (Nota Final Estimada) es: {preds[0]:.4f}")
        except Exception as e:
            st.error(f"Ocurrió un error durante el procesamiento o la predicción: {e}")

elif opcion_entrada == "Predicción Masiva (Subir Archivo Excel)":
    st.header("Predicción Masiva")
    st.write("Sube un archivo Excel (.xlsx) que contenga al menos las columnas `Felder` y `Examen_admisión` (o `Examen_admision`).")
    
    uploaded_file = st.file_uploader("Selecciona un archivo Excel", type=["xlsx"])
    
    if uploaded_file is not None:
        try:
            df_uploaded = pd.read_excel(uploaded_file)
            st.subheader("Vista previa de los datos subidos")
            st.dataframe(df_uploaded.head())
            
            if st.button("Procesar y Predecir Archivo"):
                with st.spinner("Procesando registros..."):
                    df_proc, preds = procesar_y_predecir(df_uploaded)
                    
                    # Crear dataframe con los resultados para mostrar al usuario
                    df_resultados = df_uploaded.copy()
                    df_resultados['Nota_Final_Predicha'] = preds
                    
                    st.subheader("Resultados de las Predicciones")
                    st.dataframe(df_resultados)
                    
                    # Permitir descargar el resultado
                    towrite = io.BytesIO() if 'io' in globals() else __import__('io').BytesIO()
                    df_resultados.to_excel(towrite, index=False, header=True)
                    towrite.seek(0)
                    st.download_button(
                        label="Descargar Excel con Predicciones",
                        data=towrite,
                        file_name="predicciones_resultados.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")
