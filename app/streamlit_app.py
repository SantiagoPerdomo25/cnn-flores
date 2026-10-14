"""Aplicación Streamlit: sube una imagen y obtén la clase predicha con su confianza."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st
from PIL import Image

from src.predict import load_artifacts, predict_image

st.set_page_config(page_title="Clasificador de flores", page_icon="🌼")


@st.cache_resource
def get_model():
    return load_artifacts()


st.title("🌼 Clasificador de flores con CNN")
st.write("Sube una imagen de una flor (daisy, dandelion, roses, sunflowers o tulips).")

try:
    model, class_names = get_model()
except Exception as e:
    st.error("No se pudo cargar el modelo. Ejecuta primero `python -m src.train` "
             "o copia `cnn_flores.keras` y `class_names.json` a la carpeta `models/`.")
    st.exception(e)
    st.stop()

file = st.file_uploader("Imagen", type=["jpg", "jpeg", "png"])
if file:
    image = Image.open(file)
    st.image(image, caption="Imagen cargada", use_container_width=True)
    with st.spinner("Clasificando..."):
        res = predict_image(model, image, class_names)
    st.subheader(f"Predicción: **{res['class']}**")
    st.metric("Confianza", f"{res['confidence']:.2%}")
    st.progress(res["confidence"])
    if res["confidence"] < 0.6:
        st.warning("Confianza baja: la imagen podría no pertenecer a ninguna de las clases.")
    df = pd.DataFrame({"Clase": list(res["probabilities"]),
                       "Probabilidad": list(res["probabilities"].values())}).set_index("Clase")
    st.bar_chart(df)
