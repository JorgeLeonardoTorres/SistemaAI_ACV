# *** --------------------------------- *** *** ------------------------------------- *** 
#                               CÓDIGO.: STREAMLIT_ACV_UI
#                                                               Elaborado por.: Jorge Leonardo Torres Arévalo
#                                                               Magíster en Inteligencia Artificial
#                                                               Universidad Internacional de La Rioja - UNIR
#                                                               Trabajo Fin de Estudio - TFE

import sys
import os

# Obtener el directorio actual del script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Definir la raíz del proyecto
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

# Verificar si la ruta ya está en sys.path
if project_root not in sys.path:
    sys.path.append(project_root)

# Importar BERTMultiLabelClassifier con ruta corregida
from src.utils.BERTMultiLabelClassifier import BERTMultiLabelClassifier


import os
os.system("pip install Pillow")

# Importación de librerías clave
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import torch, pydicom, random, time, base64, os, sys, re
from PIL import Image
from datetime import datetime
from transformers import BertTokenizer
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from huggingface_hub import hf_hub_download

# Definir la ruta del proyecto
project_root = "/Users/leotorres/Desktop/Modulos_Software_TFE/SistemaAI_ACV/src"

# Agregar 'src' al sys.path
if project_root not in sys.path:
    sys.path.append(project_root)

# Importar BERTMultiLabelClassifier correctamente
from src.utils.BERTMultiLabelClassifier import BERTMultiLabelClassifier

# *** --------------------------------- *** *** ------------------------------------- *** 

# Módulo 1.: Configuración inicial
device = torch.device("cpu")

# *** --------------------------------- *** *** ------------------------------------- *** 

# Módulo 2.: Manejo del estado de sesión
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "menu_option" not in st.session_state:
    st.session_state.menu_option = "Inicio"
if "records" not in st.session_state:
    st.session_state.records = []
if "zoom_level" not in st.session_state:
    st.session_state.zoom_level = 1.0
if "name" not in st.session_state:
    st.session_state.name = "Usuario"
if "role" not in st.session_state:
    st.session_state.role = "Profesional"

# *** --------------------------------- *** *** ------------------------------------- *** 

# Módulo 2.1.: Inicialización de Variables de Sesión

# Inicialización del estado de sesión si no existe
if "clinical_text" not in st.session_state:
    st.session_state.clinical_text = "No disponible"
if "symptoms" not in st.session_state:
    st.session_state.symptoms = {}
if "image" not in st.session_state:
    st.session_state.image = np.zeros((700, 700), dtype=np.uint8)  # Imagen vacía de referencia
if "diagnosis" not in st.session_state:
    st.session_state.diagnosis = "Sin diagnóstico"
if "cie_code" not in st.session_state:
    st.session_state.cie_code = "N/A"
if "confidence" not in st.session_state:
    st.session_state.confidence = 0.0
if "diagnosis_message" not in st.session_state:  
    st.session_state.diagnosis_message = ""

# *** --------------------------------- *** *** ------------------------------------- *** 

# Módulo 3.: Función para establecer el fondo
def set_background(menu_option):
    if menu_option == "Inicio":
        background_image = "/Users/leotorres/Desktop/Modulos_Software_TFE/SistemaAI_ACV/src/assets/fondo_sistema_acv.png"
    else:
        background_image = "/Users/leotorres/Desktop/Modulos_Software_TFE/SistemaAI_ACV/src/assets/fondo_sistema_acv1.png"

    with open(background_image, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()

    background_css = f"""
    <style>
    .stApp {{
        /* Fondo principal de la app con superposición */
        background: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)), 
                    url("data:image/png;base64,{encoded_image}");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
        color: #ECECEC; /* Texto claro por defecto */
        font-family: 'Roboto', sans-serif;
        padding-top: 50px;
        padding-bottom: 50px;
    }}

    h1, h2, h3 {{
        color: #FFFFFF;
        text-shadow: 2px 2px 6px black;
        margin: 20px 0;
    }}

    p, label, span, div {{
        color: #ECECEC !important;
    }}

    .stButton>button {{
        background-color: #1e3d59;
        color: #FFFFFF;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }}
    .stButton>button:hover {{
        background-color: #163447;
    }}

    .block-container {{
        background: rgba(0, 0, 0, 0.4);
        padding: 30px;
        border-radius: 20px;
        margin: 20px auto;
        max-width: 700px;
    }}

    /* Barra lateral */
    .sidebar .sidebar-content {{
        background: rgba(30, 61, 89, 0.9);
        padding: 15px;
        border-radius: 15px;
    }}

    /* Mensajes de alerta (info, success, warning, error) */
    div[data-baseweb="alert"] {{
        border-radius: 0.75rem;
        margin: 1rem 0;
        padding: 1rem;
        color: #ECECEC !important;
        background-color: rgba(255, 255, 255, 0.1) !important;
        border-left: 0.4rem solid #00c0f0;
    }}
    div[data-baseweb="alert"] p {{
        color: #ECECEC !important;
        margin: 0;
    }}
    div[data-baseweb="alert"][aria-label="info"] {{
        border-left-color: #17a2b8 !important;
    }}
    div[data-baseweb="alert"][aria-label="success"] {{
        border-left-color: #28a745 !important;
    }}
    div[data-baseweb="alert"][aria-label="warning"] {{
        border-left-color: #ffc107 !important;
    }}
    div[data-baseweb="alert"][aria-label="error"] {{
        border-left-color: #dc3545 !important;
    }}

    /* ====================== 
       Ajustes para formularios 
       ====================== */

    /* INPUTS y TEXTAREAS:
       Fondo gris-azulado oscuro, texto claro, borde sutil */
    .stTextInput input,
    .stTextArea textarea {{
        background-color: #2c3e50 !important;  /* Gris-azulado oscuro */
        color: #ecf0f1 !important;            /* Texto claro */
        border: 1px solid #5f7788 !important; /* Borde tenue */
    }}
    .stTextInput input::placeholder,
    .stTextArea textarea::placeholder {{
        color: #95a5a6 !important; /* Placeholder gris claro */
    }}

    /* SELECTBOX:
       Cuadro principal y menú desplegable en #2c3e50, texto claro */
    /* 1) Cuadro cuando el select está cerrado */
    .stSelectbox [data-baseweb="select"] > div {{
        background-color: #2c3e50 !important;
        color: #ecf0f1 !important;
        border: 1px solid #5f7788 !important;
    }}
    /* 2) Menú desplegable */
    .stSelectbox [data-baseweb="menu"] {{
        background-color: #2c3e50 !important;
        color: #ecf0f1 !important;
        border: 1px solid #5f7788 !important;
    }}
    /* 3) Opciones dentro del menú */
    .stSelectbox [data-baseweb="option"] {{
        background-color: #2c3e50 !important;
        color: #ecf0f1 !important;
    }}
    .stSelectbox [data-baseweb="option"]:hover {{
        background-color: #34495e !important; /* Un tono más claro */
        color: #ecf0f1 !important;
    }}

    /* Ajuste de label en inputs (opcional) */
    label {{
        color: #ECECEC !important;
    }}
    </style>
    """
    st.markdown(background_css, unsafe_allow_html=True)

# *** --------------------------------- *** *** ------------------------------------- *** 

# Módulo 4.: Carga de los modelos Faster R - CNN y BERT
@st.cache_resource
def load_models():
    """Carga los modelos de Faster R-CNN y BERT"""
    hemorrhagic_model_path = hf_hub_download(repo_id="JorgeLeonardo/Models_ACV", filename="hemorrhagic_best_complete_model.pth")
    ischaemic_model_path = hf_hub_download(repo_id="JorgeLeonardo/Models_ACV", filename="ischaemic_best_complete_model.pth")
    bert_model_path = hf_hub_download(repo_id="JorgeLeonardo/Models_ACV", filename="BERT_augmented_stroke_classifier_train.pth")
    
    hemorrhagic_model = torch.load(hemorrhagic_model_path, map_location=device)
    ischaemic_model = torch.load(ischaemic_model_path, map_location=device)

    num_labels = 36  # Número de etiquetas en el dataset
    bert_model = BERTMultiLabelClassifier("bert-base-multilingual-cased", num_labels)  # Modelo multi-label
    bert_model.load_state_dict(torch.load(bert_model_path, map_location=device))
    
    hemorrhagic_model.eval()
    ischaemic_model.eval()
    bert_model.eval()
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    
    return hemorrhagic_model, ischaemic_model, bert_model, tokenizer

hemorrhagic_model, ischaemic_model, bert_model, tokenizer = load_models()

# *** --------------------------------- *** *** ------------------------------------- *** 

# Módulo 5.: Transformaciones para cada tipo de ACV
transform_hemorrhagic = Compose([
    Resize((700, 700)),
    ToTensor(),
    Normalize(mean=[0.2079864740371704], std=[0.3230390250682831])
])

transform_ischaemic = Compose([
    Resize((700, 700)),
    ToTensor(),
    Normalize(mean=[0.21155431866645813], std=[0.32820257544517517])
])

# *** --------------------------------- *** *** ------------------------------------- *** 

# Módulo 6.: Funciones para clasificación de texto y extracción de entidades
# **Definición de Etiquetas**
# Definición de etiquetas según la categoría
no_acv_labels = {
    "se descarta ACV", "trauma craneoencefalico leve", "alerta", "consciente", "orientado",
    "taquicardia ventricular sostenida", "cefalea leve", "mareo leve", "laboratorios normales",
    "TAC craneal simple", "sin evidencias de lesiones específicas", "da de alta",
    "manejo analgesia", "recomendaciones de egreso"
}

si_acv_labels = {
    "trauma craneoencefalico severo", "alteracion neurologica", "perdida del estado de consciencia",
    "desorientacion", "cefalea intensa (severa)", "disartria", "hemiparesia", "taquipneico",
    "indices de saturacion limitrofes", "Glasgow bajo", "indice de masa corporal",
    "TAC", "Resonancia Magnetica craneal", "hemograma", "analisis de gases arteriales",
    "radiografias", "unidad de cuidados intensivos", "vigilancia neurologica",
    "valoracion", "reportes", "examenes", "laboratorios"
}

all_labels = [
    "se descarta ACV", "trauma craneoencefalico leve", "alerta", "consciente", "orientado",
    "taquicardia ventricular sostenida", "cefalea leve", "mareo leve", "laboratorios normales",
    "TAC craneal simple", "sin evidencias de lesiones específicas", "da de alta",
    "manejo analgesia", "recomendaciones de egreso",
    "trauma craneoencefalico severo", "alteracion neurologica", "perdida del estado de consciencia",
    "desorientacion", "cefalea intensa (severa)", "disartria", "hemiparesia", "taquipneico",
    "indices de saturacion limitrofes", "Glasgow bajo", "indice de masa corporal",
    "TAC", "Resonancia Magnetica craneal", "hemograma", "analisis de gases arteriales",
    "radiografias", "unidad de cuidados intensivos", "vigilancia neurologica",
    "valoracion", "reportes", "examenes", "laboratorios"
]
all_labels = list(no_acv_labels.union(si_acv_labels))
label_to_index = {label: idx for idx, label in enumerate(all_labels)}

# **Definición de Entidades Clínicas para NER**
# Lista de entidades para la extracción
clinical_entities = list(no_acv_labels.union(si_acv_labels).union({
    "trauma craneoencefalico", "urgencias", "analgesia", "dolor", "masculino", "femenino", "examen fisico",
    "trauma craneoencefalico leve", "alerta", "consciente", "orientado", "taquicardia ventricular sostenida",
    "cefalea leve", "mareo leve", "da de alta", "recomendaciones de egreso", "desorientacion",
    "trauma craneoencefalico severo", "alteracion neurologica", "perdida del estado de consciencia", "desorientacion", 
    "cefalea intensa", "disartria", "hemiparesia", "taquipneico", "indices de saturacion limitrofes", 
    "Glasgow bajo", "hemograma", "unidad de cuidados intensivos", "valoracion", "taquicardia", "indices de saturacion",
    "indice de masa corporal", "TAC", "Resonancia Magnetica craneal", "hemograma", "analisis de gases arteriales",
    "radiografias", "vigilancia neurologica", "reportes", "examenes", "laboratorios", "localizacion"
}))

# Diccionario de sinónimos para ayudar en la extracción
synonym_dict = {
    "masculino": ["hombre", "varon", "varonil"],
    "femenino": ["mujer", "femina", "femenil"],
    "trauma craneoencefalico leve": ["trauma leve en la cabeza", "golpe leve en la cabeza", "impacto leve en la cabeza"],
    "alerta": ["atento", "atenta", "alertado", "alertada", "presta atencion"],
    "consciente": ["despierto", "despierta", "lucido", "lucida", "coherente", "con coherencia", "que atiende ordenes sencillas"],
    "orientado": ["lucido", "lucida", "ubicado", "ubicada"],
    "taquicardia ventricular sostenida": ["arritmia", "con latidos cardiacos irregulares", "taquicardia"],
    "cefalea leve": ["dolor de cabeza leve", "cefalea tensional", "leve dolor de cabeza"],
    "mareo leve": ["debilidad", "inestabilidad", "levemente aturdido", "levemente aturdida"],
    "da de alta": ["se le da salida", "dar salida", "dar de alta", "se le da de alta"],
    "recomendaciones de egreso": ["parametros de salida", "recomendaciones de salida", "indicaciones de egreso", "indicaciones de salida"], 
    "trauma craneoencefalico severo":["golpe que ocasiona perdida del conocimiento", "golpe muy fuerte en la cabeza", "trauma fuerte en la cabeza"],
    "alteracion neurologica": ["deterioro neurológico", "compromiso neurológico", "anomalía neurológica", "Neuropatia", "trastorno neurológico", "deficit neurologico", "trastorno neurologico funcional", "sindrome organico cerebral"],
    "perdida del estado de consciencia": ["inconsciencia", "desmayo", "desmayado", "desmayada", "pérdida de conciencia", "coma", "sincope", "con obnubilacion", "obnubilado", "obnubilada", "inconsciente", "con letargo", "con aletargamiento", "con sopor", "colapsado", "colapsada", "con colapso"],
    "desorientacion": ["desorientado", "desorientada", "despistado", "despistada", "desubicado", "desubicada", "perdido", "perdida", "extraviado", "extraviada", "confundido", "confundida"],
    "cefalea intensa": ["cefalea severa", "dolor de cabeza severo", "cefalea aguda", "migraña intensa", "cefalagia", "jaqueca", "hemicranea", "neuralgia", "dolor de cabeza fuerte", "fuerte dolor de cabeza", "dolor pulsatil de cabeza", "latente dolor de cabeza"],
    "disartria": ["problemas al hablar", "no puede hablar", "alteracion en la articulacion de las palabras", "roquera", "voz entrecortada", "babeo", "escaso control de la saliva", "dificultad para deglutir", "problemas al masticar", "dificultad al tragar", "problemas al tragar"], 
    "hemiparesia": ["paralisis parcial", "debilidad muscular unilateral", "debilidad de un lado del cuerpo", "hemiparesico", "hemiparesica"],
    "taquipneico": ["taquipnea", "taquipneica", "signos de taquipnea", "patrón respiratorio rapido", "respiracion rapida y superficial", "respiracion acelerada"],
    "indices de saturacion limitrofes": ["saturacion baja", "saturacion inestable", "saturacion fluctuante"],
    "Glasgow bajo": ["se debe controlar la presion intracraneal", "situacion grave", "posible coma", "traumatismo encefalico grave"],
    "hemograma": ["cuadro hematico", "recuento de celulas sanguineas completo", "recuento sanguineo"], 
    "unidad de cuidados intensivos": ["UCI", "unidad de cuidado intensivo", "uci"],
    "valoracion": ["valoracion medica", "diagnosis", "pronostico", "evaluacion", "evaluacion medica", "dictamen medico", "examen medico"],
    "taquicardia": ["arritmia", "con latidos cardiacos irregulares", "taquicardia ventricular sostenida"],
    "indices de saturacion limitrofes": ["nivel de oxigeno anormal", "nivel anormal de oxigeno en sangre", "hipoxemia", "hiperoxia"],
    "indice de masa corporal": ["IMC", "imc", "relacion masa corporal versus estatura"],
    "TAC": ["TAC de craneo", "tomografia computarizada de craneo", "tomografia computarizada de la cabeza", "tac"],
    "Resonancia Magnetica craneal": ["IRM", "irm", "rm de la cabeza", "RM de la cabeza"],
    "analisis de gases arteriales": ["gasometria arterial", "gases arteriovenosos", "ABG", "abg"],
    "radiografias": ["rayos x", "rayos X", "roentgenografia"],
    "vigilancia neurologica": ["exploracion neurologica", "monitorizacion neurologica"]
}

# Función de extracción usando regex para buscar coincidencias exactas
def extract_clinical_entities(text, entities=clinical_entities, synonyms=synonym_dict):
    extracted_entities = set()
    text_lower = text.lower()
    for entity in entities:
        pattern = r'\b' + re.escape(entity.lower()) + r'\b'
        if re.search(pattern, text_lower):
            extracted_entities.add(entity)
        elif entity in synonyms:
            for syn in synonyms[entity]:
                pattern_syn = r'\b' + re.escape(syn.lower()) + r'\b'
                if re.search(pattern_syn, text_lower):
                    extracted_entities.add(entity)
                    break
    return list(extracted_entities)


# Función para determinar el diagnóstico en base a la asociación de entidades
def determine_diagnosis(entities, no_acv_labels, si_acv_labels):
    count_no_acv = sum(1 for ent in entities if ent in no_acv_labels)
    count_si_acv = sum(1 for ent in entities if ent in si_acv_labels)
    if count_no_acv > count_si_acv:
        diagnosis = "✅ Se descarta ACV."
    elif count_si_acv > count_no_acv:
        diagnosis = "⚠️ Paciente diagnosticado con ACV."
    else:
        diagnosis = "❌ No se pudo determinar un diagnóstico preciso."
    return diagnosis, count_no_acv, count_si_acv

# Función para el pipeline de extracción y asociación
def extract_entities_and_diagnose(text):
    detected_entities = extract_clinical_entities(text, entities=clinical_entities, synonyms=synonym_dict)
    diagnosis, count_no_acv, count_si_acv = determine_diagnosis(detected_entities, no_acv_labels, si_acv_labels)
    return detected_entities, diagnosis, count_no_acv, count_si_acv

# --- Módulo 7.: Diagnóstico clínico con BERT (utilizando la extracción y asociación de entidades) ---
def clinical_diagnosis():
    st.subheader(f"Bienvenid@ {st.session_state.name} - {st.session_state.role}")
    st.subheader("Diagnóstico Clínico con BERT")
    st.info("Ingrese el informe clínico del paciente:")
    
    clinical_text = st.text_area("Escriba aquí el informe clínico")
    
    if st.button("Evaluar con BERT"):
        detected_entities, diagnosis, count_no_acv, count_si_acv = extract_entities_and_diagnose(clinical_text)
        
        st.markdown(f"### {diagnosis}")
        st.info(f"Total en no ACV: {count_no_acv} | Total en si ACV: {count_si_acv}")
        st.write("**Entidades detectadas:**")
        st.write(", ".join(detected_entities) if detected_entities else "Ninguna detectada")
        # Puedes guardar en session_state el texto o las entidades si es necesario para la siguiente fase
        st.session_state.clinical_text = clinical_text
        st.session_state.symptoms = {ent: True for ent in detected_entities}

        # Si el paciente no tiene ACV
        if diagnosis == "✅ Se descarta ACV.":
            time.sleep(2)
            st.info("El paciente será dado de alta con recomendaciones. El paciente se dará de alta con manejo de analgesia para el dolor y se le brindarán indicaciones de egreso.")
            time.sleep(5)
            st.info("Gracias por utilizar el sistema de diagnóstico avanzado de ACV con IA... hasta pronto...!!!")
            time.sleep(5)
            # Borrar toda la sesión para reiniciar la app como si fuera la primera vez
            keys_to_clear = list(st.session_state.keys())
            for key in keys_to_clear:
                del st.session_state[key]
            st.rerun()
            #st.session_state.menu_option = "Inicio"
            #st.rerun()
        # Si el paciente tiene ACV
        elif diagnosis == "⚠️ Paciente diagnosticado con ACV.":
            time.sleep(2)
            st.info("Se recomienda llevar a cabo exámenes de TAC / Resonancia Magnética craneal, laboratorios de ingreso y traslado a la UCI para manejo integral y vigilancia neurológica.")
            time.sleep(5)
            st.session_state.menu_option = "Análisis TAC Craneal Faster R-CNN"
            st.rerun()
# *** --------------------------------- *** *** ------------------------------------- *** 

# Módulo 8.: Función para clasificar imagen
def classify_image(image, hemorrhagic_model, ischaemic_model):
    image = Image.fromarray(image).convert("RGB")

    transformed_image_hemorrhagic = transform_hemorrhagic(image).unsqueeze(0).to(device)
    transformed_image_ischaemic = transform_ischaemic(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output_hemorrhagic = hemorrhagic_model(transformed_image_hemorrhagic)
        output_ischaemic = ischaemic_model(transformed_image_ischaemic)

    num_detections_hemorrhagic = len(output_hemorrhagic[0]['boxes'])
    num_detections_ischaemic = len(output_ischaemic[0]['boxes'])

    confidence_hemorrhagic = output_hemorrhagic[0]['scores'].mean().item() if output_hemorrhagic[0]['scores'].numel() > 0 else 0
    confidence_ischaemic = output_ischaemic[0]['scores'].mean().item() if output_ischaemic[0]['scores'].numel() > 0 else 0

    area_hemorrhagic = sum((box[2] - box[0]) * (box[3] - box[1]) for box in output_hemorrhagic[0]['boxes']) if num_detections_hemorrhagic > 0 else 0
    area_ischaemic = sum((box[2] - box[0]) * (box[3] - box[1]) for box in output_ischaemic[0]['boxes']) if num_detections_ischaemic > 0 else 0

    if (num_detections_hemorrhagic > num_detections_ischaemic and confidence_hemorrhagic > 0.4) or area_hemorrhagic > area_ischaemic:
        return "ACV Hemorrágico", "I61.9", confidence_hemorrhagic, area_hemorrhagic
    elif (num_detections_ischaemic > num_detections_hemorrhagic and confidence_ischaemic > 0.4) or area_ischaemic > area_hemorrhagic:
        return "ACV Isquémico", "I63.9", confidence_ischaemic, area_ischaemic

# *** --------------------------------- *** *** ------------------------------------- *** 

# Módulo 9.: Función para generar ID de registro
def generate_id():
    return f"{random.randint(0, 9999):04}"

def load_existing_records(filepath):
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    else:
        return pd.DataFrame(columns=[
            "ID REG", "Fecha (dd-mm-aaaa)", "Informe Clínico", "Síntomas",
            "Diagnóstico", "CIE-11", "Confianza", "Nombre del Doctor", "Rol"
        ])

def save_record(new_record, filepath):
    df = load_existing_records(filepath)
    df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
    df.to_csv(filepath, index=False)

# *** --------------------------------- *** *** ------------------------------------- *** 

# Módulo 10.: Interfaz de usuario
def main():
    st.title("Sistema AI Multimodal Avanzado para la detección, diagnóstico, tipología y tratamiento de ACV")
    # Se establece el fondo según el valor actual de menu_option (o "Inicio" por defecto)
    set_background(st.session_state.get("menu_option", "Inicio"))
    st.sidebar.header("Menú")
    
    # Si el usuario no está autenticado, solo mostramos la opción "Inicio"
    if not st.session_state.authenticated:
        menu_options = ["Inicio"]
    else:
        menu_options = [
            "Inicio", 
            "Diagnóstico Clínico BERT", 
            "Análisis TAC Craneal Faster R-CNN", 
            "Esquema completo Dx ACV", 
            "Fase control y tratamiento ACV"
        ]
    
    # Si el valor de st.session_state.menu_option no está en las opciones disponibles, lo forzamos a "Inicio"
    if st.session_state.menu_option not in menu_options:
        st.session_state.menu_option = "Inicio"
    
    # Radio lateral con las opciones disponibles
    menu_option = st.sidebar.radio("Seleccione una opción:", menu_options, 
                                   index=menu_options.index(st.session_state.menu_option))
    st.session_state.menu_option = menu_option

    st.markdown("""
    <div class='block-container'>
        <p>Bienvenid@ al Sistema Avanzado de Detección de ACV.</p>
    </div>
    <div style="position: fixed; bottom: 10px; right: 10px; text-align: right; color: #ECECEC; font-family: 'Roboto', sans-serif;">
        <p style="font-size: 1.2em; margin: 0; font-weight: bold;">Versión 7.0</p>
        <p style="font-size: 0.8em; margin: 0;">JLTA-DEV-UI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Si el usuario no está autenticado y se encuentra en "Inicio", mostramos el formulario de registro.
    if not st.session_state.authenticated:
        if menu_option == "Inicio":
            st.subheader("Bienvenid@ al Sistema AI Multimodal Avanzado para detección y diagnóstico de ACV")
            st.write("Ingrese su ID médico y seleccione su rol para continuar.")
    
            medical_id = st.text_input("Ingrese su ID (Formato: ReTHUS - XXXXX)")
            # Lista de roles con opción vacía al principio
            roles = ["", "Médico de Cuidados Intensivos", "Neurólog@", "Neurocirujan@",
                     "Médico en atención primaria", "Especialista en medicina de urgencias", "Fisiatra"]
            role = st.selectbox("Seleccione su rol:", roles)
            name = st.text_input("Nombre completo")
    
            if st.button("Iniciar sesión"):
                if role == "":
                    st.error("Debe seleccionar su rol de usuario...")
                elif medical_id.startswith("ReTHUS - ") and len(medical_id.split(" - ")[1]) == 5 and medical_id.split(" - ")[1].isdigit():
                    st.session_state.authenticated = True
                    st.session_state.name = name
                    st.session_state.role = role
                    st.session_state.menu_option = "Diagnóstico Clínico BERT"
                    st.rerun()
                else:
                    st.error("ID inválido. Asegúrese de que sigue el formato correcto.")
        else:
            # Si por alguna razón se selecciona otra opción sin autenticarse, forzamos la redirección a "Inicio"
            st.error("Acceso no permitido por fallo en datos de validación de ID y rol... Por favor ingresa con tus credenciales asociadas")
            st.session_state.menu_option = "Inicio"
            st.rerun()
    else:
        # Si el usuario ya está autenticado, se muestran todas las secciones normalmente.
        if menu_option == "Inicio":
            st.subheader("Inicio")
            st.write("Bienvenid@. Seleccione alguna de las otras opciones en el menú lateral para continuar.")
    
        elif menu_option == "Diagnóstico Clínico BERT":
            clinical_diagnosis()
    
        elif menu_option == "Análisis TAC Craneal Faster R-CNN":
            st.subheader("Análisis de imágenes TAC")
            st.write("📌 Sección de análisis de imágenes TAC")
            if st.session_state.diagnosis_message:
                st.warning(st.session_state.diagnosis_message)
                st.info("Practicado el TAC Craneal, debe aportarse la imagen obtenida para determinar la tipología del ACV")
    
            st.subheader("Cargue la imagen derivada del TAC (en formato DICOM)")
            uploaded_file = st.file_uploader("Cargar imagen DICOM", type=["dcm"])
            if uploaded_file is not None:
                dicom_data = pydicom.dcmread(uploaded_file)
                image = dicom_data.pixel_array
                st.image(image, caption="Imagen TAC Craneal", use_container_width=True, channels="GRAY")
                hemorrhagic_model, ischaemic_model, _, _ = load_models()
                diagnosis, cie_code, confidence, area = classify_image(image, hemorrhagic_model, ischaemic_model)
                st.session_state.image = image
                st.session_state.diagnosis = diagnosis
                st.session_state.cie_code = cie_code
                st.session_state.confidence = confidence
                st.session_state.menu_option = "Esquema completo Dx ACV"
                st.rerun()
    
        elif menu_option == "Esquema completo Dx ACV":
            st.subheader("Consolidado del diagnóstico de ACV aplicando modelos BERT y Faster R-CNN")
            st.write(f"**Informe Clínico:** {st.session_state.clinical_text}")
            st.write("**Síntomas Seleccionados:**")
            for symptom, value in st.session_state.symptoms.items():
                st.write(f"- {symptom}: {value}")
            if st.session_state.image.any():
                fig = px.imshow(st.session_state.image, color_continuous_scale='gray')
                fig.update_traces(hoverinfo='skip')
                fig.update_layout(
                    dragmode='zoom',
                    margin=dict(l=0, r=0, t=0, b=0),
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=False),
                )
                st.plotly_chart(fig, use_container_width=True)
            st.write(f"**Diagnóstico Oficial:** {st.session_state.diagnosis}")
            st.write(f"**Código CIE-11 Asociado:** {st.session_state.cie_code}")
            if st.session_state.cie_code == "I61.9":
                st.write("**Descripción:** Hemorragia intracerebral no traumática y no especificada.")
            elif st.session_state.cie_code == "I63.9":
                st.write("**Descripción:** Infarto cerebral, no especificado.")
            st.write(f"**Confianza del Diagnóstico:** {st.session_state.confidence * 100:.2f}%")
            if st.button("Generar Registro"):
                record_id = generate_id()
                date = datetime.now().strftime("%d-%m-%Y")
                new_record = {
                    "ID REG": record_id,
                    "Fecha (dd-mm-aaaa)": date,
                    "Informe Clínico": st.session_state.clinical_text,
                    "Síntomas": ', '.join(f"{k}: {v}" for k, v in st.session_state.symptoms.items()),
                    "Diagnóstico": st.session_state.diagnosis,
                    "CIE-11": st.session_state.cie_code,
                    "Confianza": f"{st.session_state.confidence * 100:.2f}%",
                    "Nombre del Doctor": st.session_state.name,
                    "Rol": st.session_state.role
                }
                save_record(new_record, "/Users/leotorres/Desktop/Modulos_Software_TFE/SistemaAI_ACV/results/5. Data_Register_IU/ACV_Reg_Data.csv")
                st.success(f"Registro generado con ID: {record_id}")
            
            if st.button("Ir a Fase control y tratamiento ACV"):
                st.session_state.menu_option = "Fase control y tratamiento ACV"
                st.rerun()
    
        elif menu_option == "Fase control y tratamiento ACV":
            if st.session_state.diagnosis == "ACV Hemorrágico":
                st.header("Control, Manejo, Tratamientos y Procedimientos Clínicos/Quirúrgicos para ACV Hemorrágico")
                st.markdown("""
                ### 🩺 **Control Inicial:**
                - **ABC de soporte vital:** Asegurar la vía aérea, respiración y circulación.
                - **Monitoreo intensivo:** Presión arterial (PA), frecuencia cardíaca, saturación de oxígeno, y estado neurológico.
                ### 💉 **Manejo de la Presión Arterial:**
                - **Objetivo:** Reducir PA sistólica <140 mmHg (en hemorragia intracerebral espontánea) si es seguro.
                - **Fármacos:** Labetalol, nicardipino o esmolol (IV).
                ### 🩸 **Control de la Coagulación:**
                - Reversión rápida de anticoagulación si aplica (vitamina K, plasma fresco congelado, complejo de protrombina).
                - Transfusión de plaquetas si hay trombocitopenia.
                ### 🧠 **Tratamiento Quirúrgico:**
                - **Craneotomía descompresiva:** Indicada en casos de hipertensión intracerebral refractaria con desplazamiento de la línea media o herniación inminente.
                - **Evacuación del hematoma:** Recomendado en:
                  - Hemorragia cerebelosa >3 cm con deterioro neurológico o compresión del tronco encefálico.
                  - Hematomas lobares superficiales con efecto de masa y deterioro neurológico progresivo.
                - **Drenaje ventricular externo:** Para hidrocefalia aguda obstructiva o hipertensión intracraneal refractaria.
                ### 💧 **Manejo del Edema Cerebral:**
                - Manitol o solución salina hipertónica para hipertensión intracraneal.
                - Considerar sedación, hiperventilación controlada y control estricto de la glucemia.
                ### 🚑 **Prevención de Complicaciones:**
                - Profilaxis de tromboembolismo venoso (heparina de bajo peso molecular después de 48 horas si no hay sangrado activo).
                - Control estricto de la glicemia (<180 mg/dL).
                """)
    
            elif st.session_state.diagnosis == "ACV Isquémico":
                st.header("Control, Manejo, Tratamientos y Procedimientos Clínicos/Quirúrgicos para ACV Isquémico")
                st.markdown("""
                ### 🩺 **Control Inicial:**
                - **ABC de soporte vital:** Asegurar estabilidad respiratoria y hemodinámica.
                - **Evaluación rápida:** Escala NIHSS, TAC/RM para descartar hemorragia.
                ### 💉 **Trombolisis Sistémica (si es candidato):**
                - **Alteplasa (rtPA):** 0.9 mg/kg (máx. 90 mg), 10% bolo IV y el resto en infusión durante 60 min.
                - **Ventana terapéutica:** Idealmente <4.5 horas desde el inicio de los síntomas.
                ### 🧠 **Trombectomía Mecánica:**
                - Indicada en oclusión de grandes vasos (arteria cerebral media, carótida interna, etc.).
                - **Ventana:** 0-6 horas (extendida hasta 24 horas en casos seleccionados con mismatch clínico-imagenológico).
                - **Técnica:** Uso de stent retriever o aspiración directa.
                ### 🩸 **Manejo de la Presión Arterial:**
                - **Antes de trombolisis:** PA <185/110 mmHg.
                - **Sin trombolisis:** No reducir PA a menos que sea >220/120 mmHg o haya otra indicación.
                ### 💊 **Antitrombóticos:**
                - **Aspirina:** Iniciar dentro de las 24-48 h si no recibió trombolisis.
                - **Clopidogrel:** En casos específicos (ictus menor o AIT de alto riesgo).
                ### 🔬 **Control Metabólico:**
                - **Glucosa:** 140-180 mg/dL.
                - **Temperatura:** Normotermia, evitar fiebre.
                ### 🧠 **Procedimientos Quirúrgicos:**
                - **Craneotomía descompresiva:** En infartos malignos con edema cerebral severo e hipertensión intracraneal.
                - **Angioplastia y colocación de stent:** En casos seleccionados de estenosis arterial intracraneal crítica.
                - **Endarterectomía carotídea:** Para pacientes con estenosis carotídea severa (>70%) sintomática, idealmente en las primeras dos semanas post-ictus.
                ### 🚀 **Rehabilitación Temprana:**
                - Movilización precoz si es seguro.
                - Terapia física, ocupacional y del habla.
                """)
    
            st.info("Disposiciones recomendadas por parte del sistema… susceptible a validación desde el criterio y decisión del profesional a cargo.")
    
            if st.button("Finalizar Sesión"):
                st.success("Gracias por su gestión. Que este sistema avanzado haya aportado en salvar la vida de los pacientes…")
                time.sleep(5)
                keys_to_clear = list(st.session_state.keys())
                for key in keys_to_clear:
                    del st.session_state[key]
                st.rerun()

if __name__ == "__main__":
    main()

# *** --------------------------------- *** *** ------------------------------------- *** 
