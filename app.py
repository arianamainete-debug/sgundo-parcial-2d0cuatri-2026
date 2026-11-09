# -*- coding: utf-8 -*-
import io

import numpy as np
import pandas as pd
import streamlit as st
from mercado import top_fci, top_mercado, CEDEARS, ACCIONES, SinDatos

st.set_page_config(page_title="Asesor Financiero IA", page_icon=None, layout="wide")

# ───────────────────────── Diseño (oscuro, minimalista, con transiciones) ─────────────────────────
ESTILO = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@200;300;400;500&family=Roboto:wght@300;400;500&display=swap');
html { scroll-behavior: smooth; }
.stApp { background: #0a0a0a; }
.stApp, .stApp p, .stApp li, .stApp label, .stApp span, .stApp div, .stApp td, .stApp th, .stApp caption {
    font-family: 'Roboto', sans-serif; color: #cfcfcf; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4 { font-family: 'Outfit', sans-serif; font-weight: 300; font-style: italic; color: #fafafa; letter-spacing: -0.01em; }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"], footer { visibility: hidden; }
.block-container { max-width: 1060px; padding-top: 4.5rem; padding-bottom: 5rem; }
hr { border-color: #1b1b1b !important; }

/* Entrada suave de cada bloque */
@keyframes subir { from { opacity: 0; transform: translateY(18px); } to { opacity: 1; transform: none; } }
div.element-container, [data-testid="stElementContainer"] { animation: subir .6s cubic-bezier(.2,.7,.2,1) both; }

/* La animación crea un contexto que rompe position:fixed: se desactiva en el bloque de la barra */
div.element-container:has(.nav), [data-testid="stElementContainer"]:has(.nav), [data-testid="stMarkdownContainer"]:has(.nav) { animation: none !important; transform: none !important; }
/* Barra de navegación flotante tipo píldora */
.nav { position: fixed; top: 14px; left: 50%; transform: translateX(-50%); z-index: 999999; display: flex; align-items: center; gap: 4px;
       padding: 8px 10px; background: rgba(10,10,10,.82); backdrop-filter: blur(14px); border: 1px solid #2a2a2a; border-radius: 999px;
       box-shadow: 0 10px 40px rgba(0,0,0,.5); max-width: 94vw; overflow-x: auto; }
.nav .logo { width: 40px; height: 40px; min-width: 40px; border-radius: 50%; border: 2px solid #5b7fa8; display: flex; align-items: center; justify-content: center;
             font-family: 'Outfit', sans-serif; font-style: italic; font-weight: 300; color: #fff !important; margin-right: 6px; }
.nav a { color: #8d8d8d !important; text-decoration: none !important; font-weight: 500; font-size: .92rem; padding: 9px 16px; border-radius: 999px;
         transition: all .25s ease; white-space: nowrap; }
.nav a:hover { color: #fff !important; background: #1a1a1a; }

/* Portada */
.hero { min-height: 78vh; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; padding: 3rem 1rem;
        margin: 1rem 0 2rem; border-radius: 28px;
        background: radial-gradient(900px 420px at 70% 10%, rgba(91,127,168,.28), transparent 60%), linear-gradient(180deg, #1c2128 0%, #0f1114 55%, #0a0a0a 100%); }
.hero h1 { font-size: clamp(2.6rem, 8vw, 5rem); line-height: 1.05; margin: 0 0 1rem; }
.hero .sub { font-size: 1.15rem; color: #c8c8c8; font-weight: 300; margin-bottom: .8rem; }
.hero .txt { max-width: 620px; color: #a8a8a8; font-weight: 300; line-height: 1.6; margin-bottom: 1.8rem; }
.btn { display: inline-block; padding: 14px 30px; border-radius: 999px; font-weight: 500; text-decoration: none !important; margin: 0 6px 8px;
       transition: transform .25s ease, box-shadow .25s ease; }
.btn:hover { transform: translateY(-3px); box-shadow: 0 10px 30px rgba(255,255,255,.12); }
.btn.claro { background: #f5f5f5; color: #0a0a0a !important; }
.btn.oscuro { background: #0b0b0b; color: #fff !important; border: 1px solid #333; }
.deslizar { margin-top: 3rem; letter-spacing: .3em; font-size: .72rem; color: #777; }
.deslizar::after { content: ""; display: block; width: 1px; height: 46px; background: linear-gradient(#777, transparent); margin: 12px auto 0; animation: caer 1.8s ease-in-out infinite; }
@keyframes caer { 0% { transform: scaleY(0); transform-origin: top; } 50% { transform: scaleY(1); transform-origin: top; } 100% { transform: scaleY(0); transform-origin: bottom; } }

/* Encabezados de sección */
.kicker { display: flex; align-items: center; gap: 14px; letter-spacing: .3em; font-size: .78rem; color: #8a8a8a; text-transform: uppercase; margin-top: 3.2rem; }
.kicker::before { content: ""; width: 56px; height: 1px; background: #3a3a3a; }
.titulo { font-family: 'Outfit', sans-serif; font-style: italic; font-weight: 300; font-size: clamp(2rem, 5vw, 3rem); color: #fff; margin: .3rem 0 .4rem; line-height: 1.1; }
.bajada { color: #8f8f8f; font-weight: 300; max-width: 640px; margin-bottom: 1.2rem; }

/* Controles (Streamlit nuevo: react-aria) */
[data-testid="stSelectbox"] [role="group"], [data-testid="stNumberInput"] [role="group"], [data-testid="stSelectbox"] input, [data-testid="stNumberInput"] input {
    background-color: #0b0b0b !important; color: #eaeaea !important; }
[data-testid="stSelectbox"] [role="group"], [data-testid="stNumberInput"] [role="group"] { border: 1px solid #262626 !important; border-radius: 14px !important; }
[data-testid="stNumberInput"] button { background-color: #141414 !important; color: #bbb !important; }
[role="listbox"], [role="listbox"] * { background-color: #121212 !important; color: #eaeaea !important; }
[role="option"]:hover, [role="option"]:hover * { background-color: #1d1d1d !important; }
[data-testid="stRadioOption"] { transition: color .2s ease; }
[data-baseweb="select"] > div, [data-baseweb="select"] div, [data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="input"] input { background-color: #0b0b0b !important; color: #eaeaea !important; }
[data-baseweb="select"] svg { fill: #8d8d8d; }
[data-baseweb="radio"] > div:first-child { border-color: #5b7fa8 !important; background-color: #0b0b0b !important; }
[data-baseweb="select"] > div, .stNumberInput input, .stTextInput input, .stTextArea textarea {
    background: #0b0b0b !important; border: 1px solid #262626 !important; border-radius: 14px !important; color: #eaeaea !important; }
.stNumberInput button { background: #141414 !important; border-color: #262626 !important; }
[data-baseweb="popover"] * { background-color: #121212; color: #eaeaea; }
.stRadio [role="radiogroup"] label { transition: color .2s ease; }
.stRadio [role="radiogroup"] label:hover { color: #fff; }
.stButton > button { border-radius: 999px; padding: .85rem 2rem; font-weight: 500; border: 1px solid #333; background: #0b0b0b; color: #fff;
                     transition: transform .25s ease, box-shadow .25s ease, background .25s ease; }
.stButton > button[kind="primary"] { background: #f5f5f5; color: #0a0a0a; border: none; }
.stButton > button[kind="primary"] p { color: #0a0a0a; }
.stButton > button:hover { transform: translateY(-3px); box-shadow: 0 12px 34px rgba(255,255,255,.14); }

/* Tarjetas, métricas, tablas, pestañas */
[data-testid="stMetric"] { background: #121212; border: 1px solid #1f1f1f; border-radius: 22px; padding: 18px 22px; transition: transform .3s ease, border-color .3s ease; }
[data-testid="stMetric"]:hover { transform: translateY(-4px); border-color: #3a3a3a; }
[data-testid="stMetricValue"] { font-family: 'Outfit', sans-serif; font-weight: 300; color: #fff; }
[data-testid="stTable"], [data-testid="stDataFrame"] { background: #101010; border: 1px solid #1f1f1f; border-radius: 20px; overflow: hidden; }
[data-testid="stTable"] th { background: #151515 !important; color: #9a9a9a !important; text-transform: uppercase; font-size: .72rem; letter-spacing: .08em; }
[data-testid="stTable"] td { border-color: #1b1b1b !important; }
[data-testid="stExpander"] { background: #101010; border: 1px solid #1f1f1f !important; border-radius: 18px; transition: border-color .3s ease; }
[data-testid="stExpander"]:hover { border-color: #3a3a3a !important; }
[data-testid="stAlert"] { border-radius: 18px; border: 1px solid #222; }
.stTabs [data-baseweb="tab-list"] { gap: 4px; background: #0e0e0e; border: 1px solid #262626; border-radius: 999px; padding: 6px; width: fit-content; max-width: 100%; overflow-x: auto; }
.stTabs [data-baseweb="tab"] { border-radius: 999px; padding: 8px 18px; color: #8d8d8d; transition: all .25s ease; height: auto; }
.stTabs [data-baseweb="tab"]:hover { color: #fff; background: #1a1a1a; }
.stTabs [aria-selected="true"] { background: #1a1a1a; color: #fff; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.6rem; }
</style>
"""


def seccion(anclaje, kicker, titulo, bajada=""):
    st.markdown(f'<div id="{anclaje}"></div><div class="kicker">{kicker}</div><div class="titulo">{titulo}</div>'
                + (f'<div class="bajada">{bajada}</div>' if bajada else ""), unsafe_allow_html=True)


st.markdown(ESTILO, unsafe_allow_html=True)

# Parámetros del modelo (regresión logística estandarizada) entrenados con asalariados de la EPH 1T-2026.
# Se incluyen en el código para que la app no dependa de archivos .pkl ni de versiones de scikit-learn.
VARS_MODELO = ["edad", "ingresos", "universitario", "secundario", "en_pareja"]
MEDIAS = [40.07629979743417, 1034429.3354715282, 0.4034436191762323, 0.4819941480981319, 0.5275714607247355]
ESCALAS = [11.774440003673455, 812031.9410510369, 0.4905882849418803, 0.49967568411649566, 0.4992392357913232]
COEFS = [-0.4567826539137116, -2.031904715007245, -0.5982311453044415, -0.2698069151545661, -0.12853803048411974]
INTERCEPTO = -1.092551368370014


class ModeloLogit:
    def predict_proba(self, X):
        z = INTERCEPTO + sum(c * (float(X[v].iloc[0]) - m) / e for v, c, m, e in zip(VARS_MODELO, COEFS, MEDIAS, ESCALAS))
        p = 1 / (1 + np.exp(-z))
        return np.array([[1 - p, p]])


REFERENCIA_CSV = """situacion,region,educ,n,p10,p20,p30,p40,p50,p60,p70,p80,p90
Asalariado/a,Cuyo,Primario o menos,138,164816.0,263358.0,360000.0,440000.0,600000.0,600000.0,800000.0,900000.0,1165767.0
Asalariado/a,Cuyo,Secundario,534,280000.0,400000.0,491086.0,600000.0,700000.0,800000.0,950000.0,1100000.0,1400000.0
Asalariado/a,Cuyo,Superior,411,440000.0,700000.0,800000.0,900000.0,1000000.0,1000000.0,1300000.0,1500000.0,2000000.0
Asalariado/a,Gran Buenos Aires,Primario o menos,201,248742.0,400000.0,600000.0,600000.0,800000.0,900000.0,1007121.0,1200000.0,1500000.0
Asalariado/a,Gran Buenos Aires,Secundario,679,300000.0,500000.0,600000.0,800000.0,900000.0,1000000.0,1200000.0,1400000.0,1800000.0
Asalariado/a,Gran Buenos Aires,Superior,686,520286.0,720857.0,900000.0,1000000.0,1200000.0,1500000.0,1800000.0,2001300.0,3000000.0
Asalariado/a,Noreste,Primario o menos,89,170000.0,200000.0,257675.0,400000.0,500000.0,700000.0,800000.0,1000000.0,1200000.0
Asalariado/a,Noreste,Secundario,487,200000.0,300000.0,400000.0,500000.0,600000.0,755186.0,880000.0,1000000.0,1200000.0
Asalariado/a,Noreste,Superior,355,366732.0,500000.0,620000.0,769376.0,850000.0,938312.0,1000000.0,1200000.0,1500000.0
Asalariado/a,Noroeste,Primario o menos,339,200000.0,300000.0,400000.0,469570.0,518371.0,600000.0,700000.0,843032.0,1076223.0
Asalariado/a,Noroeste,Secundario,1316,250000.0,350000.0,432863.0,550000.0,700000.0,800000.0,900000.0,1089713.0,1362591.0
Asalariado/a,Noroeste,Superior,1232,360000.0,550000.0,700000.0,800000.0,900000.0,1000000.0,1200000.0,1500000.0,1900000.0
Asalariado/a,Pampeana,Primario o menos,391,200000.0,300000.0,450000.0,649132.0,800000.0,850000.0,1000000.0,1200000.0,1500000.0
Asalariado/a,Pampeana,Secundario,1558,240000.0,400000.0,600000.0,750000.0,800000.0,1000000.0,1100000.0,1200000.0,1500000.0
Asalariado/a,Pampeana,Superior,1316,500000.0,700000.0,900000.0,1000000.0,1100000.0,1300000.0,1500000.0,1800000.0,2500000.0
Asalariado/a,Patagonia,Primario o menos,132,400000.0,600000.0,700000.0,806411.0,1000000.0,1200000.0,1300000.0,1500000.0,2000000.0
Asalariado/a,Patagonia,Secundario,735,500000.0,800000.0,1000000.0,1000000.0,1200000.0,1400000.0,1500000.0,1800000.0,2200000.0
Asalariado/a,Patagonia,Superior,509,600000.0,1000000.0,1200000.0,1400000.0,1680570.0,2000000.0,2000000.0,2700000.0,3500000.0
Cuenta propia,Cuyo,Primario o menos,49,80000.0,100000.0,200000.0,280000.0,300000.0,400000.0,450000.0,558153.0,800000.0
Cuenta propia,Cuyo,Secundario,194,150000.0,200000.0,300000.0,400000.0,450000.0,500000.0,600000.0,800000.0,1000000.0
Cuenta propia,Cuyo,Superior,95,150000.0,250000.0,400000.0,600000.0,800000.0,1000000.0,1200000.0,1500000.0,3000000.0
Cuenta propia,Gran Buenos Aires,Primario o menos,52,100000.0,200000.0,301174.0,400000.0,438548.0,644310.0,791544.0,931216.0,1068911.0
Cuenta propia,Gran Buenos Aires,Secundario,208,100000.0,200000.0,300000.0,500000.0,600000.0,800000.0,1000000.0,1200000.0,1935941.0
Cuenta propia,Gran Buenos Aires,Superior,132,200000.0,300000.0,500000.0,700000.0,1000000.0,1000000.0,1500000.0,2000000.0,3000000.0
Cuenta propia,Noreste,Primario o menos,59,138052.0,150000.0,227833.0,300000.0,300000.0,376383.0,500000.0,783214.0,824873.0
Cuenta propia,Noreste,Secundario,212,100000.0,150000.0,200000.0,300000.0,360000.0,400000.0,500000.0,700000.0,950946.0
Cuenta propia,Noreste,Superior,93,200000.0,300000.0,300000.0,400000.0,500000.0,700000.0,800000.0,1000000.0,1599800.0
Cuenta propia,Noroeste,Primario o menos,153,150000.0,244674.0,300000.0,300000.0,400000.0,400000.0,500000.0,702627.0,900000.0
Cuenta propia,Noroeste,Secundario,492,150000.0,200000.0,300000.0,350000.0,400000.0,500000.0,600000.0,800000.0,1000000.0
Cuenta propia,Noroeste,Superior,292,200000.0,250000.0,350000.0,400000.0,582505.0,700000.0,900000.0,1200000.0,1600000.0
Cuenta propia,Pampeana,Primario o menos,187,100000.0,200000.0,250000.0,317606.0,400000.0,600000.0,800000.0,1000000.0,1075441.0
Cuenta propia,Pampeana,Secundario,519,150000.0,250000.0,400000.0,500000.0,600000.0,800000.0,1000000.0,1200000.0,1500000.0
Cuenta propia,Pampeana,Superior,343,300000.0,400000.0,544582.0,769326.0,900000.0,1000000.0,1500000.0,1800000.0,2500000.0
Cuenta propia,Patagonia,Primario o menos,46,255167.0,500000.0,600000.0,600000.0,824697.0,966029.0,1000000.0,1000000.0,1244872.0
Cuenta propia,Patagonia,Secundario,155,200000.0,300000.0,400000.0,500000.0,700000.0,800000.0,1000000.0,1091756.0,1800000.0
Cuenta propia,Patagonia,Superior,67,200000.0,477558.0,500000.0,700000.0,900000.0,1280831.0,2000000.0,2370578.0,3000000.0
"""


modelo = ModeloLogit()
ref = pd.read_csv(io.StringIO(REFERENCIA_CSV))


# ───────────────────────── Datos de apoyo ─────────────────────────
# Provincia -> región EPH (la tabla de referencia de ingresos está armada por región EPH)
PROVINCIAS = {
    "Ciudad Autónoma de Buenos Aires": "Gran Buenos Aires",
    "Buenos Aires – Gran Buenos Aires (conurbano)": "Gran Buenos Aires",
    "Buenos Aires – interior de la provincia": "Pampeana",
    "Catamarca": "Noroeste", "Chaco": "Noreste", "Chubut": "Patagonia", "Córdoba": "Pampeana",
    "Corrientes": "Noreste", "Entre Ríos": "Pampeana", "Formosa": "Noreste", "Jujuy": "Noroeste",
    "La Pampa": "Pampeana", "La Rioja": "Noroeste", "Mendoza": "Cuyo", "Misiones": "Noreste",
    "Neuquén": "Patagonia", "Río Negro": "Patagonia", "Salta": "Noroeste", "San Juan": "Cuyo",
    "San Luis": "Cuyo", "Santa Cruz": "Patagonia", "Santa Fe": "Pampeana",
    "Santiago del Estero": "Noroeste", "Tierra del Fuego": "Patagonia", "Tucumán": "Noroeste",
}

# Nivel educativo (categorías de NIVEL_ED de la EPH) -> (universitario, secundario, grupo de la tabla de referencia)
EDUCACION = {
    "Primario incompleto o sin instrucción": (0, 0, "Primario o menos"),
    "Primario completo": (0, 0, "Primario o menos"),
    "Secundario incompleto": (0, 1, "Secundario"),
    "Secundario completo": (0, 1, "Secundario"),
    "Universitario / terciario incompleto": (1, 0, "Superior"),
    "Universitario / terciario completo": (1, 0, "Superior"),
}

TRAMOS_USD = {"Menos de USD 500": (300, 500), "Entre USD 500 y 1.000": (500, 1000), "Entre USD 1.000 y 2.000": (1000, 2000),
              "Entre USD 2.000 y 3.000": (2000, 3000), "Entre USD 3.000 y 5.000": (3000, 5000), "Más de USD 5.000": (5000, 7000)}
TC_REF = 1500        # referencia editable; el usuario puede cambiarla en la app
FECHA_REF = "23/10/2026"
TNA_MM_REF = 21.0   # mayor TNA de los money market del ranking de referencia (CAFCI al 23/10/2026)

FUENTES = {
    "emergencia": [("CAFCI (Cámara Argentina de FCI) – valores cuotaparte oficiales", "https://www.cafci.org.ar"),
                   ("CNV – regulador de fondos y ALyCs", "https://www.cnv.gob.ar"),
                   ("Balanz – fondos comunes (para contrastar el dato)", "https://balanz.com"),
                   ("Rankia – ranking de FCI con datos de CAFCI (referencia con fecha)",
                    "https://www.rankia.com.ar/blog/fondos-comunes-de-inversion/4142161-ranking-fondos-comunes-inversion-argentina-2020"),
                   ("ArgentinaDatos – API que republica los datos de CAFCI", "https://argentinadatos.com")],
    "cedears": [("Yahoo Finance – precios históricos", "https://finance.yahoo.com"),
                ("BYMA – mercado donde cotizan CEDEARs y acciones", "https://www.byma.com.ar"),
                ("CNV – regulador del mercado de capitales (plazo de liquidación T+1)", "https://www.cnv.gob.ar"),
                ("Boletín Oficial – CNV: plazo de liquidación T+1", "https://www.boletinoficial.gob.ar/pdf/aviso/primera/316812/20250505")],
}
FUENTES["renta"] = FUENTES["emergencia"]
FUENTES["acciones"] = FUENTES["cedears"]
FUENTES["deudas"] = FUENTES["emergencia"] + [("BCRA – Central de Deudores y información al usuario financiero", "https://www.bcra.gob.ar")]

EXPLICACIONES = {
    "emergencia": ("Fondo común de inversión Money Market",
                   "Un fondo común de inversión (FCI) de **mercado de dinero** junta la plata de muchos ahorristas y la coloca en "
                   "plazos fijos, cauciones y cuentas remuneradas de muy corto plazo. Podés retirar tu dinero **el mismo día (T+0)**. "
                   "Es lo más parecido a una caja de ahorro que rinde, pero su rendimiento **no está garantizado**."),
    "renta": ("Fondo común de inversión de Renta Fija",
              "Un **Fondo Común de Inversión de Renta Fija** es un producto de inversión colectiva en el que el dinero de varios ahorristas "
              "se agrupa para comprar activos de deuda (bonos, letras y obligaciones negociables). Es como prestarle plata a gobiernos y "
              "empresas entre muchos, con un administrador profesional. Suele rescatarse en **24 horas hábiles (T+1)**. Puede perder valor "
              "si suben las tasas o si un emisor no paga."),
    "cedears": ("CEDEAR",
                "Un **CEDEAR** es un certificado que representa acciones de empresas del exterior (por ejemplo Apple o Coca-Cola). "
                "Se compra y se vende en pesos en la bolsa argentina (BYMA), pero su precio sigue al de la acción en dólares. "
                "Se liquida en **24 horas (T+1)**. Tiene riesgo de mercado: puede subir o bajar."),
    "acciones": ("Acciones argentinas",
                 "Una **acción** es una parte del capital de una empresa: al comprarla sos socio/a. No tiene un rendimiento fijo, su precio "
                 "sube y baja todos los días y puede caer con fuerza. Se liquida en **24 horas (T+1)**. Solo conviene para plazos largos y "
                 "personas que toleran perder valor en el camino."),
}

GLOSARIO = (
    "- **TNA (Tasa Nominal Anual):** lo que rendiría la inversión en un año si la tasa se mantuviera igual, sin sumar intereses sobre intereses. "
    "Sirve para comparar fondos de pesos entre sí.\n"
    "- **Plazo T+0 / T+1:** días hábiles que tarda en acreditarse tu dinero. T+0 = el mismo día; T+1 = 24 horas hábiles.\n"
    "- **Obligación Negociable (ON):** bono que emite una empresa privada para pedir plata prestada; paga intereses.\n"
    "- **Bono CER:** bono del Estado cuyo capital se ajusta por inflación.\n"
    "- **Riesgo de crédito:** que quien emitió la deuda no pueda pagar.\n"
    "- **Volatilidad:** cuánto sube y baja el precio. Más volatilidad = más riesgo."
)


def fmt(x, dec=1):
    return f"{x:.{dec}f}".replace(".", ",")


def peso(x):
    # formato argentino: $1.000.000
    return "$" + f"{x:,.0f}".replace(",", ".")


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def ranking(clave, meses="1 mes"):
    # Top 3 con datos reales, medido en la ventana del horizonte elegido. Se cachea 6 horas para no consultar las APIs en cada clic.
    if clave == "emergencia":
        return top_fci("mercadoDinero", dias=HORIZONTES[meses])
    if clave == "renta":
        return top_fci("rentaFija", dias=HORIZONTES[meses])
    periodo = "1y" if meses == "1 año" else "6mo"      # en renta variable una ventana menor a 6 meses es puro ruido
    if clave == "cedears":
        return top_mercado(CEDEARS, periodo=periodo)
    return top_mercado(ACCIONES, periodo=periodo)


def tabla_referencia(clave):
    # Referencia con fuente y fecha (Rankia con datos de CAFCI, al cierre del 23/10/2026). Se usa solo si falla la consulta en vivo.
    if clave == "emergencia":
        datos = [("ADCAP AHORRO PESOS F. DINERO - Cl. C", 21.0), ("PELLEGRINI RENTA PESOS - Cl. B", 20.9),
                 ("LOMBARD RENTA EN PESOS (Patagonia)", 20.8)]
        return pd.DataFrame({
            "Puesto": [1, 2, 3], "Instrumento": [d[0] for d in datos], "TNA (%)": [d[1] for d in datos],
            "Plazo de rescate": ["T+0 (mismo día)"] * 3,
            "Por qué está en el top 3": [f"Mayor TNA ({fmt(d[1])}%) entre los money market del ranking de CAFCI al {FECHA_REF}; se rescata el mismo día." for d in datos]})
    if clave == "renta":
        datos = [("BALANZ CAPITAL AHORRO - Cl. A", 2.25, 37.62), ("1822 RAÍCES VALORES FIDUCIARIOS", 2.18, 36.94),
                 ("ALLARIA AHORRO PLUS - Cl. B", 2.14, 34.98)]
        tna = [round(d[1] * 365 / 30, 1) for d in datos]
        return pd.DataFrame({
            "Puesto": [1, 2, 3], "Instrumento": [d[0] for d in datos], "TNA (%)": tna,
            "Plazo de rescate": ["T+1 (24 h hábiles)"] * 3,
            "Por qué está en el top 3": [f"Rindió {fmt(d[1], 2)}% en 30 días (TNA estimada {fmt(t)}%) y {fmt(d[2], 2)}% en 12 meses, "
                                         f"los mayores del grupo T+1 en pesos del ranking de CAFCI al {FECHA_REF}." for d, t in zip(datos, tna)]})
    return None


def mostrar_fuentes(clave):
    with st.expander("Fuentes verificables de estos datos"):
        st.write("El ranking se calcula con estas fuentes públicas. Podés abrirlas y comprobar el dato vos mismo:")
        for nombre, url in FUENTES[clave]:
            st.markdown(f"- [{nombre}]({url})")
        if clave in ("emergencia", "renta", "deudas"):
            st.caption(f"Referencia de contraste con fecha: al {FECHA_REF}, los money market con mayor rendimiento rendían entre 19,0% y 21,0% de TNA "
                       "(cuotapartes CAFCI, según el ranking de Rankia). Plazos: money market T+0 (mismo día); renta fija en pesos T+1 (24 horas hábiles).")
        else:
            st.caption("Plazo de liquidación de acciones y CEDEARs en BYMA: T+1 (24 horas), según la normativa de la CNV.")


def mostrar_explicacion(clave):
    if clave in EXPLICACIONES:
        titulo, texto = EXPLICACIONES[clave]
        with st.expander(titulo):
            st.markdown(texto)
            st.markdown("**Palabras que aparecen en la tabla:**\n\n" + GLOSARIO)


def mostrar_top3(clave, categoria_fallback):
    # Muestra el top 3 real (ordenado por TNA o retorno/riesgo, con su plazo). Si no hay conexión, usa la referencia con fuente y fecha.
    try:
        with st.spinner("Consultando datos reales de mercado..."):
            df_top, meta = ranking(clave)
        st.write("**Top 3 según el sistema (ordenado de mejor a peor):**")
        st.table(df_top.set_index("Puesto"))
        st.caption(f"Fuente: {meta['fuente']} · datos al {meta['fecha']} · Criterio: {meta['criterio']}. "
                   "Rendimientos pasados no garantizan resultados futuros.")
        if clave in ("cedears", "acciones"):
            st.caption("En renta variable no existe una TNA fija: se muestra el retorno anualizado de los últimos 12 meses y la relación retorno/riesgo.")
    except Exception as e:
        st.warning("No se pudo consultar el mercado en vivo en este momento. Se muestra una referencia con fuente y fecha.")
        st.caption(f"Detalle técnico: {type(e).__name__}: {str(e)[:180]}")
        tabla = tabla_referencia(clave)
        if tabla is not None:
            st.write(f"**Top 3 de referencia (datos de CAFCI al {FECHA_REF}, ordenado por TNA):**")
            st.table(tabla.set_index("Puesto"))
            st.caption("Fuente: ranking de Rankia con valores cuotaparte de CAFCI. Para renta fija la TNA se estima con el rendimiento de 30 días.")
        else:
            lista = db_activos[db_activos["Categoria"] == categoria_fallback][["Instrumento", "Riesgo principal", "Plazo de liquidación", "Se opera en"]]
            st.table(lista.reset_index(drop=True))
            st.caption("Lista ilustrativa sin ranking (no hay datos en vivo). El plazo de liquidación en BYMA es T+1 (24 horas).")
    mostrar_explicacion(clave)
    mostrar_fuentes(clave)


def mostrar_deudas(tipo_deuda, monto_deuda, cft, atraso, alto):
    st.error("**Plan para ordenar tus deudas**")
    # Tasa de referencia: la mayor TNA de money market (en vivo si se puede; si no, la de referencia con fecha)
    try:
        df_mm, meta_mm = ranking("emergencia")
        tna_ref, origen = float(df_mm["TNA (%)"].iloc[0]), f"datos en vivo de CAFCI al {meta_mm['fecha']}"
    except Exception:
        tna_ref, origen = TNA_MM_REF, f"referencia de CAFCI al {FECHA_REF}"

    if cft > 0:
        if cft > tna_ref:
            st.success(f"Tu deuda cuesta aproximadamente **{fmt(cft)}%** anual y el mejor money market rinde cerca de **{fmt(tna_ref)}%** de TNA ({origen}). "
                       "Cancelar la deuda 'rinde' más que invertir: **priorizá pagarla antes de invertir**.")
        else:
            st.info(f"Tu deuda cuesta aproximadamente **{fmt(cft)}%** anual y el mejor money market rinde cerca de **{fmt(tna_ref)}%** de TNA ({origen}). "
                    "Podés seguir pagándola normalmente e ir armando un colchón. Tené en cuenta que la TNA y el costo financiero no son exactamente comparables.")
    else:
        st.info("No indicaste el costo financiero (CFT). Pedilo a tu banco o tarjeta: es el número clave para decidir qué deuda pagar primero.")

    prioridad = {
        "Tarjeta de crédito (pago mínimo o financiación)": "Es de las deudas más caras: evitá pagar solo el mínimo y cancelala primero.",
        "Préstamo personal": "Revisá su CFT: si podés refinanciarlo a una tasa menor, conviene.",
        "Crédito hipotecario / UVA": "Suele ser la deuda más barata y de largo plazo: pagá la cuota y priorizá cancelar antes las deudas más caras.",
        "Deuda informal / prestamista": "Suele tener tasas muy altas y condiciones poco claras: es la primera a cancelar o renegociar.",
        "Otras": "Compará su costo con el de tus demás deudas y empezá por la más cara.",
    }[tipo_deuda]
    st.write(f"**Tu deuda principal ({tipo_deuda.lower()}):** {prioridad}")
    if monto_deuda > 0:
        st.write(f"Monto adeudado informado: **{peso(monto_deuda)}**.")

    st.markdown(
        "**Opciones para ordenarte:**\n\n"
        "1. **Método avalancha:** pagá primero la deuda con mayor tasa de interés (ahorrás más en intereses).\n"
        "2. **Método bola de nieve:** si necesitás motivación, cancelá primero la deuda más chica y seguí con la siguiente.\n"
        "3. **Consolidar o refinanciar:** un solo préstamo a tasa fija solo conviene si su costo financiero total (CFT) es menor al de tus deudas actuales.\n"
        "4. **Negociar con el acreedor:** pedí refinanciar, extender el plazo o bajar la tasa antes de atrasarte.\n"
        "5. **No pagar solo el mínimo de la tarjeta:** el saldo restante se financia a tasas muy altas.\n"
        "6. **Colchón mínimo:** dejá guardado alrededor de un mes de gastos en un fondo de liquidez inmediata para no volver a endeudarte ante un imprevisto.\n"
        "7. **No invertir en activos de riesgo** mientras tu deuda cueste más de lo que rinde la inversión.\n"
        "8. **Revisá tu situación crediticia** en la Central de Deudores del BCRA.")
    if atraso == "Sí":
        st.warning("Tenés cuotas atrasadas: los intereses punitorios se acumulan rápido. Contactá al acreedor cuanto antes para refinanciar.")
    if alto:
        st.warning("Tu diagnóstico de riesgo es alto: priorizá liquidez y orden de deudas antes de tomar nuevo crédito.")
    st.write("**Dónde dejar tu colchón mínimo (fondos de liquidez inmediata):**")
    mostrar_top3("emergencia", "Fondo de Emergencia")
    st.markdown("Más información sobre tu situación crediticia: [BCRA – Central de Deudores](https://www.bcra.gob.ar)")


FUENTES_TEST = [("Balanz – Test del Inversor", "https://balanz.com/test-del-inversor"),
                ("IOL InvertirOnline – Test del Inversor", "https://www.invertironline.com/herramientas/test-del-inversor"),
                ("Bull Market Brokers – test de perfil (mencionado por DolarHoy)", "https://dolarhoy.com/mercado/conocerse-a-uno-mismo-el-paso-indispensable-antes-de-empezar-a-invertir-en-el-mercado-202662212130"),
                ("Rankia – cartera con CEDEARs según perfil (rangos de renta variable)", "https://www.rankia.com.ar/blog/cedear/6887815-como-armar-cartera-inversion-cedears-segun-perfil-riesgo"),
                ("Rankia – cartera balanceada en Argentina", "https://www.rankia.com.ar/blog/trading-argentina/6956118-cartera-inversion-balanceada-que"),
                ("iProfesional – en qué invertir según cada perfil (carteras de IOL)", "https://www.iprofesional.com/finanzas/458795-finanzas-personales-en-que-me-conviene-invertir-segun-mi-perfil-de-riesgo")]


# ───────────────────────── Horizontes, liquidez, Top 3 mixto y cartera por perfil ─────────────────────────
HORIZONTES = {"1 mes": 30, "3 meses": 90, "6 meses": 180, "1 año": 365}      # días de la ventana con que se mide el rendimiento
MESES_NUM = {"1 mes": 1, "3 meses": 3, "6 meses": 6, "1 año": 12}

TIPOS = {
    "emergencia": {"nombre": "Fondo de dinero (Money Market)", "riesgo": "Bajo", "liquidez": "El mismo día (T+0)", "min": 1,
                   "que_es": "Fondo que presta tu plata a muy corto plazo. Es como una caja de ahorro que rinde más.",
                   "motivo": "te da liquidez inmediata con riesgo bajo"},
    "renta": {"nombre": "Fondo de renta fija", "riesgo": "Bajo a medio", "liquidez": "24 h hábiles (T+1)", "min": 1,
              "que_es": "Fondo que presta tu plata (junto a la de otros) al Estado y a empresas, con un administrador profesional.",
              "motivo": "busca ganarle a la inflación con un riesgo moderado"},
    "cedears": {"nombre": "CEDEAR", "riesgo": "Alto", "liquidez": "24 h (T+1)", "min": 6,
                "que_es": "Papel que representa acciones de empresas del exterior (Apple, Coca-Cola…) y se compra en pesos.",
                "motivo": "te expone a empresas del mundo y al dólar, con más potencial y más altibajos"},
    "acciones": {"nombre": "Acción argentina", "riesgo": "Muy alto", "liquidez": "24 h (T+1)", "min": 12,
                 "que_es": "Una parte de una empresa argentina: sos socio/a. Su precio sube y baja todos los días.",
                 "motivo": "tiene el mayor potencial de crecimiento, pero también las caídas más fuertes"},
}

# Orden de preferencia de los tipos de instrumento según objetivo y perfil (de más a menos adecuado)
ORDEN = {
    ("renta", "Conservador"): ["renta", "emergencia", "cedears"],
    ("renta", "Moderado"): ["renta", "cedears", "emergencia"],
    ("renta", "Agresivo"): ["renta", "cedears", "acciones", "emergencia"],
    ("crecimiento", "Conservador"): ["renta", "cedears", "emergencia"],
    ("crecimiento", "Moderado"): ["cedears", "renta", "acciones", "emergencia"],
    ("crecimiento", "Agresivo"): ["acciones", "cedears", "renta", "emergencia"],
}
ORDEN_EMERGENCIA = ["emergencia", "renta"]

# Cartera base por perfil (% en liquidez, renta fija, CEDEARs, acciones). La renta variable (CEDEARs + acciones) respeta los rangos
# publicados por Rankia: conservador 15-40 %, moderado 40-60 %, agresivo más de 70 %.
BASE_CARTERA = {
    "Conservador": {"emergencia": 30, "renta": 45, "cedears": 20, "acciones": 5},
    "Moderado": {"emergencia": 15, "renta": 30, "cedears": 30, "acciones": 25},
    "Agresivo": {"emergencia": 5, "renta": 15, "cedears": 35, "acciones": 45},
}
# Tope de renta variable y de renta fija según el plazo (a menos tiempo, menos riesgo)
LIMITES = {"1 mes": (0, 20), "3 meses": (10, 70), "6 meses": (40, 100), "1 año": (100, 100)}
LIQ_MIN = {"Sí, es probable": 30, "Tal vez": 15, "No, no la voy a tocar": 0}
NOMBRE_BUCKET = {"emergencia": "Liquidez (fondo de dinero)", "renta": "Renta fija (fondos / bonos)",
                 "cedears": "CEDEARs (empresas del exterior)", "acciones": "Acciones argentinas"}


def armar_cartera(perfil, meses, liq):
    c = dict(BASE_CARTERA[perfil])
    rv_max, rf_max = LIMITES[meses]
    rv = c["cedears"] + c["acciones"]
    if rv > rv_max:                                   # 1) tope de renta variable según el plazo
        f = rv_max / rv
        c["cedears"], c["acciones"] = c["cedears"] * f, c["acciones"] * f
        c["renta"] += rv - rv_max
    falta = LIQ_MIN[liq] - c["emergencia"]            # 2) colchón mínimo de liquidez si podés necesitar la plata
    for k in ("acciones", "cedears", "renta"):
        if falta > 0:
            t = min(c[k], falta)
            c[k] -= t
            c["emergencia"] += t
            falta -= t
    if c["renta"] > rf_max:                           # 3) tope de renta fija (a 1 mes casi todo va a liquidez)
        c["emergencia"] += c["renta"] - rf_max
        c["renta"] = rf_max
    r = {k: int(round(v)) for k, v in c.items()}
    r[max(r, key=r.get)] += 100 - sum(r.values())     # ajuste de redondeo
    return r


def items_tipo(clave, meses):
    """Devuelve (lista de instrumentos, origen). Usa datos en vivo; si no hay conexión, usa la referencia con fecha."""
    t = TIPOS[clave]
    try:
        df, meta = ranking(clave, meses)
        items = []
        for _, r in df.iterrows():
            rend = f"TNA {fmt(r['TNA (%)'])}%" if "TNA (%)" in df.columns else f"{fmt(r['Retorno anual (%)'])}% anual (histórico)"
            items.append({"Instrumento": r["Instrumento"], "Rendimiento": rend, "porque": r["Por qué está en el top 3"]})
        return items, f"en vivo · {meta['fuente']} · datos al {meta['fecha']}"
    except Exception:
        tabla = tabla_referencia(clave)
        if tabla is not None:
            items = [{"Instrumento": r["Instrumento"], "Rendimiento": f"TNA {fmt(r['TNA (%)'])}%", "porque": r["Por qué está en el top 3"]}
                     for _, r in tabla.iterrows()]
            return items, f"referencia con fecha · CAFCI al {FECHA_REF}"
        nombres = list((CEDEARS if clave == "cedears" else ACCIONES).values())[:3]
        return ([{"Instrumento": n, "Rendimiento": "sin dato en vivo", "porque": "Lista ilustrativa: no hay conexión para calcular el ranking."}
                 for n in nombres], "lista ilustrativa (sin conexión)")


def top3_mixto(obj_key, perfil_ef, meses):
    """Top 3 que combina distintos tipos de instrumento según objetivo, perfil y plazo. Devuelve (DataFrame, tipos_excluidos, origenes)."""
    orden = ORDEN_EMERGENCIA if obj_key == "emergencia" else ORDEN[(obj_key, perfil_ef)]
    elegibles = [t for t in orden if TIPOS[t]["min"] <= MESES_NUM[meses]]
    excluidos = [t for t in orden if t not in elegibles]
    cargados, origenes = {}, {}
    for t in elegibles[:3]:
        cargados[t], origenes[t] = items_tipo(t, meses)
    cola = []                                          # 1.º de cada tipo en orden de preferencia; si faltan, se completa con los siguientes del mejor tipo
    for t in elegibles[:3]:
        if cargados[t]:
            cola.append((t, cargados[t][0]))
    k = 1
    while len(cola) < 3 and k < 3:
        for t in elegibles[:3]:
            if len(cola) < 3 and len(cargados[t]) > k:
                cola.append((t, cargados[t][k]))
        k += 1
    filas = []
    for i, (t, it) in enumerate(cola[:3]):
        info = TIPOS[t]
        filas.append({"Puesto": i + 1, "Tipo": info["nombre"], "Instrumento": it["Instrumento"], "Rendimiento": it["Rendimiento"],
                      "Riesgo": info["riesgo"], "¿Cuándo tengo la plata?": info["liquidez"],
                      "Por qué está en el top 3": f"Es del tipo que mejor encaja con tu objetivo y perfil: {info['motivo']}. {it['porque']}"})
    return pd.DataFrame(filas), excluidos, origenes


def perfil_efectivo(perfil, meses, alto):
    tope = {"1 mes": "Conservador", "3 meses": "Moderado"}.get(meses)
    orden = ["Conservador", "Moderado", "Agresivo"]
    p = perfil
    if tope and orden.index(p) > orden.index(tope):
        p = tope
    if alto:
        p = "Conservador"
    return p


# Lista ILUSTRATIVA: solo se usa como respaldo para CEDEARs y acciones si no hay conexión a las fuentes (sin rendimientos inventados).
db_activos = pd.DataFrame({
    "Instrumento": ["SPY (CEDEAR S&P 500)", "KO (CEDEAR Coca-Cola)", "AAPL (CEDEAR Apple)",
                    "YPFD (acción YPF)", "GGAL (acción Galicia)", "PAMP (acción Pampa Energía)"],
    "Categoria": ["Crecimiento Conservador (CEDEAR)"] * 3 + ["Crecimiento Agresivo (Acciones)"] * 3,
    "Riesgo principal": ["Riesgo de mercado externo y de tipo de cambio"] * 3 + ["Alta volatilidad local"] * 3,
    "Plazo de liquidación": ["T+1 (24 h)"] * 6,
    "Se opera en": ["ALyC"] * 6,
})


# ───────────────────────── Interfaz ─────────────────────────
st.markdown(
    '<div class="nav"><div class="logo">AF</div><a href="#inicio" target="_self">Inicio</a><a href="#perfil" target="_self">Sobre vos</a>'
    '<a href="#test" target="_self">Test</a><a href="#objetivo" target="_self">Objetivo</a><a href="#resultados" target="_self">Resultados</a></div>'
    '<div id="inicio" class="hero"><h1>Asesor Financiero<br>con IA</h1>'
    '<div class="sub">Riesgo crediticio, perfil de inversor y cartera ideal</div>'
    '<div class="txt">Contanos tu situación y tu objetivo. Estimamos tu riesgo con datos de la EPH, descubrimos tu perfil, '
    'te mostramos un Top 3 de inversiones reales y te sugerimos cómo repartir tu plata. Todo explicado en simple.</div>'
    '<div><a class="btn claro" href="#perfil" target="_self">Empezar</a><a class="btn oscuro" href="#test" target="_self">Ir al test</a></div>'
    '<div class="deslizar">DESLIZAR</div></div>', unsafe_allow_html=True)

seccion("perfil", "Paso 1", "Sobre vos", "Estos datos alimentan el modelo de riesgo crediticio entrenado con la EPH.")
col1, col2 = st.columns(2)
with col1:
    edad = st.number_input("Edad:", min_value=18, max_value=100, value=30)
    tramo = st.selectbox("Ingreso mensual propio (en dólares):", list(TRAMOS_USD.keys()), index=2)
    tc = st.number_input("Tipo de cambio para convertir ($ por USD). Podés ajustarlo:", min_value=100, value=TC_REF, step=10)
    lo_usd, hi_usd = TRAMOS_USD[tramo]
    ingresos = (lo_usd + hi_usd) / 2 * tc          # el modelo usa el punto medio del tramo, en pesos
    st.caption(f"Se usa el punto medio del tramo: {peso(ingresos)} por mes.")
    situacion = st.radio("Situación laboral:", ["Asalariado/a (en relación de dependencia)",
                                                "Trabajo por cuenta propia / independiente",
                                                "No trabajo actualmente"])
    provincia = st.selectbox("Provincia donde vivís:", list(PROVINCIAS.keys()))
    region = PROVINCIAS[provincia]
    st.caption(f"Región EPH asignada: **{region}**")
with col2:
    edu = st.selectbox("Nivel educativo máximo alcanzado:", list(EDUCACION.keys()), index=3)
    pareja = st.radio("Estado civil / Convivencia:", ["Vivo solo/a", "Vivo con mis padres", "Vivo en pareja / Casado/a"])

st.markdown("---")
seccion("test", "Paso 2", "Tu perfil de inversor")
st.caption("8 preguntas simples. Cubren lo que miden los tests de Balanz, IOL y Bull Market: objetivo, plazo, necesidad de liquidez, "
           "tolerancia a las pérdidas, experiencia y estabilidad de tus ingresos. No hay respuestas buenas o malas.")
col3, col4 = st.columns(2)
with col3:
    q_obj = st.radio("1. ¿Qué es más importante para vos con esta plata?",
                     ["Que no se pierda, aunque gane poco", "Que gane algo más que la inflación, aceptando algunos altibajos",
                      "Que crezca lo más posible, aunque pueda perder una parte"])
    horizonte = st.radio("2. ¿En cuánto tiempo vas a necesitar esta plata?", list(HORIZONTES.keys()), index=2, horizontal=True)
    liquidez = st.radio("3. ¿Podrías necesitar sacarla antes de ese plazo (por un imprevisto)?", list(LIQ_MIN.keys()))
    tolerancia = st.radio("4. Si tu inversión cae un 15% en un mes, ¿qué hacés?",
                          ["Vendo todo para no perder más", "Espero a que se recupere", "Compro más porque está más barata"])
with col4:
    perdida = st.radio("5. ¿Cuánta pérdida podrías aguantar sin desesperarte?",
                       ["Ninguna: no quiero perder nada", "Hasta un 10%", "Más del 20%, si puedo ganar bastante más"])
    experiencia = st.radio("6. ¿Qué experiencia tenés invirtiendo?",
                           ["Ninguna", "Plazo fijo, billeteras virtuales o fondos", "Bonos, acciones o CEDEARs"])
    flujo = st.radio("7. Tus ingresos son…", ["Fijos todos los meses", "Variables (cambian mucho)"])
    compromisos = st.radio("¿Tenés gastos grandes planificados en los próximos 12 meses (viaje, mudanza, auto…)?", ["Sí", "No"])
    retencion = st.slider("8. ¿Qué porcentaje de tu ingreso podés separar sin afectar tu vida diaria?", 0, 100, 10)

st.markdown("---")
seccion("objetivo", "Paso 3", "Tu objetivo")
o1, o2 = st.columns(2)
with o1:
    objetivo = st.selectbox("¿Para qué es este dinero?", ["Fondo para Presupuesto Mensual o Emergencia",
                                                          "Generar renta y ganarle a la inflación",
                                                          "Crecimiento agresivo del capital a largo plazo",
                                                          "Priorizar, consolidar y ordenar deudas pendientes"])
with o2:
    monto = st.number_input("¿Cuánto pensás invertir? ($)", min_value=0, value=500000, step=50000)

tipo_deuda, monto_deuda, cft, atraso = "Otras", 0, 0.0, "No"
if objetivo.startswith("Priorizar"):
    st.markdown("**Contanos sobre tus deudas** (para ordenar las prioridades)")
    d1, d2 = st.columns(2)
    with d1:
        tipo_deuda = st.selectbox("Tipo de deuda principal:", ["Tarjeta de crédito (pago mínimo o financiación)", "Préstamo personal",
                                                               "Crédito hipotecario / UVA", "Deuda informal / prestamista", "Otras"])
        monto_deuda = st.number_input("Monto total adeudado ($):", min_value=0, value=0, step=50000)
    with d2:
        cft = st.number_input("Costo financiero total anual aproximado (CFT, %). Dejá 0 si no lo sabés:", min_value=0.0, value=0.0, step=1.0)
        atraso = st.radio("¿Tenés cuotas atrasadas?", ["No", "Sí"])
st.markdown("---")

if st.button("Ejecutar Análisis de Perfil", type="primary", use_container_width=True):

    univ, sec, edu_ref = EDUCACION[edu]
    es_asalariado = situacion.startswith("Asalariado")

    # ---- Riesgo crediticio (solo asalariados: población con la que se entrenó el modelo) ----
    proba_alto_riesgo = None
    if es_asalariado:
        datos_usuario = pd.DataFrame([[float(edad), float(ingresos), univ, sec, 1 if pareja == "Vivo en pareja / Casado/a" else 0]],
                                     columns=VARS_MODELO)
        proba_alto_riesgo = modelo.predict_proba(datos_usuario)[0][1]
    alto = (proba_alto_riesgo is not None and proba_alto_riesgo > 0.5)

    # ---- Perfil de inversor: 8 preguntas, puntaje máximo 16 ----
    puntos = [
        ["Que no se pierda, aunque gane poco", "Que gane algo más que la inflación, aceptando algunos altibajos",
         "Que crezca lo más posible, aunque pueda perder una parte"].index(q_obj),
        {"1 mes": 0, "3 meses": 0, "6 meses": 1, "1 año": 2}[horizonte],
        {"Sí, es probable": 0, "Tal vez": 1, "No, no la voy a tocar": 2}[liquidez],
        ["Vendo todo para no perder más", "Espero a que se recupere", "Compro más porque está más barata"].index(tolerancia),
        ["Ninguna: no quiero perder nada", "Hasta un 10%", "Más del 20%, si puedo ganar bastante más"].index(perdida),
        ["Ninguna", "Plazo fijo, billeteras virtuales o fondos", "Bonos, acciones o CEDEARs"].index(experiencia),
        (1 if flujo.startswith("Fijos") else 0) + (1 if compromisos == "No" else 0),
        2 if retencion >= 40 else (1 if retencion >= 20 else 0),
    ]
    total = sum(puntos)
    perfil = "Conservador" if total <= 5 else ("Moderado" if total <= 10 else "Agresivo")
    perfil_ef = perfil_efectivo(perfil, horizonte, alto)

    st.markdown('<div id="resultados"></div>', unsafe_allow_html=True)
    seccion("res", "Resultados", "Tu análisis")
    tab_res, tab_inv, tab_cart, tab_liq, tab_met = st.tabs([" Tu resultado", " Top 3 de inversiones", " Tu cartera ideal",
                                                            " Liquidez y glosario", " Metodología y fuentes"])

    # ================= Pestaña 1: resultado =================
    with tab_res:
        m1, m2, m3 = st.columns(3)
        if proba_alto_riesgo is None:
            m1.metric("Riesgo crediticio", "No estimado", "el modelo solo evalúa asalariados", delta_color="off")
        else:
            m1.metric("Riesgo crediticio", "Alto" if alto else "Bajo", f"probabilidad de alto riesgo: {proba_alto_riesgo:.1%}",
                      delta_color="inverse" if alto else "normal")
        m2.metric("Tu perfil de inversor", perfil, f"puntaje {total}/16", delta_color="off")
        m3.metric(f"Perfil para invertir a {horizonte}", perfil_ef, "ajustado por plazo y riesgo" if perfil_ef != perfil else "sin ajustes", delta_color="off")
        if perfil_ef != perfil:
            st.info(f"Tu perfil psicológico es **{perfil}**, pero para invertir a **{horizonte}**"
                    f"{' y con riesgo crediticio alto' if alto else ''} te conviene actuar como **{perfil_ef}**: "
                    "una cosa es cuánto riesgo aguantás y otra cuánto riesgo podés tomar según cuándo necesitás la plata.")
        st.markdown({"Conservador": "**Conservador:** priorizás cuidar tu plata. Preferís ganar poco pero seguro.",
                     "Moderado": "**Moderado:** buscás equilibrio. Aceptás algunos altibajos a cambio de ganar más.",
                     "Agresivo": "**Agresivo:** buscás hacer crecer tu plata y aceptás caídas fuertes en el camino."}[perfil])

        st.subheader("Riesgo crediticio")
        if es_asalariado:
            if alto:
                st.warning(f"**ALTO RIESGO** (probabilidad: {proba_alto_riesgo:.1%}). Priorizá la liquidez y un fondo de emergencia antes de tomar deuda.")
            else:
                st.success(f"**BAJO RIESGO** (probabilidad: {proba_alto_riesgo:.1%}). Tu perfil laboral es estable para acceder a crédito.")
            st.caption("Regresión logística entrenada con asalariados de la EPH 1T-2026. La informalidad laboral se usa como aproximación del riesgo crediticio (no mide mora).")
        else:
            st.info("El modelo se entrenó con asalariados: no estima un score para cuentapropistas ni para personas sin trabajo. "
                    "Como criterio prudente, priorizá un fondo de emergencia antes de endeudarte o invertir en activos de riesgo.")

        if not situacion.startswith("No trabajo"):
            sit_ref = "Asalariado/a" if es_asalariado else "Cuenta propia"
            fila = ref[(ref["situacion"] == sit_ref) & (ref["region"] == region) & (ref["educ"] == edu_ref)]
            st.subheader("Tu ingreso frente a la EPH")
            if len(fila) == 1:
                f = fila.iloc[0]
                cols_p = [f"p{i}" for i in range(10, 100, 10)]
                deciles = f[cols_p].astype(float).values
                def _pos(v):
                    if v < deciles[0]:
                        return 5
                    if v > deciles[-1]:
                        return 95
                    return int(np.interp(v, deciles, range(10, 100, 10)))
                p_lo, p_hi = _pos(lo_usd * tc), _pos(hi_usd * tc)
                st.write(f"Con un ingreso de **{tramo.replace('USD', 'USD ')}** (entre {peso(lo_usd * tc)} y {peso(hi_usd * tc)} por mes), estás entre el percentil "
                         f"**{p_lo}** y el **{p_hi}** de los/as ocupados/as de tu grupo ({sit_ref}, región {region}, nivel educativo {edu_ref}). "
                         f"Mediana del grupo: **{peso(f['p50'])}** (n = {int(f['n'])} casos en la EPH 1T-2026).")
                st.caption(f"Tu provincia ({provincia}) se compara con la región EPH {region}. Un percentil 70 significa que ganás más que el 70% de tu grupo.")
            else:
                st.write("No hay suficientes casos en la EPH para comparar tu grupo.")

        st.subheader("Cómo se armó tu puntaje")
        st.table(pd.DataFrame({"Pregunta": ["Objetivo", "Plazo", "Necesidad de liquidez", "Reacción a una caída", "Pérdida aceptable",
                                            "Experiencia", "Estabilidad de ingresos", "Capacidad de ahorro"],
                               "Puntos": puntos, "Máximo": [2, 2, 2, 2, 2, 2, 2, 2]}).set_index("Pregunta"))
        st.caption("0-5 puntos: Conservador · 6-10: Moderado · 11-16: Agresivo.")

    # ================= Pestaña 2: Top 3 =================
    with tab_inv:
        if objetivo.startswith("Priorizar"):
            mostrar_deudas(tipo_deuda, monto_deuda, cft, atraso, alto)
        else:
            obj_key = ("emergencia" if objetivo.startswith("Fondo") else "renta" if objetivo.startswith("Generar") else "crecimiento")
            st.subheader(f"Top 3 para tu objetivo, a {horizonte}")
            st.write(f"El sistema combina **distintos tipos de inversión** (no solo fondos) según tu objetivo, tu perfil para invertir (**{perfil_ef}**) "
                     f"y tu plazo (**{horizonte}**). Los rendimientos se miden en los últimos **{horizonte}** con datos reales.")
            with st.spinner("Consultando datos reales de mercado..."):
                df3, excluidos, origenes = top3_mixto(obj_key, perfil_ef, horizonte)
            st.table(df3.set_index("Puesto"))
            st.caption("Orden: primero el tipo de inversión que mejor encaja con tu objetivo y perfil; dentro de cada tipo, el instrumento con mejor "
                       "rendimiento (TNA) o mejor relación retorno/riesgo. Si el plazo es corto y hay pocos tipos aptos, se completa con el siguiente mejor "
                       "instrumento del mismo tipo. Rendimientos pasados no garantizan resultados futuros.")
            for t, o in origenes.items():
                st.caption(f"• {TIPOS[t]['nombre']}: {o}")
            if excluidos:
                st.info("No se muestran " + ", ".join(TIPOS[t]["nombre"] + "s" for t in excluidos) +
                        f" porque necesitan un plazo mínimo ({', '.join(str(TIPOS[t]['min']) + ' meses' for t in excluidos)}) o no encajan con tu perfil: "
                        "su precio puede estar bajo el que pagaste justo cuando necesitás la plata.")
            st.markdown("**Sobre cada instrumento**")
            for t in dict.fromkeys(df3["Tipo"].map({v["nombre"]: k for k, v in TIPOS.items()})):
                mostrar_explicacion(t)

    # ================= Pestaña 3: cartera =================
    with tab_cart:
        cartera = armar_cartera(perfil_ef, horizonte, liquidez)
        st.subheader(f"Cartera sugerida para un perfil {perfil_ef} a {horizonte}")
        st.write("No pongas todos los huevos en la misma canasta: repartir tu plata entre distintos instrumentos reduce el riesgo.")
        dfc = pd.DataFrame({"Tipo de inversión": [NOMBRE_BUCKET[k] for k in cartera], "% de tu plata": [cartera[k] for k in cartera]})
        if monto > 0:
            dfc["Monto ($)"] = [peso(monto * cartera[k] / 100) for k in cartera]
        dfc["Mejor opción hoy"] = [(items_tipo(k, horizonte)[0][0]["Instrumento"] if cartera[k] > 0 else "-") for k in cartera]
        st.table(dfc.set_index("Tipo de inversión"))
        st.bar_chart(dfc.set_index("Tipo de inversión")["% de tu plata"])
        rv = cartera["cedears"] + cartera["acciones"]
        st.caption(f"Renta variable total (CEDEARs + acciones): {rv}%. Rangos de referencia: conservador 15-40%, moderado 40-60%, agresivo más de 70%.")
        notas = []
        if horizonte in ("1 mes", "3 meses"):
            notas.append(f"Como tu plazo es de **{horizonte}**, se limitó la renta variable: en plazos cortos las caídas no alcanzan a recuperarse.")
        if LIQ_MIN[liquidez] > 0:
            notas.append(f"Como podrías necesitar la plata antes de tiempo, se reservó al menos un **{LIQ_MIN[liquidez]}%** en un fondo de dinero (disponible el mismo día).")
        if alto:
            notas.append("Tu riesgo crediticio es alto: la cartera se llevó al perfil más prudente.")
        for n in notas:
            st.info(n)
        with st.expander("Así quedan las carteras de los tres perfiles (plazo de 1 año, sin ajustes de liquidez)"):
            comp = pd.DataFrame({p: armar_cartera(p, "1 año", "No, no la voy a tocar") for p in BASE_CARTERA}).rename(index=NOMBRE_BUCKET)
            st.table(comp.astype(str) + "%")
            st.caption("Los porcentajes base se apoyan en los rangos de renta variable por perfil de Rankia y en las carteras modelo de IOL e iProfesional. Son una guía educativa, no una recomendación personalizada.")

    # ================= Pestaña 4: liquidez y glosario =================
    with tab_liq:
        st.subheader("¿Qué es la liquidez?")
        st.write("**Liquidez** es qué tan rápido podés convertir una inversión en plata disponible **sin perder valor**. "
                 "Una inversión muy líquida te da la plata el mismo día; una poco líquida puede tardar o venderse con pérdida.")
        st.table(pd.DataFrame({"Instrumento": [v["nombre"] for v in TIPOS.values()],
                               "¿Qué es?": [v["que_es"] for v in TIPOS.values()],
                               "Riesgo": [v["riesgo"] for v in TIPOS.values()],
                               "¿Cuándo tengo la plata?": [v["liquidez"] for v in TIPOS.values()],
                               "Plazo mínimo sugerido": [f"{v['min']} mes" + ("es" if v["min"] > 1 else "") for v in TIPOS.values()]}).set_index("Instrumento"))
        st.markdown("**Palabras que aparecen en la app:**\n\n" + GLOSARIO)

    # ================= Pestaña 5: metodología =================
    with tab_met:
        st.subheader("Cómo funciona")
        st.markdown(
            "1. **Riesgo crediticio:** regresión logística con microdatos de la EPH (INDEC, 1T-2026) de asalariados de 18 a 65 años (11.108 casos). "
            "Predice la probabilidad de ser informal (36,5% de la muestra), usada como aproximación del riesgo. Exactitud 80,2% y AUC 0,855.\n"
            "2. **Perfil de inversor:** 8 preguntas basadas en las dimensiones de los tests de Balanz, IOL y Bull Market (objetivo, plazo, liquidez, "
            "tolerancia, experiencia, ingresos). Puntaje 0-16.\n"
            "3. **Top 3:** datos reales de CAFCI (fondos, TNA en la ventana del plazo elegido) y Yahoo Finance (CEDEARs y acciones, retorno/volatilidad).\n"
            "4. **Cartera:** porcentajes base por perfil, recortados por plazo y por necesidad de liquidez.")
        st.warning("**Limitaciones:** la informalidad no es lo mismo que mora; el modelo solo cubre asalariados; el test no reemplaza una entrevista con un "
                   "asesor registrado; los rendimientos pasados no garantizan resultados futuros; no incluye bonos individuales ni dólar.")
        mostrar_fuentes("emergencia")
        mostrar_fuentes("cedears")
        with st.expander("Fuentes del test de perfil y de las carteras"):
            for nombre, url in FUENTES_TEST:
                st.markdown(f"- [{nombre}]({url})")
            st.caption("Los tests de Balanz, IOL y Bull Market son interactivos: de ellos se tomaron las dimensiones que miden, no el texto de sus preguntas.")

    st.markdown("---")
    seccion("operar", "Siguiente paso", "¿Dónde operar?")
    st.write("Para invertir en Argentina conviene operar con Agentes de Liquidación y Compensación (ALyCs) registrados ante la Comisión Nacional de Valores (CNV). Antes de operar, verificá su inscripción en la web de la CNV.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### **Balanz**")
        st.write("ALyC con acciones, CEDEARs, ONs y fondos comunes de inversión.")
    with c2:
        st.markdown("### **IOL (InvertirOnline)**")
        st.write("ALyC digital con cuenta remunerada y portafolios sugeridos.")
    with c3:
        st.markdown("### **Mercado Pago / Ualá**")
        st.write("Billeteras con fondos de inversión de liquidez inmediata para el dinero de uso diario.")

    st.markdown("---")
    st.caption("Herramienta educativa desarrollada para un trabajo práctico. No constituye asesoramiento financiero ni una recomendación de inversión. Los rankings usan datos históricos públicos y no garantizan resultados futuros.")
