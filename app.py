# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from mercado import top_fci, top_mercado, CEDEARS, ACCIONES, SinDatos

st.set_page_config(page_title="Asesor Financiero IA", page_icon="💳", layout="wide")

@st.cache_data(ttl=6 * 3600, show_spinner=False)
def ranking(clave):
    # Top 3 con datos reales. Se cachea 6 horas para no consultar las APIs en cada clic.
    if clave == "emergencia":
        return top_fci("mercadoDinero", dias=30)
    if clave == "renta":
        return top_fci("rentaFija", dias=90)
    if clave == "cedears":
        return top_mercado(CEDEARS)
    return top_mercado(ACCIONES)


def mostrar_top3(clave, categoria_fallback):
    # Muestra el top 3 real; si no hay conexión/datos, muestra la lista ilustrativa SIN ranking
    try:
        with st.spinner("Consultando datos reales de mercado..."):
            df_top, meta = ranking(clave)
        st.write("**Top 3 según el sistema (ordenado de mejor a peor):**")
        st.table(df_top.set_index("Puesto"))
        st.caption(f"Fuente: {meta['fuente']} · datos al {meta['fecha']} · Criterio: {meta['criterio']}. "
                   "Rendimientos pasados no garantizan resultados futuros.")
        mostrar_fuentes(clave)
    except Exception:
        st.warning("No se pudieron obtener datos de mercado en este momento. Se muestran ejemplos ilustrativos, sin ranking.")
        st.table(db_activos[db_activos["Categoria"] == categoria_fallback][["Instrumento", "Riesgo principal", "Se opera en"]].reset_index(drop=True))
        mostrar_fuentes(clave)


FUENTES = {
    "emergencia": [("CAFCI (Cámara Argentina de FCI) – valores cuotaparte oficiales", "https://www.cafci.org.ar"),
                   ("CNV – regulador de fondos y ALyCs", "https://www.cnv.gob.ar"),
                   ("Balanz – fondos comunes (para contrastar el dato)", "https://balanz.com"),
                   ("ArgentinaDatos – API que republica los datos de CAFCI", "https://argentinadatos.com")],
    "cedears": [("Yahoo Finance – precios históricos", "https://finance.yahoo.com"),
                ("BYMA – mercado donde cotizan CEDEARs y acciones", "https://www.byma.com.ar"),
                ("CNV – regulador del mercado de capitales", "https://www.cnv.gob.ar")],
}
FUENTES["renta"] = FUENTES["emergencia"]
FUENTES["acciones"] = FUENTES["cedears"]

# Referencias de contraste con fecha (no se usan para rankear; solo para que el usuario compare el orden de magnitud)
REFERENCIAS = {
    "emergencia": "Referencia de contraste: a fines de septiembre de 2026 los money market rendían entre 20,4% y 23,2% de TNA (cuotapartes CAFCI al 23/09/2026, según el ranking de Rankia).",
}


def mostrar_fuentes(clave):
    with st.expander("📚 Fuentes verificables de estos datos"):
        st.write("El ranking se calcula en vivo con estas fuentes públicas. Podés abrirlas y comprobar el dato vos mismo:")
        for nombre, url in FUENTES[clave]:
            st.markdown(f"- [{nombre}]({url})")
        if clave in REFERENCIAS:
            st.caption(REFERENCIAS[clave])


def peso(x):
    # formato argentino: $1.000.000
    return "$" + f"{x:,.0f}".replace(",", ".")

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

try:
    import io
    ref = pd.read_csv(io.StringIO(REFERENCIA_CSV))
except Exception:
    ref = None

# Lista ILUSTRATIVA: solo se usa como respaldo si no hay conexión a las fuentes de datos (sin tasas ni rendimientos inventados).
db_activos = pd.DataFrame({
    "Instrumento": [
        "FCI Money Market (de tu ALyC o billetera virtual)", "Cuenta remunerada (ALyC o billetera virtual)", "FCI de liquidez inmediata",
        "ON de YPF (elegir serie vigente)", "ON de Pampa Energía (elegir serie vigente)", "AL30 (bono soberano en USD)",
        "SPY (CEDEAR S&P 500)", "KO (CEDEAR Coca-Cola)", "AAPL (CEDEAR Apple)",
        "YPFD (acción YPF)", "GGAL (acción Galicia)", "PAMP (acción Pampa Energía)"
    ],
    "Categoria": ["Fondo de Emergencia"] * 3 + ["Renta y Crecimiento Moderado"] * 3 +
                 ["Crecimiento Conservador (CEDEAR)"] * 3 + ["Crecimiento Agresivo (Acciones)"] * 3,
    "Riesgo principal": [
        "Rendimiento variable, no garantizado", "Rendimiento variable, no garantizado", "Rendimiento variable, no garantizado",
        "Riesgo de crédito del emisor y de mercado", "Riesgo de crédito del emisor y de mercado", "Riesgo soberano y alta volatilidad",
        "Riesgo de mercado externo y de tipo de cambio", "Riesgo de mercado externo y de tipo de cambio", "Riesgo de mercado externo y de tipo de cambio",
        "Alta volatilidad local", "Alta volatilidad local", "Alta volatilidad local"
    ],
    "Se opera en": ["ALyC / billetera"] * 3 + ["ALyC"] * 9
})

st.title("💳 Plataforma de Asesoramiento Financiero y Riesgo")
st.write("Completá tu perfil para obtener un diagnóstico de riesgo y una estrategia de capital de carácter educativo.")
st.markdown("---")

st.subheader("1. Variables predictivas (riesgo crediticio)")
col1, col2 = st.columns(2)
with col1:
    edad = st.number_input("Edad:", min_value=18, max_value=100, value=30)
    ingresos = st.number_input("Ingresos mensuales propios ($):", min_value=0, value=1000000, step=10000)
    situacion = st.radio("Situación laboral:", ["Asalariado/a (en relación de dependencia)",
                                                "Trabajo por cuenta propia / independiente",
                                                "No trabajo actualmente"])
    region = st.selectbox("Región:", ["Gran Buenos Aires", "Pampeana", "Noroeste", "Noreste", "Cuyo", "Patagonia"])
with col2:
    edu = st.selectbox("Nivel educativo máximo:", ["Primario o menos", "Secundario (completo o incompleto)",
                                                   "Superior universitario / terciario (completo o incompleto)"])
    pareja = st.radio("Estado civil / Convivencia:", ["Vivo solo/a", "Vivo con mis padres", "Vivo en pareja / Casado/a"])

st.markdown("---")
st.subheader("2. Cuestionario de perfil del inversor")
col3, col4 = st.columns(2)
with col3:
    horizonte = st.selectbox("Horizonte temporal: ¿en cuánto tiempo planeás usar este dinero?", ["Meses", "Años", "Décadas"])
    tolerancia = st.selectbox("Tolerancia al riesgo: si tu inversión cae un 15% en una semana, ¿qué hacés?", ["Vendés por precaución", "Esperás a que recupere", "Comprás más"])
    retencion = st.slider("Capacidad de retención: ¿qué porcentaje de tu ingreso mensual podés separar sin afectar tu calidad de vida?", 0, 100, 10)
    conocimiento = st.radio("Conocimiento previo: ¿operaste alguna vez en el mercado de capitales?", ["Sí", "No"])
with col4:
    flujo = st.radio("Flujo de caja: ¿tus ingresos son fijos o muy variables?", ["Fijos", "Alta variabilidad"])
    compromisos = st.radio("Compromisos de capital: ¿tenés desembolsos grandes planificados en los próximos 12 meses?", ["Sí", "No"])
    psicologia = st.radio("Psicología de inversión: si obtenés una ganancia rápida e inesperada, ¿qué hacés?", ["La retiro para asegurar", "La reinvierto asumiendo más riesgo"])
    objetivo = st.selectbox("Objetivo del dinero:", ["Fondo para Presupuesto Mensual o Emergencia",
                                                     "Generar renta y ganarle a la inflación",
                                                     "Crecimiento agresivo del capital a largo plazo",
                                                     "Priorizar, consolidar y ordenar deudas pendientes"])
st.markdown("---")

if st.button("Ejecutar Análisis de Perfil", type="primary", use_container_width=True):

    # ---- Diagnóstico de riesgo (solo asalariados: es la población con la que se entrenó el modelo) ----
    proba_alto_riesgo = None
    st.subheader("📊 Diagnóstico de Riesgo Crediticio")
    es_asalariado = situacion.startswith("Asalariado")
    if es_asalariado and modelo is not None:
        datos_usuario = pd.DataFrame([[float(edad), float(ingresos),
                                       1 if edu.startswith("Superior") else 0,
                                       1 if edu.startswith("Secundario") else 0,
                                       1 if pareja == "Vivo en pareja / Casado/a" else 0]],
                                     columns=["edad", "ingresos", "universitario", "secundario", "en_pareja"])
        proba_alto_riesgo = modelo.predict_proba(datos_usuario)[0][1]
        if proba_alto_riesgo > 0.5:
            st.warning(f"⚠️ **ALTO RIESGO** (probabilidad de alto riesgo: {proba_alto_riesgo:.1%}). Priorizá la liquidez y un fondo de emergencia antes de tomar deuda.")
        else:
            st.success(f"✅ **BAJO RIESGO** (probabilidad de alto riesgo: {proba_alto_riesgo:.1%}). Tu perfil laboral es estable para acceder a crédito.")
        st.caption("Estimado con una regresión logística entrenada con asalariados de la EPH 1T-2026; la informalidad se usa como aproximación del riesgo crediticio (no mide mora).")
    else:
        st.info("El modelo se entrenó con asalariados: no estima un score para cuentapropistas ni para personas sin trabajo. "
                "Como criterio prudente, priorizá un fondo de emergencia antes de endeudarte o invertir en activos de riesgo.")

    # ---- Comparación del ingreso con la EPH ----
    if ref is not None and not situacion.startswith("No trabajo") and ingresos > 0:
        sit_ref = "Asalariado/a" if es_asalariado else "Cuenta propia"
        edu_ref = "Superior" if edu.startswith("Superior") else ("Secundario" if edu.startswith("Secundario") else "Primario o menos")
        fila = ref[(ref["situacion"] == sit_ref) & (ref["region"] == region) & (ref["educ"] == edu_ref)]
        st.subheader("📈 Tu ingreso frente a la EPH")
        if len(fila) == 1:
            f = fila.iloc[0]
            cols_p = [f"p{i}" for i in range(10, 100, 10)]
            deciles = f[cols_p].astype(float).values
            if ingresos < deciles[0]:
                pos = "dentro del 10% más bajo"
            elif ingresos > deciles[-1]:
                pos = "dentro del 10% más alto"
            else:
                pos = f"cerca del percentil {int(np.interp(ingresos, deciles, range(10, 100, 10)))}"
            st.write(f"Tu ingreso de **{peso(ingresos)}** está **{pos}** entre los/as ocupados/as de tu grupo "
                     f"({sit_ref}, {region}, nivel {edu_ref}). Mediana del grupo: **{peso(f['p50'])}** (n = {int(f['n'])} casos en la EPH 1T-2026).")
        else:
            st.write("No hay suficientes casos en la EPH para comparar tu grupo.")

    # ---- Perfil del inversor: puntaje con las 7 preguntas ----
    puntos = {"Meses": 0, "Años": 1, "Décadas": 2}[horizonte]
    puntos += {"Vendés por precaución": 0, "Esperás a que recupere": 1, "Comprás más": 2}[tolerancia]
    puntos += 2 if retencion >= 40 else (1 if retencion >= 20 else 0)
    puntos += 1 if conocimiento == "Sí" else 0
    puntos += 1 if flujo == "Fijos" else 0
    puntos += 1 if compromisos == "No" else 0
    puntos += 1 if psicologia == "La reinvierto asumiendo más riesgo" else 0
    perfil = "Conservador" if puntos <= 3 else ("Moderado" if puntos <= 6 else "Agresivo")
    st.subheader("🧭 Perfil del inversor")
    st.write(f"Puntaje: **{puntos}/10** → perfil **{perfil}**.")

    # ---- Estrategia e instrumentos ----
    st.subheader("💼 Estrategia e Instrumentos Sugeridos")
    alto = (proba_alto_riesgo is not None and proba_alto_riesgo > 0.5)

    if objetivo == "Fondo para Presupuesto Mensual o Emergencia":
        st.info("💡 **Fondos comunes de inversión Money Market:** invierten en plazos fijos y cuentas remuneradas, tienen bajo riesgo y permiten rescatar el dinero rápidamente. Su rendimiento no está garantizado.")
        mostrar_top3("emergencia", "Fondo de Emergencia")
    elif objetivo == "Generar renta y ganarle a la inflación":
        st.warning("💡 **Fondos de renta fija:** invierten en obligaciones negociables (ONs), bonos y letras, lo que permite acceder a ese mercado de forma diversificada. Tienen riesgo de crédito (el emisor puede no pagar) y de tasa de interés, y su rendimiento puede ser negativo en un período.")
        mostrar_top3("renta", "Renta y Crecimiento Moderado")
    elif objetivo == "Crecimiento agresivo del capital a largo plazo":
        if alto or perfil != "Agresivo":
            st.info("💡 **CEDEARs:** certificados que representan acciones del exterior. Cotizan en pesos pero siguen el precio de la acción en dólares, por lo que reducen la exposición al peso, aunque tienen riesgo de mercado.")
            mostrar_top3("cedears", "Crecimiento Conservador (CEDEAR)")
        else:
            st.success("💡 **Acciones argentinas:** son partes del capital de una empresa. No tienen rendimiento fijo y su precio puede caer con fuerza; son adecuadas solo para horizontes largos y perfil tolerante al riesgo.")
            mostrar_top3("acciones", "Crecimiento Agresivo (Acciones)")
    else:
        st.error("🛠️ **Plan de Saneamiento y Consolidación de Deudas:**\n\n"
                 "1. **Bola de Nieve:** ordená las deudas de menor a mayor saldo, pagá el mínimo en todas y concentrá el dinero extra en la más chica.\n\n"
                 "2. **Avalancha:** ordená las deudas por tasa de interés y liquidá primero la más cara (por ejemplo, tarjetas de crédito).\n\n"
                 "3. **Consolidación:** evaluá un préstamo a tasa fija solo si su costo financiero total (CFT) es menor que el de tus deudas actuales.")

    st.markdown("---")
    st.subheader("🛡️ ¿Dónde operar?")
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
    st.caption("⚠️ Herramienta educativa desarrollada para un trabajo práctico. No constituye asesoramiento financiero ni una recomendación de inversión. Los rankings usan datos históricos públicos y no garantizan resultados futuros.")
