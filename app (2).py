# -*- coding: utf-8 -*-
import io

import numpy as np
import pandas as pd
import streamlit as st
from mercado import top_fci, top_mercado, CEDEARS, ACCIONES, SinDatos

st.set_page_config(page_title="Asesor Financiero IA", page_icon="💳", layout="wide")

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
def ranking(clave):
    # Top 3 con datos reales. Se cachea 6 horas para no consultar las APIs en cada clic.
    if clave == "emergencia":
        return top_fci("mercadoDinero", dias=30)
    if clave == "renta":
        return top_fci("rentaFija", dias=90)
    if clave == "cedears":
        return top_mercado(CEDEARS)
    return top_mercado(ACCIONES)


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
    with st.expander("📚 Fuentes verificables de estos datos"):
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
        with st.expander(f"📖 ¿Qué es esto? {titulo} (explicación simple)"):
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
    st.error("🛠️ **Plan para ordenar tus deudas**")
    # Tasa de referencia: la mayor TNA de money market (en vivo si se puede; si no, la de referencia con fecha)
    try:
        df_mm, meta_mm = ranking("emergencia")
        tna_ref, origen = float(df_mm["TNA (%)"].iloc[0]), f"datos en vivo de CAFCI al {meta_mm['fecha']}"
    except Exception:
        tna_ref, origen = TNA_MM_REF, f"referencia de CAFCI al {FECHA_REF}"

    if cft > 0:
        if cft > tna_ref:
            st.success(f"✅ Tu deuda cuesta aproximadamente **{fmt(cft)}%** anual y el mejor money market rinde cerca de **{fmt(tna_ref)}%** de TNA ({origen}). "
                       "Cancelar la deuda 'rinde' más que invertir: **priorizá pagarla antes de invertir**.")
        else:
            st.info(f"ℹ️ Tu deuda cuesta aproximadamente **{fmt(cft)}%** anual y el mejor money market rinde cerca de **{fmt(tna_ref)}%** de TNA ({origen}). "
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
        st.warning("⚠️ Tenés cuotas atrasadas: los intereses punitorios se acumulan rápido. Contactá al acreedor cuanto antes para refinanciar.")
    if alto:
        st.warning("⚠️ Tu diagnóstico de riesgo es alto: priorizá liquidez y orden de deudas antes de tomar nuevo crédito.")
    st.write("**Dónde dejar tu colchón mínimo (fondos de liquidez inmediata):**")
    mostrar_top3("emergencia", "Fondo de Emergencia")
    st.markdown("Más información sobre tu situación crediticia: [BCRA – Central de Deudores](https://www.bcra.gob.ar)")


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
    provincia = st.selectbox("Provincia donde vivís:", list(PROVINCIAS.keys()))
    region = PROVINCIAS[provincia]
    st.caption(f"Región EPH asignada: **{region}**")
with col2:
    edu = st.selectbox("Nivel educativo máximo alcanzado:", list(EDUCACION.keys()), index=3)
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

    # ---- Diagnóstico de riesgo (solo asalariados: es la población con la que se entrenó el modelo) ----
    proba_alto_riesgo = None
    st.subheader("📊 Diagnóstico de Riesgo Crediticio")
    es_asalariado = situacion.startswith("Asalariado")
    if es_asalariado:
        datos_usuario = pd.DataFrame([[float(edad), float(ingresos), univ, sec, 1 if pareja == "Vivo en pareja / Casado/a" else 0]],
                                     columns=VARS_MODELO)
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
    if not situacion.startswith("No trabajo") and ingresos > 0:
        sit_ref = "Asalariado/a" if es_asalariado else "Cuenta propia"
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
                     f"({sit_ref}, región {region}, nivel educativo {edu_ref}). Mediana del grupo: **{peso(f['p50'])}** (n = {int(f['n'])} casos en la EPH 1T-2026).")
            st.caption(f"Tu provincia ({provincia}) se compara con la región EPH {region}.")
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
    alto = (proba_alto_riesgo is not None and proba_alto_riesgo > 0.5)

    st.subheader("🧭 Tu resultado: riesgo crediticio y perfil de inversor")
    m1, m2 = st.columns(2)
    if proba_alto_riesgo is None:
        m1.metric("Riesgo crediticio", "No estimado", "el modelo solo evalúa asalariados", delta_color="off")
    else:
        m1.metric("Riesgo crediticio", "Alto" if alto else "Bajo", f"probabilidad de alto riesgo: {proba_alto_riesgo:.1%}", delta_color="inverse" if alto else "normal")
    m2.metric("Perfil de inversor", perfil, f"puntaje {puntos}/10", delta_color="off")
    if alto:
        st.info("Como tu riesgo crediticio es alto, las recomendaciones priorizan instrumentos de bajo riesgo y liquidez aunque tu perfil sea más osado.")

    # ---- Estrategia e instrumentos ----
    st.subheader("💼 Estrategia e Instrumentos Sugeridos")

    if objetivo == "Fondo para Presupuesto Mensual o Emergencia":
        mostrar_top3("emergencia", "Fondo de Emergencia")
    elif objetivo == "Generar renta y ganarle a la inflación":
        mostrar_top3("renta", "Renta y Crecimiento Moderado")
    elif objetivo == "Crecimiento agresivo del capital a largo plazo":
        if alto or perfil != "Agresivo":
            mostrar_top3("cedears", "Crecimiento Conservador (CEDEAR)")
        else:
            mostrar_top3("acciones", "Crecimiento Agresivo (Acciones)")
    else:
        mostrar_deudas(tipo_deuda, monto_deuda, cft, atraso, alto)

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
