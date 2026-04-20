import streamlit as st
import pandas as pd
import re
import time
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Propuestas IA", layout="wide")

theme = st.get_option("theme.base")

# =========================
# 🎨 MODO DINÁMICO
# =========================
if theme == "dark":
    bg = "#0f172a"
    card = "#1e293b"
    text = "#e2e8f0"
    accent = "#3b82f6"
    shadow = "none"
else:
    bg = "#f4f7fb"
    card = "#f4f7fb"
    text = "#1a1a1a"
    accent = "#007BFF"
    shadow = "8px 8px 16px #d1d9e6, -8px -8px 16px #ffffff"

st.markdown(f"""
<style>
body {{
    background:{bg};
    color:{text};
}}

.title {{
    text-align:center;
    font-size:42px;
    font-weight:700;
}}

.subtitle {{
    text-align:center;
    opacity:0.7;
}}

.card {{
    background:{card};
    padding:20px;
    border-radius:20px;
    box-shadow:{shadow};
    margin-bottom:20px;
    transition:0.3s;
}}

.card:hover {{
    transform: translateY(-5px);
}}

.stButton>button {{
    background: linear-gradient(135deg, {accent}, #00A8FF);
    color:white;
    border-radius:12px;
    height:50px;
}}

.loader {{
    text-align:center;
    font-size:18px;
    animation: fade 1s infinite alternate;
}}

@keyframes fade {{
    from {{opacity:0.4;}}
    to {{opacity:1;}}
}}
</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚙️ Panel IA")
url = st.sidebar.text_input("📂 CSV URL")
analizar = st.sidebar.button("🚀 Analizar")

# =========================
# HEADER
# =========================
icon = "🌙" if theme == "dark" else "☀️"

st.markdown(f'<div class="title">{icon} Dashboard Inteligente</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Análisis emocional + generación de propuestas</div>', unsafe_allow_html=True)

# =========================
# FUNCIONES IA
# =========================
def limpiar(texto):
    texto = texto.lower()
    texto = re.sub(r"[^\w\s]", "", texto)
    return texto

def mejor_k(X):
    n = X.shape[0]
    if n < 4:
        return 2
    
    best_k = 2
    best_score = -1

    for k in range(2, min(5, n-1)+1):
        try:
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = model.fit_predict(X)

            if len(set(labels)) < 2:
                continue

            score = silhouette_score(X, labels)

            if score > best_score:
                best_score = score
                best_k = k
        except:
            continue

    return best_k

# =========================
# EJECUCIÓN
# =========================
if analizar:

    if not url:
        st.warning("⚠️ Ingresa un CSV")
        st.stop()

    # Loader (UX premium)
    placeholder = st.empty()
    placeholder.markdown('<div class="loader">🤖 Analizando datos...</div>', unsafe_allow_html=True)
    time.sleep(1.5)

    try:
        df = pd.read_csv(url)
        respuestas = df.iloc[:,1].dropna().astype(str)

        respuestas_limpias = [limpiar(r) for r in respuestas]

        stopwords = ["de","la","que","el","en","y","a","los","del","se"]

        temas = []

        # =========================
        # 🧠 IA HÍBRIDA
        # =========================
        if len(respuestas) >= 5:

            vectorizer = TfidfVectorizer(stop_words=stopwords, ngram_range=(1,2))
            X = vectorizer.fit_transform(respuestas_limpias)

            k = mejor_k(X)

            modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
            modelo.fit(X)

            terms = vectorizer.get_feature_names_out()
            order = modelo.cluster_centers_.argsort()[:, ::-1]

            placeholder.empty()

            st.markdown("### 🧠 Insights detectados")
            cols = st.columns(k)

            for i in range(k):
                palabras = [terms[ind] for ind in order[i, :5]]
                palabras = [p for p in palabras if len(p) > 3]

                temas.extend(palabras)

                with cols[i]:
                    st.markdown(f"""
                    <div class="card">
                    <h4>Grupo {i+1}</h4>
                    <p>{", ".join(palabras)}</p>
                    </div>
                    """, unsafe_allow_html=True)

        else:
            placeholder.empty()

            palabras = " ".join(respuestas_limpias).split()
            conteo = Counter(p for p in palabras if p not in stopwords and len(p) > 3)

            temas = [p for p, _ in conteo.most_common(5)]

            st.markdown("### ⚡ Insights rápidos")
            st.markdown(f"<div class='card'>{', '.join(temas)}</div>", unsafe_allow_html=True)

        # =========================
        # 🎯 PROPUESTA INTELIGENTE
        # =========================
        st.markdown("### 💡 Recomendación IA")

        top = temas[:3]

        texto = "Una experiencia ideal sería "
        if top:
            texto += f"basada en {', '.join(top)}, "

        texto += "diseñada para crear un momento especial y memorable."

        st.markdown(f"""
        <div class="card" style="border-left:5px solid {accent};">
        <p style="font-size:18px;">{texto}</p>
        </div>
        """, unsafe_allow_html=True)

        # =========================
        # 📊 MÉTRICAS
        # =========================
        col1, col2 = st.columns(2)
        col1.metric("Respuestas", len(respuestas))
        col2.metric("Ideas detectadas", len(set(temas)))

    except Exception as e:
        placeholder.empty()
        st.error(f"Error: {e}")
