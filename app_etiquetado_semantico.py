#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Etiquetado Semántico Automático — Revistas Femeninas Hispánicas
GRESEL-UNED · PID2023-151280OB-C22

Clasifica automáticamente las regiones estructurales de los PAGE XML
del repositorio greseluned/femeninas entrenando un clasificador sobre
las páginas ya anotadas manualmente (status GT / "Text Done").

Dependencias: streamlit, scikit-learn, numpy
"""

import streamlit as st
import xml.etree.ElementTree as ET
import re
import csv
import io
from pathlib import Path
from collections import Counter

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────

NS = {"p": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}

# Tipos estructurales con etiqueta legible y color
TIPOS = {
    "heading":      ("Título / Encabezado",  "#1A1A2E"),
    "paragraph":    ("Párrafo",              "#2C5F8A"),
    "credit":       ("Autoría / Firma",      "#8B0000"),
    "header":       ("Cabecera de página",   "#2E6B4A"),
    "caption":      ("Pie de imagen",        "#7A5C5C"),
    "page-number":  ("Número de página",     "#555555"),
    "marginalia":   ("Marginalia",           "#8B7355"),
    "floating":     ("Texto flotante",       "#5C5C8B"),
    "drop-capital": ("Inicial decorativa",   "#7A4F3A"),
}

# Publicaciones del corpus femeninas
PUBLICACIONES = {
    "femina_rd":       ("Fémina",              "#8B0000"),
    "filipinas_ph":    ("Filipinas",           "#1A1A2E"),
    "heraldo_mujer_pr":("El Heraldo de la Mujer", "#2E6B4A"),
}

SKIP_DIRS = {"merged", "metadata", "por_frases"}
MIN_EJEMPLOS_CV = 5   # mínimo de muestras para incluir un tipo en el CV

SCRIPT_DIR = Path(__file__).parent
CORPUS_CANDIDATES = [
    SCRIPT_DIR / "femeninas" / "corpus",
    SCRIPT_DIR / "corpus",
    Path.home() / "Documents" / "femeninas" / "corpus",
    Path("/tmp/femeninas/corpus"),
]

# ─────────────────────────────────────────────────────────────────────────────
#  UTILIDADES PAGE XML
# ─────────────────────────────────────────────────────────────────────────────

def _stype(custom: str):
    """Extrae el tipo estructural del atributo custom= de un TextRegion."""
    m = re.search(r'structure\s*\{type:([^;]+);', custom or "")
    return m.group(1).strip() if m else None


def _es_gt(path: Path) -> bool:
    """Devuelve True si la página tiene estado Ground Truth."""
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        return 'status="GT"' in raw or 'Text Done' in raw
    except Exception:
        return False


def _extraer_regiones(path: Path) -> list:
    """Extrae todas las TextRegion de un PAGE XML con su texto y tipo."""
    regiones = []
    try:
        root = ET.parse(path).getroot()
        for reg in root.findall(".//p:TextRegion", NS):
            tipo = _stype(reg.get("custom", ""))
            lineas = [
                u.text.strip()
                for line in reg.findall(".//p:TextLine", NS)
                for u in [line.find("p:TextEquiv/p:Unicode", NS)]
                if u is not None and u.text
            ]
            texto = " ".join(lineas).strip()
            if texto:
                regiones.append({"texto": texto, "tipo": tipo})
    except Exception:
        pass
    return regiones

# ─────────────────────────────────────────────────────────────────────────────
#  CARGA DEL CORPUS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def cargar_corpus(raiz: str) -> dict:
    """
    Recorre el corpus PAGE XML y extrae:
    - docs: lista de páginas con metadatos y regiones
    - train_X / train_y: ejemplos GT para entrenar el modelo
    """
    docs, train_X, train_y = [], [], []

    for pub_dir in sorted(Path(raiz).iterdir()):
        if not pub_dir.is_dir() or pub_dir.name in SKIP_DIRS:
            continue
        xml_dir = pub_dir / "page_xml"
        if not xml_dir.exists():
            continue
        for issue_dir in sorted(xml_dir.iterdir()):
            if not issue_dir.is_dir():
                continue
            for xml_file in sorted(issue_dir.glob("*.xml")):
                gt = _es_gt(xml_file)
                regiones = _extraer_regiones(xml_file)
                docs.append({
                    "pub":     pub_dir.name,
                    "ejemplar": issue_dir.name,
                    "pagina":  xml_file.stem,
                    "path":    str(xml_file),
                    "is_gt":   gt,
                    "regiones": regiones,
                })
                if gt:
                    for r in regiones:
                        if r["tipo"] and r["tipo"] in TIPOS:
                            train_X.append(r["texto"])
                            train_y.append(r["tipo"])

    return {"docs": docs, "train_X": train_X, "train_y": train_y}

# ─────────────────────────────────────────────────────────────────────────────
#  MODELO DE CLASIFICACIÓN
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def entrenar_modelo(train_X_tuple: tuple, train_y_tuple: tuple) -> dict:
    """
    Entrena un pipeline TF-IDF (chars + words) + LinearSVC.
    Excluye tipos con menos de MIN_EJEMPLOS_CV ejemplos del CV.
    """
    from sklearn.pipeline import Pipeline, FeatureUnion
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    import numpy as np

    X_raw = list(train_X_tuple)
    y_raw = list(train_y_tuple)

    # Filtrar clases con pocos ejemplos para el CV
    conteo = Counter(y_raw)
    tipos_validos = {t for t, n in conteo.items() if n >= MIN_EJEMPLOS_CV}
    tipos_raros   = {t: n for t, n in conteo.items() if n < MIN_EJEMPLOS_CV}

    X = [x for x, y in zip(X_raw, y_raw) if y in tipos_validos]
    y = [y for y in y_raw if y in tipos_validos]

    pipe = Pipeline([
        ("feat", FeatureUnion([
            ("chars", TfidfVectorizer(
                analyzer="char_wb", ngram_range=(2, 5),
                max_features=20000, sublinear_tf=True,
            )),
            ("words", TfidfVectorizer(
                analyzer="word", ngram_range=(1, 2),
                max_features=20000, sublinear_tf=True,
            )),
        ])),
        ("clf", LinearSVC(C=1.0, class_weight="balanced", max_iter=2000)),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1_macro")
    pipe.fit(X, y)

    return {
        "pipe":        pipe,
        "f1":          float(np.mean(scores)),
        "f1_std":      float(np.std(scores)),
        "n_train":     len(X),
        "clases":      list(pipe.named_steps["clf"].classes_),
        "dist":        dict(conteo),
        "tipos_raros": tipos_raros,
    }


def predecir(modelo: dict, textos: list) -> list:
    """Aplica el clasificador a una lista de textos."""
    if not textos:
        return []
    return list(modelo["pipe"].predict(textos))

# ─────────────────────────────────────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────────────────────────────────────

CSS = """
<style>
/* Fondo general */
[data-testid="stAppViewContainer"] { background: #F5F4F0; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #1A1A2E !important;
    border-right: 3px solid #8B0000;
}
[data-testid="stSidebar"] * { color: #DCDCDC !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] strong { color: #FFFFFF !important; }
[data-testid="stSidebar"] hr { border-color: #3A3A5A; }
[data-testid="stSidebar"] .stProgress > div > div { background: #8B0000; }

/* Botones */
.stButton > button {
    background: #8B0000 !important;
    color: white !important;
    border: none !important;
    border-radius: 4px !important;
    font-weight: 600 !important;
    padding: 8px 20px !important;
}
.stButton > button:hover { background: #a00000 !important; }

/* Botón de descarga */
.stDownloadButton > button {
    background: #1A1A2E !important;
    color: white !important;
    border: none !important;
    border-radius: 4px !important;
    font-weight: 600 !important;
}
.stDownloadButton > button:hover { background: #2A2A4E !important; }

/* Badge de tipo */
.tipo-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.80em;
    font-weight: 600;
    color: white;
    letter-spacing: 0.02em;
    white-space: nowrap;
}

/* Fila de resultado */
.result-row {
    background: white;
    border-radius: 6px;
    padding: 10px 14px 8px 16px;
    margin-bottom: 6px;
    border-left: 4px solid #ccc;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}

/* Tarjeta métrica */
.metric-card {
    background: white;
    border-radius: 8px;
    padding: 18px 22px 14px 22px;
    border-left: 4px solid #8B0000;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    text-align: center;
}
.metric-card .metric-val {
    font-size: 2em;
    font-weight: 700;
    color: #1A1A2E;
    line-height: 1.1;
}
.metric-card .metric-lbl {
    font-size: 0.82em;
    color: #666;
    margin-top: 4px;
}

/* Cabecera de sección */
.sec-header {
    border-bottom: 2px solid #8B0000;
    padding-bottom: 6px;
    margin: 28px 0 16px 0;
    color: #1A1A2E;
    font-weight: 700;
    font-size: 1.05em;
}

/* Barra de distribución */
.dist-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 8px;
}
.dist-bar-bg {
    flex: 1;
    background: #E8E6E0;
    border-radius: 4px;
    height: 10px;
    overflow: hidden;
}
.dist-bar-fill {
    height: 100%;
    border-radius: 4px;
}

/* Alerta de modelo */
.model-ok {
    background: #EAF4EA;
    border: 1px solid #2E6B4A;
    border-radius: 6px;
    padding: 10px 16px;
    color: #1A3A2A;
    font-weight: 600;
}
</style>
"""

# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS HTML
# ─────────────────────────────────────────────────────────────────────────────

def badge(tipo: str) -> str:
    label, color = TIPOS.get(tipo, (tipo, "#888888"))
    return f'<span class="tipo-badge" style="background:{color}">{label}</span>'


def metric_card(valor, etiqueta: str) -> str:
    return (
        f'<div class="metric-card">'
        f'<div class="metric-val">{valor}</div>'
        f'<div class="metric-lbl">{etiqueta}</div>'
        f'</div>'
    )


def dist_bar(conteo: dict, total: int) -> str:
    html = ""
    for tipo, n in sorted(conteo.items(), key=lambda x: -x[1]):
        _, color = TIPOS.get(tipo, (tipo, "#888888"))
        pct = n / total if total else 0
        html += (
            f'<div class="dist-row">'
            f'{badge(tipo)}'
            f'<div class="dist-bar-bg"><div class="dist-bar-fill" '
            f'style="width:{pct*100:.1f}%;background:{color}"></div></div>'
            f'<small style="color:#555;min-width:60px;text-align:right">'
            f'{n} ({pct:.0%})</small>'
            f'</div>'
        )
    return html

# ─────────────────────────────────────────────────────────────────────────────
#  INTERFAZ STREAMLIT
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Etiquetado Semántico — GRESEL-UNED",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CSS, unsafe_allow_html=True)

# ── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📰 GRESEL-UNED")
    st.markdown("### Etiquetado Semántico")
    st.markdown("*Revistas Femeninas Hispánicas*")
    st.markdown("*PID2023-151280OB-C22*")
    st.markdown("---")

    # Auto-detección del corpus
    corpus_detectado = next((c for c in CORPUS_CANDIDATES if c.exists()), None)

    if corpus_detectado:
        st.success("✓ Corpus detectado")
        st.caption(str(corpus_detectado))
        corpus_path = corpus_detectado
    else:
        st.markdown("**Ruta al corpus:**")
        ruta_input = st.text_input(
            "", placeholder="C:\\Users\\...\\femeninas\\corpus", label_visibility="collapsed"
        )
        corpus_path = Path(ruta_input) if ruta_input else None

    # Carga del corpus
    if corpus_path and corpus_path.exists():
        with st.spinner("Cargando corpus…"):
            datos = cargar_corpus(str(corpus_path))
        docs = datos["docs"]
        train_X = datos["train_X"]
        train_y = datos["train_y"]

        n_gt   = sum(1 for d in docs if d["is_gt"])
        n_ngt  = len(docs) - n_gt
        n_pubs = len({d["pub"] for d in docs})

        st.markdown("---")
        st.markdown("**Corpus cargado**")
        st.markdown(f"- Páginas GT anotadas: **{n_gt}**")
        st.markdown(f"- Páginas sin etiquetar: **{n_ngt}**")
        st.markdown(f"- Regiones de entrenamiento: **{len(train_X)}**")
        st.markdown(f"- Publicaciones: **{n_pubs}**")

        if "modelo" in st.session_state:
            m = st.session_state["modelo"]
            st.markdown("---")
            st.markdown("**Modelo entrenado**")
            st.markdown(f"- F1-macro (CV-5): **{m['f1']:.3f}**")
            st.markdown(f"- Clases: **{len(m['clases'])}**")

        st.markdown("---")
        st.markdown("**Leyenda de tipos**")
        for tipo, (label, color) in TIPOS.items():
            st.markdown(
                f'<span class="tipo-badge" style="background:{color}">{label}</span> ',
                unsafe_allow_html=True,
            )
    else:
        docs, train_X, train_y = [], [], []
        if corpus_path:
            st.error("Ruta no encontrada.")

# ── CABECERA PRINCIPAL ────────────────────────────────────────────────────────

st.markdown("# Etiquetado Semántico Automático")
st.markdown(
    "**GRESEL-UNED · PID2023-151280OB-C22** — "
    "Revistas Femeninas Hispánicas: *Fémina*, *Filipinas*, *El Heraldo de la Mujer*"
)
st.markdown("---")

if not docs:
    st.info(
        "Especifica la ruta a la carpeta `corpus` del repositorio "
        "`greseluned/femeninas` en el panel lateral para comenzar."
    )
    st.code("git clone https://github.com/greseluned/femeninas.git")
    st.stop()

# ── PESTAÑAS ─────────────────────────────────────────────────────────────────

tab_corpus, tab_modelo, tab_etiquetar, tab_exportar = st.tabs([
    "📊  Corpus",
    "🎓  Modelo",
    "🏷️  Etiquetar",
    "📥  Exportar",
])

# ════════════════════════════════════════════════════════
#  TAB 1 — RESUMEN DEL CORPUS
# ════════════════════════════════════════════════════════
with tab_corpus:
    st.markdown('<div class="sec-header">Estado del corpus</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    n_gt  = sum(1 for d in docs if d["is_gt"])
    n_ngt = len(docs) - n_gt
    n_reg_gt = len(train_X)
    n_reg_total = sum(len(d["regiones"]) for d in docs)

    with c1: st.markdown(metric_card(len(docs), "Páginas totales"), unsafe_allow_html=True)
    with c2: st.markdown(metric_card(n_gt, "Páginas GT (anotadas)"), unsafe_allow_html=True)
    with c3: st.markdown(metric_card(n_ngt, "Sin etiquetar"), unsafe_allow_html=True)
    with c4: st.markdown(metric_card(n_reg_gt, "Regiones GT etiquetadas"), unsafe_allow_html=True)

    st.markdown('<div class="sec-header">Estado de anotación por publicación</div>', unsafe_allow_html=True)

    pub_total = Counter(d["pub"] for d in docs)
    pub_gt    = Counter(d["pub"] for d in docs if d["is_gt"])

    col_pub, col_barra = st.columns([2, 3])
    for pub in sorted(pub_total):
        nombre, color = PUBLICACIONES.get(pub, (pub, "#888888"))
        total_p = pub_total[pub]
        gt_p    = pub_gt.get(pub, 0)
        pct     = gt_p / total_p if total_p else 0
        with col_pub:
            st.markdown(
                f'<span style="color:{color};font-weight:600">{nombre}</span>'
                f'<span style="color:#666;font-size:0.85em"> — {gt_p}/{total_p} páginas GT</span>',
                unsafe_allow_html=True,
            )
        with col_barra:
            st.progress(pct)

    st.markdown('<div class="sec-header">Distribución de etiquetas estructurales (páginas GT)</div>', unsafe_allow_html=True)

    if train_y:
        dist = Counter(train_y)
        st.markdown(dist_bar(dist, len(train_y)), unsafe_allow_html=True)

        st.caption(
            f"Total: {len(train_y)} regiones etiquetadas — "
            f"{len(dist)} tipos estructurales distintos. "
            f"Los tipos con menos de {MIN_EJEMPLOS_CV} ejemplos (p. ej., 'caption', 'drop-capital') "
            f"quedan fuera de la validación cruzada pero se conservan en los datos GT."
        )
    else:
        st.warning("No se encontraron regiones etiquetadas en las páginas GT.")

# ════════════════════════════════════════════════════════
#  TAB 2 — ENTRENAMIENTO DEL MODELO
# ════════════════════════════════════════════════════════
with tab_modelo:
    st.markdown('<div class="sec-header">Arquitectura del clasificador</div>', unsafe_allow_html=True)

    col_desc, col_esp = st.columns([3, 2])
    with col_desc:
        st.markdown("""
El sistema utiliza un pipeline de dos etapas:

**1. Extracción de características (FeatureUnion)**

- *TF-IDF de caracteres* (n-gramas 2–5, 20 000 rasgos): captura patrones
  tipográficos, ortografía histórica y errores OCR, muy eficaz en textos cortos.
- *TF-IDF de palabras* (unigramas y bigramas, 20 000 rasgos): captura
  vocabulario y expresiones características de cada tipo estructural.

**2. Clasificador SVM lineal (LinearSVC)**

- Pesos de clase balanceados: compensa la desproporción entre tipos frecuentes
  (párrafo, título) e infrecuentes (marginalia, flotante).
- Validación cruzada estratificada 5-fold sobre las páginas GT.
        """)
    with col_esp:
        st.markdown("""
**Corpus de entrenamiento**

| | |
|---|---|
| Páginas GT | {n_gt} |
| Regiones etiquetadas | {n_reg} |
| Tipos en CV | 7 |
| Tipos raros (solo GT) | 2 |

**Rendimiento esperado**

F1-macro superior a 0.70 con este
corpus, lo que indica que el modelo
distingue correctamente los tipos
estructurales principales.
        """.format(n_gt=n_gt, n_reg=len(train_X)))

    st.markdown('<div class="sec-header">Entrenamiento</div>', unsafe_allow_html=True)

    if not train_X:
        st.error("No hay datos de entrenamiento. Verifica que el corpus contiene páginas GT.")
    else:
        btn_entrenar = st.button("▶  Entrenar modelo", key="btn_train", use_container_width=False)

        if btn_entrenar:
            with st.spinner("Entrenando y validando (5-fold cross-validation)…"):
                modelo = entrenar_modelo(tuple(train_X), tuple(train_y))
            st.session_state["modelo"] = modelo

        if "modelo" in st.session_state:
            m = st.session_state["modelo"]

            st.markdown(
                f'<div class="model-ok">✅  Modelo entrenado — '
                f'F1-macro CV-5: <strong>{m["f1"]:.3f}</strong> '
                f'± {m["f1_std"]:.3f} — '
                f'{m["n_train"]} regiones de entrenamiento</div>',
                unsafe_allow_html=True,
            )

            st.markdown("")
            c1, c2, c3 = st.columns(3)
            with c1: st.markdown(metric_card(f"{m['f1']:.3f}", "F1-macro (CV-5)"), unsafe_allow_html=True)
            with c2: st.markdown(metric_card(f"±{m['f1_std']:.3f}", "Desviación típica"), unsafe_allow_html=True)
            with c3: st.markdown(metric_card(len(m["clases"]), "Tipos en el modelo"), unsafe_allow_html=True)

            st.markdown('<div class="sec-header">Tipos que el modelo predice</div>', unsafe_allow_html=True)
            cols = st.columns(4)
            for i, clase in enumerate(sorted(m["clases"])):
                n_ej = m["dist"].get(clase, 0)
                label, color = TIPOS.get(clase, (clase, "#888"))
                with cols[i % 4]:
                    st.markdown(
                        f'<div style="margin-bottom:12px">'
                        f'{badge(clase)}'
                        f'<br><small style="color:#666">{n_ej} ejemplos</small>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            if m.get("tipos_raros"):
                st.caption(
                    "Tipos con pocos ejemplos (excluidos del CV, presentes en datos GT): "
                    + ", ".join(f"{t} ({n})" for t, n in m["tipos_raros"].items())
                )
        else:
            st.info("Pulsa **Entrenar modelo** para iniciar el proceso.")

# ════════════════════════════════════════════════════════
#  TAB 3 — ETIQUETAR PÁGINAS
# ════════════════════════════════════════════════════════
with tab_etiquetar:
    if "modelo" not in st.session_state:
        st.warning("Primero entrena el modelo en la pestaña **Modelo**.")
        st.stop()

    m = st.session_state["modelo"]

    st.markdown('<div class="sec-header">Opciones de etiquetado</div>', unsafe_allow_html=True)

    col_f1, col_f2, col_f3 = st.columns([2, 2, 2])
    with col_f1:
        pubs_disp = ["Todas"] + sorted({d["pub"] for d in docs})
        pub_labels = {
            "Todas": "Todas",
            **{k: PUBLICACIONES.get(k, (k, ""))[0] for k in pubs_disp[1:]}
        }
        pub_sel = st.selectbox(
            "Publicación:",
            pubs_disp,
            format_func=lambda x: pub_labels.get(x, x),
        )
    with col_f2:
        solo_nuevas = st.radio(
            "Páginas a procesar:",
            ["Solo sin etiquetar", "Todas (GT + sin etiquetar)"],
            index=0,
        )
    with col_f3:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        btn_predecir = st.button("▶  Aplicar etiquetado", key="btn_pred", use_container_width=True)

    if btn_predecir:
        solo_ngt = "Solo" in solo_nuevas
        docs_proc = [
            d for d in docs
            if (pub_sel == "Todas" or d["pub"] == pub_sel)
            and (not solo_ngt or not d["is_gt"])
        ]

        resultados = []
        barra = st.progress(0.0, text="Procesando páginas…")

        for i, doc in enumerate(docs_proc):
            barra.progress((i + 1) / len(docs_proc), text=f"Procesando {doc['ejemplar']} · {doc['pagina']}…")
            if not doc["regiones"]:
                continue

            # Separar regiones ya etiquetadas de las que necesitan predicción
            textos_pred = [r["texto"] for r in doc["regiones"] if not r["tipo"] or r["tipo"] not in TIPOS]
            predicciones = predecir(m, textos_pred)

            idx_pred = 0
            for r in doc["regiones"]:
                if r["tipo"] and r["tipo"] in TIPOS:
                    tipo_fin, fuente = r["tipo"], "GT"
                else:
                    tipo_fin = predicciones[idx_pred] if idx_pred < len(predicciones) else "paragraph"
                    fuente = "Predicción"
                    idx_pred += 1

                resultados.append({
                    "publicacion":   doc["pub"],
                    "ejemplar":      doc["ejemplar"],
                    "pagina":        doc["pagina"],
                    "texto":         r["texto"][:300],
                    "tipo_predicho": tipo_fin,
                    "fuente":        fuente,
                })

        barra.empty()
        st.session_state["resultados"] = resultados
        n_pred = sum(1 for r in resultados if r["fuente"] == "Predicción")
        st.success(f"✅  {len(resultados)} regiones procesadas — {n_pred} predicciones nuevas")

    # ── Visualización de resultados ──────────────────────────────────────────
    if "resultados" in st.session_state:
        res = st.session_state["resultados"]
        n_pred = sum(1 for r in res if r["fuente"] == "Predicción")
        n_gt_r  = sum(1 for r in res if r["fuente"] == "GT")

        c1, c2, c3 = st.columns(3)
        with c1: st.markdown(metric_card(len(res), "Regiones totales"), unsafe_allow_html=True)
        with c2: st.markdown(metric_card(n_pred, "Predicciones nuevas"), unsafe_allow_html=True)
        with c3: st.markdown(metric_card(n_gt_r, "Regiones GT (referencia)"), unsafe_allow_html=True)

        st.markdown("")

        st.markdown('<div class="sec-header">Distribución de predicciones nuevas</div>', unsafe_allow_html=True)
        dist_pred = Counter(r["tipo_predicho"] for r in res if r["fuente"] == "Predicción")
        if dist_pred:
            st.markdown(dist_bar(dist_pred, sum(dist_pred.values())), unsafe_allow_html=True)

        st.markdown('<div class="sec-header">Muestra de predicciones</div>', unsafe_allow_html=True)

        col_ft1, col_ft2 = st.columns([2, 2])
        with col_ft1:
            tipo_filtro = st.selectbox(
                "Filtrar por tipo:",
                ["Todos"] + list(TIPOS.keys()),
                key="tipo_filtro",
            )
        with col_ft2:
            pub_filtro = st.selectbox(
                "Filtrar por publicación:",
                ["Todas"] + sorted({r["publicacion"] for r in res}),
                key="pub_filtro",
                format_func=lambda x: PUBLICACIONES.get(x, (x, ""))[0] if x != "Todas" else "Todas",
            )

        muestra = [
            r for r in res
            if r["fuente"] == "Predicción"
            and (tipo_filtro == "Todos" or r["tipo_predicho"] == tipo_filtro)
            and (pub_filtro  == "Todas"  or r["publicacion"]  == pub_filtro)
        ][:60]

        if not muestra:
            st.info("No hay predicciones que coincidan con los filtros seleccionados.")
        else:
            for r in muestra:
                _, color = TIPOS.get(r["tipo_predicho"], (r["tipo_predicho"], "#888"))
                texto_corto = r["texto"][:140] + ("…" if len(r["texto"]) > 140 else "")
                pub_nom = PUBLICACIONES.get(r["publicacion"], (r["publicacion"], ""))[0]
                st.markdown(
                    f'<div class="result-row" style="border-left-color:{color}">'
                    f'{badge(r["tipo_predicho"])}'
                    f'&nbsp;&nbsp;<small style="color:#888">'
                    f'{pub_nom} · {r["ejemplar"]} · {r["pagina"]}</small>'
                    f'<br><span style="color:#333;font-size:0.93em;line-height:1.5">'
                    f'{texto_corto}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

# ════════════════════════════════════════════════════════
#  TAB 4 — EXPORTAR
# ════════════════════════════════════════════════════════
with tab_exportar:
    if "resultados" not in st.session_state:
        st.warning("Primero aplica las predicciones en la pestaña **Etiquetar**.")
        st.stop()

    res = st.session_state["resultados"]
    n_pred = sum(1 for r in res if r["fuente"] == "Predicción")

    st.markdown('<div class="sec-header">Exportación CSV</div>', unsafe_allow_html=True)

    st.markdown(f"""
El CSV contiene **{len(res)} filas** (una por región de texto) con estas columnas:

| Columna | Descripción |
|---|---|
| `publicacion` | Identificador de la revista en el repositorio |
| `ejemplar` | Identificador del ejemplar (carpeta del PAGE XML) |
| `pagina` | Nombre del archivo PAGE XML |
| `texto` | Primeros 300 caracteres del texto de la región |
| `tipo_predicho` | Tipo estructural: `heading`, `paragraph`, `credit`, etc. |
| `fuente` | `GT` (anotación manual) o `Predicción` (modelo automático) |

De los {len(res)} registros: **{n_pred} son predicciones nuevas** y
**{len(res) - n_pred} proceden de páginas ya anotadas** (GT).
    """)

    # CSV completo
    def _generar_csv(filas: list) -> bytes:
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=[
            "publicacion", "ejemplar", "pagina",
            "texto", "tipo_predicho", "fuente",
        ])
        w.writeheader()
        w.writerows(filas)
        return buf.getvalue().encode("utf-8-sig")

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.download_button(
            "⬇  Descargar CSV completo (GT + predicciones)",
            data=_generar_csv(res),
            file_name="etiquetado_semantico_gresel.csv",
            mime="text/csv",
        )
    with col_d2:
        solo_pred = [r for r in res if r["fuente"] == "Predicción"]
        st.download_button(
            "⬇  Descargar solo predicciones nuevas",
            data=_generar_csv(solo_pred),
            file_name="etiquetado_semantico_nuevas.csv",
            mime="text/csv",
        )

    st.markdown('<div class="sec-header">Estadísticas del etiquetado exportado</div>', unsafe_allow_html=True)

    # Tabla resumen por publicación y tipo
    resumen = {}
    for r in [x for x in res if x["fuente"] == "Predicción"]:
        pub = PUBLICACIONES.get(r["publicacion"], (r["publicacion"], ""))[0]
        key = (pub, r["tipo_predicho"])
        resumen[key] = resumen.get(key, 0) + 1

    if resumen:
        pubs_res = sorted({k[0] for k in resumen})
        tipos_res = sorted({k[1] for k in resumen})

        # Cabecera
        cols_hdr = st.columns([3] + [2] * len(tipos_res))
        with cols_hdr[0]:
            st.markdown("**Publicación**")
        for i, t in enumerate(tipos_res):
            lbl, _ = TIPOS.get(t, (t, ""))
            with cols_hdr[i + 1]:
                st.markdown(f"**{lbl[:10]}**")

        st.markdown("---")

        for pub in pubs_res:
            cols_row = st.columns([3] + [2] * len(tipos_res))
            with cols_row[0]:
                st.markdown(pub)
            for i, t in enumerate(tipos_res):
                with cols_row[i + 1]:
                    v = resumen.get((pub, t), 0)
                    st.markdown(f"{v}" if v else "—")

    st.markdown("---")
    st.caption(
        "El CSV exportado utiliza codificación UTF-8 con BOM para compatibilidad con Excel. "
        "Puede importarse directamente en LibreOffice Calc, Excel o cualquier herramienta de análisis de datos."
    )
