# Etiquetado

Metodología para doble etiquetado de prensa histórica: estructural y semántico, para enriquecer RAG.

Proyecto **GRESEL-UNED** · PID2023-151280OB-C22

---

## Estado actual

| Línea | Estado |
|---|---|
| Etiquetado estructural automático (revistas femeninas) | Prototipo funcional con métrica de validación |
| Etiquetado semántico (temas y entidades) | Pendiente; depende de los vocabularios controlados del proyecto |

---

## Herramienta de etiquetado estructural

`app_etiquetado_semantico.py` es una aplicación Streamlit que clasifica automáticamente las **regiones de texto** de los archivos PAGE XML del repositorio [greseluned/femeninas](https://github.com/greseluned/femeninas), aprendiendo de las páginas ya anotadas manualmente en Transkribus.

> Aunque el archivo se llama «etiquetado semántico», lo que hace hoy es asignar a cada región su **tipo estructural** (título, párrafo, firma, etc.).

### Corpus

| Publicación | País | Páginas |
|---|---|---|
| *Fémina* | República Dominicana | 377 |
| *Filipinas* | Filipinas | 98 |
| *El Heraldo de la Mujer* | Puerto Rico | 192 |

### Método

1. Se leen los PAGE XML y se seleccionan las páginas con estado Ground Truth: **183 páginas**, con **365 regiones** etiquetadas manualmente.
2. Se entrena un clasificador **TF-IDF (caracteres + palabras) + LinearSVC** (scikit-learn, `class_weight="balanced"`).
3. Se valida con **validación cruzada estratificada de 5 particiones**.
4. Se aplica el modelo al resto de páginas del corpus.

### Resultados

| Indicador | Valor |
|---|---|
| F1-macro (validación cruzada 5-fold) | **0,735 ± 0,065** |
| Tipos estructurales en el modelo | 7 |
| Regiones etiquetadas en el CSV de resultados | 3.688 |
| Páginas en el CSV de resultados | 480 |

Tipos: `heading` (título/encabezado), `paragraph` (párrafo), `credit` (autoría/firma), `header` (cabecera de página), `page-number` (número de página), `marginalia` y `floating` (texto flotante). El tipo «pie de imagen» tiene muy pocos ejemplos en el Ground Truth y no forma parte del modelo por ahora.

### Archivos

| Archivo | Descripción |
|---|---|
| `app_etiquetado_semantico.py` | Aplicación Streamlit (pestañas Corpus, Modelo, Etiquetar y Exportar) |
| `etiquetado_semantico_gresel.csv` | Resultados: publicación, ejemplar, página, texto, tipo predicho y origen (GT o predicción) |
| `GUIA_ETIQUETADO_ROCIO.txt` | Guía de instalación y uso paso a paso |
| `requirements_etiquetado.txt` | Dependencias (streamlit, scikit-learn, numpy) |

### Uso

```bash
git clone https://github.com/greseluned/femeninas.git
git clone https://github.com/greseluned/Etiquetado.git
cd Etiquetado
pip install -r requirements_etiquetado.txt
streamlit run app_etiquetado_semantico.py
```

Si la app no detecta el corpus automáticamente, se puede indicar la ruta a `femeninas/corpus` en el panel lateral.

---

## Próximos pasos

- Revisar con el equipo los tipos estructurales (por ejemplo, si «poema» debe ser un tipo propio).
- Aumentar el número de páginas anotadas para mejorar el modelo.
- Valorar la escritura de las predicciones directamente en los PAGE XML para revisarlas en Transkribus.
- Definir el etiquetado semántico (temas y entidades) a partir de los vocabularios controlados del proyecto e integrarlo en el RAG.

---

Colaborador técnico: Fernando Obispo Permuy · Investigadora principal: Rocío Ortuño Casanova
