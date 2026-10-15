# Etiquetado Semántico y Estructural — GRESEL (UNED)

Repositorio de herramientas para el **etiquetado semántico y estructural** de los corpus de prensa histórica hispanófona del grupo **GRESEL-UNED**, proyecto **PID2023-151280OB-C22**.

---

## Contenido del repositorio

```
Etiquetado/
│
├── app_etiquetado_semantico.py        # Aplicación Streamlit de etiquetado
├── etiquetado_semantico_gresel.csv    # Resultados del etiquetado semántico
├── GUIA_ETIQUETADO_ROCIO.txt          # Guía de uso para el equipo investigador
├── requirements_etiquetado.txt        # Dependencias Python
└── README.md                          # Este archivo
```

---

## Descripción

### Etiquetado semántico

El etiquetado semántico consiste en identificar y clasificar fragmentos del corpus según las categorías del **vocabulario controlado GRESEL** (ver repositorio [prensa-franquista](https://github.com/greseluned/prensa-franquista)):

| Categoría | Descripción |
|---|---|
| `TÉRMINOS` | Conceptos del discurso franquista |
| `PERSONAS` | Figuras históricas relevantes |
| `EVENTOS` | Sucesos históricos del período |
| `LUGARES` | Topónimos de relevancia |

Los resultados se almacenan en `etiquetado_semantico_gresel.csv`.

### Etiquetado estructural

*(En desarrollo)* — Identificación de la estructura interna de los artículos: titulares, subtítulos, cuerpo de texto, pies de foto, anuncios publicitarios, etc.

---

## Instalación

```bash
# Clona el repositorio
git clone https://github.com/greseluned/Etiquetado.git
cd Etiquetado

# Instala las dependencias
pip install -r requirements_etiquetado.txt

# Ejecuta la aplicación
streamlit run app_etiquetado_semantico.py
```

La aplicación se abrirá en el navegador en `http://localhost:8501`.

---

## Uso

Consulta el archivo `GUIA_ETIQUETADO_ROCIO.txt` para las instrucciones detalladas de uso de la herramienta, pensadas para el equipo investigador sin perfil técnico.

En resumen:

1. Abre la aplicación con el comando de arriba
2. Carga o selecciona un fragmento del corpus
3. Asigna la categoría semántica correspondiente del vocabulario GRESEL
4. Guarda el etiquetado — se añade automáticamente al CSV de resultados

---

## Resultados actuales

El archivo `etiquetado_semantico_gresel.csv` contiene los resultados del etiquetado realizado hasta la fecha sobre el corpus disponible. El etiquetado está **en curso** — se irá ampliando a medida que se procesen más publicaciones.

---

## Relación con otros repositorios

| Repositorio | Descripción |
|---|---|
| [greseluned/prensa-franquista](https://github.com/greseluned/prensa-franquista) | Vocabulario controlado NER y sistema RAG |

---

## Proyecto

- **Grupo**: GRESEL-UNED
- **Proyecto**: PID2023-151280OB-C22
- **IP**: Rocío Ortuño Casanova (UNED)
- **Desarrollo técnico**: Fernando Obispo Permuy (UNED)
- **Web**: [gresel-uned.hypotheses.org](https://gresel-uned.hypotheses.org)
