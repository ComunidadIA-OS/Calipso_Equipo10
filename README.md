# Calipso

**Calipso analiza biografías profesionales y señala potenciales sesgos de género.** No medimos cómo son las personas, sino qué elige contar quien escribe.

La profesión importa porque una misma descripción puede ser relevante en un contexto y señalar un potencial sesgo en otro. Por ejemplo, hablar de la apariencia física puede tener sentido al describir el trabajo de una modelo, mientras que destacar la belleza de una ingeniera suele aportar poco sobre sus capacidades o sus logros profesionales. Por eso, no buscamos palabras que sean sesgadas por sí mismas: analizamos qué relación tiene lo que se cuenta con el trabajo de la persona. Comparar dentro de una profesión ayuda a reducir las diferencias debidas al tipo de trabajo, aunque no elimina todas las diferencias entre trayectorias.

Partimos de biografías reales para identificar rasgos que aparecen con distinta frecuencia en los textos de mujeres y hombres y comprobar cuáles de esas diferencias se mantienen en muestras independientes. Calipso utiliza los rasgos confirmados como referencia para analizar una nueva biografía: muestra qué elementos aparecen, cuáles faltan respecto a lo habitual en la referencia y qué frases sustentan las señales detectadas. Una diferencia no demuestra por sí sola un sesgo; su interpretación depende del contexto y de la relevancia de la información para el trabajo descrito.

## Uso

1. Escribe una biografía en castellano o inglés, de hasta **12.000 caracteres**, o carga uno de los seis ejemplos ficticios.
2. Elige una profesión, deja que el modelo la deduzca o selecciona «Otra».
3. Pulsa **Analizar biografía** y explora la métrica, el gráfico, los rasgos detectados y las frases resaltadas.

La app compara el texto con biografías reales de **28 profesiones**. Si hay una profesión equivalente, muestra su referencia y la general; en los demás casos, utiliza solo la general. Permite revisar textos antes de publicarlos y explorar distintas formas de contar una trayectoria.

## Cómo funciona

Calipso utiliza **Gradio**, **pandas**, **NumPy** y **Matplotlib**, con el modelo **Qwen3.5-397B-A17B** servido por OVH AI Endpoints mediante una API compatible con el cliente de OpenAI.

Detecta el idioma, traduce al inglés si hace falta, obtiene la profesión y anota rasgos mediante preguntas estructuradas. Parte de la anotación utiliza nombres y pronombres enmascarados. Después calcula los resultados y recupera citas exactas del texto original.

El estudio parte de **Bias in Bios** (De-Arteaga et al., 2019), versión `LabHC/bias_in_bios`: biografías en inglés procedentes de Common Crawl de 2014 a 2018. Incluye filtrado de plantillas, muestras equilibradas y confirmación de variables en muestras independientes con corrección por comparaciones múltiples.

## Dimensiones y métrica diseñadas
Agrupamos los rasgos analizados en siete dimensiones para mostrar qué aspectos de una trayectoria se destacan al contarla. Distinguimos entre rasgos **comunales**, relacionados con los vínculos, el cuidado y la vida personal, y **agénticos**, asociados con la acción, los logros y el liderazgo. A partir de los rasgos confirmados en el estudio, la métrica combina la presencia de elementos comunales y la omisión de elementos agénticos habituales en la referencia profesional.

| Dimensión | Contenido | Lado |
| --- | --- | --- |
| M1 · Marcado de género | Género y formas de nombrar | Comunal |
| M2 · Vida personal y aspecto | Familia, aficiones y apariencia | Comunal |
| M3 · Calidez y emoción | Cuidado, calidez y pasión | Comunal |
| M4 · Temas comunales añadidos | Infancia, belleza, hogar o voluntariado | Comunal |
| M5 · Agencia y estatus | Logros, liderazgo, impacto y prestigio | Agéntico |
| M6 · Perfil técnico y empresarial | Perfil científico, tecnológico o empresarial | Agéntico |
| M7 · Agencia diluida | Mérito atribuido a otros, pasiva, apoyo y duda | Comunal |

Cuando ambos componentes están disponibles:

```text
Métrica total = 0,70 × presencia + 0,30 × omisión
```

- **Presencia:** rasgos comunales detectados, ponderados por su mayor frecuencia en biografías de mujeres de referencia.
- **Omisión:** rasgos agénticos ausentes que aparecen en al menos el 10 % de las biografías de hombres de referencia.

Ningún rasgo es bueno o malo por sí mismo: importa su relevancia profesional. Las tasas por profesión se suavizan con la referencia general y, si falta un componente, se reajustan los pesos. Solo las variables confirmadas del estudio contribuyen a la métrica; su número depende de los archivos de resultados cargados.

**La puntuación de 0 a 1 es orientativa: no es una probabilidad de discriminación ni demuestra un sesgo.** Toma valores entre 0 (ningún potencial sesgo detectado) y 1 (todos los analizados). Aumenta con la presencia de rasgos comunales y con la ausencia de rasgos agénticos habituales en hombres de la misma profesión. 

## Instalación y ejecución

Puedes utilizar la aplicación web con los datos de referencia ya preparados o ejecutar los cuadernos para reproducir el estudio. **No es necesario repetir el estudio para analizar una biografía.** Necesitas Python, conexión a internet y acceso a OVH AI Endpoints; el modelo se ejecuta en el servicio remoto.

### 1. Preparar el entorno

Descarga o clona el repositorio y abre una terminal en su carpeta. Instala las dependencias de la aplicación:

```bash
python -m pip install gradio numpy pandas python-dotenv matplotlib openai pyarrow
```

Para reconocer y enmascarar nombres de personas, instala también spaCy y su modelo de inglés:

```bash
python -m pip install spacy
python -m spacy download en_core_web_sm
```

Este paso es recomendable, aunque la web puede funcionar sin él: en ese caso solo enmascara los pronombres previstos por el código. PyArrow permite leer los datos de referencia en formato Parquet.

### 2. Configurar las credenciales

Crea un archivo `.env` junto a la aplicación:

```dotenv
OVH_AI_ENDPOINTS_URL=<URL del endpoint compatible>
OVH_AI_ENDPOINTS_ACCESS_TOKEN=<token de acceso>
```

Sustituye los valores entre `<...>` por las credenciales de tu servicio. No publiques el archivo `.env` ni los tokens en GitHub. Las llamadas pueden consumir la cuota del proveedor.

### 3. Lanzar la aplicación web

Para ejecutar la aplicación utilice:

```bash
python app_gradio.py
```

Abre la dirección que Gradio muestra en la terminal, usando el enlace público temporal.


## Archivos del repositorio

El proyecto incluye el documento de metodología, los cuadernos del estudio, un detector para explorar biografías y la aplicación web:

| Archivo | Funcionalidad |
| --- | --- |
| [Metodologia_sesgo_genero.pdf](Metodologia_sesgo_genero.pdf) | Documenta el diseño del estudio, los datos, las variables, las pruebas estadísticas, la métrica, los resultados y sus limitaciones. |
| `sesgo_genero_parte_1.ipynb` | Carga y filtra las biografías, explora diferencias mediante el clasificador N05 y realiza las primeras rondas de anotación y contraste de variables. |
| `sesgo_genero_parte_2.ipynb` | Revisa y contrasta las variables en nuevas muestras de confirmación y ampliación. Genera anotaciones, resultados estadísticos, gráficos y la métrica. |
| `detector_sesgo_genero.ipynb` | Analiza una biografía propia en inglés o una seleccionada del dataset. Muestra la métrica, el perfil por dimensiones y las frases que sustentan los rasgos detectados. |
| `app_gradio.py` | Aplicación web con Gradio para analizar biografías en castellano o inglés. Incluye ejemplos, selección de profesión, comparaciones con las referencias, gráficos, tablas y texto resaltado. |

Las partes 1 y 2 documentan y ejecutan el estudio. El detector y la aplicación web utilizan los datos de referencia generados para analizar nuevos textos.

## Hoja de ruta

Pasos realizados en el desarrollo de Calipso. El detalle y las decisiones tomadas se recogen en el documento de [metodología](Metodologia_sesgo_genero.pdf).

1. **Preparación del dataset:** cargar y depurar las biografías de Bias in Bios, eliminar perfiles generados con plantillas y preparar muestras equilibradas por profesión y género.
2. **Exploración de posibles sesgos:** analizar el dataset y combinar los patrones encontrados con la literatura para definir las variables de estudio.
3. **Revisión y confirmación de variables:** distinguir hechos profesionales de encuadres añadidos y contrastar las diferencias en nuevas muestras de biografías.
4. **Diseño de dimensiones y métrica:** agrupar los rasgos en siete dimensiones y construir una puntuación que combine presencia comunal y omisión agéntica con referencias por profesión.
5. **Análisis de una biografía individual:** desarrollar un detector que calcule la métrica, compare el texto con las referencias y muestre los rasgos detectados y sus citas.
6. **Creación de la aplicación web:** integrar el análisis en una interfaz con Gradio, con entrada en castellano o inglés, ejemplos, gráficos, tablas y texto resaltado.
  
## Licencia y componentes de terceros

**Calipso se distribuye bajo MIT License**. Los datos, modelos y materiales de terceros conservan sus propias licencias y atribuciones.


---

**Calipso · Reto Mujeres por la IA Abierta**
