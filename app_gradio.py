"""
Señales de género · aplicación para analizar una biografía (en inglés o en castellano).

Usa las variables, dimensiones y métrica del estudio (parte 2: confirmación y ampliación).
Necesita estar en la misma carpeta que los cuadernos, porque lee:
  - anotaciones_ampliacion.csv, anotaciones_confirmacion.csv (biografías de referencia)
  - cache/muestra_ampliacion.parquet, cache/muestra_confirmacion.parquet
  - resultados/confirmatorio_ampliacion.csv (variables confirmadas)
La dirección y el token del modelo se leen del archivo .env.

Para lanzarla, desde un terminal en esta carpeta:  python app_gradio.py
"""
import html
import inspect
import json
import os
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import gradio as gr
from dotenv import load_dotenv
from matplotlib.figure import Figure
from openai import OpenAI

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
for env in [BASE / ".env", BASE.parent / ".env", Path("/workspace/equipo10/.env")]:
    if env.exists():
        load_dotenv(env, override=True)
        break
CACHE, RESULTADOS = BASE / "cache", BASE / "resultados"
MODELO = "Qwen3.5-397B-A17B"            # el mismo modelo que en el estudio
FASE = "ampliacion"                      # variables confirmadas en la ampliación (FDR 0,10)
REFERENCIA = ["ampliacion", "confirmacion"]
VARIABLES_METRICA = "confirmadas"
PESO_PRESENCIA, PESO_OMISION = 0.70, 0.30
MIN_TASA_HOMBRES = 10
K_SUAVIZADO = 40
MAX_CARACTERES = 12000

PROFESIONES = ["accountant", "architect", "attorney", "chiropractor", "comedian", "composer",
               "dentist", "dietitian", "dj", "filmmaker", "interior_designer", "journalist",
               "model", "nurse", "painter", "paralegal", "pastor", "personal_trainer",
               "photographer", "physician", "poet", "professor", "psychologist", "rapper",
               "software_engineer", "surgeon", "teacher", "yoga_teacher"]
NOMBRES = {
    "accountant": "contable", "architect": "arquitecto/a", "attorney": "abogado/a",
    "chiropractor": "quiropráctico/a", "comedian": "cómico/a", "composer": "compositor/a",
    "dentist": "dentista", "dietitian": "dietista", "dj": "DJ", "filmmaker": "cineasta",
    "interior_designer": "interiorista", "journalist": "periodista", "model": "modelo",
    "nurse": "enfermero/a", "painter": "pintor/a", "paralegal": "asistente legal", "pastor": "pastor/a",
    "personal_trainer": "entrenador/a personal", "photographer": "fotógrafo/a", "physician": "médico/a",
    "poet": "poeta", "professor": "profesor/a universitario/a", "psychologist": "psicólogo/a",
    "rapper": "rapero/a", "software_engineer": "ingeniero/a de software", "surgeon": "cirujano/a",
    "teacher": "docente (escuela)", "yoga_teacher": "profesor/a de yoga",
}
DEDUCIR, OTRA = "Deducir del texto", "Otra (comparar con todas)"
AZUL, ROJO = "#2a78d6", "#eb6834"
TINTA, TINTA_2, REJILLA, FONDO = "#161616", "#57534e", "#ded8ce", "#faf9f6"

# ---------------------------------------------------------------------------
# Modelo de lenguaje
# ---------------------------------------------------------------------------
_url, _token = os.getenv("OVH_AI_ENDPOINTS_URL"), os.getenv("OVH_AI_ENDPOINTS_ACCESS_TOKEN")
if not _url or not _token:
    raise RuntimeError("Faltan OVH_AI_ENDPOINTS_URL u OVH_AI_ENDPOINTS_ACCESS_TOKEN en el archivo .env.")
client = OpenAI(base_url=_url, api_key=_token, timeout=90.0, max_retries=1)


def preguntar(prompt):
    """Una petición al modelo; devuelve el JSON de la respuesta (o {} si no se pudo)."""
    for intento in range(3):
        try:
            r = client.chat.completions.create(model=MODELO, messages=[{"role": "user", "content": prompt}],
                                               temperature=0, reasoning_effort="none")
            contenido = r.choices[0].message.content or ""
            ini, fin = contenido.find("{"), contenido.rfind("}")
            return json.loads(contenido[ini:fin + 1]) if ini != -1 else {}
        except json.JSONDecodeError:
            return {}
        except Exception:
            time.sleep(2 * (intento + 1))
    return {}


# ---------------------------------------------------------------------------
# Enmascarado (igual que en el estudio)
# ---------------------------------------------------------------------------
PRONOMBRES = r"\b(she|he|her|hers|him|his|herself|himself|mrs|mr|ms|miss|mister|madam)\b"
_nlp = None


def enmascarar_lote(textos):
    global _nlp
    try:
        if _nlp is None:
            import spacy
            _nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])
        salida = []
        for doc in _nlp.pipe(textos):
            t = doc.text
            for ent in reversed(doc.ents):
                if ent.label_ == "PERSON":
                    t = t[:ent.start_char] + "[NAME]" + t[ent.end_char:]
            salida.append(t)
    except Exception:          # sin spaCy: solo se ocultan los pronombres
        salida = list(textos)
    return [re.sub(PRONOMBRES, "[X]", t, flags=re.IGNORECASE) for t in salida]


# ---------------------------------------------------------------------------
# Variables, dimensiones y codificación (copiadas de la parte 2 del estudio)
# ---------------------------------------------------------------------------
VARIABLES_FORMA = {
    # --- Control ---
    "tipo_texto": (["biografia", "no_biografia"],
        "Is this text a description of a person's work, career, education, skills or life? Answer 'biografia' if it is, even if it starts mid-story, is short, lists services or specialties, or contains some extra details. Answer 'no_biografia' ONLY if it gives almost no information about the person (just contact details, navigation text, an advertisement or an unrelated text)."),
    # --- Vida personal ---
    "menciona_pareja": ("bool",
        "Does the text mention a spouse or romantic partner (husband, wife, partner, fiancé/fiancée, boyfriend, girlfriend, significant other)?"),
    "papel_pareja": (["coprotagonista", "apoyo", "solo_mencion", "no_aparece"],
        "If a spouse or partner is mentioned, what role do they play? 'coprotagonista' if the partner shares the person's professional work or achievements ('she and her husband founded', 'works in conjunction with her husband'); 'apoyo' if the partner is described as supporting the person ('supported in ministry by his wife'); 'solo_mencion' if the partner is only mentioned as a personal detail ('lives with his wife and two children'); 'no_aparece' if no partner is mentioned."),
    "menciona_hijos": ("bool",
        "Does the text mention children or grandchildren (sons, daughters, kids, stepchildren, 'mother of', 'father of', 'mom', 'dad', 'parent of')?"),
    "menciona_estado_civil": ("bool",
        "Does the text mention marital status (married, single, divorced, widowed, engaged)?"),
    "menciona_padres_origen": ("bool",
        "Does the text mention the person's parents, siblings or family background, or where they grew up or were raised (hometown, childhood, family origins)?"),
    "presentada_por_otra_persona": ("bool",
        "Is the person introduced or defined mainly through their relationship with a partner or family member, so that this relationship is presented as a central part of who they are ('wife of', 'former wife of actor…', 'daughter of famous…', 'married to NBA player…')? Do NOT count: describing who they work for or assist as part of their job (e.g. a paralegal assisting an attorney), being discovered or helped by someone, simply mentioning family background, or simply saying they are married."),
    "menciona_aficiones": ("bool",
        "Does the text mention hobbies, free time or personal interests outside work (sports, travel, cooking, reading, music, pets, 'in their spare time', 'when not at work')?"),
    "posicion_mencion_familiar": (["primera_mitad", "segunda_mitad", "no_aparece"],
        "Where does the FIRST mention of family (partner, children, parents, marital status) appear in the text?"),
    # --- Logros, formación y trayectoria ---
    "n_logros": ("int",
        "How many distinct achievements or recognitions are mentioned (awards, prizes, honours, rankings, nominations, grants, being selected or named for something, notable publications, exhibitions, record sales, notable clients)? Do not count degrees or certifications."),
    "n_credenciales": ("int",
        "How many distinct educational or professional credentials are mentioned (degrees, diplomas, certifications, licenses, board certifications, fellowships, residencies, training programs)?"),
    "menciona_experiencia": ("bool",
        "Does the text mention length of experience or career ('over 10 years of experience', 'two decades', 'since 1995', 'veteran', 'seasoned', 'long career')?"),
    "menciona_liderazgo": ("bool",
        "Is the person described in a leadership, management or ownership role (founder, co-founder, owner, director, head, chief, CEO, president, chair, manager, partner in a firm, principal, leads or supervises a team or department)?"),
    "atribuye_logros_a_otros": (["si", "no", "sin_logros"],
        "Are any of the person's achievements credited to other people or to circumstances (a mentor, spouse, family, team, collaborators, luck, being discovered by someone, support from others)? Answer 'sin_logros' if no achievements are mentioned."),
    "logros_en_pasiva": (["si", "no", "sin_logros"],
        "Is any achievement described in the passive voice ('was awarded', 'was named', 'was selected', 'was recognized', 'was honored')? Answer 'sin_logros' if no achievements are mentioned."),
    # --- Orientación profesional ---
    "orientacion_cuidado": ("bool",
        "Does the text emphasize helping, caring for or serving others (patients, clients, students, families, community), e.g. 'dedicated to helping', 'committed to patient care', 'passion for serving'?"),
    "menciona_impacto": ("bool",
        "Does the text emphasize the person's influence, impact, reach or authority (shaped the field, widely cited, recognized authority, transformed, led change, international reputation)?"),
    "menciona_innovacion": ("bool",
        "Does the text describe the person as creating, inventing or innovating (developed a new method, patents, founded, pioneered a technique, built a product)?"),
    "menciona_docencia_mentoria": ("bool",
        "Does the text mention teaching, mentoring, training or supervising others' learning?"),
    "menciona_voluntariado": ("bool",
        "Does the text mention volunteering, charity, non-profit work, community service or pro bono work?"),
    "menciona_especializacion": ("bool",
        "Does the text mention a specific area of specialization or technical expertise (specializes in, focuses on, expert in a specific technique or area)?"),
    # --- Excepcionalidad y marcado de género ---
    "excepcionalidad": ("bool",
        "Is the person described as exceptional or a pioneer within a group or field: first, only, one of the few, pioneer, pioneering, trailblazer, groundbreaking figure, breaking barriers, rare, unique among peers, leading the way? It must describe the person, not a product or place (e.g. 'the first clinic in town' does not count)."),
    "marcado_genero": ("bool",
        "Does the text explicitly mark the person's gender in relation to their profession or role: 'female engineer', 'woman surgeon', 'male nurse', 'lady doctor', 'women in tech', 'first woman to…'? Do NOT count gendered job titles on their own (actress, actor, chairwoman, chairman, spokeswoman, spokesman, businesswoman, waitress): they are a separate question."),
    "marca_genero_persona": ("bool",
        "Does the text explicitly point out the person's own gender as relevant to who they are or what they do ('as a woman', 'as a Black Muslim woman', 'female engineer', 'male nurse', 'Best Female Comic', 'first woman to…', 'everywoman', 'ChiroMom')? Do NOT count gendered job titles alone (actress, spokeswoman) or mentions of women-only or men-only groups."),
    "titulo_con_genero": ("bool",
        "Does the text use a gendered job title or role noun for the person (actress, actor, chairwoman, chairman, spokeswoman, spokesman, businesswoman, businessman, songstress, hostess, waitress)?"),
    "espacio_un_genero": ("bool",
        "Does the text mention groups, events, awards, shows or organizations restricted to or focused on one gender ('all-female showcase', 'women's network', 'Women in Law', 'men's ministry', 'ladies' night')?"),
    # --- Lenguaje ---
    "n_adj_agencia": ("int",
        "How many words describe agency or competence (confident, ambitious, driven, independent, decisive, assertive, determined, accomplished, successful, influential, expert, outstanding, leading)?"),
    "n_adj_calidez": ("int",
        "How many words describe warmth or communal traits (warm, caring, compassionate, kind, supportive, nurturing, helpful, friendly, patient, gentle, empathetic, approachable, loving)?"),
    "n_palabras_capacidad": ("int",
        "How many words attribute success to ability or talent (talented, gifted, brilliant, intelligent, skilled, natural, exceptional, genius, innovative, creative)?"),
    "n_palabras_esfuerzo": ("int",
        "How many words attribute success to effort (hardworking, dedicated, diligent, committed, tireless, conscientious, meticulous, persistent, devoted, works hard)?"),
    "n_expresiones_duda": ("int",
        "How many expressions cast doubt or give faint praise (tries to, hopes to, aspiring, may, seems, somewhat, fairly, adequate, although, despite, still learning)?"),
    "n_terminos_emocion": ("int",
        "How many words refer to emotions or temperament (passionate, loves, enjoys, excited, proud, happy, enthusiastic, sensitive, emotional, cheerful, calm, energetic)?"),
    "n_verbos_rol_activo": ("int",
        "How many verbs present the person as the main agent who leads, decides or creates (leads, directs, manages, founded, created, developed, designed, built, launched, oversees, heads)?"),
    "n_verbos_rol_apoyo": ("int",
        "How many verbs present the person in a supporting or assisting role (assists, supports, helps, aids, contributes to, works with, works under, collaborates on, serves)?"),
    "tono": (["positivo", "neutro", "negativo"],
        "What is the overall tone of the text?"),
}

VARIABLES_CONTEXTO = {
    # --- Encuadre: el sesgo es 'encuadre' (se añade o se etiqueta sin un hecho concreto) ---
    "menciona_apariencia_ctx": (["no_aparece", "relevante", "no_relevante"],
        "Does the text comment on the person's own physical appearance (looks, beauty, attractiveness, body, height, weight, hair, clothing, style)? Answer 'no_aparece' if not. Answer 'relevante' if the person's work is directly about their image or appearance (model, actor or actress, fitness or beauty influencer, stylist, TV presenter) and the comment relates to that work. Answer 'no_relevante' if their appearance is mentioned although it has nothing to do with their work. Judge only from the text itself, whatever the profession."),
    "bienestar_sanacion_ctx": (["no_aparece", "hecho", "encuadre"],
        "Does the text mention wellness, healing or holistic well-being (healing, wellness, holistic, mindfulness, self-care, emotional well-being, nurturing)? Answer 'no_aparece' if it is not mentioned. Answer 'hecho' if it describes something concrete the person does: it is their field, specialty, clients, products or services, or a specific activity, organization or project. Answer 'encuadre' if it appears as a label, adjective or way of presenting the person or their work, without a concrete fact behind it or without being needed to describe what they do (for example: a lawyer 'who brings a holistic, healing touch to every case', an accountant who 'radiates calm and well-being'). Judge only from the text itself, whatever the profession. Do not count comments on the person's own appearance or own family."),
    "infancia_familias_ctx": (["no_aparece", "hecho", "encuadre"],
        "Does the text mention children, babies, parents or families as people the person works with or for (pediatric, early childhood, maternity, parenting, family services)? Answer 'no_aparece' if it is not mentioned. Answer 'hecho' if it describes something concrete the person does: it is their field, specialty, clients, products or services, or a specific activity, organization or project. Answer 'encuadre' if it appears as a label, adjective or way of presenting the person or their work, without a concrete fact behind it or without being needed to describe what they do (for example: a general dentist described as 'great with kids', a lawyer who 'also enjoys helping families', a 'family-friendly' practice). Judge only from the text itself, whatever the profession. Do not count comments on the person's own appearance or own family."),
    "estetica_belleza_ctx": (["no_aparece", "hecho", "encuadre"],
        "Does the text mention beauty, fashion, style, makeup, hair, flowers or decoration as a topic? Answer 'no_aparece' if it is not mentioned. Answer 'hecho' if it describes something concrete the person does: it is their field, specialty, clients, products or services, or a specific activity, organization or project. Answer 'encuadre' if it appears as a label, adjective or way of presenting the person or their work, without a concrete fact behind it or without being needed to describe what they do (for example: a lawyer 'with a passion for fashion', an engineer 'known for her sense of style', a dentist who 'loves makeup and interior decor'). Judge only from the text itself, whatever the profession. Do not count comments on the person's own appearance or own family."),
    "activismo_justicia_social_ctx": (["no_aparece", "hecho", "encuadre"],
        "Does the text mention activism or social causes (feminism, women's rights, gender equality, empowerment, civil or human rights, giving a voice to marginalized people)? Answer 'no_aparece' if not. Answer 'hecho' if it describes something concrete the person does: their job is in this field, or they founded, lead, work for or volunteer at a specific organization, campaign, case or project. Answer 'encuadre' if these words are used to label or frame the person or their ordinary work without a concrete activity behind them (for example: 'empowering women through yoga', 'a feminist voice in comedy', 'inspiring girls to dream big')."),
    # --- Puesta en valor: el sesgo es 'aplica_no_destaca' (forma parte de su trabajo y no se valora) ---
    "ciencia_cuantitativa_ctx": (["no_aplica", "aplica_no_destaca", "aplica_destaca"],
        "Is the person's work related to hard or quantitative sciences (physics, mathematics, statistics, economics, data analysis, laboratory or experimental research, engineering calculations, quantitative finance)? Answer 'no_aplica' if their work has nothing to do with it. If it does, answer 'aplica_destaca' if the text highlights or values that side of their work (methods, data, models, research, lab work, technical or scientific expertise), or 'aplica_no_destaca' if the text describes their work without mentioning or valuing that side."),
    "tecnologia_sistemas_ctx": (["no_aplica", "aplica_no_destaca", "aplica_destaca"],
        "Is the person's work related to technology, computers or technical systems (software, hardware, engineering, digital tools, technical equipment)? Answer 'no_aplica' if their work has nothing to do with it. If it does, answer 'aplica_destaca' if the text highlights or values that side of their work (technical skills, tools, systems built or managed, programming, technical expertise), or 'aplica_no_destaca' if the text describes their work without mentioning or valuing that side."),
    "empresa_negocio_ctx": (["no_aplica", "aplica_no_destaca", "aplica_destaca"],
        "Does the person own, found, run or co-run a business, company, firm, studio or private practice? Answer 'no_aplica' if the text gives no sign of that. If they do, answer 'aplica_destaca' if the text values that business (its growth, success, size, clients, achievements, reputation, or the person's role in building it), or 'aplica_no_destaca' if it is only mentioned plainly ('owns a bakery', 'has a private practice') without valuing it."),
    # --- Variables descubiertas que se mantienen (sí / no) ---
    "pasion_emocion": ("bool",
        "Does the text describe the person's work or motivation in emotional terms: passion, love, enjoyment, inspiration, dreams or feelings (passionate about, loves, enjoys, is inspired by, dreams of, heartfelt, joyful, finds fulfilment, deeply moved, emotional)? Count it even if other words are used."),
    "fama_prestigio": ("bool",
        "Does the text emphasize the person's fame, prestige or authority (famous, renowned, world-renowned, leading expert, highly regarded, recognized authority, prominent, acclaimed, legendary, reaches thousands or millions of people, medals)? Count it even if other words are used."),
    "valentia_audacia": ("bool",
        "Is the person described as brave, bold or tough (brave, courageous, fearless, bold, daring, audacious, intrepid, tough, gritty, unafraid, takes risks)? Count it even if other words are used."),
    "aficiones_hogar_creativas": ("bool",
        "Does the text mention home, craft, animal or creative leisure activities as hobbies or personal interests (dogs, cats or other pets, animals, knitting, sewing, crafts, gardening, baking, cooking, flowers, dancing)?"),
}

VARIABLES_NOMBRE = {
    "forma_nombrar": (["nombre_pila", "apellido", "nombre_completo", "solo_pronombres"],
        "How is the person mainly referred to in the text: by first name only ('nombre_pila'), by surname only, with or without a title like 'Dr. Smith' ('apellido'), by full name ('nombre_completo'), or only with pronouns, with no name at all ('solo_pronombres')?"),
    "usa_titulo_ctx": (["no_aplica", "aplica_no_usa", "aplica_usa"],
        "Does the text show that the person holds a title normally used to address them (a doctorate or PhD; a medical, dental or chiropractic degree such as MD, DO, DDS, DMD, DC; or a professorship)? Answer 'no_aplica' if the text gives no sign of such a title. If it does, answer 'aplica_usa' if the text refers to the person with the title at least once ('Dr. Smith', 'Professor Smith', 'Jane Smith, PhD'), or 'aplica_no_usa' if it never uses the title to refer to them."),
}

TIPOS = {k: t for k, (t, _) in {**VARIABLES_FORMA, **VARIABLES_CONTEXTO, **VARIABLES_NOMBRE}.items()}

def formato(tipo):
    if tipo == "bool":
        return "true or false"
    if tipo == "int":
        return "an integer, 0 if none"
    return "one of " + ", ".join(f'"{o}"' for o in tipo)

def construir_prompt(texto, variables):
    preguntas = "\n".join(f'- "{k}" ({formato(t)}): {p}' for k, (t, p) in variables.items())
    return ("You are annotating professional biographies for a research study on language.\n"
            "Some words were replaced with [X] (pronouns) or [NAME] (person names). "
            "Do not try to guess the person's gender.\n"
            "Answer every question about the biography below. When a question lists example words, count the idea "
            "even if the text expresses it with different words or synonyms.\n"
            "Return ONLY a JSON object with exactly these keys:\n"
            f"{preguntas}\n\nBiography:\n\"\"\"{texto}\"\"\"")

def limpiar(respuesta, variables):
    fila = {}
    for nombre, (tipo, _) in variables.items():
        v = respuesta.get(nombre)
        if tipo == "bool":
            fila[nombre] = v if isinstance(v, bool) else None
        elif tipo == "int":
            ok = isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0
            fila[nombre] = int(v) if ok else None
        else:
            fila[nombre] = v if v in tipo else None
    return fila

def anotar(texto, texto_enmascarado):
    fila = limpiar(preguntar(construir_prompt(texto_enmascarado, VARIABLES_FORMA)), VARIABLES_FORMA)
    fila.update(limpiar(preguntar(construir_prompt(texto_enmascarado, VARIABLES_CONTEXTO)), VARIABLES_CONTEXTO))
    fila.update(limpiar(preguntar(construir_prompt(texto, VARIABLES_NOMBRE)), VARIABLES_NOMBRE))
    fila["n_palabras"] = len(texto.split())
    return fila



DIMENSIONES = {
    "M1": ("Marcado de género", "mujeres",
           ["marcado_genero", "marca_genero_persona", "forma_nombrar"]),
    "M2": ("Vida personal y aspecto", "mujeres",
           ["menciona_pareja", "menciona_hijos", "menciona_estado_civil", "menciona_padres_origen",
            "presentada_por_otra_persona", "menciona_aficiones", "menciona_apariencia_ctx"]),
    "M3": ("Calidez y emoción", "mujeres",
           ["orientacion_cuidado", "n_adj_calidez", "n_terminos_emocion", "pasion_emocion", "bienestar_sanacion_ctx"]),
    "M4": ("Temas comunales añadidos", "mujeres",
           ["infancia_familias_ctx", "estetica_belleza_ctx", "activismo_justicia_social_ctx",
            "aficiones_hogar_creativas", "menciona_voluntariado"]),
    "M5": ("Agencia y estatus", "hombres",
           ["n_adj_agencia", "n_logros", "menciona_liderazgo", "menciona_impacto", "menciona_innovacion",
            "fama_prestigio", "valentia_audacia", "n_verbos_rol_activo", "usa_titulo_ctx", "excepcionalidad"]),
    "M6": ("Perfil técnico y empresarial puesto en valor", "hombres",
           ["ciencia_cuantitativa_ctx", "tecnologia_sistemas_ctx", "empresa_negocio_ctx"]),
    "M7": ("Agencia diluida", "mujeres",
           ["atribuye_logros_a_otros", "logros_en_pasiva", "n_verbos_rol_apoyo", "n_expresiones_duda", "papel_pareja"]),
}
DIM_DE = {v: d for d, (_, _, vs) in DIMENSIONES.items() for v in vs}
LADO = {v: DIMENSIONES[d][1] for v, d in DIM_DE.items()}
VARIABLES_SESGO = list(DIM_DE)

ETIQUETA = {   # cómo se describe cada rasgo en las explicaciones
    "marcado_genero": "Se marca su género en relación con la profesión (female engineer, male nurse…)",
    "marca_genero_persona": "Se destaca su género como algo relevante (as a woman…)",
    "excepcionalidad": "Se la presenta como excepcional en su campo (first, only, one of the few…)",
    "forma_nombrar": "Se la nombra por el nombre de pila",
    "menciona_pareja": "Se menciona a su pareja",
    "menciona_hijos": "Se mencionan sus hijos",
    "menciona_estado_civil": "Se menciona su estado civil",
    "menciona_padres_origen": "Se mencionan sus padres u origen familiar",
    "presentada_por_otra_persona": "Se la presenta a través de su pareja o su familia",
    "menciona_aficiones": "Se mencionan aficiones personales",
    "menciona_apariencia_ctx": "Se comenta su aspecto físico sin relación con su trabajo",
    "orientacion_cuidado": "Se pone el foco en cuidar o ayudar a otros",
    "n_adj_calidez": "Adjetivos de calidez (warm, caring, supportive…)",
    "n_terminos_emocion": "Palabras de emoción (passionate, loves, enjoys…)",
    "pasion_emocion": "La motivación se cuenta como emoción o pasión",
    "bienestar_sanacion_ctx": "Bienestar o sanación como etiqueta, sin ser su trabajo",
    "infancia_familias_ctx": "Niños o familias como etiqueta, sin ser su especialidad",
    "estetica_belleza_ctx": "Belleza, moda o estilo como etiqueta, sin ser su sector",
    "activismo_justicia_social_ctx": "Palabras de activismo sin una actividad concreta detrás",
    "aficiones_hogar_creativas": "Aficiones del hogar, creativas o con animales",
    "menciona_voluntariado": "Se menciona voluntariado o labor benéfica",
    "n_adj_agencia": "Adjetivos de competencia (accomplished, expert, leading…)",
    "n_logros": "Logros o reconocimientos concretos",
    "menciona_liderazgo": "Su papel de liderazgo o dirección",
    "menciona_impacto": "Su impacto o influencia",
    "menciona_innovacion": "Lo que ha creado o innovado",
    "fama_prestigio": "Su prestigio o reconocimiento",
    "valentia_audacia": "Valentía o audacia",
    "n_verbos_rol_activo": "Verbos en los que lidera, decide o crea",
    "usa_titulo_ctx": "Se la nombra con su título (Dr., Prof., PhD)",
    "ciencia_cuantitativa_ctx": "Se destaca su perfil científico o cuantitativo",
    "tecnologia_sistemas_ctx": "Se destaca su perfil técnico o tecnológico",
    "empresa_negocio_ctx": "Se pone en valor su empresa o negocio",
    "atribuye_logros_a_otros": "Su mérito se atribuye a otras personas o a la suerte",
    "logros_en_pasiva": "Sus logros se cuentan en pasiva (was awarded…)",
    "n_verbos_rol_apoyo": "Verbos de apoyo (assists, supports, works under…)",
    "n_expresiones_duda": "Expresiones de duda o elogio tibio (tries to, hopes to…)",
    "papel_pareja": "Su mérito se comparte con la pareja",
}

# Valores que cuentan como 1 y como "no aplica" en las variables de opciones
UNO = {"menciona_apariencia_ctx": {"no_relevante"}, "forma_nombrar": {"nombre_pila"},
       "papel_pareja": {"coprotagonista", "apoyo"}, "atribuye_logros_a_otros": {"si"}, "logros_en_pasiva": {"si"},
       "usa_titulo_ctx": {"aplica_usa"}}
NO_APLICA = {"forma_nombrar": {"solo_pronombres"}, "papel_pareja": {"no_aparece"},
             "atribuye_logros_a_otros": {"sin_logros"}, "logros_en_pasiva": {"sin_logros"},
             "usa_titulo_ctx": {"no_aplica"}}
for v in ["bienestar_sanacion_ctx", "infancia_familias_ctx", "estetica_belleza_ctx", "activismo_justicia_social_ctx"]:
    UNO[v] = {"encuadre"}
for v in ["ciencia_cuantitativa_ctx", "tecnologia_sistemas_ctx", "empresa_negocio_ctx"]:
    UNO[v], NO_APLICA[v] = {"aplica_destaca"}, {"no_aplica"}

def codificar_valor(variable, x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    tipo = TIPOS[variable]
    if tipo == "bool":
        return 1.0 if str(x).strip().lower() in ("true", "1", "1.0") else 0.0
    if tipo == "int":
        return float(float(x) > 0)
    if x in NO_APLICA.get(variable, ()):
        return np.nan
    return float(x in UNO[variable])

def codificar(df):
    """Tabla 0 / 1 / vacío de las variables de sesgo."""
    return pd.DataFrame({v: df[v].map(lambda x: codificar_valor(v, x)) for v in VARIABLES_SESGO if v in df.columns},
                        index=df.index)

# Hipótesis: lado mujeres -> más en mujeres; lado hombres -> más en hombres
HIPOTESIS_FINAL = {v: ("más en mujeres" if LADO[v] == "mujeres" else "más en hombres") for v in VARIABLES_SESGO}

# Variables cuyo lado se cambió después de ver los resultados de la confirmación. En el confirmatorio se
# mantiene la hipótesis que se fijó de antemano (y queda como no confirmada); no entran en la métrica.
# Para usarlas con su nuevo lado habría que confirmarlas en otra muestra.
POSTHOC = {"excepcionalidad": "más en mujeres"} if FASE == "confirmacion" else {}   # en la ampliación ya es hipótesis previa
HIPOTESIS_FINAL.update(POSTHOC)

DESCRIPTIVAS = ["espacio_un_genero", "titulo_con_genero", "posicion_mencion_familiar = primera_mitad",
                "n_credenciales", "menciona_experiencia", "menciona_docencia_mentoria", "menciona_especializacion",
                "tono = neutro", "n_palabras_esfuerzo", "n_palabras_capacidad"]
print(len(VARIABLES_SESGO), "variables de sesgo en", len(DIMENSIONES), "dimensiones;", len(DESCRIPTIVAS), "descriptivas.")

# ---------------------------------------------------------------------------
# Datos de referencia (biografías anotadas en la parte 2)
# ---------------------------------------------------------------------------
partes = []
for _fase in REFERENCIA:
    _a, _m = BASE / f"anotaciones_{_fase}.csv", CACHE / f"muestra_{_fase}.parquet"
    if _a.exists() and _m.exists():
        _d = pd.read_parquet(_m).drop(columns=["texto_enmascarado"], errors="ignore").merge(pd.read_csv(_a), on="idx")
        partes.append(_d)
if not partes:
    raise FileNotFoundError("No encuentro anotaciones_ampliacion.csv / anotaciones_confirmacion.csv: "
                            "pon la app en la carpeta de los cuadernos y ejecuta antes la parte 2.")
REF = pd.concat(partes, ignore_index=True)
REF = REF[REF["tipo_texto"].notna() & (REF["tipo_texto"] != "no_biografia")].reset_index(drop=True)
REF["mujer"] = (REF["genero"] == "woman").astype(int)
REF_COD = codificar(REF)
_conf = pd.read_csv(RESULTADOS / f"confirmatorio_{FASE}.csv")
CONFIRMADAS = _conf.loc[_conf["confirmada"] == True, "variable"].tolist()

# ---------------------------------------------------------------------------
# Métrica (copiada de la parte 2)
# ---------------------------------------------------------------------------
TODAS = "(todas)"
VARS_METRICA = CONFIRMADAS if VARIABLES_METRICA == "confirmadas" else VARIABLES_SESGO
print("Variables en la métrica:", len(VARS_METRICA),
      f"({sum(LADO[v] == 'mujeres' for v in VARS_METRICA)} de presencia, "
      f"{sum(LADO[v] == 'hombres' for v in VARS_METRICA)} de omisión)")
if not VARS_METRICA:
    print("⚠ No hay variables confirmadas: la métrica no se puede calcular. "
          "Para explorar, pon VARIABLES_METRICA = 'hipotesis' en la configuración.")

def tabla_tasas(validas, cod, variables):
    """Tasa de cada rasgo en mujeres y hombres, por profesión y en general (sobre las biografías a las que aplica)."""
    filas = []
    base = validas[["profesion", "mujer"]].join(cod[variables])
    for v in variables:
        d = base[["profesion", "mujer", v]].dropna()
        grupos = [(TODAS, d)] + list(d.groupby("profesion"))
        for prof, g in grupos:
            h, m = g.loc[g.mujer == 0, v], g.loc[g.mujer == 1, v]
            filas.append({"variable": v, "profesion": prof, "tasa_h": h.mean() if len(h) else np.nan,
                          "tasa_m": m.mean() if len(m) else np.nan, "n_h": len(h), "n_m": len(m)})
    return pd.DataFrame(filas).set_index(["variable", "profesion"])

def calcular_pesos(tasas, variables, profesion):
    filas = []
    for v in variables:
        g = tasas.loc[(v, TODAS)]
        if profesion is not None and (v, profesion) in tasas.index:
            p = tasas.loc[(v, profesion)]
            n = min(p["n_h"], p["n_m"])
            lam = n / (n + K_SUAVIZADO) if n > 0 else 0.0
            th = p["tasa_h"] if pd.notna(p["tasa_h"]) else g["tasa_h"]
            tm = p["tasa_m"] if pd.notna(p["tasa_m"]) else g["tasa_m"]
        else:
            lam, th, tm = 0.0, g["tasa_h"], g["tasa_m"]
        th_s = lam * th + (1 - lam) * g["tasa_h"]
        tm_s = lam * tm + (1 - lam) * g["tasa_m"]
        if LADO[v] == "mujeres":
            peso = max(0.0, tm_s - th_s)
        else:
            peso = th_s if th_s * 100 >= MIN_TASA_HOMBRES else 0.0
        filas.append({"variable": v, "dimension": DIM_DE[v], "lado": LADO[v], "peso": peso,
                      "tasa_h_ref": th_s, "tasa_m_ref": tm_s, "tasa_h_prof": th, "tasa_m_prof": tm,
                      "tasa_h_gen": g["tasa_h"], "tasa_m_gen": g["tasa_m"], "suavizado_prof": lam})
    return pd.DataFrame(filas).set_index("variable")

def metrica_biografia(codigos, profesion, pesos=None):
    """codigos: Serie 0/1/vacío de una biografía. Devuelve las dos partes, el total y el detalle."""
    pesos = PESOS if pesos is None else pesos
    prof = profesion if profesion in pesos else None
    P = pesos[prof].copy()
    P["x"] = [codigos.get(v, np.nan) for v in P.index]
    partes = {}
    for lado, nombre in [("mujeres", "presencia"), ("hombres", "omision")]:
        q = P[(P["lado"] == lado) & (P["peso"] > 0) & P["x"].notna()]
        suma = q["peso"].sum()
        if suma > 0:
            mal = q["x"] if lado == "mujeres" else 1 - q["x"]
            partes[nombre] = float((q["peso"] * mal).sum() / suma)
        elif lado == "hombres":
            partes[nombre] = 0.0   # en su profesión no hay ningún rasgo agéntico habitual: no falta nada
        else:
            partes[nombre] = np.nan
        P.loc[q.index, "peso_relativo"] = q["peso"] / suma if suma > 0 else np.nan
    disponibles = {k: w for k, w in [("presencia", PESO_PRESENCIA), ("omision", PESO_OMISION)] if pd.notna(partes[k])}
    total_w = sum(disponibles.values())
    total = sum(partes[k] * w for k, w in disponibles.items()) / total_w if total_w > 0 else np.nan
    escala = {k: (w / total_w if total_w > 0 else 0) for k, w in [("presencia", PESO_PRESENCIA), ("omision", PESO_OMISION)]}

    def estado(f):
        if pd.isna(f["x"]):
            return "no aplica"
        if f["peso"] <= 0:
            return "sin peso en esta profesión"
        if f["lado"] == "mujeres":
            return "presente (sesgo)" if f["x"] == 1 else "ausente (bien)"
        return "presente (bien)" if f["x"] == 1 else "falta (omisión)"

    P["estado"] = P.apply(estado, axis=1)
    P["aporta_al_total"] = 0.0
    mp = P["estado"] == "presente (sesgo)"
    mo = P["estado"] == "falta (omisión)"
    P.loc[mp, "aporta_al_total"] = escala["presencia"] * P.loc[mp, "peso_relativo"]
    P.loc[mo, "aporta_al_total"] = escala["omision"] * P.loc[mo, "peso_relativo"]
    return {"profesion": prof, "presencia": partes["presencia"], "omision": partes["omision"], "total": total,
            "detalle": P.reset_index()}

def pct(x):
    return "—" if pd.isna(x) else f"{x * 100:.0f} %"

def explicar(res, mostrar_bien=True):
    prof = res["profesion"]
    nombre_prof = NOMBRES.get(prof, prof) if prof else "todas las profesiones (no hay una equivalente)"
    det = res["detalle"]
    print("=" * 88)
    print("MÉTRICA DE SESGO DE GÉNERO   (0 = ningún sesgo · 1 = todos los sesgos detectados)")
    print("Profesión de referencia:", nombre_prof)
    print("=" * 88)
    print(f"TOTAL: {res['total']:.2f}" if pd.notna(res["total"]) else "TOTAL: no se puede calcular")
    print(f"  · Sesgo por presencia: {res['presencia']:.2f}  (pesa {PESO_PRESENCIA * 100:.0f} %)"
          if pd.notna(res["presencia"]) else "  · Sesgo por presencia: sin rasgos aplicables")
    print(f"  · Omisión agéntica:    {res['omision']:.2f}  (pesa {PESO_OMISION * 100:.0f} %)"
          if pd.notna(res["omision"]) else "  · Omisión agéntica: sin rasgos aplicables")

    def minus(t):
        return t[0].lower() + t[1:]

    def linea(f, simbolo):
        print(f"  {simbolo} {ETIQUETA[f['variable']]}  [{f['dimension']}]   +{f['aporta_al_total']:.2f} al total")
        print(f"      referencia {'para esta profesión' if prof else 'general'}: mujeres {pct(f['tasa_m_ref'])}, "
              f"hombres {pct(f['tasa_h_ref'])}")

    def lista(q, simbolo):
        grandes = q[q["aporta_al_total"] >= 0.005]
        for _, f in grandes.iterrows():
            linea(f, simbolo)
        if len(q) > len(grandes):
            print(f"  (y {len(q) - len(grandes)} más con un peso casi nulo en esta profesión: "
                  + "; ".join(minus(ETIQUETA[v]) for v in q.loc[~q.index.isin(grandes.index), "variable"]) + ")")

    print("\nPARTE 1 · SESGO POR PRESENCIA — rasgos que suben la puntuación")
    q = det[det["estado"] == "presente (sesgo)"].sort_values("aporta_al_total", ascending=False)
    if q.empty:
        print("  ✓ Ninguno.")
    lista(q, "✗")
    if mostrar_bien:
        q = det[(det["lado"] == "mujeres") & (det["estado"] == "ausente (bien)")]
        if not q.empty:
            print("  No aparecen (bien):", "; ".join(minus(ETIQUETA[v]) for v in q["variable"]))

    print("\nPARTE 2 · OMISIÓN AGÉNTICA — lo que suele aparecer en los hombres de su profesión y falta")
    q = det[det["estado"] == "falta (omisión)"].sort_values("aporta_al_total", ascending=False)
    if q.empty:
        print("  ✓ No falta nada relevante.")
    lista(q, "○")
    if mostrar_bien:
        q = det[(det["lado"] == "hombres") & (det["estado"] == "presente (bien)")]
        if not q.empty:
            print("  Sí aparecen (bien):", "; ".join(minus(ETIQUETA[v]) for v in q["variable"]))
    na = det[det["estado"] == "no aplica"]
    if not na.empty:
        print("\nNo aplican a este texto:", "; ".join(minus(ETIQUETA[v]) for v in na["variable"]))
    print("\nSon señales, no un veredicto: un rasgo no es incorrecto en sí mismo; el sesgo es que se use")
    print("sistemáticamente más con un género que con otro.")

TASAS = tabla_tasas(REF, REF_COD, VARS_METRICA)
PESOS = {p: calcular_pesos(TASAS, VARS_METRICA, p) for p in REF["profesion"].unique()}
PESOS[None] = calcular_pesos(TASAS, VARS_METRICA, None)
TASAS_TODAS = tabla_tasas(REF, REF_COD, VARIABLES_SESGO)
REF["met_prof"] = [metrica_biografia(REF_COD.loc[i], REF.at[i, "profesion"])["total"] for i in REF.index]
REF["met_gen"] = [metrica_biografia(REF_COD.loc[i], None)["total"] for i in REF.index]
REF_DIM = pd.DataFrame({d: REF_COD[[v for v in vs if v in REF_COD]].mean(axis=1)
                        for d, (_, _, vs) in DIMENSIONES.items()})
LADO_DIM = {d: lado for d, (_, lado, _) in DIMENSIONES.items()}
ETIQ_DIM = {d: f"{d} · {nombre}" for d, (nombre, _, _) in DIMENSIONES.items()}
CORTO = {"M1": "Marcado\nde género", "M2": "Vida personal\ny aspecto", "M3": "Calidez y\nemoción",
         "M4": "Temas comunales\nañadidos", "M5": "Agencia y\nestatus", "M6": "Perfil técnico\ny empresarial",
         "M7": "Agencia\ndiluida"}
PREGUNTA = {k: p for k, (_, p) in {**VARIABLES_FORMA, **VARIABLES_CONTEXTO, **VARIABLES_NOMBRE}.items()}


# ---------------------------------------------------------------------------
# Funciones del análisis
# ---------------------------------------------------------------------------
_ES = {"el", "la", "los", "las", "de", "del", "que", "y", "en", "es", "un", "una", "con", "por", "para", "su",
       "sus", "ha", "como", "se", "al", "lo", "más", "años", "trabaja", "fue", "está"}
_EN = {"the", "and", "of", "in", "is", "a", "an", "with", "for", "his", "her", "has", "as", "to", "at", "she",
       "he", "years", "works", "was", "from", "on"}


def idioma(texto):
    palabras = re.findall(r"[a-záéíóúñü]+", texto.lower())
    es, en = sum(p in _ES for p in palabras), sum(p in _EN for p in palabras)
    return "es" if es > en else "en"


def traducir(texto):
    r = preguntar("Translate this professional biography into English. Be faithful: keep every name, pronoun, "
                  "gendered word, adjective and the original tone; do not add, remove or soften anything. "
                  'Return ONLY a JSON object like {"translation": "..."}.\n\nBiography:\n"""' + texto + '"""')
    t = r.get("translation")
    if not isinstance(t, str) or not t.strip():
        raise ValueError("No se pudo traducir el texto al inglés.")
    return t.strip()


def profesion_de_referencia(texto_en, eleccion):
    if eleccion == OTRA:
        return None
    if eleccion and eleccion != DEDUCIR:
        return next((en for en, es in NOMBRES.items() if es == eleccion), None)
    r = preguntar(f'Which of these professions is the closest match to the profession of the person described in '
                  f'this text: """{texto_en[:1500]}"""? Options: {", ".join(PROFESIONES)}. If none is a reasonable '
                  'match, answer "none". Return ONLY a JSON object like {"profesion": "<option or none>"}.')
    p = r.get("profesion")
    return p if p in PROFESIONES else None


def citas(texto_original, variables):
    if not variables:
        return {}
    lista = "\n".join(f'- "{v}" (a short exact quote or ""): {PREGUNTA[v]}' for v in variables)
    r = preguntar("For each feature below, copy the SHORTEST exact phrase from the biography that shows it, word for "
                  "word and in the biography's original language (it may be Spanish or English). If you cannot "
                  f'find one, use "". Return ONLY a JSON object with these keys:\n{lista}\n\n'
                  f'Biography:\n"""{texto_original}"""')
    return {v: r[v].strip() for v in variables
            if isinstance(r.get(v), str) and r[v].strip() and r[v].strip() in texto_original}


def tasa_ref(v, prof, mujer):
    clave = (v, prof if prof is not None else TODAS)
    return TASAS_TODAS.loc[clave, "tasa_m" if mujer else "tasa_h"] if clave in TASAS_TODAS.index else np.nan


def medias_dimension(prof):
    g = REF if prof is None else REF[REF["profesion"] == prof]
    d = REF_DIM.loc[g.index]
    return d[g["mujer"] == 0].mean(), d[g["mujer"] == 1].mean()


def pct(x):
    return None if pd.isna(x) else round(float(x) * 100, 1)


def tabla_metrica(res_prof, res_gen, prof):
    filas = []
    for nombre, res, p in [(f"Su profesión ({NOMBRES.get(prof, '')})", res_prof, prof),
                           ("Todas las profesiones", res_gen, None)]:
        if res is None:
            continue
        col = "met_gen" if p is None else "met_prof"
        g = REF if p is None else REF[REF["profesion"] == p]
        filas.append([nombre, round(res["total"], 2), round(res["presencia"], 2), round(res["omision"], 2),
                      round(g.loc[g.mujer == 0, col].mean(), 2), round(g.loc[g.mujer == 1, col].mean(), 2)])
    return pd.DataFrame(filas, columns=["Comparación", "Métrica total", "Presencia (70 %)", "Omisión (30 %)",
                                        "Media hombres", "Media mujeres"])


def tabla_dimensiones(cod, prof):
    refs = ([(prof, "prof.")] if prof else []) + [(None, "todas")]
    medias = {etq: medias_dimension(p) for p, etq in refs}
    filas = []
    for d, (nombre, lado, vs) in DIMENSIONES.items():
        aplican = [v for v in vs if pd.notna(cod.get(v, np.nan))]
        det = int(sum(cod.get(v) == 1 for v in aplican))
        fila = {"Dimensión": ETIQ_DIM[d], "Lado": "comunal" if lado == "mujeres" else "agéntico",
                "Detectados": det, "Aplican": len(aplican),
                "% texto": round(det / len(aplican) * 100, 1) if aplican else None}
        for etq, (h, m) in medias.items():
            fila[f"% hombres ({etq})"] = pct(h[d])
            fila[f"% mujeres ({etq})"] = pct(m[d])
        filas.append(fila)
    return pd.DataFrame(filas)


def filas_rasgos(cod, res_prof, res_gen, prof, encontradas, dimension=None):
    aporta = (res_prof or res_gen)["detalle"].set_index("variable")["aporta_al_total"]
    filas = []
    for d, (_, _, vs) in DIMENSIONES.items():
        if dimension and d != dimension:
            continue
        for v in vs:
            x = cod.get(v, np.nan)
            filas.append({"Dimensión": d, "Rasgo": ETIQUETA[v],
                          "Resultado": "no aplica" if pd.isna(x) else ("detectado" if x == 1 else "no detectado"),
                          "En la métrica": "sí" if v in VARS_METRICA else "no",
                          "Aporta": round(float(aporta.get(v, 0.0)), 3),
                          "% hombres (ref.)": pct(tasa_ref(v, prof, 0)), "% mujeres (ref.)": pct(tasa_ref(v, prof, 1)),
                          "Frase del texto": encontradas.get(v, "")})
    return pd.DataFrame(filas)


def grafico(dims, prof):
    refs = ([(prof, f"Comparado con: {NOMBRES[prof]}")] if prof else []) + [(None, "Comparado con: todas las profesiones")]
    derecha = [d for d in DIMENSIONES if LADO_DIM[d] == "mujeres"]
    izquierda = [d for d in DIMENSIONES if LADO_DIM[d] == "hombres"]
    ang = {d: np.radians(a) for d, a in zip(derecha, np.linspace(25, 155, len(derecha)))}
    ang.update({d: np.radians(a) for d, a in zip(izquierda, np.linspace(235, 305, len(izquierda)))})
    fig = Figure(figsize=(8, 7.2 * len(refs) + 0.6), facecolor=FONDO)
    for k, (p, subt) in enumerate(refs):
        ax = fig.add_subplot(len(refs), 1, k + 1, projection="polar")
        hom, muj = medias_dimension(p)
        ax.set_facecolor(FONDO)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 1.22)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["25 %", "50 %", "75 %", "100 %"], color=TINTA_2, fontsize=7.5)
        ax.set_rlabel_position(0)
        ax.set_xticks([ang[d] for d in DIMENSIONES])
        ax.set_xticklabels([CORTO[d] for d in DIMENSIONES], fontsize=9, color=TINTA)
        ax.tick_params(axis="x", pad=14)
        ax.grid(color=REJILLA, linewidth=0.8)
        ax.spines["polar"].set_visible(False)
        for borde in (0, np.pi):
            ax.plot([borde, borde], [0, 1.22], color="#b9b8b4", lw=1, ls="--", zorder=1)
        for d in DIMENSIONES:
            a, v = ang[d], dims.get(d, np.nan)
            col = ROJO if LADO_DIM[d] == "mujeres" else AZUL
            if pd.notna(v) and v > 0:
                ax.plot([a, a], [0, v], color=col, lw=3, solid_capstyle="butt", zorder=3)
                ax.plot(a, v, marker=(3, 0, -np.degrees(a)), markersize=11, color=col, zorder=4)
            ax.text(a, (v if pd.notna(v) else 0) + 0.13, "–" if pd.isna(v) else f"{v * 100:.0f} %",
                    ha="center", va="center", fontsize=9.5, fontweight="bold", color=col, zorder=5)
            ax.plot(a, hom[d], marker="D", markersize=6.5, color="#7c7b78", zorder=6, linestyle="none")
            ax.plot(a, muj[d], marker="o", markersize=6.5, markerfacecolor="white", markeredgecolor="#7c7b78",
                    markeredgewidth=1.5, zorder=6, linestyle="none")
        ax.plot(0, 0, "o", color=TINTA_2, markersize=5, zorder=5)
        ax.set_title(subt + "\n← agéntico (azul)      comunal (rojo) →", color=TINTA, fontsize=10.5, pad=22)
    fig.text(0.5, 0.25 / (7.2 * len(refs) + 0.6), "Flecha: % de rasgos de la dimensión presentes en el texto.  «–»: no aplica.\nMarcas grises: ◆ media de hombres "
             "de referencia  ○ media de mujeres de referencia.", ha="center", fontsize=8.3, color=TINTA_2)
    fig.subplots_adjust(left=0.12, right=0.88, top=1 - 1.1 / (7.2 * len(refs) + 0.6),
                        bottom=0.9 / (7.2 * len(refs) + 0.6), hspace=0.4)
    return fig


def resaltar(texto, encontradas):
    agrupadas = {}
    for v, frase in encontradas.items():
        col = ROJO if LADO[v] == "mujeres" else AZUL
        c0, etqs = agrupadas.get(frase, (col, []))
        agrupadas[frase] = (ROJO if ROJO in (c0, col) else c0, etqs + [ETIQUETA[v]])
    t, huecos = html.escape(texto), {}
    for n, (frase, (col, etqs)) in enumerate(sorted(agrupadas.items(), key=lambda x: -len(x[0]))):
        f = html.escape(frase)
        if f in t:
            clave = f"@@{n}@@"
            t = t.replace(f, clave, 1)
            huecos[clave] = (f'<mark style="background:{col}2e;border-bottom:2px solid {col};padding:0 2px;'
                             f'color:inherit" title="{html.escape(" · ".join(etqs))}">{f}</mark>')
    for clave, marca in huecos.items():
        t = t.replace(clave, marca)
    leyenda = (f'<span style="border-bottom:2px solid {ROJO}">rojo</span>: rasgo comunal detectado · '
               f'<span style="border-bottom:2px solid {AZUL}">azul</span>: rasgo agéntico detectado. '
               'Pasa el ratón por encima para ver el rasgo.')
    return (f'<div style="font-size:15px;line-height:1.7">{t}</div>'
            f'<div style="font-size:12px;color:#586b76;margin-top:8px">{leyenda}</div>')


def texto_dimension(dimension, datos):
    if not datos:
        return "Analiza una biografía para explorar sus dimensiones."
    nombre, lado, vs = DIMENSIONES[dimension]
    cod = datos["codigos"]
    aplican = [v for v in vs if pd.notna(cod.get(v, np.nan))]
    det = sum(cod.get(v) == 1 for v in aplican)
    porc = f" ({det / len(aplican) * 100:.1f} %)" if aplican else ""
    return (f"### {ETIQ_DIM[dimension]}\n**Lado:** {'comunal' if lado == 'mujeres' else 'agéntico'} · "
            f"**{det} de {len(aplican)} rasgos que aplican detectados{porc}** · {len(vs)} rasgos en total.")


# ---------------------------------------------------------------------------
# Análisis completo (lo que hace el botón)
# ---------------------------------------------------------------------------
def analizar(texto, eleccion_profesion):
    vacio = (None, None, None, None, {}, gr.update(), "", None, "", None)
    if not texto or not texto.strip():
        return ("Introduce una biografía.",) + vacio
    if len(texto) > MAX_CARACTERES:
        return (f"Máximo: {MAX_CARACTERES:,} caracteres. El texto no se ha enviado.".replace(",", "."),) + vacio
    try:
        lengua = idioma(texto)
        texto_en = traducir(texto) if lengua == "es" else texto
        prof = profesion_de_referencia(texto_en, eleccion_profesion)
        prof = prof if prof in PESOS else None
        fila = anotar(texto_en, enmascarar_lote([texto_en])[0])
        if all(v is None for k, v in fila.items() if k != "n_palabras"):
            raise ValueError("El modelo no devolvió anotaciones. Inténtalo de nuevo.")
        cod = codificar(pd.DataFrame([fila])).iloc[0]
        dims = {d: cod[[v for v in vs if v in cod]].mean() for d, (_, _, vs) in DIMENSIONES.items()}
        res_gen = metrica_biografia(cod, None)
        res_prof = metrica_biografia(cod, prof) if prof else None
        presentes = [v for v in VARIABLES_SESGO if cod.get(v) == 1]
        encontradas = citas(texto, presentes)
    except Exception as e:
        return (f"No se pudo analizar el texto: {e}",) + vacio

    avisos = [f"**Idioma:** {'castellano (traducido al inglés para analizarlo)' if lengua == 'es' else 'inglés'} · "
              f"**Profesión de referencia:** {NOMBRES[prof] if prof else 'ninguna de las 28 (solo comparación con todas)'}"]
    if fila.get("tipo_texto") == "no_biografia":
        avisos.append("⚠ El modelo no reconoce el texto como una biografía: los resultados pueden no tener sentido.")
    datos = {"codigos": cod, "res_prof": res_prof, "res_gen": res_gen, "prof": prof, "citas": encontradas}
    dim_inicial = max(DIMENSIONES, key=lambda d: -1 if pd.isna(dims[d]) else dims[d])
    return ("\n\n".join(avisos), tabla_metrica(res_prof, res_gen, prof), grafico(dims, prof),
            tabla_dimensiones(cod, prof), datos, gr.update(value=ETIQ_DIM[dim_inicial]),
            texto_dimension(dim_inicial, datos), filas_rasgos(cod, res_prof, res_gen, prof, encontradas, dim_inicial),
            resaltar(texto, encontradas), filas_rasgos(cod, res_prof, res_gen, prof, encontradas))


def ver_dimension(etiqueta, datos):
    if not datos or not etiqueta:
        return "Analiza una biografía para explorar sus dimensiones.", None
    d = etiqueta.split(" · ")[0]
    return texto_dimension(d, datos), filas_rasgos(datos["codigos"], datos["res_prof"], datos["res_gen"],
                                                   datos["prof"], datos["citas"], d)


def seleccionar_dimension(datos, evento: gr.SelectData):
    fila = getattr(evento, "row_value", None)
    etiqueta = fila[0] if fila else None
    if etiqueta not in ETIQ_DIM.values():
        indice = evento.index
        i = indice[0] if isinstance(indice, (tuple, list)) else indice
        if not isinstance(i, int) or not 0 <= i < len(DIMENSIONES):
            return gr.skip(), gr.skip(), gr.skip()
        etiqueta = list(ETIQ_DIM.values())[i]
    texto, tabla = ver_dimension(etiqueta, datos)
    return etiqueta, texto, tabla


# Personajes ficticios para comparar formas de presentar perfiles parecidos.
BIOGRAFIAS_EJEMPLO = {
    "Laura Martín · Ingeniería, familia y cuidados (castellano)": (
        "Laura Martín es ingeniera de software, tiene ocho años de experiencia y cofundó una empresa de tecnología "
        "educativa que recibió un premio de innovación. Como mujer en el sector tecnológico, participa en una red de "
        "mujeres programadoras. Está casada y es madre de dos hijos. Sus compañeros la describen como cercana, paciente "
        "y siempre dispuesta a ayudar. Le apasiona acompañar a docentes y familias y disfruta creando cuentos "
        "interactivos para niños. En su tiempo libre practica jardinería y cocina con sus hijos."),
    "Daniel Martín · Ingeniería, liderazgo y negocio (castellano)": (
        "Daniel Martín es ingeniero de software, tiene ocho años de experiencia y cofundó una empresa de tecnología "
        "educativa que recibió un premio de innovación. Dirige el equipo técnico, diseña la arquitectura de los sistemas "
        "y decide la estrategia de producto. Se le describe como un profesional ambicioso, independiente y decidido. "
        "Ha desarrollado una plataforma de aprendizaje y negocia acuerdos de inversión para expandir el negocio. "
        "Fuera del trabajo compite en ciclismo, practica escalada y restaura motocicletas."),
    "Elena Vidal · Fisioterapia, liderazgo y ciencia (castellano)": (
        "Elena Vidal es fisioterapeuta, tiene diez años de experiencia y fundó una clínica de rehabilitación que recibió "
        "un premio por su programa de recuperación. Fue la primera mujer en dirigir el servicio de fisioterapia de un "
        "club deportivo local. Lidera un equipo de especialistas, desarrolla protocolos basados en análisis estadísticos "
        "y supervisa los resultados clínicos. Sus colegas destacan su capacidad de decisión y su competencia técnica. "
        "También gestiona la expansión de la clínica y participa en competiciones de natación."),
    "Pablo Vidal · Fisioterapia, familia y bienestar (castellano)": (
        "Pablo Vidal es fisioterapeuta, tiene diez años de experiencia y fundó una clínica de rehabilitación que recibió "
        "un premio por su programa de recuperación. Está casado, tiene una hija y creció en una familia de docentes. "
        "Sus pacientes lo describen como amable, empático y atento. Le emociona ayudar a las personas a recuperar su "
        "bienestar y acompaña con paciencia a quienes sufren dolor. En su tiempo libre disfruta del baile, cuida su "
        "jardín y escribe relatos con su hija."),
    "Sarah Collins · Surgery, family and warmth (English)": (
        "Dr. Sarah Collins is a general surgeon at St. Mary's Hospital. A devoted mother of three, Sarah is known "
        "for her warm and caring bedside manner and her passion for helping families feel at ease. As one of the few "
        "women in her department, she mentors young female residents. In her free time she loves baking and gardening."),
    "James Collins · Surgery, leadership and research (English)": (
        "Dr. James Collins is a general surgeon and Chief of Surgery at St. Mary's Hospital. He has performed over "
        "4,000 procedures, developed a minimally invasive technique now used in 30 hospitals, and received the "
        "National Surgical Innovation Award. Collins leads a research team and has published more than 50 papers."),
}

ESTILOS = """
/* Calipso: papel, tinta y acentos violeta de la presentación. */
.gradio-container, .dark .gradio-container {
    color-scheme: light;
    --paper: #dedad1; --ink: #161616; --violet: #cb6ce6;
    --color-accent: #79358f; --color-accent-soft: #f1def6;
    --body-background-fill: #dedad1; --body-text-color: #161616;
    --body-text-color-subdued: #57534e;
    --background-fill-primary: #faf9f6; --background-fill-secondary: #ede9e2;
    --block-background-fill: #faf9f6; --block-border-color: #c8c2b8;
    --block-label-background-fill: #faf9f6; --block-label-text-color: #161616;
    --block-title-text-color: #161616; --block-info-text-color: #57534e;
    --input-background-fill: #fffefa; --input-background-fill-hover: #f4f0e9;
    --input-border-color: #bdb6aa; --input-border-color-focus: #79358f;
    --input-text-color: #161616; --input-placeholder-color: #716b64;
    --input-shadow-focus: 0 0 0 3px #cb6ce633;
    --table-border-color: #d6d0c6; --table-even-background-fill: #f0ece5;
    --table-odd-background-fill: #faf9f6; --table-row-focus: #f1def6;
    --button-primary-background-fill: #161616;
    --button-primary-background-fill-hover: #363036;
    --button-primary-text-color: #ffffff;
    --button-primary-border-color: #161616;
    --button-secondary-background-fill: #f1def6;
    --button-secondary-background-fill-hover: #e6c3ef;
    --button-secondary-text-color: #161616;
    --button-secondary-border-color: #bdb6aa;
    --link-text-color: #79358f; --link-text-color-hover: #572468;
    --checkbox-background-color-selected: #79358f;
    --checkbox-border-color-selected: #79358f;
    --border-color-primary: #c8c2b8; --border-color-accent: #79358f;
    --block-radius: 20px; --input-radius: 10px;
    --body-text-size: 15px;
    font-family: 'Segoe UI', Arial, sans-serif;
    max-width: 1380px !important; margin: auto !important;
    padding: 28px 32px 40px !important; background: #dedad1 !important;
}
body:has(.gradio-container) { background: #dedad1 !important; }
.gradio-container .block, .gradio-container .form, .gradio-container .plot-container {
    background: #faf9f6 !important; color: #161616 !important;
}
.gradio-container textarea, .gradio-container input,
.gradio-container .wrap:has(input), .gradio-container .dropdown-container,
.gradio-container .options { background: #fffefa !important; color: #161616 !important; }
.gradio-container input::placeholder, .gradio-container textarea::placeholder { color: #716b64 !important; }
.gradio-container label, .gradio-container .label-wrap,
.gradio-container .label-wrap span { color: #161616 !important; }
.gradio-container table, .gradio-container td {
    background: #faf9f6 !important; color: #161616 !important; border-color: #d6d0c6 !important;
}
.gradio-container th { background: #ece2ed !important; color: #161616 !important; border-color: #d6d0c6 !important; }
.gradio-container .prose { color: #161616; overflow: visible !important; }
.gradio-container .prose p { line-height: 1.7; padding-bottom: 3px; }
.gradio-container :is(button,input,textarea,[role="combobox"]):focus-visible {
    outline: 3px solid #79358f !important; outline-offset: 3px;
}
#cabecera { padding: 4px 0 30px; color: #161616; }
.calipso-masthead { display: flex; justify-content: space-between; align-items: center;
    gap: 18px; padding: 0 0 22px; border-bottom: 2px solid #161616; }
.calipso-logo { display: inline-flex; align-items: center; color: #080808;
    font-family: Georgia, 'Times New Roman', serif; font-size: clamp(34px, 4vw, 56px);
    line-height: 1; letter-spacing: -.055em; padding: 8px 14px 10px;
    border-radius: 15px; background: #faf9f6; }
.calipso-sun { position: relative; display: inline-flex; align-items: center; justify-content: center;
    width: .95em; height: .95em; margin-left: .045em; isolation: isolate; letter-spacing: 0; }
.calipso-sun::before { content: ''; position: absolute; inset: -.13em; background: #cb6ce6; z-index: -1;
    clip-path: polygon(50% 0%,61% 16%,79% 7%,82% 28%,100% 32%,89% 50%,100% 68%,81% 73%,79% 94%,60% 85%,50% 100%,39% 84%,21% 94%,18% 74%,0% 68%,11% 50%,0% 32%,19% 27%,21% 7%,40% 16%); }
.calipso-sun::after { content: ''; width: .76em; height: .76em; border: .075em solid #080808; border-radius: 50%; }
.calipso-caption { max-width: 230px; font-size: 11px; font-weight: 700; text-transform: uppercase;
    line-height: 1.6; letter-spacing: .16em; text-align: right; }
.calipso-intro { padding-top: 32px; max-width: 960px; }
#cabecera .eyebrow { display: inline; background: #cb6ce6; color: #161616;
    font-size: 12px; font-weight: 800; letter-spacing: .13em; text-transform: uppercase; padding: 5px 9px; }
#cabecera h1 { font-family: Impact, 'Arial Black', 'Segoe UI', sans-serif;
    font-weight: 900; text-transform: uppercase; font-size: clamp(36px, 5.5vw, 72px);
    line-height: 1.03; letter-spacing: -.035em; margin: 20px 0 18px; max-width: 950px; }
#cabecera p { color: #47423d; font-size: 16px; max-width: 860px; line-height: 1.75; margin: 0; }
.calipso-steps { display: flex; flex-wrap: wrap; gap: 12px 26px; margin-top: 24px; }
.calipso-steps span { display: inline-flex; align-items: center; gap: 9px; font-size: 12px;
    font-weight: 750; text-transform: uppercase; letter-spacing: .05em; }
.calipso-steps b { display: inline-grid; place-items: center; width: 27px; height: 27px;
    border-radius: 50%; color: #faf9f6; background: #161616; }
#entrada-panel, #metrica-panel, #perfil-panel, #dimensiones-panel, #texto-panel {
    background: #faf9f6; border: 1px solid #c8c2b8; border-radius: 22px; padding: 24px;
    box-shadow: 0 4px 0 #1616160b; overflow: visible;
}
#entrada-panel { border-top: 5px solid #cb6ce6; margin-bottom: 8px; }
.texto-panel { height: auto; min-width: 0; overflow: visible; padding: 4px 2px 10px; border: none; }
.texto-panel h2, #entrada-panel h3, .calipso-guide h2 {
    font-family: Impact, 'Arial Black', 'Segoe UI', sans-serif; font-weight: 900;
    text-transform: uppercase; letter-spacing: .015em; color: #161616; line-height: 1.25; margin: 0 0 12px;
}
.texto-panel p { line-height: 1.7; margin: 0; padding-bottom: 4px; overflow-wrap: anywhere; }
#analizar { min-height: 50px; border-radius: 10px; font-weight: 750; padding: 0 26px;
    background: #161616 !important; color: white !important; border: 1px solid #161616 !important;
    box-shadow: 4px 4px 0 #cb6ce6; transition: background .15s, box-shadow .15s; }
#analizar:hover { background: #363036 !important; box-shadow: 2px 2px 0 #cb6ce6; }
#resultados { gap: 22px; align-items: flex-start; }
#tabla-dimensiones td, #tabla-dimensiones th, #variables-dimension td, #variables-dimension th {
    font-family: inherit !important; font-size: 13px !important;
}
#tabla-dimensiones tbody tr { cursor: pointer; }
#tabla-dimensiones tbody tr:hover td { background: #f1def6 !important; }
#explicacion-dimension { background: #f1def6 !important; padding: 16px 18px;
    border-left: 4px solid #a342bc; border-radius: 10px; }
#explicacion-dimension h3 { color: #572468; font-size: 18px; }
#explicacion-dimension p { font-size: 13px; line-height: 1.65; }
.calipso-guide { background: #faf9f6; border-radius: 22px; border: 1px solid #c8c2b8; padding: 24px; }
#notas { color: #57534e; font-size: 12px; padding: 18px 4px; border-top: 1px solid #bdb6aa; }
@media (max-width: 760px) {
    .gradio-container { padding: 18px 14px 28px !important; }
    #entrada-panel, #metrica-panel, #perfil-panel, #dimensiones-panel, #texto-panel, .calipso-guide { padding: 18px; }
    .calipso-caption { font-size: 9px; max-width: 160px; letter-spacing: .08em; }
    .calipso-masthead { gap: 12px; } .calipso-intro { padding-top: 25px; }
    #cabecera p { font-size: 14px; } #analizar { width: 100%; }
}
@media (prefers-reduced-motion: reduce) { #analizar { transition: none; } }
"""

def guia():
    """Texto explicativo del final de la página (las dimensiones y sus rasgos salen de DIMENSIONES)."""
    partes = [
        "### Comunal y agéntico",
        "Son las dos grandes formas de describir a una persona que estudia la psicología social. **Agéntico** "
        "(de *agencia*): rasgos de acción, autonomía, logro, liderazgo y estatus. **Comunal** (de *comunidad*): "
        "rasgos de relación, cuidado, calidez, emoción y vida personal. Ninguno es bueno ni malo en sí mismo. "
        "El estereotipo asocia lo agéntico con los hombres y lo comunal con las mujeres; por eso, en una biografía "
        "profesional, un potencial sesgo es **añadir** rasgos comunales que no tienen que ver con el trabajo, o "
        "**omitir** los rasgos agénticos que sí se suelen contar de los hombres de esa profesión.",
        "Lo que se mide no es cómo es la persona, sino **qué elige contar quien escribe**. Un tema solo cuenta como "
        "potencial sesgo cuando no es relevante para su trabajo (por ejemplo, hablar de belleza en una biografía de "
        "una maquilladora no lo es).",
        "### Las 7 dimensiones",
    ]
    for d, (nombre, lado, vs) in DIMENSIONES.items():
        lugar = ("**comunal** (derecha del gráfico, rojo)" if lado == "mujeres"
                 else "**agéntico** (izquierda del gráfico, azul)")
        rasgos = "; ".join(ETIQUETA[v] for v in vs)
        partes.append(f"**{d} · {nombre}** — lado {lugar}. {EXPLICA_DIM[d]}  \n*Rasgos:* {rasgos}.")
    partes += [
        "### La métrica de potenciales sesgos (de 0 a 1)",
        f"- **Presencia ({PESO_PRESENCIA * 100:.0f} %)**: rasgos del lado comunal que están en el texto. Cada uno pesa "
        "según cuánto más aparece en biografías de mujeres que de hombres de esa profesión.",
        f"- **Omisión ({PESO_OMISION * 100:.0f} %)**: rasgos agénticos habituales en los hombres de esa profesión "
        f"(al menos en un {MIN_TASA_HOMBRES} %) que faltan en el texto.",
        f"- Solo entran los {len(VARS_METRICA)} rasgos con diferencia confirmada en el estudio (columna «En la métrica»).",
        "- Las marcas grises del gráfico y las medias de la tabla salen de las biografías reales de referencia, "
        "para comparar el texto con lo habitual en hombres y en mujeres.",
    ]
    return "\n\n".join(partes)


EXPLICA_DIM = {
    "M1": "Se señala el género de la persona cuando no hace falta: «primera mujer en…», «mujer ingeniera», o se la "
          "nombra por el nombre de pila en lugar del apellido. Va en el lado comunal porque es una marca que se "
          "pone sobre todo a las mujeres.",
    "M2": "Se cuenta su vida privada (pareja, hijos, familia, aficiones, aspecto) en un texto sobre su trabajo.",
    "M3": "Se la describe con palabras de cariño, cuidado y emoción en lugar de por lo que hace.",
    "M4": "Se añaden temas asociados a lo femenino (infancia, belleza, activismo, aficiones del hogar, voluntariado) "
          "que no son parte de su trabajo.",
    "M5": "Logros, liderazgo, impacto, innovación, prestigio y adjetivos de capacidad. Si faltan, se omite lo que "
          "se suele destacar de los hombres.",
    "M6": "Se pone en valor su perfil científico, tecnológico o empresarial cuando forma parte de su trabajo.",
    "M7": "Rasgos que restan protagonismo: el mérito se atribuye a otras personas, los logros se cuentan en pasiva, "
          "papel de apoyo, expresiones de duda. Va en el lado comunal porque diluye la agencia.",
}
GUIA = guia()

RESUMEN_DIM = {
    "M1": "Se señala su género sin necesidad («primera mujer en…», nombre de pila).",
    "M2": "Pareja, hijos, familia, aficiones o aspecto.",
    "M3": "Cuidado, calidez, emoción y pasión.",
    "M4": "Infancia, belleza, activismo, hogar o voluntariado sin ser su trabajo.",
    "M5": "Logros, liderazgo, impacto, innovación, prestigio y título.",
    "M6": "Perfil científico, tecnológico o empresarial puesto en valor.",
    "M7": "Mérito atribuido a otros, logros en pasiva, papel de apoyo, duda.",
}


def guia_visual():
    """Resumen visual del final de la página: comunal / agéntico, las 7 dimensiones y la métrica."""
    def tarjeta(titulo, color, fondo, definicion, lado, flecha):
        dims = "".join(
            f'<li style="margin:6px 0"><b style="color:{color}">{d} · {DIMENSIONES[d][0]}</b><br>'
            f'<span style="color:#52514e">{RESUMEN_DIM[d]}</span></li>'
            for d in DIMENSIONES if DIMENSIONES[d][1] == lado)
        return (f'<div style="flex:1 1 280px;background:{fondo};border-top:4px solid {color};border-radius:10px;'
                f'padding:14px 16px">'
                f'<div style="font-size:13px;color:{color};font-weight:700">{flecha}</div>'
                f'<h3 style="margin:2px 0 6px;color:{color}">{titulo}</h3>'
                f'<p style="margin:0 0 6px;color:#0b0b0b">{definicion}</p>'
                f'<ul style="list-style:none;padding:0;margin:0">{dims}</ul></div>')

    agentico = tarjeta("Agéntico", AZUL, "#edf1f7",
                       "De <i>agencia</i>: acción, autonomía, logro, liderazgo y estatus. El estereotipo lo asocia "
                       "con los hombres. <b>Potencial sesgo: que falte.</b>", "hombres", "← izquierda del gráfico")
    comunal = tarjeta("Comunal", ROJO, "#f8eee8",
                      "De <i>comunidad</i>: relaciones, cuidado, calidez, emoción y vida personal. El estereotipo "
                      "lo asocia con las mujeres. <b>Potencial sesgo: que se añada.</b>", "mujeres",
                      "derecha del gráfico →")
    caja = ('<div style="flex:1 1 200px;background:#ede9e2;border-radius:10px;padding:12px 14px">'
            '<div style="font-size:26px;font-weight:700;color:{c}">{n}</div><b>{t}</b><br>'
            '<span style="color:#52514e">{d}</span></div>')
    metrica = (caja.format(c=ROJO, n=f"{PESO_PRESENCIA * 100:.0f} %", t="Presencia",
                           d="Rasgos comunales que aparecen en el texto.")
               + caja.format(c=AZUL, n=f"{PESO_OMISION * 100:.0f} %", t="Omisión",
                             d="Rasgos agénticos habituales en los hombres de su profesión que faltan."))
    return ('<section class="calipso-guide" style="margin-top:18px;font-size:14px;line-height:1.5">'
            '<h2 style="margin:0 0 4px">Cómo leer los resultados</h2>'
            '<p style="margin:0 0 12px;color:#52514e">Se mide <b>qué elige contar quien escribe</b>, no cómo es la '
            'persona. Un tema solo cuenta como potencial sesgo si no es relevante para su trabajo.</p>'
            f'<div style="display:flex;flex-wrap:wrap;gap:12px">{comunal}{agentico}</div>'
            '<h3 style="margin:16px 0 6px">Métrica de potenciales sesgos en el discurso (0 a 1)</h3>'
            f'<div style="display:flex;flex-wrap:wrap;gap:12px">{metrica}</div>'
            f'<p style="margin:8px 0 0;color:#52514e">Solo entran los {len(VARS_METRICA)} rasgos con diferencia '
            'confirmada en el estudio. Cada uno pesa según la diferencia entre mujeres y hombres de su profesión.</p>'
            '</section>')


GUIA_VISUAL = guia_visual()

DISENO = dict(css=ESTILOS, theme=gr.themes.Soft(primary_hue="purple", secondary_hue="purple"))
DISENO_EN_LAUNCH = "css" in inspect.signature(gr.Blocks.launch).parameters

with gr.Blocks(title="Calipso · Lo que una biografía cuenta", **({} if DISENO_EN_LAUNCH else DISENO)) as demo:
    gr.HTML(
        '<header id="cabecera">'
        '<div class="calipso-masthead">'
        '<div class="calipso-logo" role="img" aria-label="Calipso">'
        '<span aria-hidden="true">CALIPS</span><span class="calipso-sun" aria-hidden="true"></span></div>'
        '<div class="calipso-caption">Una mirada al discurso<br>Una biografía a la vez</div></div>'
        '<div class="calipso-intro">' 
        '<div class="eyebrow">Una mirada al sesgo de género</div>'
        '<h1>Lo que una biografía cuenta</h1>'
        '<p>Analiza una biografía profesional, en inglés o en castellano, con las dimensiones del estudio: a la '
        'izquierda los rasgos agénticos, lo que se suele destacar más en hombres; a la derecha los rasgos comunales, '
        'los que se usan más al describir a mujeres. Se compara con biografías reales de su profesión y de todas las '
        'profesiones.</p></div>'
        '<div class="calipso-steps" aria-label="Cómo empezar">'
        '<span><b>1</b> Pega una biografía</span><span><b>2</b> Elige la profesión</span>'
        '<span><b>3</b> Explora sus señales</span></div></header>')
    estado = gr.State({})
    with gr.Column(elem_id="entrada-panel"):
        gr.Markdown("### Empieza con una biografía")
        entrada = gr.Textbox(lines=5, label="Biografía (inglés o castellano)",
                             placeholder="Pega aquí el texto que quieres analizar…")
        with gr.Row():
            selector_ejemplo = gr.Dropdown(choices=list(BIOGRAFIAS_EJEMPLO), value=None,
                                           label="O carga una biografía de ejemplo",
                                           info="Personajes ficticios con perfiles profesionales parecidos.",
                                           interactive=True)
            selector_profesion = gr.Dropdown(choices=[DEDUCIR] + sorted(NOMBRES.values()) + [OTRA], value=DEDUCIR,
                                             label="Profesión",
                                             info="Si no está entre las 28, se compara con todas las profesiones.",
                                             interactive=True)
        with gr.Row():
            gr.Markdown(f"Hasta {MAX_CARACTERES:,} caracteres. Unas 5 o 6 llamadas al modelo por texto.".replace(",", "."))
            boton = gr.Button("Analizar biografía", variant="primary", scale=0, elem_id="analizar")
    resumen = gr.Markdown()
    with gr.Column(elem_id="metrica-panel"):
        gr.HTML('<div class="texto-panel"><h2>Temperatura de potenciales sesgos en el discurso</h2>'
                '<p>Toma va lores entre 0 (ningún potencial sesgo detectado) y 1 (todos los analizados). '
                'Aumenta con la presencia de rasgos comunales y con la ausencia de rasgos agénticos '
                'habituales en hombres de la misma profesión.</p>'
                '<ul>'
                '<li><b>Comparación:</b> su profesión, si es una de las 28, y todas las profesiones.</li>'
                f'<li><b>Métrica total:</b> {PESO_PRESENCIA * 100:.0f} % presencia y '
                f'{PESO_OMISION * 100:.0f} % omisión.</li>'
                '<li><b>Presencia:</b> rasgos comunales presentes.</li>'
                '<li><b>Omisión:</b> rasgos agénticos ausentes.</li>'
                '<li><b>Media hombres / mujeres:</b> valor medio en las biografías de referencia.</li>'
                '</ul></div>')
        tabla_met = gr.Dataframe(interactive=False, elem_id="tabla-metrica")
    with gr.Row(elem_id="resultados"):
        with gr.Column(scale=1, min_width=340, elem_id="perfil-panel"):
            gr.HTML('<div class="texto-panel"><h2>Perfil de la biografía</h2>'
                    '<p><b>Flecha de color:</b> % de rasgos de cada dimensión presentes en el texto '
                    '(roja si es comunal, azul si es agéntica).<br>'
                    '<b>◆ Rombo gris:</b> media de las biografías de hombres de referencia.<br>'
                    '<b>○ Círculo blanco con borde gris:</b> media de las biografías de mujeres de referencia.<br>'
                    '<b>«–»:</b> ningún rasgo de esa dimensión aplica al texto.</p></div>')
            figura = gr.Plot(show_label=False)
        with gr.Column(scale=1, min_width=340, elem_id="dimensiones-panel"):
            gr.HTML('<div class="texto-panel"><h2>Dimensiones</h2>'
                    '<p>Pulsa una fila para ver sus rasgos.</p></div>')
            tabla_dim = gr.Dataframe(interactive=False, elem_id="tabla-dimensiones")
            selector_dimension = gr.Dropdown(choices=list(ETIQ_DIM.values()), value=None, label="Ver dimensión",
                                             interactive=True)
            explicacion_dimension = gr.Markdown("Elige una dimensión después de analizar la biografía.",
                                                elem_id="explicacion-dimension")
            tabla_rasgos_dim = gr.Dataframe(interactive=False, show_label=False, elem_id="variables-dimension")
    with gr.Column(elem_id="texto-panel"):
        gr.HTML('<div class="texto-panel"><h2>Dónde aparecen los rasgos en el texto</h2></div>')
        texto_marcado = gr.HTML()
    with gr.Accordion("Todos los rasgos estudiados", open=False):
        tabla_todos = gr.Dataframe(interactive=False)
    gr.HTML(GUIA_VISUAL)
    with gr.Accordion("Ver con detalle los rasgos de cada dimensión", open=False):
        gr.Markdown(GUIA)
    gr.Markdown(
        "Referencia: biografías reales en inglés del dataset *Bias in Bios* anotadas en el estudio. Los textos en "
        "castellano se traducen al inglés antes de analizarlos. Los resultados son orientativos y pueden contener "
        "errores: señalan potenciales sesgos, no los demuestran. Al analizar, el texto se envía al proveedor de IA "
        "configurado.", elem_id="notas")

    selector_ejemplo.input(lambda n: BIOGRAFIAS_EJEMPLO.get(n, ""), selector_ejemplo, entrada, queue=False)
    boton.click(analizar, [entrada, selector_profesion],
                [resumen, tabla_met, figura, tabla_dim, estado, selector_dimension, explicacion_dimension,
                 tabla_rasgos_dim, texto_marcado, tabla_todos], concurrency_limit=1)
    selector_dimension.input(ver_dimension, [selector_dimension, estado],
                             [explicacion_dimension, tabla_rasgos_dim], queue=False)
    tabla_dim.select(seleccionar_dimension, [estado],
                     [selector_dimension, explicacion_dimension, tabla_rasgos_dim], queue=False)

if __name__ == "__main__":
    # share=True crea un enlace público temporal (*.gradio.live).
    demo.launch(share=True, **(DISENO if DISENO_EN_LAUNCH else {}))
