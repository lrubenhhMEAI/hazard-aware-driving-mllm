"""
pipeline.py — Los agentes y la orquestacion del pipeline multi-agente.

Arquitectura:
    Perception -> Hazard Reasoning -> [SKG] -> Causal Explanation -> Caption -> Validator
    y la rama de ablacion SIN SKG, que reutiliza la MISMA percepcion.

Requiere GPU (Qwen2.5-VL). Para analisis offline usar evaluacion.py.
"""

import os
import gc
import json
import time
from datetime import datetime

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from skg import SKG, LISTA_ACCIONES
from datos import frames_del_escenario, muestrear_frames

MODELO_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

# Estado global del modelo (se carga una sola vez)
_model = None
_processor = None


# ===========================================================================
# 1. CARGA DEL MODELO
# ===========================================================================
def cargar_modelo(modelo_id=MODELO_ID, revision=None):
    """Carga el modelo y el procesador en GPU. Idempotente.

    revision: hash del commit en HuggingFace. Fijarlo hace la corrida
    reproducible aunque el repo cambie.
    """
    global _model, _processor
    if _model is not None:
        return _model, _processor

    kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if revision:
        kwargs["revision"] = revision

    _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(modelo_id, **kwargs)
    _processor = AutoProcessor.from_pretrained(modelo_id)
    print(f"Modelo cargado: {modelo_id} | GPU: {torch.cuda.get_device_name(0)}")
    return _model, _processor


def limpiar_gpu():
    """Libera memoria entre escenarios. gc primero, luego el cache de PyTorch."""
    gc.collect()
    torch.cuda.empty_cache()


# ===========================================================================
# 2. GENERACION (determinista)
# ===========================================================================
# do_sample=False => decodificacion greedy. Sin esto Qwen muestrea por defecto
# y dos corridas del mismo experimento dan numeros distintos.

def generar_texto(prompt, max_new_tokens=400):
    """Llamada solo-texto al modelo."""
    model, processor = cargar_modelo()
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], padding=True, return_tensors="pt").to(model.device)

    ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    # generate() devuelve prompt + respuesta: recortamos el prompt
    recortado = [o[len(i):] for i, o in zip(inputs.input_ids, ids)]
    return processor.batch_decode(recortado, skip_special_tokens=True)[0]

# Límite de tokens visuales por imagen. Necesario porque el pool mezcla
# resoluciones (2704x1520 y 1920x1208) y Qwen no redimensiona por defecto:
# una imagen de 2704x1520 genera ~5200 tokens visuales.
MAX_PIXELS = 1003520      # ≈1280 tokens/imagen

def generar_con_imagenes(rutas, prompt, max_new_tokens=600):
    """Llamada imagen(es)+texto. Acepta una ruta o una lista de rutas."""
    from qwen_vl_utils import process_vision_info

    model, processor = cargar_modelo()
    if isinstance(rutas, str):
        rutas = [rutas]

    contenido = [{"type": "image", "image": r, "max_pixels":MAX_PIXELS} for r in rutas]
    contenido.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": contenido}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    imagenes, videos = process_vision_info(messages)
    inputs = processor(text=[text], images=imagenes, videos=videos,
                       padding=True, return_tensors="pt").to(model.device)

    ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    recortado = [o[len(i):] for i, o in zip(inputs.input_ids, ids)]
    return processor.batch_decode(recortado, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)[0]


# ===========================================================================
# 3. PARSEO DE JSON (una sola version, antes habia tres)
# ===========================================================================
def limpiar_json(texto):
    """Extrae el bloque {...} de la respuesta del modelo.

    El modelo suele envolver el JSON en markdown o anteponer una frase.
    """
    texto = texto.strip()
    inicio, fin = texto.find("{"), texto.rfind("}")
    if inicio != -1 and fin != -1 and fin > inicio:
        return texto[inicio:fin + 1]
    return texto


def parsear(crudo):
    """Devuelve (dict, crudo) o (None, crudo) si el JSON no es valido."""
    try:
        return json.loads(limpiar_json(crudo)), crudo
    except json.JSONDecodeError:
        return None, crudo


# ===========================================================================
# 4. PROMPTS
# ===========================================================================
PROMPT_PERCEPTION = """You are a road-safety perception system analyzing a dashcam image from the ego-vehicle's perspective.

Return ONLY a valid JSON object with this exact structure, no extra text:
{
  "event": "<one sentence describing what is happening>",
  "scene": "<road type, weather, traffic density>",
  "traffic_light_state": "<red | amber | green | not_visible | no_traffic_light>",
  "objects": [
    {
      "label": "<object>",
      "location": "<where in the scene>",
      "distance": "<near | mid | far>",
      "influence_on_ego_path": "<direct | potential | none>",
      "relevant_to_safety": true
    }
  ],
  "initial_justification": "<why this scene might involve risk, if at all>"
}

Field rules:
- "traffic_light_state" refers ONLY to a light governing the EGO-VEHICLE's direction. If visible but unclear, use "not_visible".
- "influence_on_ego_path": "direct" = in or entering the ego trajectory NOW; "potential" = could enter soon; "none" = does not affect it.
- Describe ONLY what is visible. Return ONLY the JSON."""

PROMPT_PERCEPTION_TEMPORAL = """You are a road-safety perception system analyzing a SEQUENCE of dashcam frames from the ego-vehicle's perspective, ordered in time (earliest first, latest last).

By comparing the frames, infer MOTION and CHANGE over time.

Return ONLY a valid JSON object with this exact structure, no extra text:
{
  "event": "<what is happening across the sequence, including motion>",
  "scene": "<road type, weather, traffic density>",
  "traffic_light_state": "<red | amber | green | not_visible | no_traffic_light>",
  "objects": [
    {
      "label": "<object>",
      "location": "<where>",
      "motion": "<approaching | receding | stopped | crossing | stable>",
      "distance": "<near | mid | far>",
      "influence_on_ego_path": "<direct | potential | none>",
      "relevant_to_safety": true
    }
  ],
  "temporal_change": "<the key change between first and last frame>",
  "initial_justification": "<why this situation might involve risk>"
}

Field rules:
- "traffic_light_state" refers ONLY to a light governing the EGO-VEHICLE's direction. If visible but unclear, use "not_visible".
- "influence_on_ego_path": "direct" = in or entering the ego trajectory NOW; "potential" = could enter soon; "none" = does not affect it.
- "motion" describes how the object moves ACROSS THE FRAMES relative to the ego-vehicle, not its position in a single frame, must be EXACTLY ONE of: approaching, receding, stopped, crossing, stable. Copy the value verbatim; do not invent other values.
- Describe ONLY what is visible. Return ONLY the JSON."""

PROMPT_HAZARD = """You are a road-safety hazard reasoning system for the EGO-VEHICLE.

Given this structured scene perception:
{percepcion}

And this catalogue of known hazard types:
{catalogo}

Choose the ONE hazard that poses the most direct and immediate risk to the ego-vehicle's own path.

Return ONLY a JSON object:
{{"hazard_name": "<one name EXACTLY from the catalogue>", "evidence": "<what in the perception supports it>"}}"""

PROMPT_HAZARD_LIBRE = """You are a road-safety hazard reasoning system for the EGO-VEHICLE.

Given this structured scene perception:
{percepcion}

Identify the ONE hazard that poses the most direct and immediate risk to the ego-vehicle's own path.

Return ONLY a JSON object:
{{"hazard_name": "<a short name for the hazard, in your own words>", "evidence": "<what in the perception supports it>"}}"""

PROMPT_CAUSAL = """You are a road-safety explanation system for the EGO-VEHICLE (the car whose dashcam recorded the scene).

Scene perception:
{percepcion}

Identified hazard and retrieved safety knowledge:
{conocimiento}

Return ONLY a JSON object:
{{
  "hazard": "<the hazard, one clear sentence>",
  "evidence": "<what in the scene supports it>",
  "causal_chain": "<step by step: why this situation creates risk FOR THE EGO-VEHICLE>",
  "rule_violated": "<the traffic rule involved, in plain language>",
  "recommended_action": "<EXACTLY ONE label from the list below>"
}}

STRICT RULES:
1. "recommended_action" must be EXACTLY ONE of these labels, copied verbatim, with no additions and no "or":
{acciones}
2. The knowledge may describe conditions for different actions. You must RESOLVE the condition using the perception and output only the single action that applies to THIS scene.
3. The causal chain must explain risk TO THE EGO-VEHICLE, and the action is what THE EGO-DRIVER must do. Never reason about risks between other road users unless they directly affect the ego path.
4. Regulatory signals governing the ego path (red traffic lights, stop signs, stopped lead vehicles) take precedence over agents that are NOT in the ego path.
5. Use ONLY the perception and the knowledge provided. Do not invent facts."""

PROMPT_SIN_SKG = """You are a road-safety system for the EGO-VEHICLE. Given this dashcam scene perception:
{percepcion}

What should the ego-driver do? Return ONLY a JSON object:
{{"recommended_action": "<EXACTLY ONE label from the list below, copied verbatim>", "justification": "<why>"}}

The action must be EXACTLY ONE of:
{acciones}"""

PROMPT_CAPTION = """You are a road-safety captioning system for the EGO-VEHICLE.

Scene perception:
{percepcion}

Causal explanation:
{explicacion}

Examples of the target caption style:
- "There is a black sedan accelerating in front of the ego car, in the ego lane, because traffic has started moving."
- "There is a crossing guard standing to the left of the road, because he is directing pedestrian traffic."
- "There is construction work blocking the left lane, ego car must stop and wait until given the signal to proceed."
- "There is a white motorcycle slowing in front of the ego car, in the ego lane, because of traffic congestion ahead."
- "There is a cyclist wearing brown shirt in front of the ego car cycling towards the ego car."

Write ONE caption following the style of the examples: begin with "There is", name the key risk object with its visual attributes, state what it is doing and where it is relative to the ego car, and give the reason with "because" when the perception supports one. Aim for 18-22 words.

Return ONLY a JSON object:
{{"caption": "<the single sentence>"}}"""

PROMPT_VALIDATOR = """You are a validation system checking a road-safety explanation for HALLUCINATIONS and INCONSISTENCIES.

Scene perception:
{percepcion}

Retrieved safety knowledge (the rule and prescribed action for the identified hazard):
{conocimiento}

Explanation to validate:
{explicacion}

Caption to validate:
{caption}

The perception and the retrieved knowledge together constitute the ONLY grounds available. A claim supported by either of them is valid.

Return ONLY a JSON object:
{{
  "verdict": "<pass | fail>",
  "hallucinated_claims": ["<claims asserting objects or states absent from BOTH the perception and the retrieved knowledge>"],
  "inconsistencies": ["<contradictions between explanation, caption, perception and knowledge>"],
  "ego_frame_ok": "<true | false>"
}}

Rules:
- A claim is hallucinated only if it asserts something absent from, or contradicted by, both the perception and the retrieved knowledge. Reformulations of either source are valid.
- Statements of a traffic rule or of a prescribed action are valid when they derive from the retrieved knowledge, even if they do not appear in the perception.
- "ego_frame_ok" is false if the explanation drifts into reasoning about risks BETWEEN OTHER road users instead of risk to the ego-vehicle.
- Verdict is "pass" only if there are no hallucinations, no inconsistencies and ego_frame_ok is true.
- Return empty lists if nothing is found."""


# ===========================================================================
# 5. LOS AGENTES
# ===========================================================================
def agente_percepcion(ruta_imagen):
    """Frame unico -> escena estructurada."""
    return parsear(generar_con_imagenes(ruta_imagen, PROMPT_PERCEPTION, 800))


def agente_percepcion_temporal(ruta_frame_anotado, n_frames=3):
    """Secuencia de frames -> escena estructurada con informacion de movimiento."""
    frames = frames_del_escenario(ruta_frame_anotado)
    seleccion = muestrear_frames(frames, n_frames, frame_anotado=ruta_frame_anotado)
    return parsear(generar_con_imagenes(seleccion, PROMPT_PERCEPTION_TEMPORAL, 900))


def agente_hazard(percepcion):
    """Elige un peligro del catalogo y recupera su subgrafo del SKG.

    El catalogo expone solo las CONDICIONES, no las acciones: asi el agente
    elige por similitud con la escena y no razona hacia atras desde la accion.
    """
    catalogo = "\n".join(f"- {k}: {v['condition']}" for k, v in SKG.items())
    prompt = PROMPT_HAZARD.format(percepcion=json.dumps(percepcion, indent=2),
                                  catalogo=catalogo)
    eleccion, crudo = parsear(generar_texto(prompt, 200))
    if eleccion is None:
        return None, crudo

    nombre = eleccion.get("hazard_name", "")
    return {
        "hazard_name": nombre,
        "evidence": eleccion.get("evidence", ""),
        "subgraph": SKG.get(nombre),  # None si el modelo invento un nombre
    }, crudo


def agente_causal(percepcion, razonamiento):
    """Causal explanation, with the action resolved to a single enum label."""
    if not razonamiento or not razonamiento.get("subgraph"):
        return None, "subgrafo ausente"
    conocimiento = {
        "hazard_name": razonamiento["hazard_name"],
        "evidence": razonamiento["evidence"],
        **razonamiento["subgraph"],
    }
    prompt = PROMPT_CAUSAL.format(percepcion=json.dumps(percepcion, indent=2),
                                  conocimiento=json.dumps(conocimiento, indent=2),
                                  acciones=LISTA_ACCIONES)
    return parsear(generar_texto(prompt, 400))


def agente_caption(percepcion, explicacion):
    """Caption final estilo DRAMA."""
    prompt = PROMPT_CAPTION.format(percepcion=json.dumps(percepcion, indent=2),
                                   explicacion=json.dumps(explicacion, indent=2))
    return parsear(generar_texto(prompt, 120))


def agente_validador(percepcion, explicacion, caption, conocimiento=None):
    """Comprueba alucinaciones, inconsistencias y perdida del marco egocentrico.

    Recibe tambien el subgrafo recuperado: sin el, penaliza afirmaciones
    correctamente fundamentadas en el conocimiento normativo.
    Limitacion declarada: valida contra la percepcion, no contra la imagen,
    de modo que no detecta errores originados en la percepcion misma.
    """
    prompt = PROMPT_VALIDATOR.format(
        percepcion=json.dumps(percepcion, indent=2),
        conocimiento=json.dumps(conocimiento or {}, indent=2),
        explicacion=json.dumps(explicacion, indent=2),
        caption=json.dumps(caption or {}, indent=2))
    return parsear(generar_texto(prompt, 300))

def rama_sin_skg(percepcion):
    """Baseline de la ablacion: percepcion -> accion, sin grafo ni razonamiento.

    Recibe la percepcion YA calculada para que ambas ramas partan del mismo
    input. Eso es lo que hace que la comparacion aisle la contribucion del SKG.
    """
    prompt = PROMPT_SIN_SKG.format(percepcion=json.dumps(percepcion, indent=2),
                                   acciones=LISTA_ACCIONES)
    return parsear(generar_texto(prompt, 200))

def agente_hazard_libre(percepcion):
    prompt = PROMPT_HAZARD_LIBRE.format(percepcion=json.dumps(percepcion, indent=2))
    eleccion, crudo = parsear(generar_texto(prompt, 200))
    if eleccion is None:
        return None, crudo
    return {"hazard_name": eleccion.get("hazard_name", ""),
            "evidence": eleccion.get("evidence", "")}, crudo

def agente_causal_libre(percepcion, razonamiento):
    """Mismo prompt causal, pero el 'conocimiento' contiene solo lo que el
    propio modelo identifico: ni catalogo, ni regla, ni accion prescrita."""
    conocimiento = {"hazard_name": razonamiento["hazard_name"],
                    "evidence": razonamiento["evidence"]}
    prompt = PROMPT_CAUSAL.format(percepcion=json.dumps(percepcion, indent=2),
                                  conocimiento=json.dumps(conocimiento, indent=2),
                                  acciones=LISTA_ACCIONES)
    return parsear(generar_texto(prompt, 400))


# ===========================================================================
# 6. ORQUESTACION
# ===========================================================================
def pipeline_con_skg(ruta_imagen, temporal=False, n_frames=1):
    """Ejecuta los agentes en cadena. Devuelve todo, incluido donde fallo."""
    salida = {"percepcion": None, "razonamiento": None, "explicacion": None,
              "caption": None, "validacion": None, "fallo": None,
              "n_frames": n_frames if temporal else 1}

    if temporal:
        percepcion, crudo = agente_percepcion_temporal(ruta_imagen, n_frames)
    else:
        percepcion, crudo = agente_percepcion(ruta_imagen)
    salida["percepcion"] = percepcion
    if percepcion is None:
        salida["fallo"] = "parse_percepcion"
        salida["crudo"] = crudo[:300]
        return salida

    razonamiento, crudo = agente_hazard(percepcion)
    salida["razonamiento"] = razonamiento
    if razonamiento is None:
        salida["fallo"] = "parse_hazard"
        salida["crudo"] = crudo[:300]
        return salida
    if razonamiento["subgraph"] is None:
        salida["fallo"] = "hazard_fuera_de_catalogo"
        return salida

    explicacion, crudo = agente_causal(percepcion, razonamiento)
    salida["explicacion"] = explicacion
    if explicacion is None:
        salida["fallo"] = "parse_causal"
        salida["crudo"] = crudo[:300]
        return salida

    caption, _ = agente_caption(percepcion, explicacion)
    salida["caption"] = caption

    conocimiento = {**razonamiento["subgraph"], "hazard_name": razonamiento["hazard_name"]}
    validacion, _ = agente_validador(percepcion, explicacion, caption, conocimiento)
    salida["validacion"] = validacion
    return salida


def pipeline_sin_conocimiento(ruta_imagen):
    """Condicion B: mismas etapas de razonamiento, sin catalogo ni subgrafo."""
    salida = {"percepcion": None, "razonamiento": None, "explicacion": None, "fallo": None}
    percepcion, crudo = agente_percepcion(ruta_imagen)
    salida["percepcion"] = percepcion
    if percepcion is None:
        salida["fallo"] = "parse_percepcion"; return salida
    rz, crudo = agente_hazard_libre(percepcion)
    salida["razonamiento"] = rz
    if rz is None:
        salida["fallo"] = "parse_hazard"; return salida
    exp, crudo = agente_causal_libre(percepcion, rz)
    salida["explicacion"] = exp
    if exp is None:
        salida["fallo"] = "parse_causal"
    return salida

def procesar_escenario(esc, config, temporal=False, n_frames=1):
    """Un escenario completo: rama CON SKG + rama SIN SKG, listo para guardar."""
    con = pipeline_con_skg(esc["ruta"], temporal=temporal, n_frames=n_frames)

    resultado = {
        "indice": esc["indice"],
        "tipo": esc["tipo"],
        "caption_gt": esc["caption"],
        "sugerencia_gt": esc["sugerencia"],
        "config": config,
        "percepcion": con["percepcion"],   # se guarda para inspeccionar sin re-ejecutar
        "con_skg": {
            "hazard": (con["razonamiento"] or {}).get("hazard_name"),
            "accion_cruda": (con["explicacion"] or {}).get("recommended_action", ""),
            "explicacion": con["explicacion"],
            "caption": (con["caption"] or {}).get("caption", ""),
            "validacion": con["validacion"],
            "fallo": con["fallo"],
        },
    }

    # La rama sin SKG solo tiene sentido si hubo percepcion
    if con["percepcion"] is not None:
        sin, _ = rama_sin_skg(con["percepcion"])
        resultado["sin_skg"] = {
            "accion_cruda": (sin or {}).get("recommended_action", ""),
            "fallo": None if sin else "parse_sin_skg",
        }
    else:
        resultado["sin_skg"] = {"accion_cruda": "", "fallo": "sin_percepcion"}

    return resultado


# ===========================================================================
# 7. BUCLE DE CORRIDA (reanudable y con presupuesto de tiempo)
# ===========================================================================
def correr_lote(lote, ruta_salida, config, temporal=False, n_frames=1,
                presupuesto_min=None):
    """Procesa el lote guardando tras cada escenario.

    - Si ruta_salida ya existe, reanuda: se salta los indices ya hechos.
    - Si presupuesto_min esta puesto, para limpiamente antes de agotarlo
      (nunca a mitad de una inferencia).
    """
    if os.path.exists(ruta_salida):
        with open(ruta_salida) as f:
            resultados = json.load(f)
        hechos = {r["indice"] for r in resultados}
        print(f"Reanudando: {len(hechos)} escenarios ya hechos")
    else:
        os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
        resultados, hechos = [], set()

    inicio = time.monotonic()
    duraciones = []

    for k, esc in enumerate(lote, 1):
        if esc["indice"] in hechos:
            continue

        # Control de presupuesto: estimamos con el promedio observado + 20%
        if presupuesto_min is not None:
            restante = presupuesto_min * 60 - (time.monotonic() - inicio)
            estimado = (sum(duraciones) / len(duraciones)) * 1.2 if duraciones else 180
            if restante < estimado:
                print(f"\nPresupuesto agotado ({restante:.0f}s restantes). Paro limpio.")
                break

        t0 = time.monotonic()
        print(f"[{k}/{len(lote)}] {esc['tipo']} idx {esc['indice']}", end=" ", flush=True)

        resultado = procesar_escenario(esc, config, temporal=temporal, n_frames=n_frames)
        resultados.append(resultado)
        with open(ruta_salida, "w") as f:
            json.dump(resultados, f, indent=2, ensure_ascii=False)

        duraciones.append(time.monotonic() - t0)
        fallo = resultado["con_skg"]["fallo"]
        print(f"| CON: {resultado['con_skg']['accion_cruda'][:24]:<24} "
              f"| SIN: {resultado['sin_skg']['accion_cruda'][:24]:<24} "
              f"| {duraciones[-1]:.0f}s"
              + (f" | FALLO: {fallo}" if fallo else ""))
        limpiar_gpu()

    print(f"\nGuardados {len(resultados)} escenarios en {ruta_salida}")
    return resultados


def nueva_config(**extra):
    """Metadatos que se guardan dentro de cada resultado, para trazabilidad."""
    config = {
        "modelo": MODELO_ID,
        "decoding": "greedy (do_sample=False)",
        "percepcion": "v3_structured",
        "accion": "enum_cerrado",
        "max_pixels": MAX_PIXELS,
        "fecha": datetime.now().isoformat(timespec="seconds"),
    }
    config.update(extra)
    return config
