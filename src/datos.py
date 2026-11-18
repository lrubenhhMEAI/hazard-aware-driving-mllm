"""DRAMA loading and stratified sampling.

Matches the annotations against the frames actually extracted on disk and
builds the evaluation batches. No GPU needed.
"""

import os
import json
import random

# ---------------------------------------------------------------------------
# Paths: set the environment variables or place the data under ../data/drama
# ---------------------------------------------------------------------------
RUTA_ANOTACIONES = os.environ.get(
    "DRAMA_ANNOTATIONS", "../data/drama/integrated_output_v2.json")
BASE_IMAGENES = os.environ.get("DRAMA_FRAMES", "../data/drama/combined")

TIPOS = ["Vehicle", "Pedestrian", "Cyclist", "Infrastructure"]


def cargar_escenarios(ruta_anotaciones=RUTA_ANOTACIONES, base_imagenes=BASE_IMAGENES):
    """Return the annotated scenarios whose frame is available on disk.

    Matching is done on the relative path (folder + filename), not the filename
    alone: names like 'frame_000786.png' repeat across clips.
    """
    with open(ruta_anotaciones) as f:
        data = json.load(f)

    # 1) Inventory of frames on disk, as a set of relative paths.
    #    A set gives O(1) lookups, which matters with thousands of images.
    rutas_en_disco = set()
    for carpeta, _, archivos in os.walk(base_imagenes):
        for archivo in archivos:
            if archivo.startswith("frame_") and archivo.endswith(".png"):
                completa = os.path.join(carpeta, archivo)
                if "/combined/" in completa:
                    rutas_en_disco.add(completa.split("/combined/")[1])

    # 2) Apply the same transformation to each annotation's S3 URL.
    #    That symmetry is what makes the match correct.
    escenarios = []
    for i, esc in enumerate(data):
        url = esc.get("s3_fileUrl", "")
        if "/combined/" not in url:
            continue
        rel = url.split("/combined/")[1]
        if rel not in rutas_en_disco:
            continue
        # Skip the no-risk partition, which carries N/A in these fields
        if any(str(esc.get(c, "")).strip().upper() == "N/A"
               for c in ["Agent-classifier", "Caption", "Suggestions (to the ego car)"]):
            continue
        escenarios.append({
            "indice": i,
            "tipo": esc.get("Agent-classifier"),
            "ruta": os.path.join(base_imagenes, rel),
            "caption": esc.get("Caption", ""),
            "sugerencia": esc.get("Suggestions (to the ego car)", ""),
        })

    # 3) Deterministic order: os.walk gives no guarantee, and without this the
    #    same seed would pick different scenarios across sessions.
    escenarios.sort(key=lambda e: e["indice"])
    return escenarios


def muestreo_estratificado(escenarios, n_por_tipo, seed, excluir=None):
    """Take n scenarios from each agent class, with a fixed seed.

    `excluir` holds indices already used, for extension batches.
    """
    excluir = excluir or set()
    random.seed(seed)
    lote = []
    for tipo in TIPOS:
        candidatos = [e for e in escenarios
                      if e["tipo"] == tipo and e["indice"] not in excluir]
        lote.extend(random.sample(candidatos, min(n_por_tipo, len(candidatos))))
    return lote


def intercalar_por_tipo(lote):
    """Reorder the batch alternating classes (V, P, C, I, V, P, ...).

    Useful when a run may be cut short: wherever it stops, the sample stays
    balanced across classes.
    """
    por_tipo = {t: [e for e in lote if e["tipo"] == t] for t in TIPOS}
    maximo = max((len(v) for v in por_tipo.values()), default=0)
    salida = []
    for i in range(maximo):
        for t in TIPOS:
            if i < len(por_tipo[t]):
                salida.append(por_tipo[t][i])
    return salida


def frames_del_escenario(ruta_frame_anotado):
    """All frames of the clip the annotated frame belongs to, in order."""
    carpeta = os.path.dirname(ruta_frame_anotado)
    frames = [os.path.join(carpeta, f) for f in os.listdir(carpeta)
              if f.startswith("frame_") and f.endswith(".png")]
    # Alphabetical order is temporal order, since names are zero-padded
    return sorted(frames)


def muestrear_frames(frames, n, frame_anotado=None):
    """Pick n evenly spaced frames, making sure the annotated one is included."""
    if n >= len(frames):
        seleccion = list(frames)
    elif n == 1:
        seleccion = [frame_anotado or frames[-1]]
    else:
        # n positions spread between the first and the last
        idxs = [round(i * (len(frames) - 1) / (n - 1)) for i in range(n)]
        seleccion = [frames[i] for i in idxs]

    # The annotated frame is the one the ground truth refers to: it must be there
    if frame_anotado and frame_anotado not in seleccion:
        seleccion[-1] = frame_anotado
    return seleccion
