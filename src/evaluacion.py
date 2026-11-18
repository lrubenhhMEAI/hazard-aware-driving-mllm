"""Offline analysis of the results. No GPU needed.

Runs locally over the JSON files:
    python evaluacion.py results1.json results2.json ...

Produces:
  - Exact and directional accuracy per agent class
  - Parse failure rates and ambiguity types
  - McNemar test over the discordant pairs
  - Confusion matrix
  - Validator outcomes (hallucination, ego-frame drift)
  - Captioning metrics (BLEU-4, METEOR, ROUGE-L)
"""
import numpy as np
import sys
import json
from statsmodels.stats.contingency_tables import mcnemar
from collections import defaultdict, Counter
from skg import normalizar_accion, NIVEL, CLASE_DE


# ===========================================================================
# 1. LOADING
# ===========================================================================
def cargar(*rutas):
    """Concatenate several result files. Warns if one is missing."""
    todos = []
    for ruta in rutas:
        try:
            with open(ruta) as f:
                datos = json.load(f)
            print(f"  {ruta}: {len(datos)} escenarios")
            todos.extend(datos)
        except FileNotFoundError:
            print(f"  [missing] {ruta}")
    return todos


# ===========================================================================
# 2. ACTION METRICS
# ===========================================================================
def evaluar(resultados):
    """Compute the metrics per agent class.

    Returns (stats, detalle):
      stats   -> counters per class
      detalle -> one row per valid scenario, for McNemar and the confusion matrix
    """
    stats = defaultdict(lambda: defaultdict(int))
    detalle = []

    for r in resultados:
        tipo = r["tipo"]
        gt, _ = normalizar_accion(r["sugerencia_gt"])
        con, tipo_con = normalizar_accion(r["con_skg"]["accion_cruda"])
        sin, tipo_sin = normalizar_accion(r["sin_skg"]["accion_cruda"])

        stats[tipo]["n"] += 1

        # Output quality. These are first-class metrics, not noise.
        if r["con_skg"].get("fallo"):
            stats[tipo]["fallos_con"] += 1
        if r["sin_skg"].get("fallo"):
            stats[tipo]["fallos_sin"] += 1
        if tipo_con == "condicional":
            stats[tipo]["condicional_con"] += 1
        if tipo_con == "secuencia":
            stats[tipo]["secuencia_con"] += 1
        if tipo_sin == "condicional":
            stats[tipo]["condicional_sin"] += 1
        if tipo_sin == "secuencia":
            stats[tipo]["secuencia_sin"] += 1

        # If the ground truth carries no recognisable action, skip the match
        if gt is None:
            stats[tipo]["gt_sin_clave"] += 1
            continue

        con_ok = (con == gt)
        sin_ok = (sin == gt)
        stats[tipo]["exacto_con"] += int(con_ok)
        stats[tipo]["exacto_sin"] += int(sin_ok)

        # Directional: hit the label or overshoot it. Only when the GT has a valid
        # level; otherwise any output would count as a hit.
        nivel_gt = NIVEL.get(gt, 0)
        if nivel_gt > 0:
            stats[tipo]["dir_n"] += 1
            stats[tipo]["dir_con"] += int(con is not None and NIVEL.get(con, 0) >= nivel_gt)
            stats[tipo]["dir_sin"] += int(sin is not None and NIVEL.get(sin, 0) >= nivel_gt)

        detalle.append({
            "indice": r["indice"], "tipo": tipo, "gt": gt,
            "con": con, "sin": sin, "con_ok": con_ok, "sin_ok": sin_ok,
            "hazard": r["con_skg"].get("hazard"),
        })

    return stats, detalle


def tabla_acciones(stats):
    """Print the main results table."""
    print(f"\n{'Clase':<16}{'n':>4} | {'Ex CON':>7}{'Ex SIN':>7} | "
          f"{'Dir CON':>8}{'Dir SIN':>8} | {'Cond':>5}{'Sec':>5}{'Fallo':>6}")
    print("-" * 72)

    total = defaultdict(int)
    for clase, s in stats.items():
        print(f"{clase:<16}{s['n']:>4} | {s['exacto_con']:>7}{s['exacto_sin']:>7} | "
              f"{s['dir_con']:>8}{s['dir_sin']:>8} | "
              f"{s['condicional_con']:>5}{s['secuencia_con']:>5}"
              f"{s['fallos_con'] + s['fallos_sin']:>6}")
        for k, v in s.items():
            total[k] += v

    print("-" * 72)
    print(f"{'TOTAL':<16}{total['n']:>4} | {total['exacto_con']:>7}{total['exacto_sin']:>7} | "
          f"{total['dir_con']:>8}{total['dir_sin']:>8} | "
          f"{total['condicional_con']:>5}{total['secuencia_con']:>5}"
          f"{total['fallos_con'] + total['fallos_sin']:>6}")

    n = total["n"] or 1
    dn = total["dir_n"] or 1
    print(f"\n  Exacta:      CON {100*total['exacto_con']/n:.1f}%  "
          f"SIN {100*total['exacto_sin']/n:.1f}%")
    print(f"  Direccional: CON {100*total['dir_con']/dn:.1f}%  "
          f"SIN {100*total['dir_sin']/dn:.1f}%  (sobre {total['dir_n']} casos validos)")
    print(f"  GT sin accion reconocible (excluidos): {total['gt_sin_clave']}")
    print(f"  Condicionales sin resolver: CON {total['condicional_con']}  "
          f"SIN {total['condicional_sin']}")
    print(f"  Secuencias compuestas:      CON {total['secuencia_con']}  "
          f"SIN {total['secuencia_sin']}")
    return total


# ===========================================================================
# 3. McNEMAR (paired design)
# ===========================================================================
def mcnemar_test(detalle, etiqueta=""):
    """McNemar test over the discordant pairs.

    Only the cells where the two conditions differ say anything about which is
    better; the agreements carry no information.
    """
    a = sum(1 for d in detalle if d["con_ok"] and d["sin_ok"])          # both right
    b = sum(1 for d in detalle if d["con_ok"] and not d["sin_ok"])      # only WITH
    c = sum(1 for d in detalle if not d["con_ok"] and d["sin_ok"])      # only WITHOUT
    e = sum(1 for d in detalle if not d["con_ok"] and not d["sin_ok"])  # both wrong

    print(f"\n--- McNemar {etiqueta} (n={len(detalle)}) ---")
    print(f"  both right: {a:<4} only WITH: {b}")
    print(f"  only WITHOUT: {c:<4} both wrong: {e}")

    # exact=True gives the binomial test, which is right when b+c is small
    res = mcnemar(np.array([[a, b], [c, e]]), exact=True)
    print(f"  p = {res.pvalue:.4f} -> "
          f"{'significant' if res.pvalue < 0.05 else 'not significant'} (alpha=0.05)")
    return res.pvalue


# ===========================================================================
# 4. FAILURE DIAGNOSIS
# ===========================================================================
def matriz_confusion(detalle, solo_ambos_fallan=False, top=12):
    """Show the most frequent (ground truth -> predicted) pairs."""
    filas = [d for d in detalle if not (d["con_ok"] or d["sin_ok"])] if solo_ambos_fallan else detalle
    titulo = "both wrong" if solo_ambos_fallan else "all cases"
    print(f"\n--- Confusion CON SKG ({titulo}, {len(filas)} casos) ---")
    for (gt, pred), cuenta in Counter((d["gt"], d["con"]) for d in filas).most_common(top):
        marca = "ok" if gt == pred else "  "
        print(f"  {marca} {str(gt):<24} -> {str(pred):<24} ({cuenta})")


def precision_router(detalle):
    """Hazard agent accuracy: class of the chosen hazard vs the annotated class."""
    router = defaultdict(lambda: {"ok": 0, "n": 0})
    for d in detalle:
        clase_elegida = CLASE_DE.get(d.get("hazard"), "?")
        router[d["tipo"]]["n"] += 1
        router[d["tipo"]]["ok"] += int(clase_elegida == d["tipo"])

    print("\n--- Router accuracy (class of the chosen hazard) ---")
    for clase, s in router.items():
        pct = 100 * s["ok"] / s["n"] if s["n"] else 0
        print(f"  {clase:<16} {s['ok']:>3}/{s['n']:<3} ({pct:.0f}%)")


def distribucion_acciones(resultados):
    """Compare the action distribution of GT, WITH and WITHOUT.

    Shows the commitment collapse: without the graph the system falls back on
    non-committal actions.
    """
    gt = Counter(normalizar_accion(r["sugerencia_gt"])[0] for r in resultados)
    con = Counter(normalizar_accion(r["con_skg"]["accion_cruda"])[0] for r in resultados)
    sin = Counter(normalizar_accion(r["sin_skg"]["accion_cruda"])[0] for r in resultados)

    print("\n--- Action distribution ---")
    print(f"  {'Action':<26}{'GT':>5}{'WITH':>6}{'W/O':>6}")
    for accion in sorted(set(gt) | set(con) | set(sin), key=lambda a: -NIVEL.get(a, 0)):
        print(f"  {str(accion):<26}{gt.get(accion,0):>5}{con.get(accion,0):>6}{sin.get(accion,0):>6}")


# ===========================================================================
# 5. VALIDATOR
# ===========================================================================
def resultados_validador(resultados):
    """Pass rate, hallucination rate and ego-frame drift."""
    total = aprobados = ego_ok = con_alucinacion = 0
    for r in resultados:
        v = r["con_skg"].get("validacion")
        if not v:
            continue
        total += 1
        aprobados += int(v.get("verdict") == "pass")
        ego_ok += int(v.get("ego_frame_ok") is True)
        con_alucinacion += int(bool(v.get("hallucinated_claims")))

    if not total:
        print("\n--- Validator: no verdicts ---")
        return

    print(f"\n--- Validator ({total} cases) ---")
    print(f"  Passed:           {aprobados}/{total} ({100*aprobados/total:.0f}%)")
    print(f"  With hallucination: {con_alucinacion}/{total} ({100*con_alucinacion/total:.0f}%)")
    print(f"  Ego-frame drift:  {total-ego_ok}/{total} ({100*(total-ego_ok)/total:.0f}%)")


# ===========================================================================
# 6. CAPTIONING
# ===========================================================================
def metricas_caption(resultados):
    """BLEU-4, METEOR and ROUGE-L of the generated captions against DRAMA's."""
    try:
        import nltk
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from nltk.translate.meteor_score import meteor_score
        from rouge_score import rouge_scorer
    except ImportError:
        print("\n[captioning: pip install nltk rouge-score]")
        return

    suavizado = SmoothingFunction().method1
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    bleu = meteor = rouge = 0.0
    n = 0
    for r in resultados:
        hip = r["con_skg"].get("caption", "")
        ref = r.get("caption_gt", "")
        if not hip or not ref:
            continue
        n += 1
        hip_tok, ref_tok = hip.split(), ref.split()
        bleu += sentence_bleu([ref_tok], hip_tok, weights=(0.25,) * 4,
                              smoothing_function=suavizado)
        meteor += meteor_score([ref_tok], hip_tok)
        rouge += scorer.score(ref, hip)["rougeL"].fmeasure

    if not n:
        print("\n--- Captioning: no captions generated ---")
        return

    print(f"\n--- Captioning ({n} captions) ---")
    print(f"  BLEU-4:  {bleu/n:.4f}")
    print(f"  METEOR:  {meteor/n:.4f}")
    print(f"  ROUGE-L: {rouge/n:.4f}")
    print("  (CIDEr needs pycocoevalcap, optional)")


# ===========================================================================
# 7. SINGLE-FRAME vs TEMPORAL
# ===========================================================================
def comparar_temporal(base, temporal, clase=None):
    """Compare the same sample under single-frame and multi-frame perception."""
    idx_temporal = {r["indice"]: r for r in temporal}
    pares = [(b, idx_temporal[b["indice"]]) for b in base
             if b["indice"] in idx_temporal and (clase is None or b["tipo"] == clase)]

    if not pares:
        print("\n--- Temporal: no comparable pairs ---")
        return

    sf = tf = 0
    for b, t in pares:
        gt, _ = normalizar_accion(b["sugerencia_gt"])
        a_sf, _ = normalizar_accion(b["con_skg"]["accion_cruda"])
        a_tf, _ = normalizar_accion(t["con_skg"]["accion_cruda"])
        sf += int(a_sf == gt)
        tf += int(a_tf == gt)

    etiqueta = clase or "all classes"
    print(f"\n--- Single-frame vs temporal ({etiqueta}, {len(pares)} pares) ---")
    print(f"  Exacta 1 frame:  {sf}/{len(pares)}")
    print(f"  Exacta temporal: {tf}/{len(pares)}")

    
def tests(datos, etiqueta):
    """Exact and directional match for one condition against its baseline.

    Prints both metrics with their McNemar p-value in a single block.
    """
    ec, es, dc, ds = [], [], [], []
    for r in datos:
        gt, _ = normalizar_accion(r["sugerencia_gt"])
        if not gt: continue
        c, _ = normalizar_accion(r["con_skg"]["accion_cruda"])
        s, _ = normalizar_accion(r["sin_skg"]["accion_cruda"])
        ec.append(c == gt); es.append(s == gt)
        if NIVEL.get(gt, 0) > 0:
            dc.append(c is not None and NIVEL.get(c,0) >= NIVEL[gt])
            ds.append(s is not None and NIVEL.get(s,0) >= NIVEL[gt])

    def mc(x, y, nombre):
        a=sum(1 for p,q in zip(x,y) if p and q);     b=sum(1 for p,q in zip(x,y) if p and not q)
        c_=sum(1 for p,q in zip(x,y) if not p and q); e=sum(1 for p,q in zip(x,y) if not p and not q)
        pv = mcnemar(np.array([[a,b],[c_,e]]), exact=True).pvalue
        print(f"  {nombre:<13} CON {sum(x):>3}/{len(x)} ({100*sum(x)/len(x):>4.1f}%)  "
              f"SIN {sum(y):>3}/{len(y)} ({100*sum(y)/len(y):>4.1f}%)  |  "
              f"soloCON={b:<3} soloSIN={c_:<3} ambos✗={e:<3} p={pv:.4f}{'  ***' if pv<0.05 else ''}")

    print(f"\n=== {etiqueta} (n={len(ec)}) ===")
    mc(ec, es, "Exacta"); mc(dc, ds, "Direccional")

# ===========================================================================
# 8. FULL REPORT
# ===========================================================================
def informe(resultados, temporal=None):
    """Run the whole analysis in one pass."""
    stats, detalle = evaluar(resultados)
    tabla_acciones(stats)
    mcnemar_test(detalle, "(CON vs SIN)")
    matriz_confusion(detalle, solo_ambos_fallan=True)
    precision_router(detalle)
    distribucion_acciones(resultados)
    resultados_validador(resultados)
    metricas_caption(resultados)
    if temporal:
        comparar_temporal(resultados, temporal, clase="Vehicle")
        comparar_temporal(resultados, temporal)
    return stats, detalle


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    print("Loading results:")
    datos = cargar(*sys.argv[1:])
    print(f"Total: {len(datos)} escenarios")
    informe(datos)
