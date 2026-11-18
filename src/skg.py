
"""Safety Knowledge Graph and action taxonomy.

Pure Python, no GPU needed. Imported by the other notebooks.
"""

import re

# ---------------------------------------------------------------------------
# Safety Knowledge Graph
# ---------------------------------------------------------------------------
# Each entry is a hazard node with its relations flattened:
#   agent_class -> the DRAMA agent class it belongs to
#   condition   -> what the hazard agent sees in the catalogue when choosing
#   rule        -> the Vienna Convention provision that governs it
#   action      -> the manoeuvre that provision prescribes, stated conditionally

SKG = {
    # ---- Vehicle ----------------------------------------------------------
    "lead_vehicle_slowing": {
        "agent_class": "Vehicle",
        "condition": "A vehicle ahead in the ego lane is decelerating or slowing",
        "rule": "Vienna Convention Art. 13.1 and 13.5 (control of vehicle and sufficient distance)",
        "action": "If the lead vehicle is stopped or braking hard: (must) Stop. "
                  "If it is merely slowing: Follow the vehicle ahead.",
    },
    "lead_vehicle_stopped": {
        "agent_class": "Vehicle",
        "condition": "A vehicle ahead in the ego lane is fully stopped",
        "rule": "Vienna Convention Art. 13.5 (sufficient distance to avoid collision if the vehicle ahead stops)",
        "action": "(must) Stop",
    },
    "vehicle_cutting_in": {
        "agent_class": "Vehicle",
        "condition": "A vehicle from an adjacent lane is merging into the ego lane",
        "rule": "Vienna Convention Art. 13.1 (control of vehicle at all times)",
        "action": "If it is already in the ego lane: Slow down. Otherwise: Be aware or cautious.",
    },
    "oncoming_vehicle": {
        "agent_class": "Vehicle",
        "condition": "An oncoming vehicle occupies or approaches the ego path",
        "rule": "Vienna Convention Art. 12.1 (passing oncoming traffic)",
        "action": "If it invades the ego lane: Slow down. Otherwise: Be aware or cautious.",
    },
    "parked_vehicle_obstruction": {
        "agent_class": "Vehicle",
        "condition": "A parked or stopped vehicle obstructs part of the ego lane",
        "rule": "Vienna Convention Art. 12.1 (obstruction in the carriageway)",
        "action": "Carefully manoeuvre",
    },

    # ---- Pedestrian -------------------------------------------------------
    "pedestrian_crossing": {
        "agent_class": "Pedestrian",
        "condition": "A pedestrian is crossing or about to cross the ego path",
        "rule": "Vienna Convention Art. 21.2 (behaviour at pedestrian crossings)",
        "action": "If the pedestrian is on the ego path: (must) Stop. "
                  "If only near it: Slow down.",
    },
    "pedestrian_at_roadside": {
        "agent_class": "Pedestrian",
        "condition": "A pedestrian stands at the roadside, not on the ego path",
        "rule": "Vienna Convention Art. 21.1 and 7.3 (care towards vulnerable road-users)",
        "action": "If they step toward the ego path: Slow down. Otherwise: Be aware or cautious.",
    },
    "pedestrian_in_lane": {
        "agent_class": "Pedestrian",
        "condition": "A pedestrian is walking inside the ego lane",
        "rule": "Vienna Convention Art. 21.1 (avoid endangering pedestrians)",
        "action": "(must) Stop",
    },

    # ---- Cyclist ----------------------------------------------------------
    "cyclist_crossing": {
        "agent_class": "Cyclist",
        "condition": "A cyclist is crossing the ego path",
        "rule": "Vienna Convention Art. 16.2 (yield to cyclists when changing direction)",
        "action": "If already on the ego path: (must) Stop. Otherwise: Slow down.",
    },
    "cyclist_on_road": {
        "agent_class": "Cyclist",
        "condition": "A cyclist rides in or beside the ego lane, same direction",
        "rule": "Vienna Convention Art. 11.4 (sufficient lateral space when overtaking)",
        "action": "If overtaking is needed: Carefully manoeuvre. Otherwise: Be aware or cautious.",
    },

    # ---- Infrastructure ---------------------------------------------------
    "traffic_light": {
        "agent_class": "Infrastructure",
        "condition": "A traffic light governs the ego direction",
        "rule": "Vienna Convention Art. 5.1 and 5.2 (compliance with light signals; signals take precedence)",
        "action": "If red or amber for the ego direction: (must) Stop. "
                  "If green or state not visible: Be aware or cautious.",
    },
    "stop_sign": {
        "agent_class": "Infrastructure",
        "condition": "A stop sign governs the ego direction",
        "rule": "Vienna Convention Art. 5.1 (compliance with traffic signs)",
        "action": "(must) Stop",
    },
    "give_way_sign": {
        "agent_class": "Infrastructure",
        "condition": "A give-way / yield sign governs the ego direction",
        "rule": "Vienna Convention Art. 5.1 and 18 (priority at intersections)",
        "action": "Yield",
    },
    "zebra_crossing": {
        "agent_class": "Infrastructure",
        "condition": "A marked pedestrian crossing lies ahead on the ego path",
        "rule": "Vienna Convention Art. 21.2 and 11.9 (pedestrian crossings; no overtaking a stopped vehicle)",
        "action": "If pedestrians are on or entering it: (must) Stop. Otherwise: Slow down.",
    },
    "road_works": {
        "agent_class": "Infrastructure",
        "condition": "Road works, cones or a narrowed lane affect the ego path",
        "rule": "Vienna Convention Art. 34.3 and 34.4 (road works)",
        "action": "Carefully manoeuvre",
    },
}

# Derived from the graph so the two never drift apart.
CLASE_DE = {nombre: datos["agent_class"] for nombre, datos in SKG.items()}


# ---------------------------------------------------------------------------
# Action taxonomy
# ---------------------------------------------------------------------------
ACCIONES = [
    "(must) Stop",
    "Yield",
    "Slow down",
    "Carefully manoeuvre",
    "Follow the vehicle ahead",
    "Be aware or cautious",
]

# Formatted for injection into the prompts.
LISTA_ACCIONES = "\n".join(f'- "{a}"' for a in ACCIONES)

# Severity level, used by the directional metric. Higher means more cautious.
NIVEL = {
    "(must) Stop": 4,
    "Yield": 3,
    "Slow down": 3,
    "Carefully manoeuvre": 2,
    "Follow the vehicle ahead": 2,
    "Be aware or cautious": 1,
}

# Keywords mapped to canonical labels, ordered from most to least severe.
# When several match, the first one wins, which is the most cautious reading.
_MAPEO = [
    (["must) stop", "must stop", "stop"], "(must) Stop"),
    (["yield", "give way"],               "Yield"),
    (["slow"],                            "Slow down"),
    (["manoeuvre", "maneuver"],           "Carefully manoeuvre"),
    (["follow"],                          "Follow the vehicle ahead"),
    (["cautious", "aware", "caution"],    "Be aware or cautious"),
]


def normalizar_accion(texto):
    """Turn free text into a canonical action label.

    Returns (label, ambiguity):
      label     -> the most severe action found in the text, or None
      ambiguity -> "none"        : a single clean action
                   "condicional" : an unresolved menu, e.g. "slow down OR stop"
                   "secuencia"   : a compound manoeuvre, e.g. "slow down AND stop"

    Only the label is used for matching against the ground truth. The ambiguity
    flag is diagnostic and does not affect any metric.
    """
    if not texto:
        return None, "none"

    t = texto.lower()
    encontradas = [etiq for claves, etiq in _MAPEO if any(c in t for c in claves)]
    if not encontradas:
        return None, "none"

    etiqueta = encontradas[0]

    # Ambiguity only exists if the text points at two different actions. This is
    # why "Be aware or cautious" is never flagged, even though it contains "or".
    if len(set(encontradas)) <= 1:
        return etiqueta, "none"

    # \b keeps "or" from matching inside words like "corner" or "before".
    tiene_or = re.search(r"\bor\b", t) is not None
    tiene_and = re.search(r"\b(and|then)\b", t) is not None

    if tiene_or and not tiene_and:
        return etiqueta, "condicional"
    if tiene_and and not tiene_or:
        return etiqueta, "secuencia"
    return etiqueta, "condicional"
