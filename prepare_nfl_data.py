"""
Generación del dataset sintético NFL 2014
==========================================
Genera edges.csv y nodes.csv con la misma estructura que el dataset original
del Mundial 2026, pero ambientado en la temporada 2014 de la NFL.

Estructura de salida:
  edges.csv  ->  source, target, weight
  nodes.csv  ->  node, tipo, equipo

Uso:
    python prepare_nfl_data.py

Requerimientos:
    pip install pandas

Para usar después:
    python mundial2026_analisis_red.py --edges edges.csv --nodes nodes.csv
"""

import pandas as pd
import random

random.seed(42)

# ---------------------------------------------------------------------------
# Equipos de la NFL temporada 2014 (con abreviatura y nombre real)
# ---------------------------------------------------------------------------
EQUIPOS = {
    "patriots": "NE",
    "seahawks": "SEA",
    "packers": "GB",
    "cowboys": "DAL",
    "broncos": "DEN",
    "colts": "IND",
    "steelers": "PIT",
    "cardinals": "ARI",
    "lions": "DET",
    "ravens": "BAL",
    "panthers": "CAR",
    "chiefs": "KC",
}

# ---------------------------------------------------------------------------
# Construcción de nodos
# ---------------------------------------------------------------------------

nodes = []

# 1. Cuentas oficiales de equipo (tipo = equipo)
for equipo, abbr in EQUIPOS.items():
    nodes.append({"node": equipo, "tipo": "equipo", "equipo": abbr})

# 2. Medios de comunicación (tipo = medio)
medios = {
    "nfl": "global",
    "espn": "global",
    "nflnetwork": "global",
    "sportscenter": "global",
    "bleacherreport": "global",
    "cbssports": "global",
}
for medio, _ in medios.items():
    nodes.append({"node": medio, "tipo": "medio", "equipo": "global"})

# 3. Jugadores / Influencers (tipo = influencer)
influencers = {
    "tombrady": "NE",
    "aaronrodgers": "GB",
    "russellwilson": "SEA",
    "dezbyrant": "DAL",
    "peytonmanning": "DEN",
    "andrewluck": "IND",
    "jjwatt": "global",   # Houston no en la lista, pero DPOY 2014
    "obj": "global",       # Odell Beckham Jr, NYG
}
for inf, equipo in influencers.items():
    nodes.append({"node": inf, "tipo": "influencer", "equipo": equipo})

# 4. Aficionados (tipo = fan) — 3 por equipo
for abbr in sorted(EQUIPOS.values()):
    for i in range(1, 4):
        nodes.append({
            "node": f"fan_{abbr.lower()}_{i:02d}",
            "tipo": "fan",
            "equipo": abbr,
        })

# 5. Bots (tipo = bot)
for i in range(1, 6):
    nodes.append({"node": f"bot_nfl_{i:02d}", "tipo": "bot", "equipo": "global"})

# 6. Hashtags (tipo = hashtag)
hashtags = [
    ("#SBXLIX", "global"),
    ("#NFL", "global"),
    ("#Patriots", "NE"),
    ("#Seahawks", "SEA"),
    ("#GoPackGo", "GB"),
    ("#CowboysNation", "DAL"),
    ("#BroncosCountry", "DEN"),
    ("#ForTheShoe", "IND"),
]
for h, equipo in hashtags:
    nodes.append({"node": h, "tipo": "hashtag", "equipo": equipo})

nodes_df = pd.DataFrame(nodes)

# ---------------------------------------------------------------------------
# Construcción de aristas (edges)
# ---------------------------------------------------------------------------
# Cada tupla: (source, target, weight, [opcional comunidad/contexto])
# weight = número de menciones (fuerza de la conexión)
# ---------------------------------------------------------------------------

edges = []

def add_edge(s, t, w=1):
    edges.append({"source": s, "target": t, "weight": w})

# --- Aficionados mencionan a su equipo ---
pairs_fan_team = [
    ("fan_ne_01", "patriots", 3),
    ("fan_ne_02", "patriots", 2),
    ("fan_ne_03", "patriots", 1),
    ("fan_sea_01", "seahawks", 3),
    ("fan_sea_02", "seahawks", 2),
    ("fan_sea_03", "seahawks", 1),
    ("fan_gb_01", "packers", 3),
    ("fan_gb_02", "packers", 2),
    ("fan_gb_03", "packers", 1),
    ("fan_dal_01", "cowboys", 3),
    ("fan_dal_02", "cowboys", 2),
    ("fan_dal_03", "cowboys", 1),
    ("fan_den_01", "broncos", 2),
    ("fan_den_02", "broncos", 1),
    ("fan_ind_01", "colts", 2),
    ("fan_pit_01", "steelers", 2),
    ("fan_ari_01", "cardinals", 1),
    ("fan_det_01", "lions", 1),
]
for s, t, w in pairs_fan_team:
    add_edge(s, t, w)

# --- Aficionados mencionan a jugadores estrella ---
pairs_fan_player = [
    ("fan_ne_01", "tombrady", 5),
    ("fan_ne_02", "tombrady", 3),
    ("fan_sea_01", "russellwilson", 4),
    ("fan_sea_02", "russellwilson", 2),
    ("fan_gb_01", "aaronrodgers", 5),
    ("fan_gb_02", "aaronrodgers", 3),
    ("fan_dal_01", "dezbyrant", 3),
    ("fan_dal_02", "dezbyrant", 2),
    ("fan_den_01", "peytonmanning", 3),
    ("fan_ind_01", "andrewluck", 3),
]
for s, t, w in pairs_fan_player:
    add_edge(s, t, w)

# --- Aficionados mencionan hashtags ---
pairs_fan_hashtag = [
    ("fan_ne_01", "#Patriots", 2),
    ("fan_ne_02", "#Patriots", 1),
    ("fan_sea_01", "#Seahawks", 2),
    ("fan_gb_01", "#GoPackGo", 2),
    ("fan_dal_01", "#CowboysNation", 2),
    ("fan_den_01", "#BroncosCountry", 1),
    ("fan_ind_01", "#ForTheShoe", 1),
    ("fan_ne_01", "#SBXLIX", 1),
    ("fan_sea_01", "#SBXLIX", 1),
]
for s, t, w in pairs_fan_hashtag:
    add_edge(s, t, w)

# --- Aficionados se mencionan entre sí (comunidad de fans) ---
pairs_fan_fan = [
    ("fan_ne_01", "fan_ne_02", 1),
    ("fan_ne_02", "fan_ne_03", 1),
    ("fan_sea_01", "fan_sea_02", 1),
    ("fan_sea_02", "fan_sea_03", 1),
    ("fan_gb_01", "fan_gb_02", 1),
    ("fan_dal_01", "fan_dal_02", 1),
    ("fan_ne_01", "fan_sea_01", 1),   # rivalidad SB
    ("fan_gb_01", "fan_dal_01", 1),   # rivalidad divisional (en playoffs 2014)
]
for s, t, w in pairs_fan_fan:
    add_edge(s, t, w)

# --- Medios mencionan equipos y jugadores ---
pairs_media = [
    ("nfl", "patriots", 4),
    ("nfl", "seahawks", 4),
    ("nfl", "packers", 3),
    ("nfl", "tombrady", 5),
    ("nfl", "aaronrodgers", 4),
    ("nfl", "russellwilson", 3),
    ("espn", "patriots", 3),
    ("espn", "seahawks", 3),
    ("espn", "cowboys", 2),
    ("espn", "tombrady", 4),
    ("nflnetwork", "patriots", 3),
    ("nflnetwork", "seahawks", 2),
    ("nflnetwork", "dezbyrant", 2),
    ("sportscenter", "obj", 3),      # la famosa recepción de OBJ
    ("sportscenter", "jjwatt", 2),
    ("bleacherreport", "dezbyrant", 2),
    ("bleacherreport", "obj", 2),
    ("cbssports", "packers", 2),
    ("cbssports", "broncos", 2),
    ("cbssports", "peytonmanning", 2),
]
for s, t, w in pairs_media:
    add_edge(s, t, w)

# --- Medios mencionan hashtags ---
pairs_media_h = [
    ("nfl", "#NFL", 3),
    ("espn", "#NFL", 2),
    ("espn", "#SBXLIX", 2),
    ("nflnetwork", "#SBXLIX", 2),
]
for s, t, w in pairs_media_h:
    add_edge(s, t, w)

# --- Jugadores se mencionan entre sí ---
pairs_player_player = [
    ("tombrady", "russellwilson", 2),   # respeto mutuo antes del SB
    ("russellwilson", "tombrady", 2),
    ("aaronrodgers", "dezbyrant", 1),
    ("peytonmanning", "tombrady", 2),   # la gran rivalidad Manning-Brady
    ("tombrady", "peytonmanning", 2),
    ("andrewluck", "tombrady", 1),
    ("dezbyrant", "aaronrodgers", 1),
]
for s, t, w in pairs_player_player:
    add_edge(s, t, w)

# --- Jugadores mencionan hashtags ---
pairs_player_h = [
    ("tombrady", "#SBXLIX", 1),
    ("russellwilson", "#SBXLIX", 1),
    ("tombrady", "#Patriots", 1),
    ("aaronrodgers", "#GoPackGo", 1),
]
for s, t, w in pairs_player_h:
    add_edge(s, t, w)

# --- Bots: amplificación artificial ---
pairs_bot = [
    ("bot_nfl_01", "tombrady", 10),        # bot amplifica a Brady
    ("bot_nfl_02", "patriots", 8),         # bot amplifica a Patriots
    ("bot_nfl_03", "seahawks", 6),
    ("bot_nfl_04", "dezbyrant", 7),
    ("bot_nfl_05", "cowboys", 5),
    ("bot_nfl_02", "tombrady", 4),
    ("bot_nfl_04", "patriots", 3),
    ("fan_ne_01", "bot_nfl_01", 1),        # fan sin saberlo menciona a un bot
    ("fan_dal_01", "bot_nfl_04", 1),
    ("bot_nfl_01", "#Patriots", 5),
    ("bot_nfl_02", "#SBXLIX", 4),
    ("bot_nfl_03", "#Seahawks", 3),
]
for s, t, w in pairs_bot:
    add_edge(s, t, w)

# --- Aristas adicionales para dar más realismo ---
extra_edges = [
    # Rivalidad Packers-Cowboys (playoffs divisionales 2014)
    ("fan_gb_01", "cowboys", 1),
    ("fan_dal_01", "packers", 1),
    # El catch de Dez (controversia)
    ("dezbyrant", "packers", 1),
    ("fan_dal_01", "#NFL", 1),
    # Deflategate (controversia Patriots)
    ("espn", "#NFL", 1),
    ("nflnetwork", "tombrady", 2),
    # Lesión de Manning
    ("peytonmanning", "broncos", 1),
    # JJ Watt siendo JJ Watt
    ("jjwatt", "nfl", 1),
    ("jjwatt", "espn", 1),
    # OBJ catch
    ("obj", "sportscenter", 1),
    ("obj", "cowboys", 1),   # famoso catch vs DAL
    # Aficionado menciona a medio
    ("fan_ne_01", "espn", 1),
    ("fan_sea_01", "nflnetwork", 1),
    # AFC Championship (NE vs IND)
    ("tombrady", "colts", 1),
    ("andrewluck", "patriots", 1),
    ("fan_ind_01", "patriots", 1),
]
for s, t, w in extra_edges:
    add_edge(s, t, w)

edges_df = pd.DataFrame(edges)

# ---------------------------------------------------------------------------
# Guardar
# ---------------------------------------------------------------------------
nodes_df.to_csv("nodes.csv", index=False)
edges_df.to_csv("edges.csv", index=False)

print(f"Dataset NFL 2014 generado:")
print(f"  nodes.csv -> {len(nodes_df)} nodos ({nodes_df['tipo'].value_counts().to_dict()})")
print(f"  edges.csv -> {len(edges_df)} aristas")
print(f"\nEquipos representados: {sorted(nodes_df[nodes_df['tipo']=='equipo']['equipo'].unique())}")
print(f"\nListo para ejecutar:")
print(f"  python mundial2026_analisis_red.py --edges edges.csv --nodes nodes.csv")
