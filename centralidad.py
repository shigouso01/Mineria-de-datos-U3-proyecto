"""
Tarea 3 — Centralidad e influencers
Tarea 4 — Detección de bots
Tarea 5 — Visualización interactiva (PyVis)
================================================================================

Todo lo relacionado con centralidad, detección de bots y visualización vive
en este módulo, para que se importe desde mundial2026_analisis_red.py (que
ya tiene el grafo, la partición de comunidades y los metadatos de nodos).

Nota general de diseño: las funciones de este módulo reciben el grafo, la
partición y los DataFrames ya construidos en el script principal — no cargan
archivos ni recalculan nada por su cuenta — para evitar que T3, T4 y T5
usen versiones distintas del grafo o de las comunidades sin querer.
"""

import pandas as pd
import networkx as nx


def calcular_centralidades(G: nx.DiGraph) -> pd.DataFrame:
    """
    Calcula tres métricas de centralidad sobre el grafo:

    - degree centrality: cuántas conexiones directas tiene cada cuenta
      (proxy de volumen de interacción / alcance directo).
    - betweenness centrality: qué tanto actúa cada cuenta como "puente"
      entre otras cuentas (proxy de control del flujo de información
      entre comunidades).
    - eigenvector centrality: qué tan conectada está una cuenta a otras
      cuentas que a su vez son influyentes (proxy de influencia real,
      no solo de volumen).

    Nota sobre el peso en betweenness_centrality:
    NetworkX interpreta el parámetro `weight` como una DISTANCIA para el
    cálculo de caminos más cortos, no como fuerza de conexión (fuente:
    documentación oficial de networkx.algorithms.centrality.betweenness_centrality:
    "Weights are used to calculate weighted shortest paths, so they are
    interpreted as distances").
    Como en nuestro caso `weight` = número de menciones (a mayor valor,
    conexión más fuerte, no más "lejana"), usamos su inverso como distancia
    para que las cuentas con más interacción se traten como más cercanas.
    Para eigenvector_centrality esto no aplica: ahí el peso sí se usa
    directamente como fuerza de conexión, que es el comportamiento deseado.
    """
    G_und = G.to_undirected()

    # Distancia = inverso del peso, para que "más menciones" = "más cerca"
    for _, _, data in G_und.edges(data=True):
        data["distancia"] = 1 / data["weight"] if data.get("weight", 0) else 1

    degree_c = nx.degree_centrality(G_und)
    betweenness_c = nx.betweenness_centrality(G_und, weight="distancia")
    try:
        eigenvector_c = nx.eigenvector_centrality(
            G_und, weight="weight", max_iter=1000
        )
    except nx.PowerIterationFailedConvergence:
        # Fallback numérico si el grafo no converge en el límite de iteraciones
        eigenvector_c = nx.eigenvector_centrality_numpy(G_und, weight="weight")

    df = pd.DataFrame({
        "node": list(G_und.nodes()),
    })
    df["degree_centrality"] = df["node"].map(degree_c)
    df["betweenness_centrality"] = df["node"].map(betweenness_c)
    df["eigenvector_centrality"] = df["node"].map(eigenvector_c)

    return df


def reportar_centralidades(
    centralidad_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    particion: dict,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Construye el Top n de cuentas más influyentes según eigenvector
    centrality (la métrica que mejor captura "influencia real" y no solo
    volumen de menciones), y justifica cada entrada cruzando sus tres
    métricas de centralidad con los metadatos de nodes.csv (tipo, país)
    y la comunidad a la que pertenece (Tarea 2).
    """
    print("=" * 70)
    print("TAREA 3 — CENTRALIDAD E INFLUENCERS")
    print("=" * 70)

    nodes_idx = nodes_df.set_index("node")
    info = centralidad_df.set_index("node").join(nodes_idx, how="left")
    info["comunidad"] = info.index.map(particion)

    top = info.sort_values("eigenvector_centrality", ascending=False).head(top_n)

    print(f"\nTop {top_n} cuentas más influyentes (por eigenvector centrality):\n")
    resumen = []
    for nodo, fila in top.iterrows():
        tipo = fila.get("tipo", "N/D")
        equipo = fila.get("equipo", "N/D")
        comunidad = fila.get("comunidad", "N/D")

        # Justificación cualitativa según qué métrica domina
        justificacion = []
        if fila["degree_centrality"] >= info["degree_centrality"].quantile(0.75):
            justificacion.append("alto volumen de interacción directa")
        if fila["betweenness_centrality"] >= info["betweenness_centrality"].quantile(0.75):
            justificacion.append("actúa como puente entre comunidades")
        if fila["eigenvector_centrality"] >= info["eigenvector_centrality"].quantile(0.90):
            justificacion.append("conectada a otras cuentas muy influyentes")
        if not justificacion:
            justificacion.append("influencia moderada y distribuida")

        print(f"  {nodo}  [{tipo} · {equipo} · comunidad {comunidad}]")
        print(f"    degree={fila['degree_centrality']:.4f}  "
              f"betweenness={fila['betweenness_centrality']:.4f}  "
              f"eigenvector={fila['eigenvector_centrality']:.4f}")
        print(f"    Justificación: {', '.join(justificacion)}")

        resumen.append({
            "node": nodo,
            "tipo": tipo,
            "equipo": equipo,
            "comunidad": comunidad,
            "degree_centrality": round(fila["degree_centrality"], 4),
            "betweenness_centrality": round(fila["betweenness_centrality"], 4),
            "eigenvector_centrality": round(fila["eigenvector_centrality"], 4),
            "justificacion": "; ".join(justificacion),
        })
    print()

    return pd.DataFrame(resumen)


# ---------------------------------------------------------------------------
# Tarea 4 — Detección de bots
# ---------------------------------------------------------------------------

# Tipos de cuenta que nunca deben marcarse como bot, sin importar sus métricas
# (fuentes oficiales/medios/influencers verificados según nodes.csv, y los
# hashtags porque no son cuentas).
TIPOS_WHITELIST = {"equipo", "medio", "influencer", "hashtag"}


def detectar_bots(
    G: nx.DiGraph,
    centralidad_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    particion: dict,
):
    """
    Detector de bots "ajustado": lista de bots conocidos + reglas suaves,
    tal como pide el documento del proyecto.

    - Lista conocida: nodos ya etiquetados como tipo == 'bot' en nodes.csv.
    - Reglas suaves: candidatos adicionales no etiquetados que muestran un
      patrón estructural atípico (alta centralidad de grado relativa a su
      tipo de cuenta + conexión con varias comunidades a la vez + nulo
      coeficiente de clustering).

    Importante (calibrado sobre este dataset): con un grafo tan pequeño y
    disperso como este, el coeficiente de clustering por sí solo NO separa
    bien bots de cuentas reales (la gran mayoría de fans también tiene
    clustering = 0), y una regla de "conecta con >=3 comunidades" puede
    marcar como sospechosas a cuentas reales que simplemente fueron
    mencionadas por un bot amplificador (falso positivo). Por eso los
    candidatos por regla suave se devuelven APARTE, para revisión manual,
    en vez de mezclarse automáticamente con los bots confirmados.

    Devuelve:
        bots_confirmados: set de nodos ya etiquetados como bot en nodes.csv
        candidatos_revision: set de nodos sospechosos por regla suave,
            que NO están en la lista conocida y requieren revisión manual
        detalle_df: DataFrame con el análisis de cada bot confirmado:
            comunidad a la que pertenece, a quién menciona (amplifica) y
            quién lo menciona a él.
    """
    print("=" * 70)
    print("TAREA 4 — DETECCIÓN DE BOTS")
    print("=" * 70)

    nodes_idx = nodes_df.set_index("node")
    G_und = G.to_undirected()
    clustering = nx.clustering(G_und, weight="weight")
    cent_idx = centralidad_df.set_index("node")

    bots_confirmados = set(nodes_idx[nodes_idx["tipo"] == "bot"].index)

    # --- Reglas suaves para candidatos NO etiquetados ---
    pool = nodes_idx[
        (~nodes_idx["tipo"].isin(TIPOS_WHITELIST))
        & (~nodes_idx.index.isin(bots_confirmados))
    ].index

    umbral_degree = cent_idx.loc[
        cent_idx.index.isin(pool), "degree_centrality"
    ].quantile(0.90)

    candidatos_revision = set()
    for nodo in pool:
        if nodo not in G_und:
            continue
        deg = cent_idx.loc[nodo, "degree_centrality"] if nodo in cent_idx.index else 0
        clust = clustering.get(nodo, 0)
        comunidades_vecinas = {particion[v] for v in G_und.neighbors(nodo)}

        if deg >= umbral_degree and clust == 0 and len(comunidades_vecinas) >= 3:
            candidatos_revision.add(nodo)

    print(f"\nBots confirmados (lista conocida, nodes.csv): {len(bots_confirmados)}")
    for b in sorted(bots_confirmados):
        print(f"  - {b}")

    print(f"\nCandidatos adicionales por regla suave (revisión manual, NO confirmados): "
          f"{len(candidatos_revision)}")
    for c in sorted(candidatos_revision):
        print(f"  - {c}  (posible cuenta amplificada por un bot, no necesariamente un bot)")

    # --- Análisis de cada bot confirmado: comunidad, a quién menciona, quién lo menciona ---
    filas = []
    for bot in sorted(bots_confirmados):
        comunidad = particion.get(bot, "N/D")

        # A quién menciona el bot (lo que amplifica activamente)
        menciona_a = []
        if bot in G:
            menciona_a = [
                f"{t} (peso {G[bot][t].get('weight', 1)})" for t in G.successors(bot)
            ]

        # Quién menciona al bot (quién lo amplifica a él / le da alcance)
        mencionado_por = []
        if bot in G:
            mencionado_por = [
                f"{s} (peso {G[s][bot].get('weight', 1)})" for s in G.predecessors(bot)
            ]

        print(f"\n--- {bot} ---")
        print(f"Comunidad: {comunidad}")
        print(f"Menciona a: {', '.join(menciona_a) if menciona_a else '(nadie)'}")
        print(f"Es mencionado por: {', '.join(mencionado_por) if mencionado_por else '(nadie)'}")

        filas.append({
            "bot": bot,
            "comunidad": comunidad,
            "menciona_a": "; ".join(menciona_a),
            "mencionado_por": "; ".join(mencionado_por),
        })
    print()

    detalle_df = pd.DataFrame(filas)
    return bots_confirmados, candidatos_revision, detalle_df


# ---------------------------------------------------------------------------
# Tarea 5 — Visualización interactiva (PyVis)
# ---------------------------------------------------------------------------

def generar_visualizacion(
    G: nx.DiGraph,
    particion: dict,
    centralidad_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    bots_confirmados: set,
    modularidad: float,
    output_path: str = "grafo_nfl2014.html",
):
    """
    Genera el HTML interactivo con PyVis pedido en la Tarea 5:
    - colores por comunidad,
    - tamaño de nodo según centralidad (eigenvector, la misma métrica
      usada para el top de influencers en la Tarea 3, para mantener
      consistencia entre tareas),
    - bots en rojo,
    - panel lateral con estadísticas globales, top influencers y
      número de bots.
    """
    import matplotlib.pyplot as plt
    from pyvis.network import Network

    print("=" * 70)
    print("TAREA 5 — VISUALIZACIÓN INTERACTIVA (PyVis)")
    print("=" * 70)

    G_und = G.to_undirected()
    cent_idx = centralidad_df.set_index("node")
    nodes_idx = nodes_df.set_index("node")

    net = Network(
        height="800px",
        width="100%",
        bgcolor="#ffffff",
        font_color="black",
        notebook=False,
        directed=False,
    )
    net.from_nx(G_und)

    comunidades = sorted(set(particion.values()))
    palette = plt.colormaps["tab20"].resampled(max(len(comunidades), 1))
    color_map = {com: palette(i) for i, com in enumerate(comunidades)}

    def escalar_tamano(valor):
        # eigenvector_centrality suele estar entre 0 y ~0.4 en este grafo
        return 10 + (valor * 80)

    for node in net.nodes:
        node_id = str(node["id"])
        comunidad = particion.get(node_id)
        eig = cent_idx.loc[node_id, "eigenvector_centrality"] if node_id in cent_idx.index else 0
        tipo = nodes_idx.loc[node_id, "tipo"] if node_id in nodes_idx.index else "N/D"
        equipo = nodes_idx.loc[node_id, "equipo"] if node_id in nodes_idx.index else "N/D"

        if comunidad is not None:
            rgba = color_map[comunidad]
            hex_color = "#%02x%02x%02x" % (
                int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
            )
            node["color"] = hex_color

        node["size"] = escalar_tamano(eig)
        node["title"] = (
            f"Nodo: {node_id}<br>"
            f"Tipo: {tipo} · Equipo: {equipo}<br>"
            f"Comunidad: {comunidad}<br>"
            f"Eigenvector centrality: {eig:.4f}"
        )

        if node_id in bots_confirmados:
            node["color"] = "#ff0000"
            node["title"] += "<br><b>⚠ BOT DETECTADO</b>"

    net.write_html(output_path)

    # --- Panel lateral con estadísticas + top influencers ---
    top_influencers = (
        cent_idx.sort_values("eigenvector_centrality", ascending=False).head(10)
    )
    influencers_html = "<ul>"
    for nodo, fila in top_influencers.iterrows():
        influencers_html += f"<li>{nodo}: {fila['eigenvector_centrality']:.4f}</li>"
    influencers_html += "</ul>"

    with open(output_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    panel_html = f"""
<div id="panel-estadisticas"
style="
position: fixed;
top: 20px;
right: 20px;
width: 320px;
background: #ffffff;
border: 2px solid #333;
padding: 15px;
border-radius: 10px;
box-shadow: 0px 0px 10px rgba(0,0,0,0.3);
font-family: Arial;
z-index: 9999;
">
<h3 style="margin-top:0;">📊 Estadísticas Globales</h3>
<p><b>Nodos:</b> {G.number_of_nodes()}</p>
<p><b>Enlaces:</b> {G.number_of_edges()}</p>
<p><b>Comunidades:</b> {len(comunidades)}</p>
<p><b>Modularidad:</b> {modularidad:.4f}</p>
<p><b>Bots detectados:</b> {len(bots_confirmados)}</p>
<hr>
<h4>🌟 Top 10 Influencers (eigenvector)</h4>
{influencers_html}
<hr>
<button onclick="mostrarSoloBots()"
style="padding:8px; background:#ff0000; color:white; border:none; border-radius:5px;
cursor:pointer;">
Mostrar solo bots
</button>
<button onclick="mostrarTodos()"
style="padding:8px; margin-top:5px; background:#333; color:white; border:none; border-radius:5px; cursor:pointer;">
Mostrar todos
</button>
</div>
<script>
function mostrarSoloBots() {{
    var allNodes = nodes.get();
    allNodes.forEach(function(n) {{
        if (n.title && n.title.includes('BOT DETECTADO')) {{
            nodes.update({{id: n.id, hidden: false}});
        }} else {{
            nodes.update({{id: n.id, hidden: true}});
        }}
    }});
}}

function mostrarTodos() {{
    var allNodes = nodes.get();
    allNodes.forEach(function(n) {{
        nodes.update({{id: n.id, hidden: false}});
    }});
}}
</script>
</body>
"""

    html_content = html_content.replace("</body>", panel_html)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\nHTML interactivo generado en: {output_path}")
