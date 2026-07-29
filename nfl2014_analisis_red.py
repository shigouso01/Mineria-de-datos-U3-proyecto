# %%
"""
Mapa de influencia en la conversación sobre la NFL temporada 2014
==================================================================

Tarea 1: Construcción del grafo de menciones (edges.csv) con NetworkX.
Tarea 2: Detección de comunidades con Louvain, cálculo de modularidad
         e interpretación de cada comunidad usando los atributos de nodes.csv
         (tipo de cuenta, equipo).

--------------------------------------------------------------------------------
DEPENDENCIAS EXTERNAS (instalar antes de ejecutar):

    pip install pandas networkx python-louvain

Se usa la librería 'python-louvain' (se importa como `community`) porque es
el estándar de facto para Louvain en Python y permite fijar random_state
para resultados reproducibles.
--------------------------------------------------------------------------------

Uso:
    python mundial2026_analisis_red.py
    python mundial2026_analisis_red.py --edges edges.csv --nodes nodes.csv

Requiere en el mismo directorio (o rutas indicadas por --edges / --nodes):
    edges.csv  -> columnas: source, target, weight
    nodes.csv  -> columnas: node, tipo, equipo
"""

import sys
import argparse

import pandas as pd
import networkx as nx

try:
    import community as community_louvain  # python-louvain
except ImportError:
    sys.exit(
        "Falta la dependencia 'python-louvain'.\n"
        "Instálala con: pip install python-louvain"
    )

from centralidad import (
    calcular_centralidades,
    reportar_centralidades,
    detectar_bots,
    generar_visualizacion,
)


# ---------------------------------------------------------------------------
# Carga y validación de datos
# ---------------------------------------------------------------------------

def cargar_datos(edges_path: str, nodes_path: str):
    try:
        edges_df = pd.read_csv(edges_path)
        nodes_df = pd.read_csv(nodes_path)
    except FileNotFoundError as e:
        sys.exit(f"No se encontró el archivo: {e.filename}")

    columnas_edges_esperadas = {"source", "target", "weight"}
    columnas_nodes_esperadas = {"node", "tipo", "equipo"}

    faltantes_edges = columnas_edges_esperadas - set(edges_df.columns)
    faltantes_nodes = columnas_nodes_esperadas - set(nodes_df.columns)

    if faltantes_edges:
        sys.exit(f"edges.csv no contiene las columnas requeridas: {faltantes_edges}")
    if faltantes_nodes:
        sys.exit(f"nodes.csv no contiene las columnas requeridas: {faltantes_nodes}")

    return edges_df, nodes_df


# ---------------------------------------------------------------------------
# Tarea 1 — Construcción del grafo
# ---------------------------------------------------------------------------

def construir_grafo(edges_df: pd.DataFrame) -> nx.DiGraph:
    """Construye un grafo dirigido: las menciones tienen dirección (quién menciona a quién)."""
    G = nx.from_pandas_edgelist(
        edges_df,
        source="source",
        target="target",
        edge_attr="weight",
        create_using=nx.DiGraph(),
    )
    return G


def reportar_grafo(G: nx.DiGraph) -> None:
    print("=" * 70)
    print("TAREA 1 — CONSTRUCCIÓN DEL GRAFO")
    print("=" * 70)
    print(f"Número de nodos : {G.number_of_nodes()}")
    print(f"Número de enlaces (aristas): {G.number_of_edges()}")
    print(f"Densidad del grafo: {nx.density(G):.6f}")

    if G.number_of_nodes() > 0:
        grados = dict(G.degree(weight="weight"))
        top5 = sorted(grados.items(), key=lambda x: x[1], reverse=True)[:5]
        print("\nTop 5 cuentas por grado ponderado (volumen de interacción):")
        for nodo, grado in top5:
            print(f"  {nodo}: {grado}")
    print()


# ---------------------------------------------------------------------------
# Tarea 2 — Detección de comunidades (Louvain)
# ---------------------------------------------------------------------------

def detectar_comunidades(G: nx.DiGraph):
    print("=" * 70)
    print("TAREA 2 — DETECCIÓN DE COMUNIDADES (LOUVAIN)")
    print("=" * 70)

    # Louvain está definido para grafos no dirigidos; convertimos preservando pesos.
    G_und = G.to_undirected()

    particion = community_louvain.best_partition(G_und, weight="weight", random_state=42)
    modularidad = community_louvain.modularity(particion, G_und, weight="weight")

    comunidades = {}
    for nodo, com_id in particion.items():
        comunidades.setdefault(com_id, []).append(nodo)

    print(f"Modularidad global: {modularidad:.4f}")
    print(f"Número de comunidades detectadas: {len(comunidades)}\n")

    print("Tamaño de cada comunidad:")
    for com_id, miembros in sorted(comunidades.items(), key=lambda x: -len(x[1])):
        print(f"  Comunidad {com_id}: {len(miembros)} nodos")
    print()

    return particion, comunidades, modularidad


def interpretar_comunidades(comunidades: dict, nodes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cruza cada comunidad detectada con los metadatos de nodes.csv (tipo, equipo)
    para dar una lectura cualitativa: ¿es una comunidad de un equipo, de medios
    globales?, y si hay concentración sospechosa de bots.
    """
    print("=" * 70)
    print("INTERPRETACIÓN DE COMUNIDADES")
    print("=" * 70)

    nodes_idx = nodes_df.set_index("node")
    resumen = []

    for com_id, miembros in sorted(comunidades.items(), key=lambda x: -len(x[1])):
        info = nodes_idx.reindex(miembros)

        equipos = info["equipo"].value_counts(dropna=True)
        tipos = info["tipo"].value_counts(dropna=True)

        equipo_dominante = equipos.idxmax() if not equipos.empty else "N/D"
        pct_equipo = (equipos.max() / len(miembros) * 100) if not equipos.empty else 0.0
        pct_bots = (tipos.get("bot", 0) / len(miembros) * 100) if len(miembros) else 0.0

        print(f"\n--- Comunidad {com_id} ({len(miembros)} nodos) ---")
        print(f"Equipo dominante: {equipo_dominante} ({pct_equipo:.1f}% de la comunidad)")

        print("Distribución por equipo:")
        for equipo, cnt in equipos.items():
            print(f"    {equipo}: {cnt}")

        print("Distribución por tipo de cuenta:")
        for tipo, cnt in tipos.items():
            print(f"    {tipo}: {cnt}")

        etiqueta = f"Comunidad dominada por {equipo_dominante}"
        if pct_bots > 30:
            etiqueta += f" — ALERTA: {pct_bots:.1f}% de bots, posible amplificación artificial"
        elif tipos.get("medio", 0) >= 2 and len(equipos) > 2:
            etiqueta = "Comunidad de medios/cobertura global (conecta varios equipos)"
        elif tipos.get("influencer", 0) >= 1 and tipos.get("influencer", 0) / len(miembros) > 0.1:
            etiqueta += " — presencia relevante de influencers que marcan agenda"

        print(f"Interpretación: {etiqueta}")

        resumen.append({
            "comunidad": com_id,
            "tamano": len(miembros),
            "equipo_dominante": equipo_dominante,
            "pct_equipo_dominante": round(pct_equipo, 1),
            "pct_bots": round(pct_bots, 1),
            "interpretacion": etiqueta,
        })

    return pd.DataFrame(resumen)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Análisis de red social sobre la NFL temporada 2014 (grafo + comunidades Louvain)."
    )
    parser.add_argument("--edges", default="edges.csv", help="Ruta al archivo edges.csv")
    parser.add_argument("--nodes", default="nodes.csv", help="Ruta al archivo nodes.csv")
    parser.add_argument(
        "--out",
        default="resumen_comunidades_nfl.csv",
        help="Ruta de salida para el resumen de comunidades",
    )
    parser.add_argument(
        "--out-centralidad",
        default="centralidad_nfl.csv",
        help="Ruta de salida para el top de centralidad/influencers",
    )
    parser.add_argument(
        "--out-bots",
        default="bots_detectados_nfl.csv",
        help="Ruta de salida para el detalle de bots (Tarea 4)",
    )
    parser.add_argument(
        "--out-html",
        default="grafo_nfl2014.html",
        help="Ruta de salida para la visualización interactiva (Tarea 5)",
    )
    args = parser.parse_args()

    edges_df, nodes_df = cargar_datos(args.edges, args.nodes)

    G = construir_grafo(edges_df)
    reportar_grafo(G)

    particion, comunidades, modularidad = detectar_comunidades(G)
    resumen_df = interpretar_comunidades(comunidades, nodes_df)

    resumen_df.to_csv(args.out, index=False)
    print(f"\nResumen de comunidades guardado en: {args.out}")

    centralidad_df = calcular_centralidades(G)
    top_influencers_df = reportar_centralidades(centralidad_df, nodes_df, particion, top_n=10)

    top_influencers_df.to_csv(args.out_centralidad, index=False)
    print(f"Top de influencers guardado en: {args.out_centralidad}")

    bots_confirmados, candidatos_revision, bots_detalle_df = detectar_bots(
        G, centralidad_df, nodes_df, particion
    )
    bots_detalle_df.to_csv(args.out_bots, index=False)
    print(f"Detalle de bots guardado en: {args.out_bots}")

    generar_visualizacion(
        G, particion, centralidad_df, nodes_df, bots_confirmados, modularidad,
        output_path=args.out_html,
    )


if __name__ == "__main__":
    main()
# %%
