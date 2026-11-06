"""
Reúne os algoritmos de busca em um único dicionário, para que os experimentos possam percorrer todos da mesma forma.
"""

from collections.abc import Callable

from algoritmos.a_estrela import busca_a_estrela
from algoritmos.best_first import busca_best_first
from algoritmos.bfs import busca_largura
from algoritmos.dfs import busca_profundidade
from algoritmos.dijkstra import busca_dijkstra
from algoritmos.hill_climbing import busca_hill_climbing
from rede import Grafo
from resultado import ResultadoBusca

# Assinatura comum a todos os algoritmos: (grafo, origem, destino, gravarEventos) -> ResultadoBusca
FuncaoBusca = Callable[[Grafo, int, int, bool], ResultadoBusca]

# Associa o nome usado nas tabelas e figuras à função de cada algoritmo
ALGORITMOS: dict[str, FuncaoBusca] = {
    "DFS": busca_profundidade,
    "BFS": busca_largura,
    "Best-First": busca_best_first,
    "A*": busca_a_estrela,
    "Hill Climbing": busca_hill_climbing,
    "Dijkstra": busca_dijkstra,
}