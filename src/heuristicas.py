"""
Heurísticas usadas pelos algoritmos de busca informada (Best-First, A* e Hill Climbing).
"""

import math

from rede import Grafo, gera_rede_knn

def distancia_euclidiana(pontoA: tuple[float, float], pontoB: tuple[float, float]) -> float:
    """
    Calcula a distância euclidiana entre dois pontos do plano.

    Args:
        pontoA: Coordenada (x, y) do primeiro ponto.
        pontoB: Coordenada (x, y) do segundo ponto.

    Returns:
        A distância em linha reta entre os dois pontos.
    """
    # Calcula a diferença entre as coordenadas x e entre as coordenadas y dos dois pontos
    diferencaX = pontoA[0] - pontoB[0]
    diferencaY = pontoA[1] - pontoB[1]

    # Aplica o teorema de Pitágoras: raiz da soma dos quadrados das diferenças
    return math.sqrt(diferencaX ** 2 + diferencaY ** 2)

def heuristica_euclidiana(grafo: Grafo, vertice: int, destino: int) -> float:
    """
    Estima o custo restante de um vértice até o destino pela distância em linha reta.

    A heurística é admissível, porque o peso de cada aresta é a distância
    euclidiana entre suas pontas e, pela desigualdade triangular, nenhum
    caminho pelo grafo é mais curto que a linha reta. Com isso, o A* sempre
    encontra o caminho de menor distância.

    Args:
        grafo: Grafo que contém as coordenadas dos vértices.
        vertice: Vértice a partir do qual o custo é estimado.
        destino: Vértice de destino da busca.

    Returns:
        A distância euclidiana entre o vértice e o destino.
    """
    # Busca a posição dos dois vértices no plano e mede a distância entre elas
    return distancia_euclidiana(grafo.coordenadas[vertice], grafo.coordenadas[destino])

if __name__ == "__main__":
    # Gera uma rede pequena para conferir a heurística
    grafoTeste = gera_rede_knn(300, 5, semente=42)

    # Confere que a heurística de um vértice até ele mesmo é zero
    print("h(0, 0) =", heuristica_euclidiana(grafoTeste, 0, 0))

    # Confere a admissibilidade nas arestas: o peso de cada aresta nunca é menor que a heurística entre suas pontas
    arestasVioladas = 0
    for vertice in range(grafoTeste.quantidade_vertices()):
        for vizinho, peso in grafoTeste.vizinhos(vertice):
            # Usa uma tolerância pequena para ignorar erros de arredondamento de ponto flutuante
            if peso < heuristica_euclidiana(grafoTeste, vertice, vizinho) - 1e-9:
                arestasVioladas += 1

    print("arestas com peso menor que a heurística:", arestasVioladas)