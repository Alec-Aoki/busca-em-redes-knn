"""
Heurísticas usadas pelos algoritmos de busca informada (Best-First, A* e Hill Climbing).
"""

import math
from collections.abc import Callable

from rede import Grafo, gera_rede_knn

# Assinatura comum a todas as heurísticas: (grafo, vértice, destino) -> custo estimado do vértice até o destino
FuncaoHeuristica = Callable[[Grafo, int, int], float]

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

def distancia_chebyshev(pontoA: tuple[float, float], pontoB: tuple[float, float]) -> float:
    """
    Calcula a distância de Chebyshev entre dois pontos do plano.

    É a maior das diferenças entre as coordenadas, também chamada de
    distância do rei, por ser o número de movimentos que o rei do xadrez faz
    entre duas casas. Num triângulo retângulo com catetos iguais às
    diferenças em x e em y, ela é o cateto maior, enquanto a distância
    euclidiana é a hipotenusa.

    Args:
        pontoA: Coordenada (x, y) do primeiro ponto.
        pontoB: Coordenada (x, y) do segundo ponto.

    Returns:
        A maior entre a diferença em x e a diferença em y, em valor absoluto.
    """
    # Calcula o tamanho da diferença em x e em y, sem sinal, já que só importa o quanto os pontos se afastam
    diferencaX = abs(pontoA[0] - pontoB[0])
    diferencaY = abs(pontoA[1] - pontoB[1])

    # Fica com a maior das duas diferenças
    return max(diferencaX, diferencaY)

def heuristica_chebyshev(grafo: Grafo, vertice: int, destino: int) -> float:
    """
    Estima o custo restante de um vértice até o destino pela distância de Chebyshev.

    A distância de Chebyshev nunca é maior que a euclidiana (o cateto maior
    nunca passa da hipotenusa), que por sua vez nunca é maior que o custo
    real. Por isso a heurística também é admissível, e o A* continua
    encontrando o caminho de menor distância. Ela coincide com a euclidiana
    quando o destino está na mesma linha ou coluna do vértice e subestima mais
    quando ele está na diagonal (cerca de 29% a menos a 45 graus), o que a
    torna menos informativa e faz o A* expandir mais vértices.

    Args:
        grafo: Grafo que contém as coordenadas dos vértices.
        vertice: Vértice a partir do qual o custo é estimado.
        destino: Vértice de destino da busca.

    Returns:
        A distância de Chebyshev entre o vértice e o destino.
    """
    # Busca a posição dos dois vértices no plano e mede a distância de Chebyshev entre elas
    return distancia_chebyshev(grafo.coordenadas[vertice], grafo.coordenadas[destino])

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

    # Confere que Chebyshev nunca passa da euclidiana, o que garante a admissibilidade dela também
    paresViolados = 0
    for vertice in range(grafoTeste.quantidade_vertices()):
        # Compara as duas heurísticas do vértice atual até todos os outros vértices
        for destino in range(grafoTeste.quantidade_vertices()):
            if heuristica_chebyshev(grafoTeste, vertice, destino) > heuristica_euclidiana(grafoTeste, vertice, destino) + 1e-9:
                paresViolados += 1

    print("pares em que Chebyshev passa da euclidiana:", paresViolados)