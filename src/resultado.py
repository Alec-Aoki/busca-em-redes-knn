"""
Formato padrão de retorno dos algoritmos de busca e funções para montá-lo.
"""

import math
from dataclasses import dataclass, field

from rede import Grafo

# Tipo de evento gravado quando um vértice entra na fronteira (fila, pilha ou heap)
EVENTO_FRONTEIRA = "fronteira"

# Tipo de evento gravado quando um vértice é retirado da fronteira e expandido
EVENTO_EXPANDIDO = "expandido"

@dataclass
class ResultadoBusca:
    """
    Resultado de uma execução de algoritmo de busca.

    Attributes:
        encontrouSolucao: Indica se o algoritmo chegou ao destino.
        caminho: Vértices do caminho encontrado, da origem ao destino. Vazio se falhou.
        distancia: Soma dos pesos das arestas do caminho. Infinito se falhou.
        nosExpandidos: Número de vértices retirados da fronteira e expandidos.
        eventos: Sequência de eventos (tipo, vértice) usada pela visualização.
            Fica vazia quando a busca roda sem gravar eventos.
    """

    encontrouSolucao: bool
    caminho: list[int]
    distancia: float
    nosExpandidos: int
    eventos: list[tuple[str, int]] = field(default_factory=list)

def peso_aresta(grafo: Grafo, verticeA: int, verticeB: int) -> float:
    """
    Retorna o peso da aresta entre dois vértices.

    Args:
        grafo: Grafo que contém a aresta.
        verticeA: Uma das pontas da aresta.
        verticeB: A outra ponta da aresta.

    Returns:
        O peso da aresta (verticeA, verticeB).

    Raises:
        ValueError: Se os dois vértices não forem adjacentes.
    """
    # Procura o verticeB na lista de vizinhos do verticeA
    for vizinho, peso in grafo.vizinhos(verticeA):
        # Retorna o peso assim que encontra o vizinho procurado
        if vizinho == verticeB:
            return peso

    # Chegar aqui significa que o caminho usa uma aresta que não existe, o que indica bug no algoritmo
    raise ValueError(f"não existe aresta entre {verticeA} e {verticeB}")

def calcula_distancia_caminho(grafo: Grafo, caminho: list[int]) -> float:
    """
    Soma os pesos das arestas ao longo de um caminho.

    Args:
        grafo: Grafo que contém o caminho.
        caminho: Sequência de vértices, onde cada par consecutivo é uma aresta.

    Returns:
        A distância total percorrida pelo caminho.
    """
    # Acumula o peso de cada aresta do caminho
    distanciaTotal = 0.0

    # Percorre os pares consecutivos (v0, v1), (v1, v2), ..., que são as arestas usadas
    for verticeAtual, proximoVertice in zip(caminho, caminho[1:]):
        distanciaTotal += peso_aresta(grafo, verticeAtual, proximoVertice)

    return distanciaTotal

def reconstroi_caminho(pais: dict[int, int | None], destino: int) -> list[int]:
    """
    Reconstrói o caminho da origem até o destino a partir do dicionário de pais.

    Durante a busca, cada vértice alcançado guarda o vértice de onde veio. A
    origem guarda None, porque não veio de ninguém. Seguindo os pais a partir
    do destino, chega-se à origem, e basta inverter a sequência.

    Args:
        pais: Dicionário que associa cada vértice alcançado ao vértice anterior.
        destino: Vértice final do caminho.

    Returns:
        Lista de vértices da origem ao destino.
    """
    # Começa a reconstrução pelo destino
    caminhoInvertido = [destino]

    # Pega o pai do destino como próximo vértice a visitar
    verticeAtual = pais[destino]

    # Sobe pelos pais até chegar na origem, cujo pai é None
    while verticeAtual is not None:
        caminhoInvertido.append(verticeAtual)
        verticeAtual = pais[verticeAtual]

    # Inverte a lista, porque ela foi montada do destino para a origem
    return caminhoInvertido[::-1]

def monta_resultado_sucesso(
    grafo: Grafo,
    pais: dict[int, int | None],
    destino: int,
    nosExpandidos: int,
    eventos: list[tuple[str, int]],
) -> ResultadoBusca:
    """
    Monta o resultado de uma busca que chegou ao destino.

    Args:
        grafo: Grafo onde a busca foi feita.
        pais: Dicionário de pais preenchido durante a busca.
        destino: Vértice de destino da busca.
        nosExpandidos: Número de vértices expandidos.
        eventos: Eventos gravados durante a busca.

    Returns:
        O resultado com o caminho e a distância calculados.
    """
    # Recupera a sequência de vértices da origem ao destino
    caminho = reconstroi_caminho(pais, destino)

    # Soma os pesos das arestas do caminho para obter a distância percorrida
    distancia = calcula_distancia_caminho(grafo, caminho)

    return ResultadoBusca(
        encontrouSolucao=True,
        caminho=caminho,
        distancia=distancia,
        nosExpandidos=nosExpandidos,
        eventos=eventos,
    )

def monta_resultado_falha(nosExpandidos: int, eventos: list[tuple[str, int]]) -> ResultadoBusca:
    """
    Monta o resultado de uma busca que não chegou ao destino.

    Args:
        nosExpandidos: Número de vértices expandidos até a busca desistir.
        eventos: Eventos gravados durante a busca.

    Returns:
        O resultado com caminho vazio e distância infinita.
    """
    # Usa distância infinita para que a falha nunca seja confundida com um caminho válido
    return ResultadoBusca(
        encontrouSolucao=False,
        caminho=[],
        distancia=math.inf,
        nosExpandidos=nosExpandidos,
        eventos=eventos,
    )

if __name__ == "__main__":
    # Monta à mão um grafo de 4 vértices em linha: 0 - 1 - 2 - 3, com pesos 1, 2 e 3
    grafoTeste = Grafo(
        coordenadas=[(0.0, 0.0), (1.0, 0.0), (3.0, 0.0), (6.0, 0.0)],
        adjacencias=[
            [(1, 1.0)],
            [(0, 1.0), (2, 2.0)],
            [(1, 2.0), (3, 3.0)],
            [(2, 3.0)],
        ],
    )

    # Simula o dicionário de pais que uma busca de 0 até 3 deixaria
    paisTeste: dict[int, int | None] = {0: None, 1: 0, 2: 1, 3: 2}

    # Monta o resultado e confere caminho [0, 1, 2, 3] e distância 6
    print(monta_resultado_sucesso(grafoTeste, paisTeste, 3, nosExpandidos=4, eventos=[]))

    # Confere o formato de uma falha
    print(monta_resultado_falha(nosExpandidos=2, eventos=[]))