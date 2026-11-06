"""
Algoritmo de Dijkstra.
"""

import heapq
import math

from rede import Grafo, gera_rede_knn
from resultado import (
    EVENTO_EXPANDIDO,
    EVENTO_FRONTEIRA,
    ResultadoBusca,
    monta_resultado_falha,
    monta_resultado_sucesso,
)
from algoritmos.a_estrela import busca_a_estrela
from utils import sorteia_pares

def busca_dijkstra(grafo: Grafo, origem: int, destino: int, gravarEventos: bool = False) -> ResultadoBusca:
    """
    Executa o algoritmo de Dijkstra da origem até o destino.

    A fronteira é um heap ordenado por g(n), a distância já percorrida da
    origem até n. O vértice mais perto da origem é sempre o próximo a ser
    expandido, então a busca cresce em círculos de distância a partir da
    origem, sem nenhuma informação sobre onde está o destino. Encontra sempre
    o caminho de menor distância e para assim que o destino é expandido.

    Args:
        grafo: Grafo onde a busca é feita.
        origem: Vértice de partida.
        destino: Vértice que se quer alcançar.
        gravarEventos: Se verdadeiro, grava os eventos de fronteira e expansão
            para a visualização. Deve ficar falso ao medir tempo.

    Returns:
        O resultado da busca, com caminho, distância e nós expandidos.
    """
    # Lista de eventos para a visualização, preenchida apenas se gravarEventos for verdadeiro
    eventos: list[tuple[str, int]] = []

    # Guarda de onde veio cada vértice pelo melhor caminho conhecido até agora
    pais: dict[int, int | None] = {origem: None}

    # Guarda o menor g conhecido até cada vértice; a origem está a distância zero dela mesma
    melhorCusto: dict[int, float] = {origem: 0.0}

    # Conjunto dos vértices já expandidos, cujo menor custo já é definitivo
    expandidos: set[int] = set()

    # Heap da fronteira, onde cada item é (g, vértice); em caso de empate em g, compara o vértice
    fronteira: list[tuple[float, int]] = [(0.0, origem)]

    # Registra a entrada da origem na fronteira
    if gravarEventos:
        eventos.append((EVENTO_FRONTEIRA, origem))

    # Conta quantos vértices foram retirados da fronteira e expandidos
    nosExpandidos = 0

    # Continua enquanto houver itens no heap
    while fronteira:
        # Retira o item de menor g, ou seja, o vértice mais perto da origem entre os da fronteira
        custoAtual, verticeAtual = heapq.heappop(fronteira)

        # Descarta entradas obsoletas, deixadas no heap quando um caminho melhor até o vértice foi encontrado
        if verticeAtual in expandidos or custoAtual > melhorCusto[verticeAtual]:
            continue

        # Marca o vértice como expandido
        expandidos.add(verticeAtual)
        nosExpandidos += 1

        # Registra a expansão do vértice atual
        if gravarEventos:
            eventos.append((EVENTO_EXPANDIDO, verticeAtual))

        # Para ao expandir o destino, já que a partir daí o seu custo é definitivo
        if verticeAtual == destino:
            return monta_resultado_sucesso(grafo, pais, destino, nosExpandidos, eventos)

        # Percorre os vizinhos do vértice atual junto com o peso de cada aresta
        for vizinho, peso in grafo.vizinhos(verticeAtual):
            # Ignora vizinhos já expandidos, cujo menor custo já é conhecido
            if vizinho in expandidos:
                continue

            # Calcula o custo de chegar no vizinho passando pelo vértice atual
            novoCusto = custoAtual + peso

            # Só atualiza se o novo caminho for melhor que o melhor conhecido (infinito se o vizinho nunca foi visto)
            if novoCusto >= melhorCusto.get(vizinho, math.inf):
                continue

            # Guarda o novo melhor custo e o novo pai do vizinho
            melhorCusto[vizinho] = novoCusto
            pais[vizinho] = verticeAtual

            # Insere a nova entrada do vizinho no heap, com prioridade igual ao custo desde a origem
            heapq.heappush(fronteira, (novoCusto, vizinho))

            # Registra a entrada do vizinho na fronteira
            if gravarEventos:
                eventos.append((EVENTO_FRONTEIRA, vizinho))

    # O heap esvaziou sem alcançar o destino, então ele está em outra componente
    return monta_resultado_falha(nosExpandidos, eventos)

if __name__ == "__main__":
    # Gera a mesma rede e o mesmo par dos testes anteriores, para comparar
    grafoTeste = gera_rede_knn(300, 5, semente=42)
    origemTeste, destinoTeste = sorteia_pares(grafoTeste, 1, distanciaMinima=200)[0]

    # Roda a busca gravando eventos e mostra o resultado principal
    resultadoTeste = busca_dijkstra(grafoTeste, origemTeste, destinoTeste, gravarEventos=True)
    print(f"origem = {origemTeste} | destino = {destinoTeste}")
    print(f"encontrou = {resultadoTeste.encontrouSolucao} | arestas no caminho = {len(resultadoTeste.caminho) - 1}")
    print(f"distância = {resultadoTeste.distancia:.2f} | nós expandidos = {resultadoTeste.nosExpandidos}")

    # Confere que A* e Dijkstra dão a mesma distância nas três configurações do experimento ii
    for valorK in (3, 7, 11):
        grafoGrande = gera_rede_knn(2500, valorK, semente=42)
        paresGrandes = sorteia_pares(grafoGrande, 30)

        # Conta os pares em que as distâncias divergem e soma os nós expandidos de cada algoritmo
        divergencias = 0
        expandidosDijkstra = 0
        expandidosAEstrela = 0
        for origem, destino in paresGrandes:
            resultadoDijkstra = busca_dijkstra(grafoGrande, origem, destino)
            resultadoAEstrela = busca_a_estrela(grafoGrande, origem, destino)

            # Usa uma tolerância pequena para ignorar erros de arredondamento de ponto flutuante
            if abs(resultadoDijkstra.distancia - resultadoAEstrela.distancia) > 1e-6:
                divergencias += 1

            expandidosDijkstra += resultadoDijkstra.nosExpandidos
            expandidosAEstrela += resultadoAEstrela.nosExpandidos

        print(
            f"n = 2500, k = {valorK:2d} | divergências = {divergencias} de {len(paresGrandes)} "
            f"| expandidos (média): Dijkstra = {expandidosDijkstra / len(paresGrandes):.0f}, "
            f"A* = {expandidosAEstrela / len(paresGrandes):.0f}"
        )