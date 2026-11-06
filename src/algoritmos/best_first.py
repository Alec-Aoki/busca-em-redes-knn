"""
Busca gulosa pela melhor escolha (Best-First).
"""

import heapq

from heuristicas import heuristica_euclidiana
from rede import Grafo, gera_rede_knn
from resultado import (
    EVENTO_EXPANDIDO,
    EVENTO_FRONTEIRA,
    ResultadoBusca,
    monta_resultado_falha,
    monta_resultado_sucesso,
)
from utils import sorteia_pares

def busca_best_first(grafo: Grafo, origem: int, destino: int, gravarEventos: bool = False) -> ResultadoBusca:
    """
    Executa a busca gulosa Best-First da origem até o destino.

    A fronteira é um heap ordenado apenas pela heurística h(n), a distância em
    linha reta até o destino. O vértice que parece mais perto do destino é
    sempre o próximo a ser expandido, sem levar em conta o quanto já se
    andou para chegar nele. Costuma expandir poucos vértices, mas não garante
    o caminho de menor distância.

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

    # Guarda de onde veio cada vértice descoberto; a origem não veio de ninguém
    pais: dict[int, int | None] = {origem: None}

    #
    # Heap da fronteira, onde cada item é (h, vértice)
    # O heapq sempre retira o item de menor h; em caso de empate, compara o
    # vértice, o que deixa a ordem determinística
    #
    fronteira: list[tuple[float, int]] = [(heuristica_euclidiana(grafo, origem, destino), origem)]

    # Registra a entrada da origem na fronteira
    if gravarEventos:
        eventos.append((EVENTO_FRONTEIRA, origem))

    # Conta quantos vértices foram retirados da fronteira e expandidos
    nosExpandidos = 0

    # Continua enquanto houver vértices no heap
    while fronteira:
        # Retira o vértice com menor h, ou seja, o que parece mais perto do destino
        _, verticeAtual = heapq.heappop(fronteira)
        nosExpandidos += 1

        # Registra a expansão do vértice atual
        if gravarEventos:
            eventos.append((EVENTO_EXPANDIDO, verticeAtual))

        # Testa o objetivo ao expandir, como nos outros algoritmos
        if verticeAtual == destino:
            return monta_resultado_sucesso(grafo, pais, destino, nosExpandidos, eventos)

        # Percorre os vizinhos do vértice atual, ignorando o peso da aresta
        for vizinho, _ in grafo.vizinhos(verticeAtual):
            #
            # Ignora vizinhos já descobertos. Como h depende só do vértice e não do
            # caminho até ele, descobrir o vizinho de novo por outro pai não mudaria
            # a sua prioridade, então não há motivo para colocá-lo no heap outra vez
            #
            if vizinho in pais:
                continue

            # Anota que o vizinho foi descoberto a partir do vértice atual
            pais[vizinho] = verticeAtual

            # Insere o vizinho no heap com prioridade igual à sua distância em linha reta até o destino
            heapq.heappush(fronteira, (heuristica_euclidiana(grafo, vizinho, destino), vizinho))

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
    resultadoTeste = busca_best_first(grafoTeste, origemTeste, destinoTeste, gravarEventos=True)
    print(f"origem = {origemTeste} | destino = {destinoTeste}")
    print(f"encontrou = {resultadoTeste.encontrouSolucao} | arestas no caminho = {len(resultadoTeste.caminho) - 1}")
    print(f"distância = {resultadoTeste.distancia:.2f} | nós expandidos = {resultadoTeste.nosExpandidos}")
    print(f"eventos gravados = {len(resultadoTeste.eventos)}")