"""
Busca A*.
"""

import heapq
import math

from heuristicas import heuristica_euclidiana
from rede import Grafo
from resultado import (
    EVENTO_EXPANDIDO,
    EVENTO_FRONTEIRA,
    ResultadoBusca,
    monta_resultado_falha,
    monta_resultado_sucesso,
)

def busca_a_estrela(grafo: Grafo, origem: int, destino: int, gravarEventos: bool = False) -> ResultadoBusca:
    """
    Executa a busca A* da origem até o destino.

    A fronteira é um heap ordenado por f(n) = g(n) + h(n), onde g(n) é a
    distância já percorrida da origem até n e h(n) é a distância em linha
    reta de n até o destino. Como a heurística euclidiana é admissível e
    consistente nesta rede, o primeiro caminho que chega ao destino ao ser
    expandido é o de menor distância.

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

    #
    # Heap da fronteira, onde cada item é (f, g, vértice)
    # O heapq retira sempre o menor f; em caso de empate, prefere o menor g e
    # depois o menor vértice, o que deixa a ordem determinística
    #
    fronteira: list[tuple[float, float, int]] = [(heuristica_euclidiana(grafo, origem, destino), 0.0, origem)]

    # Registra a entrada da origem na fronteira
    if gravarEventos:
        eventos.append((EVENTO_FRONTEIRA, origem))

    # Conta quantos vértices foram retirados da fronteira e expandidos
    nosExpandidos = 0

    # Continua enquanto houver itens no heap
    while fronteira:
        # Retira o item de menor f, ou seja, o mais promissor considerando o caminho já feito e o que falta
        _, custoAtual, verticeAtual = heapq.heappop(fronteira)

        #
        # Descarta entradas obsoletas. O heapq não permite atualizar a prioridade de
        # um item, então quando um caminho melhor até um vértice é encontrado, uma
        # nova entrada é inserida e a antiga fica no heap. Ela é reconhecida aqui por
        # já ter sido expandida ou por ter um g maior que o melhor conhecido
        #
        if verticeAtual in expandidos or custoAtual > melhorCusto[verticeAtual]:
            continue

        # Marca o vértice como expandido
        expandidos.add(verticeAtual)
        nosExpandidos += 1

        # Registra a expansão do vértice atual
        if gravarEventos:
            eventos.append((EVENTO_EXPANDIDO, verticeAtual))

        # Testa o objetivo ao expandir, que é o momento em que o custo até o destino passa a ser definitivo
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

            # Calcula f somando o custo já percorrido com a estimativa do que falta até o destino
            prioridade = novoCusto + heuristica_euclidiana(grafo, vizinho, destino)

            # Insere a nova entrada do vizinho no heap
            heapq.heappush(fronteira, (prioridade, novoCusto, vizinho))

            # Registra a entrada do vizinho na fronteira
            if gravarEventos:
                eventos.append((EVENTO_FRONTEIRA, vizinho))

    # O heap esvaziou sem alcançar o destino, então ele está em outra componente
    return monta_resultado_falha(nosExpandidos, eventos)

if __name__ == "__main__":
    from rede import gera_rede_knn
    from utils import sorteia_pares

    # Gera a mesma rede e o mesmo par dos testes anteriores, para comparar
    grafoTeste = gera_rede_knn(300, 5, semente=42)
    origemTeste, destinoTeste = sorteia_pares(grafoTeste, 1, distanciaMinima=200)[0]

    # Roda a busca gravando eventos e mostra o resultado principal
    resultadoTeste = busca_a_estrela(grafoTeste, origemTeste, destinoTeste, gravarEventos=True)
    print(f"origem = {origemTeste} | destino = {destinoTeste}")
    print(f"encontrou = {resultadoTeste.encontrouSolucao} | arestas no caminho = {len(resultadoTeste.caminho) - 1}")
    print(f"distância = {resultadoTeste.distancia:.2f} | nós expandidos = {resultadoTeste.nosExpandidos}")
    print(f"eventos gravados = {len(resultadoTeste.eventos)}")