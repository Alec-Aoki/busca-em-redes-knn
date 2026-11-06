"""
Subida de encosta (Hill Climbing) sem backtracking.
"""

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

def busca_hill_climbing(grafo: Grafo, origem: int, destino: int, gravarEventos: bool = False) -> ResultadoBusca:
    """
    Executa a busca Hill Climbing da origem até o destino, sem backtracking.

    A busca não mantém fronteira: guarda apenas o vértice atual e, a cada
    passo, pula para o vizinho com menor h(n), a distância em linha reta até o
    destino. Se nenhum vizinho estiver mais perto do destino que o vértice
    atual, a busca está presa em um mínimo local e para, reportando falha.
    Como h diminui a cada passo, um vértice nunca é visitado duas vezes.

    Args:
        grafo: Grafo onde a busca é feita.
        origem: Vértice de partida.
        destino: Vértice que se quer alcançar.
        gravarEventos: Se verdadeiro, grava os eventos de avaliação e expansão
            para a visualização. Deve ficar falso ao medir tempo.

    Returns:
        O resultado da busca, com caminho, distância e nós expandidos.
    """
    # Lista de eventos para a visualização, preenchida apenas se gravarEventos for verdadeiro
    eventos: list[tuple[str, int]] = []

    # Guarda de onde veio cada vértice do caminho; a origem não veio de ninguém
    pais: dict[int, int | None] = {origem: None}

    # A busca começa parada na origem
    verticeAtual = origem

    # Calcula o quão longe a origem está do destino em linha reta
    heuristicaAtual = heuristica_euclidiana(grafo, verticeAtual, destino)

    # Registra a origem como avaliada, para a visualização mostrar o ponto de partida
    if gravarEventos:
        eventos.append((EVENTO_FRONTEIRA, origem))

    # Conta quantos vértices foram expandidos, que aqui são os vértices por onde a busca passou
    nosExpandidos = 0

    # Repete até chegar no destino ou ficar presa; o laço sempre termina, porque h diminui a cada passo
    while True:
        nosExpandidos += 1

        # Registra a expansão do vértice atual
        if gravarEventos:
            eventos.append((EVENTO_EXPANDIDO, verticeAtual))

        # Testa o objetivo ao expandir, como nos outros algoritmos
        if verticeAtual == destino:
            return monta_resultado_sucesso(grafo, pais, destino, nosExpandidos, eventos)

        # Guarda o melhor vizinho encontrado até agora e a sua heurística, começando sem nenhum
        melhorVizinho: int | None = None
        melhorHeuristica = heuristicaAtual

        # Avalia todos os vizinhos do vértice atual, ignorando o peso da aresta
        for vizinho, _ in grafo.vizinhos(verticeAtual):
            # Calcula o quão longe o vizinho está do destino em linha reta
            heuristicaVizinho = heuristica_euclidiana(grafo, vizinho, destino)

            # Registra o vizinho como avaliado, para a visualização mostrar as opções consideradas
            if gravarEventos:
                eventos.append((EVENTO_FRONTEIRA, vizinho))

            # Fica com o vizinho se ele for estritamente melhor que o vértice atual e que os vizinhos anteriores
            if heuristicaVizinho < melhorHeuristica:
                melhorVizinho = vizinho
                melhorHeuristica = heuristicaVizinho

        # Nenhum vizinho está mais perto do destino, então a busca está presa em um mínimo local
        if melhorVizinho is None:
            return monta_resultado_falha(nosExpandidos, eventos)

        # Anota de onde o melhor vizinho foi alcançado e move a busca para ele
        pais[melhorVizinho] = verticeAtual
        verticeAtual = melhorVizinho
        heuristicaAtual = melhorHeuristica

if __name__ == "__main__":
    # Gera a mesma rede e o mesmo par dos testes anteriores, para comparar
    grafoTeste = gera_rede_knn(300, 5, semente=42)
    origemTeste, destinoTeste = sorteia_pares(grafoTeste, 1, distanciaMinima=200)[0]

    # Roda a busca gravando eventos e mostra o resultado principal
    resultadoTeste = busca_hill_climbing(grafoTeste, origemTeste, destinoTeste, gravarEventos=True)
    print(f"origem = {origemTeste} | destino = {destinoTeste}")
    print(f"encontrou = {resultadoTeste.encontrouSolucao} | nós expandidos = {resultadoTeste.nosExpandidos}")

    # Mostra o caminho só se a busca chegou ao destino, já que em caso de falha ele fica vazio
    if resultadoTeste.encontrouSolucao:
        print(f"arestas no caminho = {len(resultadoTeste.caminho) - 1} | distância = {resultadoTeste.distancia:.2f}")

    # Mede a taxa de sucesso nas três configurações do experimento ii
    for valorK in (3, 7, 11):
        grafoGrande = gera_rede_knn(2500, valorK, semente=42)
        paresGrandes = sorteia_pares(grafoGrande, 30)
        sucessos = sum(busca_hill_climbing(grafoGrande, origem, destino).encontrouSolucao for origem, destino in paresGrandes)
        print(f"n = 2500, k = {valorK:2d} | sucesso em {sucessos} de {len(paresGrandes)} pares")