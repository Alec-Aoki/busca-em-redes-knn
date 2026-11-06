"""
Busca em largura (BFS).
"""

from collections import deque

from rede import Grafo, gera_rede_knn
from resultado import (
    EVENTO_EXPANDIDO,
    EVENTO_FRONTEIRA,
    ResultadoBusca,
    monta_resultado_falha,
    monta_resultado_sucesso,
)
from utils import sorteia_pares

def busca_largura(grafo: Grafo, origem: int, destino: int, gravarEventos: bool = False) -> ResultadoBusca:
    """
    Executa a busca em largura da origem até o destino.

    A fronteira é uma fila: o vértice descoberto há mais tempo é o primeiro a
    ser expandido. Com isso a busca avança em camadas a partir da origem e
    encontra o caminho com o menor número de arestas, que não é necessariamente
    o de menor distância, já que os pesos são ignorados.

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

    # Fila da fronteira, começando apenas com a origem
    fronteira = deque([origem])

    # Registra a entrada da origem na fronteira
    if gravarEventos:
        eventos.append((EVENTO_FRONTEIRA, origem))

    # Conta quantos vértices foram retirados da fronteira e expandidos
    nosExpandidos = 0

    # Continua enquanto houver vértices descobertos e ainda não expandidos
    while fronteira:
        # Retira o vértice mais antigo da fila (o primeiro que entrou)
        verticeAtual = fronteira.popleft()
        nosExpandidos += 1

        # Registra a expansão do vértice atual
        if gravarEventos:
            eventos.append((EVENTO_EXPANDIDO, verticeAtual))

        # Testa o objetivo ao expandir, como nos outros algoritmos, para que a contagem de nós seja comparável
        if verticeAtual == destino:
            return monta_resultado_sucesso(grafo, pais, destino, nosExpandidos, eventos)

        # Percorre os vizinhos do vértice atual, ignorando o peso da aresta
        for vizinho, _ in grafo.vizinhos(verticeAtual):
            # Ignora vizinhos já descobertos, porque a primeira descoberta na BFS já é pelo menor número de arestas
            if vizinho in pais:
                continue

            # Anota que o vizinho foi descoberto a partir do vértice atual
            pais[vizinho] = verticeAtual

            # Coloca o vizinho no fim da fila, para ser expandido depois dos que já estão nela
            fronteira.append(vizinho)

            # Registra a entrada do vizinho na fronteira
            if gravarEventos:
                eventos.append((EVENTO_FRONTEIRA, vizinho))

    # A fila esvaziou sem alcançar o destino, então ele está em outra componente
    return monta_resultado_falha(nosExpandidos, eventos)

if __name__ == "__main__":
    # Gera uma rede pequena e sorteia um par afastado para testar
    grafoTeste = gera_rede_knn(300, 5, semente=42)
    origemTeste, destinoTeste = sorteia_pares(grafoTeste, 1, distanciaMinima=200)[0]

    # Roda a busca gravando eventos e mostra o resultado principal
    resultadoTeste = busca_largura(grafoTeste, origemTeste, destinoTeste, gravarEventos=True)
    print(f"origem = {origemTeste} | destino = {destinoTeste}")
    print(f"encontrou = {resultadoTeste.encontrouSolucao} | arestas no caminho = {len(resultadoTeste.caminho) - 1}")
    print(f"distância = {resultadoTeste.distancia:.2f} | nós expandidos = {resultadoTeste.nosExpandidos}")
    print(f"eventos gravados = {len(resultadoTeste.eventos)}")
    print("caminho:", resultadoTeste.caminho)