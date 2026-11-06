"""
Busca em profundidade (DFS).
"""

from rede import Grafo, gera_rede_knn
from resultado import (
    EVENTO_EXPANDIDO,
    EVENTO_FRONTEIRA,
    ResultadoBusca,
    monta_resultado_falha,
    monta_resultado_sucesso,
)
from utils import sorteia_pares

def busca_profundidade(grafo: Grafo, origem: int, destino: int, gravarEventos: bool = False) -> ResultadoBusca:
    """
    Executa a busca em profundidade da origem até o destino.

    A fronteira é uma pilha: o vértice descoberto mais recentemente é o
    primeiro a ser expandido. Com isso a busca segue um ramo até não ter mais
    para onde ir e só então volta para tentar outro. A implementação é
    iterativa, com pilha explícita, para não estourar o limite de recursão do
    Python em redes grandes. O caminho encontrado não tem garantia de ser o
    mais curto, nem em arestas nem em distância.

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

    # Guarda de onde veio cada vértice expandido
    pais: dict[int, int | None] = {}

    # Conjunto dos vértices já expandidos, que nunca são expandidos de novo
    expandidos: set[int] = set()

    #
    # Pilha da fronteira, onde cada item é (vértice, vértice de onde ele foi descoberto)
    # O pai fica junto do vértice porque o mesmo vértice pode ser empilhado mais de uma
    # vez, por pais diferentes, e o pai que vale é o da entrada que for desempilhada
    #
    fronteira: list[tuple[int, int | None]] = [(origem, None)]

    # Registra a entrada da origem na fronteira
    if gravarEventos:
        eventos.append((EVENTO_FRONTEIRA, origem))

    # Conta quantos vértices foram retirados da fronteira e expandidos
    nosExpandidos = 0

    # Continua enquanto houver itens na pilha
    while fronteira:
        # Retira o item do topo da pilha (o último que entrou)
        verticeAtual, paiAtual = fronteira.pop()

        # Descarta entradas repetidas de um vértice que já foi expandido por outro caminho
        if verticeAtual in expandidos:
            continue

        # Marca o vértice como expandido e anota de onde ele veio
        expandidos.add(verticeAtual)
        pais[verticeAtual] = paiAtual
        nosExpandidos += 1

        # Registra a expansão do vértice atual
        if gravarEventos:
            eventos.append((EVENTO_EXPANDIDO, verticeAtual))

        # Testa o objetivo ao expandir, como nos outros algoritmos
        if verticeAtual == destino:
            return monta_resultado_sucesso(grafo, pais, destino, nosExpandidos, eventos)

        # Percorre os vizinhos de trás para frente, para que o primeiro vizinho da lista fique no topo da pilha
        for vizinho, _ in reversed(grafo.vizinhos(verticeAtual)):
            # Só empilha vizinhos que ainda não foram expandidos
            if vizinho in expandidos:
                continue

            # Empilha o vizinho junto com o vértice atual como pai
            fronteira.append((vizinho, verticeAtual))

            # Registra a entrada do vizinho na fronteira
            if gravarEventos:
                eventos.append((EVENTO_FRONTEIRA, vizinho))

    # A pilha esvaziou sem alcançar o destino, então ele está em outra componente
    return monta_resultado_falha(nosExpandidos, eventos)

if __name__ == "__main__":
    # Gera a mesma rede e o mesmo par do teste da BFS, para comparar
    grafoTeste = gera_rede_knn(300, 5, semente=42)
    origemTeste, destinoTeste = sorteia_pares(grafoTeste, 1, distanciaMinima=200)[0]

    # Roda a busca gravando eventos e mostra o resultado principal
    resultadoTeste = busca_profundidade(grafoTeste, origemTeste, destinoTeste, gravarEventos=True)
    print(f"origem = {origemTeste} | destino = {destinoTeste}")
    print(f"encontrou = {resultadoTeste.encontrouSolucao} | arestas no caminho = {len(resultadoTeste.caminho) - 1}")
    print(f"distância = {resultadoTeste.distancia:.2f} | nós expandidos = {resultadoTeste.nosExpandidos}")
    print(f"eventos gravados = {len(resultadoTeste.eventos)}")

    # Confere que a DFS iterativa aguenta a rede grande sem estourar recursão
    grafoGrande = gera_rede_knn(2500, 3, semente=42)
    origemGrande, destinoGrande = sorteia_pares(grafoGrande, 1)[0]
    resultadoGrande = busca_profundidade(grafoGrande, origemGrande, destinoGrande)
    print(
        f"n = 2500, k = 3 | encontrou = {resultadoGrande.encontrouSolucao} "
        f"| arestas no caminho = {len(resultadoGrande.caminho) - 1} "
        f"| nós expandidos = {resultadoGrande.nosExpandidos}"
    )