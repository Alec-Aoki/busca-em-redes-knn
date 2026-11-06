"""
Funções auxiliares dos experimentos: componentes conexas e sorteio de pares (origem, destino).
"""

import math
from collections import deque

import numpy as np

from rede import Grafo, gera_rede_knn

# Semente usada por padrão em todos os experimentos, para que os resultados sejam reproduzíveis
SEMENTE_PADRAO = 42

def encontra_componentes_conexas(grafo: Grafo) -> list[list[int]]:
    """
    Encontra as componentes conexas do grafo usando busca em largura.

    Args:
        grafo: Grafo a ser analisado.

    Returns:
        Lista de componentes, onde cada componente é a lista dos seus vértices.
        As componentes vêm ordenadas da maior para a menor.
    """
    # Marca quais vértices já foram atribuídos a alguma componente
    visitados = [False] * grafo.quantidade_vertices()

    # Lista que acumula as componentes encontradas
    componentes: list[list[int]] = []

    # Tenta iniciar uma nova componente a partir de cada vértice
    for verticeInicial in range(grafo.quantidade_vertices()):
        # Pula o vértice se ele já pertence a uma componente encontrada antes
        if visitados[verticeInicial]:
            continue

        # Começa uma nova componente contendo apenas o vértice inicial
        componenteAtual = [verticeInicial]
        visitados[verticeInicial] = True

        # Fila da busca em largura, começando pelo vértice inicial
        filaVertices = deque([verticeInicial])

        # Expande a busca enquanto houver vértices na fila
        while filaVertices:
            # Retira o vértice mais antigo da fila
            verticeAtual = filaVertices.popleft()

            # Percorre os vizinhos do vértice atual, ignorando o peso da aresta
            for vizinho, _ in grafo.vizinhos(verticeAtual):
                # Só processa vizinhos que ainda não foram alcançados
                if not visitados[vizinho]:
                    # Marca o vizinho e o coloca na componente atual
                    visitados[vizinho] = True
                    componenteAtual.append(vizinho)

                    # Coloca o vizinho no fim da fila para expandi-lo depois
                    filaVertices.append(vizinho)

        # A fila esvaziou, então todos os vértices alcançáveis já estão na componente
        componentes.append(componenteAtual)

    # Ordena as componentes pelo tamanho, da maior para a menor
    componentes.sort(key=len, reverse=True)

    return componentes

def maior_componente(grafo: Grafo) -> list[int]:
    """
    Retorna os vértices da maior componente conexa do grafo.

    Args:
        grafo: Grafo a ser analisado.

    Returns:
        Lista dos vértices da maior componente.
    """
    # Como as componentes vêm ordenadas por tamanho, a primeira é a maior
    return encontra_componentes_conexas(grafo)[0]

def sorteia_pares(
    grafo: Grafo,
    quantidadePares: int,
    semente: int | None = SEMENTE_PADRAO,
    distanciaMinima: float = 0.0,
) -> list[tuple[int, int]]:
    """
    Sorteia pares (origem, destino) distintos dentro da maior componente conexa.

    Usar a maior componente garante que sempre existe caminho entre origem e
    destino, o que é necessário principalmente para k = 3, em que a rede pode
    ser desconexa. Os pares são distintos sem considerar a ordem, ou seja,
    (A, B) e (B, A) contam como o mesmo par.

    Args:
        grafo: Grafo de onde os pares são sorteados.
        quantidadePares: Número de pares a sortear.
        semente: Semente do gerador aleatório, para reprodutibilidade.
        distanciaMinima: Distância euclidiana mínima entre origem e destino.
            Útil na visualização, para evitar pares próximos demais.

    Returns:
        Lista de pares (origem, destino).

    Raises:
        ValueError: Se não for possível sortear a quantidade pedida de pares.
    """
    # Restringe o sorteio à maior componente, onde todos os vértices se alcançam
    componente = maior_componente(grafo)

    # Calcula quantos pares sem ordem existem na componente, que é o limite de pares distintos
    maximoPares = len(componente) * (len(componente) - 1) // 2

    # Recusa pedidos impossíveis antes de entrar no laço de sorteio
    if quantidadePares > maximoPares:
        raise ValueError("a componente não tem pares distintos suficientes")

    # Cria um gerador próprio com a semente, para que a mesma semente produza sempre os mesmos pares
    geradorAleatorio = np.random.default_rng(semente)

    # Lista dos pares na ordem em que foram sorteados
    pares: list[tuple[int, int]] = []

    # Conjunto auxiliar para checar repetição rapidamente, guardando cada par como (menor, maior)
    paresVistos: set[tuple[int, int]] = set()

    # Limita o número de tentativas para não travar se a distância mínima for grande demais
    tentativasRestantes = 1000 * quantidadePares

    # Sorteia até completar a quantidade pedida ou acabarem as tentativas
    while len(pares) < quantidadePares and tentativasRestantes > 0:
        tentativasRestantes -= 1

        # Sorteia dois vértices diferentes da componente (replace=False impede repetir o mesmo vértice)
        origem, destino = geradorAleatorio.choice(componente, size=2, replace=False)

        # Converte os inteiros do numpy para int nativo
        origem, destino = int(origem), int(destino)

        # Normaliza o par para que (A, B) e (B, A) sejam a mesma chave
        chavePar = (min(origem, destino), max(origem, destino))

        # Descarta o par se ele já foi sorteado antes
        if chavePar in paresVistos:
            continue

        # Calcula a distância em linha reta entre origem e destino
        distanciaPar = math.dist(grafo.coordenadas[origem], grafo.coordenadas[destino])

        # Descarta o par se origem e destino estiverem próximos demais
        if distanciaPar < distanciaMinima:
            continue

        # Aceita o par
        paresVistos.add(chavePar)
        pares.append((origem, destino))

    # Avisa se as tentativas acabaram antes de completar os pares
    if len(pares) < quantidadePares:
        raise ValueError("não foi possível sortear pares suficientes com a distância mínima pedida")

    return pares

if __name__ == "__main__":
    # Mostra quantas componentes cada configuração do experimento ii tem e o tamanho da maior
    for valorK in (3, 7, 11):
        grafoTeste = gera_rede_knn(2500, valorK, semente=SEMENTE_PADRAO)
        componentesTeste = encontra_componentes_conexas(grafoTeste)
        print(
            f"k = {valorK:2d} | componentes = {len(componentesTeste)} "
            f"| maior componente = {len(componentesTeste[0])} vértices"
        )

    # Sorteia alguns pares na rede de k = 3, que é a mais propensa a ser desconexa
    grafoTeste = gera_rede_knn(2500, 3, semente=SEMENTE_PADRAO)
    print("pares sorteados (k = 3):", sorteia_pares(grafoTeste, 5))

    # Sorteia um par afastado em uma rede pequena, como no experimento i
    grafoPequeno = gera_rede_knn(300, 5, semente=SEMENTE_PADRAO)
    print("par afastado (n = 300):", sorteia_pares(grafoPequeno, 1, distanciaMinima=200))