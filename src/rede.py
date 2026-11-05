"""
Geração de redes KNN e estrutura de grafo usada pelos algoritmos de busca.
"""

from dataclasses import dataclass

import numpy as np

@dataclass
class Grafo:
    """
    Grafo não-direcionado e ponderado, com vértices posicionados no plano.

    Os vértices são identificados por inteiros de 0 a quantidade - 1, e esse
    identificador é o índice em `coordenadas` e em `adjacencias`.

    Attributes:
        coordenadas: Posição (x, y) de cada vértice no plano.
        adjacencias: Para cada vértice, a lista de pares (vizinho, peso), onde
            o peso é a distância euclidiana entre os dois vértices.
    """

    coordenadas: list[tuple[float, float]]
    adjacencias: list[list[tuple[int, float]]]

    def quantidade_vertices(self) -> int:
        """
        Retorna o número de vértices do grafo.
        """
        # Cada vértice tem exatamente uma coordenada, então o tamanho da lista é o número de vértices
        return len(self.coordenadas)

    def quantidade_arestas(self) -> int:
        """
        Retorna o número de arestas do grafo.

        Cada aresta aparece duas vezes nas listas de adjacência (uma em cada
        extremidade), por isso a soma dos graus é dividida por dois.
        """
        # Soma o tamanho de todas as listas de adjacência e divide por dois, já que cada aresta é contada nas duas pontas
        return sum(len(listaVizinhos) for listaVizinhos in self.adjacencias) // 2

    def grau_medio(self) -> float:
        """
        Retorna o grau médio dos vértices do grafo.
        """
        # A soma dos graus é o dobro do número de arestas, e a média divide essa soma pelo número de vértices
        return 2 * self.quantidade_arestas() / self.quantidade_vertices()

    def vizinhos(self, vertice: int) -> list[tuple[int, float]]:
        """
        Retorna os pares (vizinho, peso) adjacentes ao vértice informado.
        """
        # O identificador do vértice é o próprio índice na lista de adjacências
        return self.adjacencias[vertice]

def gera_vertices(quantidade: int, semente: int | None = None) -> list[tuple[float, float]]:
    """
    Gera vértices com coordenadas aleatórias uniformes em [0, quantidade] x [0, quantidade].

    O limite do plano cresce com o número de vértices, como pede o enunciado,
    o que mantém a densidade de pontos aproximadamente constante.

    Args:
        quantidade: Número de vértices a gerar.
        semente: Semente do gerador aleatório, para reprodutibilidade.

    Returns:
        Lista com a coordenada (x, y) de cada vértice.
    """

    # Cria um gerador próprio com a semente, para que a mesma semente produza sempre os mesmos pontos
    geradorAleatorio = np.random.default_rng(semente)

    # Sorteia uma matriz quantidade x 2, onde cada linha é o par (x, y) de um vértice
    matrizCoordenadas = geradorAleatorio.uniform(0, quantidade, size=(quantidade, 2))

    # Converte cada linha da matriz numpy em uma tupla de floats nativos do Python
    return [(float(coordenadaX), float(coordenadaY)) for coordenadaX, coordenadaY in matrizCoordenadas]

def calcula_matriz_distancias(coordenadas: list[tuple[float, float]]) -> np.ndarray:
    """
    Calcula a matriz de distâncias euclidianas entre todos os pares de vértices.

    Args:
        coordenadas: Posição (x, y) de cada vértice.

    Returns:
        Matriz simétrica quantidade x quantidade, onde a posição [origem, alvo]
        contém a distância entre os dois vértices.
    """

    # Converte a lista de tuplas em uma matriz numpy quantidade x 2
    matrizCoordenadas = np.asarray(coordenadas, dtype=np.float64)

    # Separa a primeira coluna (todos os x) e a segunda coluna (todos os y)
    eixoX = matrizCoordenadas[:, 0]
    eixoY = matrizCoordenadas[:, 1]

    # Monta a matriz em que a posição [linha, coluna] vale x_linha - x_coluna, para todos os pares de uma vez
    matrizDistancias = np.subtract.outer(eixoX, eixoX)

    # Eleva as diferenças em x ao quadrado, alterando a própria matriz para não alocar outra
    matrizDistancias **= 2

    # Faz o mesmo para as diferenças em y, em uma matriz separada
    diferencasY = np.subtract.outer(eixoY, eixoY)
    diferencasY **= 2

    # Soma as duas parcelas, obtendo (x_i - x_j)^2 + (y_i - y_j)^2 em cada posição
    matrizDistancias += diferencasY

    # Tira a raiz quadrada de cada posição no lugar, resultando na distância euclidiana
    np.sqrt(matrizDistancias, out=matrizDistancias)

    return matrizDistancias

def gera_rede_knn(quantidade: int, k: int, semente: int | None = None) -> Grafo:
    """
    Gera uma rede KNN não-direcionada por força bruta.

    Cada vértice é ligado aos seus k vizinhos mais próximos. Como o grafo é
    não-direcionado, a aresta (A, B) existe se B está entre os k mais próximos
    de A ou se A está entre os k mais próximos de B. Por isso o grau médio
    resultante fica entre k e 2k, tipicamente um pouco acima de k.

    Args:
        quantidade: Número de vértices da rede.
        k: Número de vizinhos mais próximos ligados a cada vértice.
        semente: Semente do gerador aleatório, para reprodutibilidade.

    Returns:
        O grafo gerado, com pesos iguais à distância euclidiana.

    Raises:
        ValueError: Se k não estiver entre 1 e quantidade - 1.
    """

    # Garante que existem vizinhos suficientes, já que um vértice não pode ser vizinho de si mesmo
    if not 1 <= k < quantidade:
        raise ValueError("k deve estar entre 1 e quantidade - 1")

    # Sorteia a posição de cada vértice no plano
    coordenadas = gera_vertices(quantidade, semente)

    # Calcula a distância de cada vértice para todos os outros
    matrizDistancias = calcula_matriz_distancias(coordenadas)

    # Coloca infinito na diagonal (distância de cada vértice a ele mesmo), para que ele nunca seja escolhido como vizinho
    np.fill_diagonal(matrizDistancias, np.inf)

    # Em cada linha, move os índices dos k menores valores para as k primeiras posições, sem ordená-los entre si
    # Isso custa O(n) por linha, mais barato que ordenar a linha inteira com argsort, que custaria O(n log n)
    matrizParticionada = np.argpartition(matrizDistancias, k - 1, axis=1)

    # Fica só com as k primeiras colunas, que são os índices dos k vizinhos mais próximos de cada vértice
    indicesVizinhosProximos = matrizParticionada[:, :k]

    # Conjunto de arestas, onde cada aresta é guardada como (menor, maior) para que A-B e B-A sejam a mesma entrada
    conjuntoArestas: set[tuple[int, int]] = set()

    # Percorre cada vértice junto com a linha dos seus k vizinhos mais próximos
    for vertice, vizinhosProximos in enumerate(indicesVizinhosProximos):
        for vizinho in vizinhosProximos:
            # Converte o inteiro do numpy para int nativo, para não misturar tipos dentro do conjunto
            vizinho = int(vizinho)

            # Adiciona a aresta normalizada; se ela já existir vinda do outro lado, o conjunto ignora a repetição
            conjuntoArestas.add((min(vertice, vizinho), max(vertice, vizinho)))

    # Cria uma lista de adjacência vazia para cada vértice
    adjacencias: list[list[tuple[int, float]]] = [[] for _ in range(quantidade)]

    # Percorre as arestas em ordem, para que as listas de adjacência saiam sempre iguais para a mesma semente
    for verticeA, verticeB in sorted(conjuntoArestas):
        # Usa a distância já calculada na matriz como peso da aresta
        peso = float(matrizDistancias[verticeA, verticeB])

        # Registra a aresta nas duas pontas, já que o grafo é não-direcionado
        adjacencias[verticeA].append((verticeB, peso))
        adjacencias[verticeB].append((verticeA, peso))

    return Grafo(coordenadas=coordenadas, adjacencias=adjacencias)

if __name__ == "__main__":
    # Gera as três configurações do experimento ii e mostra o tamanho e o grau médio de cada uma
    for valorK in (3, 7, 11):
        
        grafoTeste = gera_rede_knn(2500, valorK, semente=42)

        print(
            f"k = {valorK:2d} | vértices = {grafoTeste.quantidade_vertices()} "
            f"| arestas = {grafoTeste.quantidade_arestas()} "
            f"| grau médio = {grafoTeste.grau_medio():.2f}"
        )