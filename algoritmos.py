"""
Heurísticas, resultado das buscas e os seis algoritmos de busca.

Os algoritmos (DFS, BFS, Best-First, A*, Hill Climbing e Dijkstra) têm a
mesma assinatura, (grafo, origem, destino, gravarEventos), e devolvem um
ResultadoBusca. O dicionário ALGORITMOS, no fim do arquivo, reúne todos para
que os experimentos possam percorrê-los da mesma forma.
"""

import heapq
import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from rede import Grafo

#
# Heurísticas
#

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

#
# Resultado das buscas
#

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

#
# Algoritmos de busca
#

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

def busca_a_estrela(
    grafo: Grafo,
    origem: int,
    destino: int,
    gravarEventos: bool = False,
    funcaoHeuristica: FuncaoHeuristica = heuristica_euclidiana,
) -> ResultadoBusca:
    """
    Executa a busca A* da origem até o destino.

    A fronteira é um heap ordenado por f(n) = g(n) + h(n), onde g(n) é a
    distância já percorrida da origem até n e h(n) é a estimativa da
    heurística de n até o destino. Por padrão, h(n) é a distância em linha
    reta. Como as heurísticas disponíveis (euclidiana e Chebyshev) são
    admissíveis e consistentes nesta rede, o primeiro caminho que chega ao
    destino ao ser expandido é o de menor distância.

    Args:
        grafo: Grafo onde a busca é feita.
        origem: Vértice de partida.
        destino: Vértice que se quer alcançar.
        gravarEventos: Se verdadeiro, grava os eventos de fronteira e expansão
            para a visualização. Deve ficar falso ao medir tempo.
        funcaoHeuristica: Heurística usada para estimar o custo até o destino.
            Fica por último e com valor padrão para que a assinatura comum dos
            algoritmos continue funcionando sem mudanças.

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
    fronteira: list[tuple[float, float, int]] = [(funcaoHeuristica(grafo, origem, destino), 0.0, origem)]

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
            prioridade = novoCusto + funcaoHeuristica(grafo, vizinho, destino)

            # Insere a nova entrada do vizinho no heap
            heapq.heappush(fronteira, (prioridade, novoCusto, vizinho))

            # Registra a entrada do vizinho na fronteira
            if gravarEventos:
                eventos.append((EVENTO_FRONTEIRA, vizinho))

    # O heap esvaziou sem alcançar o destino, então ele está em outra componente
    return monta_resultado_falha(nosExpandidos, eventos)

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

# Assinatura comum a todos os algoritmos: (grafo, origem, destino, gravarEventos) -> ResultadoBusca
FuncaoBusca = Callable[[Grafo, int, int, bool], ResultadoBusca]

# Associa o nome usado nas tabelas e figuras à função de cada algoritmo, na mesma ordem do enunciado
ALGORITMOS: dict[str, FuncaoBusca] = {
    "DFS": busca_profundidade,
    "BFS": busca_largura,
    "Best-First": busca_best_first,
    "A*": busca_a_estrela,
    "Hill Climbing": busca_hill_climbing,
    "Dijkstra": busca_dijkstra,
}
