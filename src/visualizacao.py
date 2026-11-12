"""
Visualização da execução dos algoritmos de busca: animação passo a passo e figura em grade.
"""

import math
from pathlib import Path

import matplotlib

from algoritmos import ALGORITMOS
from rede import gera_rede_knn
from utils import sorteia_pares

# Usa o backend Agg, que desenha direto em arquivo e funciona sem janela gráfica
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D

from rede import Grafo
from resultado import EVENTO_EXPANDIDO, ResultadoBusca

# Cor de fundo da figura
COR_FUNDO = "#fcfcfb"

# Cor das arestas da rede
COR_ARESTA = "#e1e0d9"

# Cor dos textos e dos marcadores de origem e destino
COR_TEXTO = "#0b0b0b"

# Cor dos vértices que a busca ainda não viu
COR_NAO_VISTO = "#c3c2b7"

# Cor dos vértices que estão na fronteira
COR_FRONTEIRA = "#eda100"

# Cor dos vértices já expandidos
COR_EXPANDIDO = "#2a78d6"

# Cor do vértice sendo expandido no passo atual
COR_ATUAL = "#e34948"

# Cor do caminho final encontrado
COR_CAMINHO = "#008300"

# Cor da trajetória de uma busca que falhou, como o Hill Climbing preso em um mínimo local
COR_TRAJETORIA_FALHA = "#e34948"

# Códigos de estado de cada vértice, usados como índice na tabela de cores
ESTADO_NAO_VISTO = 0
ESTADO_FRONTEIRA = 1
ESTADO_EXPANDIDO = 2

# Tabela que converte o código de estado na cor RGBA do vértice
CORES_POR_ESTADO = np.array([to_rgba(COR_NAO_VISTO), to_rgba(COR_FRONTEIRA), to_rgba(COR_EXPANDIDO)])

# Tamanho dos vértices comuns e do vértice atual, em pontos ao quadrado (unidade do scatter)
TAMANHO_VERTICE = 28
TAMANHO_VERTICE_ATUAL = 90

# Tamanho dos vértices que fazem parte do caminho final, maior que o dos vértices comuns para cada um ficar visível
TAMANHO_VERTICE_CAMINHO = 60

# Tamanho dos marcadores de origem e destino
TAMANHO_MARCADOR_EXTREMO = 220

def calcula_passos(eventos: list[tuple[str, int]], quantidadeVertices: int) -> list[tuple[int, np.ndarray]]:
    """
    Converte a lista de eventos em uma sequência de passos, um por vértice expandido.

    Cada passo guarda o vértice expandido naquele momento e o estado de todos
    os vértices logo depois da expansão, já com os vizinhos descobertos
    colocados na fronteira.

    Args:
        eventos: Eventos (tipo, vértice) gravados pelo algoritmo.
        quantidadeVertices: Número de vértices do grafo.

    Returns:
        Lista de pares (vértice expandido, vetor de estados dos vértices).
    """
    # Vetor com o estado de cada vértice, começando com todos não vistos
    estados = np.full(quantidadeVertices, ESTADO_NAO_VISTO, dtype=np.int8)

    # Lista de passos montada ao longo dos eventos
    passos: list[tuple[int, np.ndarray]] = []

    # Vértice sendo expandido no passo em construção; None antes da primeira expansão
    verticeAtual: int | None = None

    # Percorre os eventos na ordem em que aconteceram
    for tipoEvento, vertice in eventos:
        if tipoEvento == EVENTO_EXPANDIDO:
            # Uma nova expansão fecha o passo anterior, que é salvo com uma cópia dos estados
            if verticeAtual is not None:
                passos.append((verticeAtual, estados.copy()))

            # Marca o vértice como expandido e começa um novo passo com ele
            estados[vertice] = ESTADO_EXPANDIDO
            verticeAtual = vertice
        else:
            # Um evento de fronteira só muda vértices não vistos, para nunca rebaixar um vértice já expandido
            if estados[vertice] == ESTADO_NAO_VISTO:
                estados[vertice] = ESTADO_FRONTEIRA

    # Salva o último passo, que não foi fechado por uma expansão seguinte
    if verticeAtual is not None:
        passos.append((verticeAtual, estados.copy()))

    return passos

def extrai_trajetoria(grafo: Grafo, eventos: list[tuple[str, int]]) -> list[int] | None:
    """
    Extrai a sequência de vértices expandidos, se ela formar um caminho no grafo.

    Serve para mostrar onde uma busca sem fronteira, como o Hill Climbing,
    parou ao falhar. Nos outros algoritmos a ordem de expansão salta entre
    vértices não adjacentes, e nesse caso nenhuma trajetória é devolvida.

    Args:
        grafo: Grafo onde a busca foi feita.
        eventos: Eventos (tipo, vértice) gravados pelo algoritmo.

    Returns:
        A lista de vértices expandidos em ordem, ou None se ela não for um caminho.
    """
    # Filtra apenas os eventos de expansão, mantendo a ordem
    expandidos = [vertice for tipoEvento, vertice in eventos if tipoEvento == EVENTO_EXPANDIDO]

    # Confere cada par de vértices consecutivos
    for verticeAtual, proximoVertice in zip(expandidos, expandidos[1:]):
        # Monta o conjunto dos vizinhos do vértice atual, ignorando os pesos
        vizinhosAtual = {vizinho for vizinho, _ in grafo.vizinhos(verticeAtual)}

        # Se o próximo vértice não for vizinho, a sequência não é um caminho
        if proximoVertice not in vizinhosAtual:
            return None

    return expandidos

def desenha_base(eixo: plt.Axes, grafo: Grafo, origem: int, destino: int):
    """
    Desenha a rede no eixo: arestas, vértices e marcadores de origem e destino.

    Args:
        eixo: Eixo do matplotlib onde a rede é desenhada.
        grafo: Grafo a ser desenhado.
        origem: Vértice de partida, marcado com uma estrela.
        destino: Vértice de destino, marcado com um quadrado.

    Returns:
        O objeto de pontos dos vértices, cujas cores e tamanhos são alterados a cada quadro.
    """
    # Converte as coordenadas em uma matriz numpy, para acessar todos os x e todos os y de uma vez
    matrizCoordenadas = np.asarray(grafo.coordenadas)

    # Monta a lista de segmentos ((x1, y1), (x2, y2)), um por aresta, sem repetir A-B e B-A
    segmentos = [
        (grafo.coordenadas[vertice], grafo.coordenadas[vizinho])
        for vertice in range(grafo.quantidade_vertices())
        for vizinho, _ in grafo.vizinhos(vertice)
        if vertice < vizinho
    ]

    # Desenha todas as arestas de uma vez com LineCollection, bem mais rápido que uma linha por aresta
    eixo.add_collection(LineCollection(segmentos, colors=COR_ARESTA, linewidths=0.8, zorder=1))

    # Desenha os vértices, todos começando com a cor de não visto
    pontosVertices = eixo.scatter(
        matrizCoordenadas[:, 0],
        matrizCoordenadas[:, 1],
        s=TAMANHO_VERTICE,
        c=[COR_NAO_VISTO] * grafo.quantidade_vertices(),
        # Contorno escuro fino, para que os vértices amarelos continuem visíveis sobre o fundo claro
        edgecolors=COR_TEXTO,
        linewidths=0.3,
        zorder=2,
    )

    # Marca a origem com uma estrela, maior que o quadrado porque a estrela preenche menos área com o mesmo tamanho
    eixo.scatter(*grafo.coordenadas[origem], s=TAMANHO_MARCADOR_EXTREMO * 1.6, marker="*", c=COR_TEXTO, edgecolors=COR_FUNDO, linewidths=1.5, zorder=5)

    # Marca o destino com um quadrado
    eixo.scatter(*grafo.coordenadas[destino], s=TAMANHO_MARCADOR_EXTREMO * 0.6, marker="s", c=COR_TEXTO, edgecolors=COR_FUNDO, linewidths=1.5, zorder=5)

    # Mantém a proporção 1:1 entre os eixos, para que distâncias no desenho correspondam às distâncias reais
    eixo.set_aspect("equal")

    # Remove marcações e bordas, que não acrescentam informação a um desenho de rede
    eixo.set_xticks([])
    eixo.set_yticks([])
    for borda in eixo.spines.values():
        borda.set_visible(False)

    eixo.set_facecolor(COR_FUNDO)

    return pontosVertices

def desenha_caminho(eixo: plt.Axes, grafo: Grafo, caminho: list[int], falhou: bool) -> tuple[Line2D, PathCollection]:
    """
    Desenha um caminho como uma linha grossa ligando os vértices em ordem, com cada vértice destacado.

    O destaque nos vértices deixa visível cada vértice que faz parte do
    caminho, e não só a linha que passa por eles.

    Args:
        eixo: Eixo do matplotlib onde o caminho é desenhado.
        grafo: Grafo que contém as coordenadas dos vértices.
        caminho: Sequência de vértices a desenhar.
        falhou: Se verdadeiro, desenha como trajetória de falha (vermelha e tracejada).

    Returns:
        A linha desenhada e o objeto de pontos dos vértices destacados.
    """
    # Separa as coordenadas x e y dos vértices do caminho, na ordem do caminho
    coordenadasX = [grafo.coordenadas[vertice][0] for vertice in caminho]
    coordenadasY = [grafo.coordenadas[vertice][1] for vertice in caminho]

    # Escolhe cor e estilo conforme o caminho seja uma solução ou uma trajetória de falha
    cor = COR_TRAJETORIA_FALHA if falhou else COR_CAMINHO
    estilo = "--" if falhou else "-"

    # Desenha a linha acima dos vértices e abaixo dos marcadores de origem e destino
    (linhaCaminho,) = eixo.plot(coordenadasX, coordenadasY, color=cor, linestyle=estilo, linewidth=3, solid_capstyle="round", zorder=4)

    # Desenha cada vértice do caminho por cima da linha, na mesma cor, com contorno da cor do fundo para separá-lo da linha
    pontosCaminho = eixo.scatter(coordenadasX, coordenadasY, s=TAMANHO_VERTICE_CAMINHO, c=cor, edgecolors=COR_FUNDO, linewidths=1, zorder=4.5)

    return linhaCaminho, pontosCaminho

def texto_resumo(resultado: ResultadoBusca) -> str:
    """
    Monta o texto com o resultado final de uma busca.

    Args:
        resultado: Resultado da busca.

    Returns:
        Texto com a distância e o número de vértices do caminho, ou o aviso de falha.
    """
    # Em caso de falha, não há caminho nem distância para mostrar
    if not resultado.encontrouSolucao:
        return f"Não encontrou o destino | expandidos: {resultado.nosExpandidos}"

    return (
        f"Distância: {resultado.distancia:.2f} | vértices no caminho: {len(resultado.caminho)} "
        f"| expandidos: {resultado.nosExpandidos}"
    )

def cria_legenda(figura: plt.Figure) -> None:
    """
    Adiciona à figura a legenda com o significado de cada cor e marcador.

    Args:
        figura: Figura do matplotlib que recebe a legenda.
    """
    # Cria elementos falsos só para a legenda, já que os vértices reais mudam de cor ao longo da animação
    elementosLegenda = [
        Line2D([], [], marker="o", linestyle="", markersize=6, markerfacecolor=COR_NAO_VISTO, markeredgecolor=COR_TEXTO, markeredgewidth=0.3, label="Não visto"),
        Line2D([], [], marker="o", linestyle="", markersize=6, markerfacecolor=COR_FRONTEIRA, markeredgecolor=COR_TEXTO, markeredgewidth=0.3, label="Fronteira"),
        Line2D([], [], marker="o", linestyle="", markersize=6, markerfacecolor=COR_EXPANDIDO, markeredgecolor=COR_TEXTO, markeredgewidth=0.3, label="Expandido"),
        Line2D([], [], marker="o", linestyle="", markersize=10, markerfacecolor=COR_ATUAL, markeredgecolor=COR_TEXTO, markeredgewidth=0.3, label="Expandindo agora"),
        Line2D([], [], color=COR_CAMINHO, linewidth=3, marker="o", markersize=7, markeredgecolor=COR_FUNDO, label="Caminho final"),
        Line2D([], [], color=COR_TRAJETORIA_FALHA, linewidth=3, linestyle="--", marker="o", markersize=7, markeredgecolor=COR_FUNDO, label="Trajetória até falhar"),
        Line2D([], [], marker="*", linestyle="", markersize=12, color=COR_TEXTO, label="Origem"),
        Line2D([], [], marker="s", linestyle="", markersize=8, color=COR_TEXTO, label="Destino"),
    ]

    # Coloca a legenda em uma faixa na parte de baixo da figura, em quatro colunas
    figura.legend(handles=elementosLegenda, loc="lower center", ncol=4, frameon=False, fontsize=9, labelcolor=COR_TEXTO)

def desenha_estado_final(
    eixo: plt.Axes,
    grafo: Grafo,
    origem: int,
    destino: int,
    nomeAlgoritmo: str,
    resultado: ResultadoBusca,
) -> None:
    """
    Desenha em um eixo o estado da rede ao fim da busca, com o caminho encontrado.

    Args:
        eixo: Eixo do matplotlib onde o estado é desenhado.
        grafo: Grafo onde a busca foi feita.
        origem: Vértice de partida.
        destino: Vértice de destino.
        nomeAlgoritmo: Nome mostrado no título do eixo.
        resultado: Resultado da busca, que precisa ter sido executada com gravarEventos=True.
    """
    # Desenha a rede vazia
    pontosVertices = desenha_base(eixo, grafo, origem, destino)

    # Calcula os passos e pega o estado dos vértices no último deles
    passos = calcula_passos(resultado.eventos, grafo.quantidade_vertices())
    if passos:
        _, estadosFinais = passos[-1]

        # Pinta cada vértice com a cor do seu estado final
        pontosVertices.set_facecolors(CORES_POR_ESTADO[estadosFinais])

    # Desenha o caminho final ou, em caso de falha, a trajetória até onde a busca chegou
    if resultado.encontrouSolucao:
        desenha_caminho(eixo, grafo, resultado.caminho, falhou=False)
    else:
        trajetoria = extrai_trajetoria(grafo, resultado.eventos)
        if trajetoria is not None and len(trajetoria) > 1:
            desenha_caminho(eixo, grafo, trajetoria, falhou=True)

    # Título com o nome do algoritmo e, abaixo, o resumo do resultado
    eixo.set_title(f"{nomeAlgoritmo}\n{texto_resumo(resultado)}", fontsize=10, color=COR_TEXTO)

def cria_animacao(
    grafo: Grafo,
    origem: int,
    destino: int,
    nomeAlgoritmo: str,
    resultado: ResultadoBusca,
    passosPorQuadro: int = 1,
    quadrosFinais: int = 30,
) -> tuple[plt.Figure, animation.FuncAnimation]:
    """
    Cria a animação passo a passo de uma busca.

    Cada quadro mostra o estado dos vértices depois de uma (ou mais) expansões,
    com o vértice expandido no momento destacado. Ao fim, o caminho final
    aparece e fica parado por alguns quadros.

    Args:
        grafo: Grafo onde a busca foi feita.
        origem: Vértice de partida.
        destino: Vértice de destino.
        nomeAlgoritmo: Nome mostrado no título.
        resultado: Resultado da busca, que precisa ter sido executada com gravarEventos=True.
        passosPorQuadro: Quantas expansões avançam a cada quadro. Valores maiores encurtam a animação.
        quadrosFinais: Quantos quadros o estado final fica parado no fim da animação.

    Returns:
        A figura e o objeto de animação, que ainda precisa ser salvo.
    """
    # Calcula os passos da busca a partir dos eventos gravados
    passos = calcula_passos(resultado.eventos, grafo.quantidade_vertices())

    # Seleciona um passo a cada passosPorQuadro, para acelerar animações longas
    indicesPassos = list(range(0, len(passos), passosPorQuadro))

    # Garante que o último passo sempre aparece, mesmo que o salto não caia exatamente nele
    if indicesPassos and indicesPassos[-1] != len(passos) - 1:
        indicesPassos.append(len(passos) - 1)

    # Cria a figura quadrada, com espaço embaixo para a legenda
    figura, eixo = plt.subplots(figsize=(8, 8.8), facecolor=COR_FUNDO)
    figura.subplots_adjust(left=0.03, right=0.97, top=0.9, bottom=0.1)

    # Desenha a rede e guarda o objeto dos vértices para atualizar a cada quadro
    pontosVertices = desenha_base(eixo, grafo, origem, destino)
    cria_legenda(figura)

    # Vetor de tamanhos dos vértices, alterado para destacar o vértice atual
    tamanhosBase = np.full(grafo.quantidade_vertices(), TAMANHO_VERTICE, dtype=float)

    # Escolhe o desenho final: o caminho encontrado ou a trajetória até a falha; a lista guarda a linha e os vértices destacados
    elementosFinais: list = []
    if resultado.encontrouSolucao:
        elementosFinais = list(desenha_caminho(eixo, grafo, resultado.caminho, falhou=False))
    else:
        trajetoria = extrai_trajetoria(grafo, resultado.eventos)
        if trajetoria and len(trajetoria) > 1:
            elementosFinais = list(desenha_caminho(eixo, grafo, trajetoria, falhou=True))

    # O caminho final fica escondido até o fim da animação
    for elemento in elementosFinais:
        elemento.set_visible(False)

    # Título que muda a cada quadro
    titulo = eixo.set_title("", fontsize=12, color=COR_TEXTO)

    # Total de quadros: um por passo selecionado mais os quadros parados no fim
    totalQuadros = len(indicesPassos) + quadrosFinais

    def atualiza_quadro(indiceQuadro: int) -> list:
        """
        Atualiza cores, tamanhos, linha final e título para o quadro informado.

        Args:
            indiceQuadro: Número do quadro sendo desenhado.

        Returns:
            Lista dos elementos alterados, exigida pelo FuncAnimation.
        """
        # Quadros depois do último passo repetem o estado final
        indicePasso = indicesPassos[min(indiceQuadro, len(indicesPassos) - 1)]
        verticeAtual, estados = passos[indicePasso]

        # Converte os estados em cores usando a tabela, de uma vez para todos os vértices
        cores = CORES_POR_ESTADO[estados].copy()
        tamanhos = tamanhosBase.copy()

        # Verifica se o quadro ainda está mostrando a busca ou já está no estado final
        mostrandoBusca = indiceQuadro < len(indicesPassos)

        # Enquanto a busca acontece, destaca o vértice expandido no passo com outra cor e tamanho maior
        if mostrandoBusca:
            cores[verticeAtual] = to_rgba(COR_ATUAL)
            tamanhos[verticeAtual] = TAMANHO_VERTICE_ATUAL

        pontosVertices.set_facecolors(cores)
        pontosVertices.set_sizes(tamanhos)

        # Mostra o caminho final (linha e vértices destacados) apenas nos quadros parados do fim
        for elemento in elementosFinais:
            elemento.set_visible(not mostrandoBusca)

        # Conta os vértices expandidos até este passo
        expandidosAteAgora = int(np.count_nonzero(estados == ESTADO_EXPANDIDO))

        # Durante a busca, mostra o progresso; no fim, mostra o resumo do resultado
        if mostrandoBusca:
            titulo.set_text(f"{nomeAlgoritmo}\npasso {indicePasso + 1} de {len(passos)} | expandidos: {expandidosAteAgora}")
        else:
            titulo.set_text(f"{nomeAlgoritmo}\n{texto_resumo(resultado)}")

        return [pontosVertices, titulo] + elementosFinais

    # Cria a animação chamando atualiza_quadro para cada quadro, sem blit para o título também ser redesenhado
    animacaoBusca = animation.FuncAnimation(figura, atualiza_quadro, frames=totalQuadros, blit=False)

    return figura, animacaoBusca

def salva_animacao(
    figura: plt.Figure,
    animacaoBusca: animation.FuncAnimation,
    caminhoBase: str | Path,
    quadrosPorSegundo: int = 15,
) -> list[Path]:
    """
    Salva a animação em GIF e, se o ffmpeg estiver instalado, também em MP4.

    Args:
        figura: Figura da animação, fechada depois de salvar para liberar memória.
        animacaoBusca: Animação criada por cria_animacao.
        caminhoBase: Caminho do arquivo sem extensão.
        quadrosPorSegundo: Velocidade da animação.

    Returns:
        Lista dos arquivos gerados.
    """
    # Converte para Path e cria a pasta de destino se ela ainda não existir
    caminhoBase = Path(caminhoBase)
    caminhoBase.parent.mkdir(parents=True, exist_ok=True)

    arquivosGerados: list[Path] = []

    # Salva o GIF com o Pillow, que vem instalado pelo requirements.txt
    caminhoGif = caminhoBase.with_suffix(".gif")
    animacaoBusca.save(caminhoGif, writer=animation.PillowWriter(fps=quadrosPorSegundo), dpi=80)
    arquivosGerados.append(caminhoGif)

    # Salva o MP4 apenas se o ffmpeg estiver disponível no sistema
    if animation.writers.is_available("ffmpeg"):
        caminhoMp4 = caminhoBase.with_suffix(".mp4")
        animacaoBusca.save(caminhoMp4, writer=animation.FFMpegWriter(fps=quadrosPorSegundo), dpi=150)
        arquivosGerados.append(caminhoMp4)

    # Fecha a figura, já que o matplotlib mantém todas as figuras abertas na memória
    plt.close(figura)

    return arquivosGerados

def salva_grade(
    grafo: Grafo,
    origem: int,
    destino: int,
    resultados: dict[str, ResultadoBusca],
    caminhoArquivo: str | Path,
) -> Path:
    """
    Salva uma figura com o estado final de cada algoritmo, lado a lado em grade.

    Args:
        grafo: Grafo onde as buscas foram feitas.
        origem: Vértice de partida.
        destino: Vértice de destino.
        resultados: Dicionário que associa o nome de cada algoritmo ao seu resultado.
        caminhoArquivo: Caminho do arquivo de imagem a salvar.

    Returns:
        O caminho do arquivo salvo.
    """
    # Converte para Path e cria a pasta de destino se ela ainda não existir
    caminhoArquivo = Path(caminhoArquivo)
    caminhoArquivo.parent.mkdir(parents=True, exist_ok=True)

    # Calcula a grade com até 3 colunas e quantas linhas forem necessárias (2 x 3 para seis algoritmos)
    quantidadeColunas = min(3, len(resultados))
    quantidadeLinhas = math.ceil(len(resultados) / quantidadeColunas)

    # Cria a figura com um eixo por algoritmo; squeeze=False garante uma matriz de eixos mesmo com uma linha só
    figura, matrizEixos = plt.subplots(
        quantidadeLinhas,
        quantidadeColunas,
        figsize=(6 * quantidadeColunas, 6.4 * quantidadeLinhas + 0.6),
        facecolor=COR_FUNDO,
        squeeze=False,
    )

    # Achata a matriz de eixos em uma lista, para percorrer junto com os resultados
    listaEixos = matrizEixos.flatten()

    # Desenha o estado final de cada algoritmo em seu eixo
    for eixo, (nomeAlgoritmo, resultado) in zip(listaEixos, resultados.items()):
        desenha_estado_final(eixo, grafo, origem, destino, nomeAlgoritmo, resultado)

    # Esconde os eixos que sobraram, caso o número de algoritmos não preencha a grade
    for eixo in listaEixos[len(resultados):]:
        eixo.set_visible(False)

    cria_legenda(figura)

    # Ajusta o espaçamento deixando espaço embaixo para a legenda
    figura.tight_layout(rect=(0, 0.05, 1, 1))

    figura.savefig(caminhoArquivo, dpi=150, facecolor=COR_FUNDO)
    plt.close(figura)

    return caminhoArquivo

if __name__ == "__main__":
    # Gera a rede e o par dos testes anteriores
    grafoTeste = gera_rede_knn(300, 5, semente=42)
    origemTeste, destinoTeste = sorteia_pares(grafoTeste, 1, distanciaMinima=200)[0]

    # Roda todos os algoritmos gravando eventos
    resultadosTeste = {
        nomeAlgoritmo: funcaoBusca(grafoTeste, origemTeste, destinoTeste, True)
        for nomeAlgoritmo, funcaoBusca in ALGORITMOS.items()
    }

    # Pasta onde os arquivos de teste são salvos
    pastaTeste = Path(__file__).resolve().parent.parent / "resultados" / "teste_visualizacao"

    # Salva a grade com o estado final dos seis algoritmos
    print("grade:", salva_grade(grafoTeste, origemTeste, destinoTeste, resultadosTeste, pastaTeste / "grade.png"))

    # Salva a animação do A* como exemplo
    figuraTeste, animacaoTeste = cria_animacao(grafoTeste, origemTeste, destinoTeste, "A*", resultadosTeste["A*"])
    print("animação:", salva_animacao(figuraTeste, animacaoTeste, pastaTeste / "a_estrela"))