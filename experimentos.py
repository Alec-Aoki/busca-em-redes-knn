"""
Visualização das buscas e pipeline com os três experimentos do trabalho.

A primeira parte gera as animações passo a passo de cada busca e a figura em
grade com o estado final de todos os algoritmos. A segunda parte roda os
experimentos em sequência:

Experimento i: visualização dos seis algoritmos em uma rede KNN pequena,
com dois pares (origem, destino), um em que todos chegam ao destino e outro
em que o Hill Climbing fica preso em um mínimo local.

Experimento ii: estatísticas de distância, tempo e nós expandidos em redes
de 2500 vértices com k = 3, 7 e 11, sobre 30 pares.

Experimento iii: comparação entre A* e Dijkstra sobre as execuções do
experimento ii, incluindo o A* com a heurística de Chebyshev.

Os resultados de cada experimento são salvos em resultados/exp1, exp2 e exp3,
ao lado deste arquivo.
"""

import csv
import gc
import itertools
import math
import threading
import time
from functools import partial
from pathlib import Path

import matplotlib

# Usa o backend Agg, que desenha direto em arquivo e funciona sem janela gráfica
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D

from algoritmos import ALGORITMOS, EVENTO_EXPANDIDO, FuncaoBusca, ResultadoBusca, busca_a_estrela, heuristica_chebyshev
from rede import SEMENTE_PADRAO, Grafo, gera_rede_knn, sorteia_pares

#
# Visualização
#

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

# Pasta onde os resultados são salvos, ao lado deste arquivo
PASTA_RESULTADOS = Path(__file__).resolve().parent / "resultados"

#
# Experimento i: visualização
#

# Número de vértices da rede, dentro da faixa de 100 a 500 pedida pelo enunciado
VERTICES_EXP1 = 300

# Número de vizinhos mais próximos de cada vértice
K_EXP1 = 5

# Distância mínima em linha reta entre origem e destino, para que a busca atravesse boa parte da rede
DISTANCIA_MINIMA_PAR = 200

# Quantos pares candidatos são sorteados para procurar os dois cenários
QUANTIDADE_CANDIDATOS = 200

# Velocidade das animações
QUADROS_POR_SEGUNDO = 15

# Pasta onde os resultados do experimento são salvos
PASTA_EXP1 = PASTA_RESULTADOS / "exp1"

# Nome de cada cenário, usado como nome da subpasta de saída
CENARIO_SUCESSO = "sucesso"
CENARIO_FALHA = "falha_hill_climbing"

# Quadros do indicador de progresso, mostrados em sequência para dar a impressão de um traço girando
QUADROS_INDICADOR = "|/-\\"

# Intervalo entre dois quadros do indicador, em segundos
INTERVALO_INDICADOR = 0.15

class IndicadorProgresso:
    """
    Mostra no terminal um traço girando e o tempo decorrido enquanto uma tarefa demorada roda.

    Deve ser usado com `with`: o indicador começa ao entrar no bloco e é
    apagado ao sair, deixando no lugar uma linha com o tempo total gasto. A
    animação roda em uma thread separada, para que a tarefa principal não
    precise parar para atualizar o terminal.

    Attributes:
        mensagem: Texto mostrado ao lado do traço, descrevendo a tarefa.
    """

    def __init__(self, mensagem: str) -> None:
        """
        Prepara o indicador sem iniciá-lo.

        Args:
            mensagem: Texto mostrado ao lado do traço, descrevendo a tarefa.
        """
        self.mensagem = mensagem

        # Sinal usado para avisar a thread do indicador que ela deve parar
        self.sinalParada = threading.Event()

        # Thread que desenha o indicador; daemon=True faz ela morrer junto com o programa se ele for interrompido
        self.threadIndicador = threading.Thread(target=self.anima, daemon=True)

        # Instante em que o indicador começou, preenchido ao entrar no bloco with
        self.instanteInicio = 0.0

    def anima(self) -> None:
        """
        Redesenha a linha do indicador até receber o sinal de parada.
        """
        # Percorre os quadros do traço em ciclo infinito: |, /, -, \, |, /, ...
        for quadroAtual in itertools.cycle(QUADROS_INDICADOR):
            # Para o laço assim que a tarefa principal terminar
            if self.sinalParada.is_set():
                break

            # Calcula há quantos segundos a tarefa começou
            segundosDecorridos = time.perf_counter() - self.instanteInicio

            # O \r volta o cursor para o começo da linha, então cada desenho sobrescreve o anterior
            print(f"\r{quadroAtual} {self.mensagem} ({segundosDecorridos:.0f}s)", end="", flush=True)

            # Espera o próximo quadro; o wait retorna antes do tempo se o sinal de parada chegar
            self.sinalParada.wait(INTERVALO_INDICADOR)

    def __enter__(self) -> "IndicadorProgresso":
        """
        Marca o instante de início e começa a animação.
        """
        self.instanteInicio = time.perf_counter()
        self.threadIndicador.start()
        return self

    def __exit__(self, tipoExcecao, valorExcecao, rastreamento) -> None:
        """
        Para a animação e troca a linha do indicador pela mensagem final.

        Args:
            tipoExcecao: Tipo da exceção levantada dentro do bloco, ou None.
            valorExcecao: Exceção levantada dentro do bloco, ou None.
            rastreamento: Rastreamento da exceção, ou None.
        """
        # Avisa a thread que ela deve parar e espera ela terminar o desenho atual
        self.sinalParada.set()
        self.threadIndicador.join()

        segundosDecorridos = time.perf_counter() - self.instanteInicio

        # Escolhe o símbolo final conforme a tarefa tenha terminado normalmente ou com erro
        simboloFinal = "ok" if tipoExcecao is None else "erro"

        # Sobrescreve a linha do indicador; os espaços extras apagam restos de uma linha anterior mais longa
        print(f"\r[{simboloFinal}] {self.mensagem} ({segundosDecorridos:.1f}s)" + " " * 10)

def escolhe_pares(grafo: Grafo) -> dict[str, tuple[int, int]]:
    """
    Escolhe um par para cada cenário do experimento.

    O par de sucesso é o primeiro candidato em que o Hill Climbing chega ao
    destino, e o de falha é o primeiro em que ele não chega. Basta testar o
    Hill Climbing, porque os outros algoritmos sempre chegam ao destino
    dentro de uma mesma componente conexa.

    Args:
        grafo: Rede onde os pares são sorteados.

    Returns:
        Dicionário que associa o nome do cenário ao seu par (origem, destino).

    Raises:
        RuntimeError: Se algum dos cenários não for encontrado entre os candidatos.
    """
    # Sorteia os candidatos com a semente padrão, para que os pares escolhidos sejam sempre os mesmos
    candidatos = sorteia_pares(grafo, QUANTIDADE_CANDIDATOS, semente=SEMENTE_PADRAO, distanciaMinima=DISTANCIA_MINIMA_PAR)

    # Pega a função do Hill Climbing no dicionário de algoritmos
    funcaoHillClimbing = ALGORITMOS["Hill Climbing"]

    paresEscolhidos: dict[str, tuple[int, int]] = {}

    # Testa os candidatos em ordem até encontrar os dois cenários
    for origem, destino in candidatos:
        # Roda o Hill Climbing sem gravar eventos, já que aqui só interessa saber se ele chegou
        chegou = funcaoHillClimbing(grafo, origem, destino, False).encontrouSolucao

        # Classifica o par conforme o resultado e só guarda o primeiro de cada cenário
        cenario = CENARIO_SUCESSO if chegou else CENARIO_FALHA
        paresEscolhidos.setdefault(cenario, (origem, destino))

        # Para assim que os dois cenários tiverem um par
        if len(paresEscolhidos) == 2:
            return paresEscolhidos

    raise RuntimeError("não foi possível encontrar os dois cenários entre os candidatos sorteados")

def salva_csv(caminhoArquivo: Path, cenario: str, origem: int, destino: int, resultados: dict[str, ResultadoBusca]) -> None:
    """
    Salva em CSV o resultado de cada algoritmo em um cenário.

    Args:
        caminhoArquivo: Caminho do arquivo CSV.
        cenario: Nome do cenário, repetido em cada linha.
        origem: Vértice de partida.
        destino: Vértice de destino.
        resultados: Dicionário que associa o nome de cada algoritmo ao seu resultado.
    """
    # Abre o arquivo com newline vazio, como pede o módulo csv, e utf-8 para os acentos
    with open(caminhoArquivo, "w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.writer(arquivo)

        # Escreve o cabeçalho
        escritor.writerow(["cenario", "algoritmo", "origem", "destino", "encontrou", "distancia", "vertices_no_caminho", "nos_expandidos", "caminho"])

        # Escreve uma linha por algoritmo
        for nomeAlgoritmo, resultado in resultados.items():
            escritor.writerow([
                cenario,
                nomeAlgoritmo,
                origem,
                destino,
                resultado.encontrouSolucao,
                # Deixa a distância vazia em caso de falha, em vez de escrever infinito
                f"{resultado.distancia:.4f}" if resultado.encontrouSolucao else "",
                len(resultado.caminho),
                resultado.nosExpandidos,
                # Junta os vértices do caminho com hífen, para caber em uma única célula
                "-".join(str(vertice) for vertice in resultado.caminho),
            ])

def monta_nome_arquivo(nomeAlgoritmo: str) -> str:
    """
    Converte o nome de um algoritmo em um nome de arquivo sem espaços, hífens nem asterisco.

    Args:
        nomeAlgoritmo: Nome do algoritmo, como no dicionário ALGORITMOS.

    Returns:
        O nome do arquivo sem extensão, por exemplo "hill_climbing" e "a_estrela".
    """
    # Troca o asterisco por "_estrela" e os espaços e hífens por underline, tudo em minúsculas
    return nomeAlgoritmo.lower().replace("*", "_estrela").replace(" ", "_").replace("-", "_")

def salva_estado_final(
    grafo: Grafo,
    origem: int,
    destino: int,
    nomeAlgoritmo: str,
    resultado: ResultadoBusca,
    caminhoArquivo: Path,
) -> None:
    """
    Salva em PNG o estado final de um único algoritmo, com a legenda embaixo.

    Args:
        grafo: Rede onde a busca foi feita.
        origem: Vértice de partida.
        destino: Vértice de destino.
        nomeAlgoritmo: Nome mostrado no título da figura.
        resultado: Resultado da busca, que precisa ter sido executada com gravarEventos=True.
        caminhoArquivo: Caminho do arquivo de imagem a salvar.
    """
    # Cria uma figura quadrada com um único eixo, deixando espaço embaixo para a legenda
    figura, eixo = plt.subplots(figsize=(8, 8.8), facecolor=COR_FUNDO)
    figura.subplots_adjust(left=0.03, right=0.97, top=0.9, bottom=0.1)

    # Desenha a rede com o estado final da busca e o caminho encontrado (ou a trajetória até a falha)
    desenha_estado_final(eixo, grafo, origem, destino, nomeAlgoritmo, resultado)
    cria_legenda(figura)

    # Salva recortando as sobras de margem, sem cortar o título de duas linhas
    figura.savefig(caminhoArquivo, dpi=150, facecolor=COR_FUNDO, bbox_inches="tight", pad_inches=0.15)

    # Fecha a figura, já que o matplotlib mantém todas as figuras abertas na memória
    plt.close(figura)

def imprime_tabela(cenario: str, origem: int, destino: int, resultados: dict[str, ResultadoBusca]) -> None:
    """
    Mostra no terminal o caminho e a distância final de cada algoritmo, como pede o enunciado.

    Args:
        cenario: Nome do cenário.
        origem: Vértice de partida.
        destino: Vértice de destino.
        resultados: Dicionário que associa o nome de cada algoritmo ao seu resultado.
    """
    print(f"\n=== Cenário: {cenario} | origem = {origem} | destino = {destino} ===")

    for nomeAlgoritmo, resultado in resultados.items():
        # Em caso de falha, mostra só o aviso e os nós expandidos
        if not resultado.encontrouSolucao:
            print(f"{nomeAlgoritmo:14s} | não encontrou o destino | expandidos = {resultado.nosExpandidos}")
            continue

        print(
            f"{nomeAlgoritmo:14s} | distância = {resultado.distancia:9.2f} "
            f"| vértices = {len(resultado.caminho):4d} | expandidos = {resultado.nosExpandidos:4d}"
        )
        print(f"{'':14s} | caminho: {resultado.caminho}")

def roda_cenario(grafo: Grafo, cenario: str, origem: int, destino: int) -> None:
    """
    Roda os seis algoritmos em um par e salva animações, grade e CSV do cenário.

    Args:
        grafo: Rede onde as buscas são feitas.
        cenario: Nome do cenário, usado como nome da subpasta.
        origem: Vértice de partida.
        destino: Vértice de destino.
    """
    # Cria a subpasta do cenário
    pastaCenario = PASTA_EXP1 / cenario
    pastaCenario.mkdir(parents=True, exist_ok=True)

    # Roda todos os algoritmos gravando eventos, que são necessários para a visualização
    resultados = {
        nomeAlgoritmo: funcaoBusca(grafo, origem, destino, True)
        for nomeAlgoritmo, funcaoBusca in ALGORITMOS.items()
    }

    imprime_tabela(cenario, origem, destino, resultados)
    salva_csv(pastaCenario / "resultados.csv", cenario, origem, destino, resultados)
    print()

    # Salva a grade mostrando o indicador de progresso, já que ela leva alguns segundos
    with IndicadorProgresso(f"{cenario}: salvando a grade"):
        salva_grade(grafo, origem, destino, resultados, pastaCenario / "grade.png")

    # Gera a imagem final e a animação de cada algoritmo, numerando para o usuário saber quanto falta
    for indiceAlgoritmo, (nomeAlgoritmo, resultado) in enumerate(resultados.items(), start=1):
        # Monta um nome de arquivo sem espaços nem asterisco, por exemplo "hill_climbing" e "a_estrela"
        nomeArquivo = monta_nome_arquivo(nomeAlgoritmo)

        # Mostra o indicador enquanto a imagem e a animação são salvas, que é a parte demorada do script
        with IndicadorProgresso(f"{cenario}: {nomeAlgoritmo} [{indiceAlgoritmo}/{len(resultados)}]"):
            # Salva o estado final deste algoritmo sozinho, além da grade com todos
            salva_estado_final(grafo, origem, destino, nomeAlgoritmo, resultado, pastaCenario / f"{nomeArquivo}.png")

            figura, animacaoBusca = cria_animacao(grafo, origem, destino, nomeAlgoritmo, resultado)
            salva_animacao(figura, animacaoBusca, pastaCenario / nomeArquivo, QUADROS_POR_SEGUNDO)

def roda_experimento_1() -> None:
    """
    Executa o experimento i completo.
    """
    # Gera a rede pequena com a semente padrão
    grafo = gera_rede_knn(VERTICES_EXP1, K_EXP1, semente=SEMENTE_PADRAO)
    print(f"rede: {grafo.quantidade_vertices()} vértices, {grafo.quantidade_arestas()} arestas, grau médio {grafo.grau_medio():.2f}")

    # Escolhe um par para cada cenário e roda os seis algoritmos em cada um
    for cenario, (origem, destino) in escolhe_pares(grafo).items():
        roda_cenario(grafo, cenario, origem, destino)

    print(f"\nresultados salvos em {PASTA_EXP1}")

#
# Experimento ii: estatísticas em redes grandes
#

# Número de vértices das redes, fixado pelo enunciado
VERTICES_EXP2 = 2500

# Valores de k pedidos pelo enunciado
VALORES_K = (3, 7, 11)

# Número de pares (origem, destino); o enunciado pede no mínimo 10
QUANTIDADE_PARES = 30

# Quantas vezes cada busca é repetida para medir o tempo; o tempo do par é a média dessas repetições
# As buscas levam menos de alguns milissegundos, então 20 repetições custam pouco e reduzem bastante o ruído
REPETICOES_TEMPO = 20

# Pasta onde os resultados do experimento são salvos
PASTA_EXP2 = PASTA_RESULTADOS / "exp2"

def mede_busca(funcaoBusca: FuncaoBusca, grafo: Grafo, origem: int, destino: int) -> tuple[ResultadoBusca, float]:
    """
    Executa uma busca várias vezes e mede o tempo médio de execução.

    Args:
        funcaoBusca: Algoritmo de busca a medir.
        grafo: Rede onde a busca é feita.
        origem: Vértice de partida.
        destino: Vértice de destino.

    Returns:
        O resultado da última execução e o tempo médio, em milissegundos.
    """
    #
    # Desliga o coletor de lixo durante a medição, para que uma coleta disparada
    # no meio de uma busca não seja contada como tempo do algoritmo. É o mesmo
    # cuidado que o módulo timeit da biblioteca padrão toma
    #
    gc.disable()

    try:
        # Lista com o tempo de cada repetição, em segundos
        temposRepeticoes = []

        for _ in range(REPETICOES_TEMPO):
            # Marca o instante antes da busca com o relógio de maior precisão disponível
            instanteInicio = time.perf_counter()

            # Executa a busca sem gravar eventos, para não somar o custo da visualização ao tempo
            resultado = funcaoBusca(grafo, origem, destino, False)

            # Guarda a diferença entre o instante final e o inicial
            temposRepeticoes.append(time.perf_counter() - instanteInicio)
    finally:
        # Religa o coletor de lixo mesmo que a busca levante uma exceção
        gc.enable()

    # Calcula a média das repetições e converte de segundos para milissegundos
    tempoMedioMs = 1000 * sum(temposRepeticoes) / len(temposRepeticoes)

    return resultado, tempoMedioMs

def coleta_execucoes(pares: list[tuple[int, int]]) -> pd.DataFrame:
    """
    Roda todos os algoritmos em todos os pares, para cada valor de k.

    Args:
        pares: Pares (origem, destino) usados em todas as redes.

    Returns:
        Tabela com uma linha por execução (k, par, algoritmo).
    """
    # Lista de linhas da tabela, montada como dicionários e convertida em DataFrame no fim
    linhas = []

    for valorK in VALORES_K:
        # Gera a rede com a semente padrão, o que mantém os vértices nas mesmas posições para todo k
        grafo = gera_rede_knn(VERTICES_EXP2, valorK, semente=SEMENTE_PADRAO)
        print(f"k = {valorK:2d} | arestas = {grafo.quantidade_arestas()} | grau médio = {grafo.grau_medio():.2f}")

        for indicePar, (origem, destino) in enumerate(pares):
            for nomeAlgoritmo, funcaoBusca in ALGORITMOS.items():
                resultado, tempoMs = mede_busca(funcaoBusca, grafo, origem, destino)

                linhas.append({
                    "k": valorK,
                    "par": indicePar,
                    "origem": origem,
                    "destino": destino,
                    "algoritmo": nomeAlgoritmo,
                    "encontrou": resultado.encontrouSolucao,
                    # Deixa a distância vazia (NaN) em caso de falha, para que ela seja ignorada nas médias
                    "distancia": resultado.distancia if resultado.encontrouSolucao else float("nan"),
                    # Também deixa vazio o número de vértices do caminho em caso de falha, pelo mesmo motivo
                    "vertices_no_caminho": len(resultado.caminho) if resultado.encontrouSolucao else float("nan"),
                    "nos_expandidos": resultado.nosExpandidos,
                    "tempo_ms": tempoMs,
                })

            # Mostra o progresso a cada 10 pares, já que a execução inteira leva alguns minutos
            if (indicePar + 1) % 10 == 0:
                print(f"  {indicePar + 1} de {len(pares)} pares concluídos")

    return pd.DataFrame(linhas)

def resume_execucoes(execucoes: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula as estatísticas por valor de k e algoritmo.

    A distância média, o seu desvio padrão e o número médio de vértices no
    caminho consideram apenas as execuções que chegaram ao destino, porque
    uma falha não tem caminho. Por isso, para o Hill Climbing, essas médias
    são calculadas sobre um subconjunto dos pares e devem ser lidas junto com
    a coluna quantidade_sucessos.
    Os desvios padrão são amostrais (divididos por n - 1), o padrão do pandas.

    Args:
        execucoes: Tabela com uma linha por execução, gerada por coleta_execucoes.

    Returns:
        Tabela com uma linha por (k, algoritmo).
    """
    # Agrupa as execuções por k e algoritmo, mantendo a ordem dos algoritmos do enunciado (sort=False)
    grupos = execucoes.groupby(["k", "algoritmo"], sort=False)

    # Calcula cada estatística em uma coluna; mean e std já ignoram os NaN das falhas
    resumo = grupos.agg(
        distancia_media=("distancia", "mean"),
        distancia_desvio=("distancia", "std"),
        tempo_medio_ms=("tempo_ms", "mean"),
        tempo_desvio_ms=("tempo_ms", "std"),
        nos_expandidos_medio=("nos_expandidos", "mean"),
        vertices_no_caminho_medio=("vertices_no_caminho", "mean"),
        # A soma de uma coluna de verdadeiro/falso conta os verdadeiros, ou seja, os pares em que a busca chegou
        quantidade_sucessos=("encontrou", "sum"),
        # A média de uma coluna de verdadeiro/falso é a fração de verdadeiros
        taxa_sucesso=("encontrou", "mean"),
    )

    # Transforma k e algoritmo de volta em colunas comuns, em vez de índice
    return resumo.reset_index()

def roda_experimento_2() -> pd.DataFrame:
    """
    Executa o experimento ii completo.

    Returns:
        A tabela com todas as execuções, usada em seguida pelo experimento iii.
    """
    PASTA_EXP2.mkdir(parents=True, exist_ok=True)

    # Gera a rede com o menor k só para sortear os pares, que ficam válidos para os outros k
    grafoMenorK = gera_rede_knn(VERTICES_EXP2, min(VALORES_K), semente=SEMENTE_PADRAO)
    pares = sorteia_pares(grafoMenorK, QUANTIDADE_PARES, semente=SEMENTE_PADRAO)

    execucoes = coleta_execucoes(pares)
    resumo = resume_execucoes(execucoes)

    # Salva a tabela completa com uma linha por execução
    execucoes.to_csv(PASTA_EXP2 / "execucoes.csv", index=False)

    # Salva o resumo com quatro casas decimais
    resumo.to_csv(PASTA_EXP2 / "resumo.csv", index=False, float_format="%.4f")

    # Mostra o resumo no terminal, com todas as colunas e duas casas decimais
    with pd.option_context("display.width", 200, "display.max_columns", None, "display.float_format", "{:.2f}".format):
        print()
        print(resumo.to_string(index=False))

    print(f"\nresultados salvos em {PASTA_EXP2}")

    return execucoes

#
# Experimento iii: A* vs Dijkstra
#

# Pasta onde os resultados do experimento são salvos
PASTA_EXP3 = PASTA_RESULTADOS / "exp3"

# Nomes dos dois algoritmos lidos do experimento ii, iguais às chaves do dicionário ALGORITMOS
NOME_A_ESTRELA = "A*"
NOME_DIJKSTRA = "Dijkstra"

# Nome do A* com a heurística de Chebyshev, que é rodado neste experimento
NOME_A_ESTRELA_CHEBYSHEV = "A* Chebyshev"

# Tolerância para considerar duas distâncias iguais, ignorando erros de arredondamento de ponto flutuante
TOLERANCIA_DISTANCIA = 1e-6

# Cores do gráfico: as três primeiras posições da paleta categórica validada
COR_A_ESTRELA = "#2a78d6"
COR_DIJKSTRA = "#eb6834"
COR_A_ESTRELA_CHEBYSHEV = "#1baf7a"

# Cores secundárias de texto e da grade do gráfico (fundo e texto principal vêm da visualização)
COR_TEXTO_SECUNDARIO = "#52514e"
COR_GRADE = "#e1e0d9"

def carrega_comparacao(execucoes: pd.DataFrame) -> pd.DataFrame:
    """
    Coloca A* e Dijkstra lado a lado a partir das execuções do experimento ii.

    Args:
        execucoes: Tabela com uma linha por execução, devolvida pelo experimento ii.

    Returns:
        Tabela com uma linha por (k, par), com colunas de distância, nós
        expandidos e tempo de cada um dos dois algoritmos.
    """
    # Mantém só as linhas dos dois algoritmos comparados
    execucoes = execucoes[execucoes["algoritmo"].isin([NOME_A_ESTRELA, NOME_DIJKSTRA])]

    #
    # Transforma a tabela de formato longo (uma linha por algoritmo) em formato
    # largo (uma linha por par, com uma coluna para cada algoritmo e métrica),
    # o que permite comparar os dois algoritmos no mesmo par com uma subtração
    #
    comparacao = execucoes.pivot_table(
        index=["k", "par", "origem", "destino"],
        columns="algoritmo",
        values=["distancia", "nos_expandidos", "tempo_ms"],
    )

    # Junta os dois níveis de nome de coluna em um só, por exemplo ("distancia", "A*") vira "distancia_A*"
    comparacao.columns = [f"{metrica}_{algoritmo}" for metrica, algoritmo in comparacao.columns]

    return comparacao.reset_index()

def adiciona_a_estrela_chebyshev(comparacao: pd.DataFrame) -> pd.DataFrame:
    """
    Roda o A* com a heurística de Chebyshev nos mesmos pares e redes do experimento ii.

    O tempo é medido da mesma forma que no experimento ii, reaproveitando a
    função mede_busca de lá. Para que os tempos sejam comparáveis com os do
    A* euclidiano e do Dijkstra, este experimento deve rodar na mesma máquina
    e logo depois do experimento ii.

    Args:
        comparacao: Tabela gerada por carrega_comparacao.

    Returns:
        A mesma tabela com as colunas de distância, nós expandidos e tempo do A* com Chebyshev.
    """
    #
    # Fixa a heurística de Chebyshev no A*, criando uma função que recebe só
    # (grafo, origem, destino, gravarEventos), a mesma assinatura que o
    # mede_busca espera dos algoritmos do dicionário ALGORITMOS
    #
    buscaChebyshev = partial(busca_a_estrela, funcaoHeuristica=heuristica_chebyshev)

    # Listas com os resultados de cada linha da tabela, na mesma ordem das linhas
    distancias = []
    nosExpandidos = []
    tempos = []

    # Guarda a rede de cada k já gerada, para não gerar a mesma rede uma vez por par
    redesPorK = {}

    # Percorre os pares na ordem das linhas da tabela
    for valorK, origem, destino in zip(comparacao["k"], comparacao["origem"], comparacao["destino"]):
        # Gera a rede deste k só na primeira vez em que ele aparece, com a mesma semente do experimento ii
        if valorK not in redesPorK:
            redesPorK[valorK] = gera_rede_knn(VERTICES_EXP2, int(valorK), semente=SEMENTE_PADRAO)

        resultado, tempoMs = mede_busca(buscaChebyshev, redesPorK[valorK], int(origem), int(destino))

        distancias.append(resultado.distancia)
        nosExpandidos.append(resultado.nosExpandidos)
        tempos.append(tempoMs)

    # Adiciona as três colunas novas, seguindo o mesmo padrão de nome das outras
    comparacao[f"distancia_{NOME_A_ESTRELA_CHEBYSHEV}"] = distancias
    comparacao[f"nos_expandidos_{NOME_A_ESTRELA_CHEBYSHEV}"] = nosExpandidos
    comparacao[f"tempo_ms_{NOME_A_ESTRELA_CHEBYSHEV}"] = tempos

    return comparacao

def adiciona_distancia_euclidiana(comparacao: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona a distância em linha reta entre origem e destino de cada par.

    Args:
        comparacao: Tabela gerada por carrega_comparacao.

    Returns:
        A mesma tabela com a coluna distancia_euclidiana.
    """
    #
    # Gera a rede de novo só para obter as coordenadas. Como a semente é a mesma
    # do experimento ii, os vértices ficam nas mesmas posições, e qualquer k serve
    #
    coordenadas = gera_rede_knn(VERTICES_EXP2, min(VALORES_K), semente=SEMENTE_PADRAO).coordenadas

    # Calcula a distância em linha reta de cada par usando as coordenadas da origem e do destino
    comparacao["distancia_euclidiana"] = [
        math.dist(coordenadas[origem], coordenadas[destino])
        for origem, destino in zip(comparacao["origem"], comparacao["destino"])
    ]

    return comparacao

def resume_comparacao(comparacao: pd.DataFrame) -> pd.DataFrame:
    """
    Resume a comparação entre os dois A* e o Dijkstra para cada valor de k.

    Args:
        comparacao: Tabela com as colunas dos três algoritmos.

    Returns:
        Tabela com uma linha por k.
    """
    # Calcula, par a par, a diferença absoluta de distância de cada A* em relação ao Dijkstra
    comparacao["diferenca_distancia"] = (comparacao[f"distancia_{NOME_A_ESTRELA}"] - comparacao[f"distancia_{NOME_DIJKSTRA}"]).abs()
    comparacao["diferenca_distancia_chebyshev"] = (comparacao[f"distancia_{NOME_A_ESTRELA_CHEBYSHEV}"] - comparacao[f"distancia_{NOME_DIJKSTRA}"]).abs()

    # Calcula, par a par, quantas vezes o Dijkstra expande mais vértices que cada A*
    comparacao["razao_expandidos"] = comparacao[f"nos_expandidos_{NOME_DIJKSTRA}"] / comparacao[f"nos_expandidos_{NOME_A_ESTRELA}"]
    comparacao["razao_expandidos_chebyshev"] = comparacao[f"nos_expandidos_{NOME_DIJKSTRA}"] / comparacao[f"nos_expandidos_{NOME_A_ESTRELA_CHEBYSHEV}"]

    # Calcula, par a par, quantas vezes o Dijkstra é mais lento que cada A*
    comparacao["razao_tempo"] = comparacao[f"tempo_ms_{NOME_DIJKSTRA}"] / comparacao[f"tempo_ms_{NOME_A_ESTRELA}"]
    comparacao["razao_tempo_chebyshev"] = comparacao[f"tempo_ms_{NOME_DIJKSTRA}"] / comparacao[f"tempo_ms_{NOME_A_ESTRELA_CHEBYSHEV}"]

    # Função que conta quantas diferenças de distância ficaram dentro da tolerância
    def conta_iguais(diferencas: pd.Series) -> int:
        """
        Conta os pares em que a diferença de distância está dentro da tolerância.
        """
        return int((diferencas <= TOLERANCIA_DISTANCIA).sum())

    # Agrupa por k e calcula as estatísticas de cada grupo
    resumo = comparacao.groupby("k").agg(
        pares=("par", "count"),
        distancias_iguais=("diferenca_distancia", conta_iguais),
        distancias_iguais_chebyshev=("diferenca_distancia_chebyshev", conta_iguais),
        distancia_media=(f"distancia_{NOME_A_ESTRELA}", "mean"),
        expandidos_medio_a_estrela=(f"nos_expandidos_{NOME_A_ESTRELA}", "mean"),
        expandidos_medio_a_estrela_chebyshev=(f"nos_expandidos_{NOME_A_ESTRELA_CHEBYSHEV}", "mean"),
        expandidos_medio_dijkstra=(f"nos_expandidos_{NOME_DIJKSTRA}", "mean"),
        razao_expandidos_media=("razao_expandidos", "mean"),
        razao_expandidos_media_chebyshev=("razao_expandidos_chebyshev", "mean"),
        tempo_medio_ms_a_estrela=(f"tempo_ms_{NOME_A_ESTRELA}", "mean"),
        tempo_medio_ms_a_estrela_chebyshev=(f"tempo_ms_{NOME_A_ESTRELA_CHEBYSHEV}", "mean"),
        tempo_medio_ms_dijkstra=(f"tempo_ms_{NOME_DIJKSTRA}", "mean"),
        razao_tempo_media=("razao_tempo", "mean"),
        razao_tempo_media_chebyshev=("razao_tempo_chebyshev", "mean"),
    )

    # Calcula a redução percentual de nós expandidos de cada A* em relação ao Dijkstra, a partir das médias
    resumo["reducao_expandidos_pct"] = 100 * (1 - resumo["expandidos_medio_a_estrela"] / resumo["expandidos_medio_dijkstra"])
    resumo["reducao_expandidos_chebyshev_pct"] = 100 * (1 - resumo["expandidos_medio_a_estrela_chebyshev"] / resumo["expandidos_medio_dijkstra"])

    return resumo.reset_index()

def salva_grafico(comparacao: pd.DataFrame, caminhoArquivo: Path) -> None:
    """
    Salva o gráfico de nós expandidos em função da distância entre origem e destino.

    Gera um painel por valor de k, lado a lado e com o mesmo eixo y, para que
    os três possam ser comparados diretamente.

    Args:
        comparacao: Tabela com a coluna distancia_euclidiana e as colunas dos três algoritmos.
        caminhoArquivo: Caminho do arquivo de imagem a salvar.
    """
    # Lista das séries do gráfico, do algoritmo que mais expande para o que menos expande, como (nome, cor)
    series = [
        (NOME_DIJKSTRA, COR_DIJKSTRA),
        (NOME_A_ESTRELA_CHEBYSHEV, COR_A_ESTRELA_CHEBYSHEV),
        (NOME_A_ESTRELA, COR_A_ESTRELA),
    ]

    # Cria um painel por k; sharey=True faz os três usarem a mesma escala vertical
    figura, listaEixos = plt.subplots(1, len(VALORES_K), figsize=(15, 5), sharey=True, facecolor=COR_FUNDO)

    for eixo, valorK in zip(listaEixos, VALORES_K):
        # Seleciona os pares deste valor de k
        dadosK = comparacao[comparacao["k"] == valorK]

        # Desenha um ponto por par para cada algoritmo, com contorno da cor do fundo para separar pontos sobrepostos
        for nomeAlgoritmo, cor in series:
            eixo.scatter(dadosK["distancia_euclidiana"], dadosK[f"nos_expandidos_{nomeAlgoritmo}"], s=40, color=cor, edgecolors=COR_FUNDO, linewidths=1, label=nomeAlgoritmo, zorder=3)

        eixo.set_title(f"k = {valorK}", fontsize=12, color=COR_TEXTO)
        eixo.set_xlabel("Distância em linha reta entre origem e destino", fontsize=10, color=COR_TEXTO_SECUNDARIO)

        # Deixa a grade horizontal discreta e atrás dos pontos
        eixo.grid(axis="y", color=COR_GRADE, linewidth=0.8, zorder=0)
        eixo.set_facecolor(COR_FUNDO)

        # Remove as bordas de cima e da direita, que não acrescentam informação
        eixo.spines["top"].set_visible(False)
        eixo.spines["right"].set_visible(False)

        # Pinta as bordas e marcações restantes com o tom secundário, para não competir com os dados
        for borda in ("left", "bottom"):
            eixo.spines[borda].set_color(COR_TEXTO_SECUNDARIO)
        eixo.tick_params(colors=COR_TEXTO_SECUNDARIO)

    # O eixo y é compartilhado, então só o primeiro painel precisa do rótulo
    listaEixos[0].set_ylabel("Nós expandidos", fontsize=10, color=COR_TEXTO_SECUNDARIO)

    # Coloca uma legenda só para os três painéis, no canto superior esquerdo do primeiro
    listaEixos[0].legend(frameon=False, fontsize=10, loc="upper left", labelcolor=COR_TEXTO)

    figura.suptitle("Nós expandidos por A* (euclidiana e Chebyshev) e Dijkstra em redes KNN com 2500 vértices", fontsize=13, color=COR_TEXTO)
    figura.tight_layout()
    figura.savefig(caminhoArquivo, dpi=150, facecolor=COR_FUNDO)
    plt.close(figura)

def roda_experimento_3(execucoes: pd.DataFrame) -> None:
    """
    Executa o experimento iii completo.

    Args:
        execucoes: Tabela com uma linha por execução, devolvida pelo experimento ii.
    """
    PASTA_EXP3.mkdir(parents=True, exist_ok=True)

    comparacao = carrega_comparacao(execucoes)
    comparacao = adiciona_a_estrela_chebyshev(comparacao)
    comparacao = adiciona_distancia_euclidiana(comparacao)
    resumo = resume_comparacao(comparacao)

    # Salva a comparação par a par e o resumo por k
    comparacao.to_csv(PASTA_EXP3 / "comparacao_por_par.csv", index=False, float_format="%.4f")
    resumo.to_csv(PASTA_EXP3 / "resumo.csv", index=False, float_format="%.4f")

    salva_grafico(comparacao, PASTA_EXP3 / "expandidos_por_distancia.png")

    # Mostra o resumo no terminal transposto (uma linha por métrica), já que agora são muitas colunas
    with pd.option_context("display.width", 250, "display.max_columns", None, "display.float_format", "{:.2f}".format):
        print(resumo.set_index("k").T.to_string())

    print(f"\nresultados salvos em {PASTA_EXP3}")

def main() -> None:
    """
    Executa os três experimentos em sequência.
    """
    print("=== Experimento i: visualização ===")
    roda_experimento_1()

    print("\n=== Experimento ii: estatísticas ===")
    execucoes = roda_experimento_2()

    # O experimento iii usa as execuções do ii, então roda logo em seguida, na mesma máquina
    print("\n=== Experimento iii: A* vs Dijkstra ===")
    roda_experimento_3(execucoes)

if __name__ == "__main__":
    main()
