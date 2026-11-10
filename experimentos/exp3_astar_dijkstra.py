"""
Experimento iii: comparação entre A* e Dijkstra.

Usa as execuções salvas pelo experimento ii, que precisa ter sido rodado
antes. Além do A* com a heurística euclidiana e do Dijkstra, roda nos mesmos
pares um A* com a heurística de Chebyshev, que também é admissível mas
estima pior o custo restante. Para cada valor de k, confere que os três
encontram a mesma distância, compara nós expandidos e tempo, e gera um
gráfico de nós expandidos em função da distância em linha reta entre origem
e destino.
"""

import math
import sys
from functools import partial
from pathlib import Path

import matplotlib

# Usa o backend Agg, que desenha direto em arquivo e funciona sem janela gráfica
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

# Coloca a pasta experimentos no caminho de busca, para reaproveitar as constantes do experimento ii
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp2_estatisticas import PASTA_RESULTADOS as PASTA_EXP2
from exp2_estatisticas import QUANTIDADE_VERTICES, RAIZ_PROJETO, VALORES_K, mede_busca
from algoritmos.a_estrela import busca_a_estrela
from heuristicas import heuristica_chebyshev
from rede import gera_rede_knn
from utils import SEMENTE_PADRAO

# Pasta onde os resultados do experimento são salvos
PASTA_RESULTADOS = RAIZ_PROJETO / "resultados" / "exp3"

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

# Cores de fundo, texto e grade do gráfico
COR_FUNDO = "#fcfcfb"
COR_TEXTO = "#0b0b0b"
COR_TEXTO_SECUNDARIO = "#52514e"
COR_GRADE = "#e1e0d9"

def carrega_comparacao() -> pd.DataFrame:
    """
    Carrega as execuções do experimento ii e coloca A* e Dijkstra lado a lado.

    Returns:
        Tabela com uma linha por (k, par), com colunas de distância, nós
        expandidos e tempo de cada um dos dois algoritmos.

    Raises:
        FileNotFoundError: Se o experimento ii ainda não tiver sido rodado.
    """
    caminhoExecucoes = PASTA_EXP2 / "execucoes.csv"

    # Avisa com uma mensagem clara se o arquivo do experimento ii não existir
    if not caminhoExecucoes.exists():
        raise FileNotFoundError(f"{caminhoExecucoes} não encontrado; rode experimentos/exp2_estatisticas.py antes")

    execucoes = pd.read_csv(caminhoExecucoes)

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
            redesPorK[valorK] = gera_rede_knn(QUANTIDADE_VERTICES, int(valorK), semente=SEMENTE_PADRAO)

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
    coordenadas = gera_rede_knn(QUANTIDADE_VERTICES, min(VALORES_K), semente=SEMENTE_PADRAO).coordenadas

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

def main() -> None:
    """
    Executa o experimento iii completo.
    """
    PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)

    comparacao = carrega_comparacao()
    comparacao = adiciona_a_estrela_chebyshev(comparacao)
    comparacao = adiciona_distancia_euclidiana(comparacao)
    resumo = resume_comparacao(comparacao)

    # Salva a comparação par a par e o resumo por k
    comparacao.to_csv(PASTA_RESULTADOS / "comparacao_por_par.csv", index=False, float_format="%.4f")
    resumo.to_csv(PASTA_RESULTADOS / "resumo.csv", index=False, float_format="%.4f")

    salva_grafico(comparacao, PASTA_RESULTADOS / "expandidos_por_distancia.png")

    # Mostra o resumo no terminal transposto (uma linha por métrica), já que agora são muitas colunas
    with pd.option_context("display.width", 250, "display.max_columns", None, "display.float_format", "{:.2f}".format):
        print(resumo.set_index("k").T.to_string())

    print(f"\nresultados salvos em {PASTA_RESULTADOS}")

if __name__ == "__main__":
    main()