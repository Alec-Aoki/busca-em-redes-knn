"""
Experimento iii: comparação entre A* e Dijkstra.

Usa as execuções salvas pelo experimento ii, que precisa ter sido rodado
antes. Para cada valor de k, confere que os dois algoritmos encontram a mesma
distância, compara nós expandidos e tempo, e gera um gráfico de nós
expandidos em função da distância em linha reta entre origem e destino.
"""

import math
import sys
from pathlib import Path

import matplotlib

# Usa o backend Agg, que desenha direto em arquivo e funciona sem janela gráfica
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

# Coloca a pasta experimentos no caminho de busca, para reaproveitar as constantes do experimento ii
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp2_estatisticas import PASTA_RESULTADOS as PASTA_EXP2
from exp2_estatisticas import QUANTIDADE_VERTICES, RAIZ_PROJETO, VALORES_K
from rede import gera_rede_knn
from utils import SEMENTE_PADRAO

# Pasta onde os resultados do experimento são salvos
PASTA_RESULTADOS = RAIZ_PROJETO / "resultados" / "exp3"

# Nomes dos dois algoritmos comparados, iguais às chaves do dicionário ALGORITMOS
NOME_A_ESTRELA = "A*"
NOME_DIJKSTRA = "Dijkstra"

# Tolerância para considerar duas distâncias iguais, ignorando erros de arredondamento de ponto flutuante
TOLERANCIA_DISTANCIA = 1e-6

# Cores do gráfico: as duas primeiras posições da paleta categórica validada
COR_A_ESTRELA = "#2a78d6"
COR_DIJKSTRA = "#eb6834"

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
    Resume a comparação entre A* e Dijkstra para cada valor de k.

    Args:
        comparacao: Tabela gerada por carrega_comparacao.

    Returns:
        Tabela com uma linha por k.
    """
    # Calcula, par a par, a diferença absoluta de distância entre os dois algoritmos
    comparacao["diferenca_distancia"] = (comparacao[f"distancia_{NOME_A_ESTRELA}"] - comparacao[f"distancia_{NOME_DIJKSTRA}"]).abs()

    # Calcula, par a par, quantas vezes o Dijkstra expande mais vértices que o A*
    comparacao["razao_expandidos"] = comparacao[f"nos_expandidos_{NOME_DIJKSTRA}"] / comparacao[f"nos_expandidos_{NOME_A_ESTRELA}"]

    # Calcula, par a par, quantas vezes o Dijkstra é mais lento que o A*
    comparacao["razao_tempo"] = comparacao[f"tempo_ms_{NOME_DIJKSTRA}"] / comparacao[f"tempo_ms_{NOME_A_ESTRELA}"]

    # Agrupa por k e calcula as estatísticas de cada grupo
    resumo = comparacao.groupby("k").agg(
        pares=("par", "count"),
        # Conta os pares em que as distâncias ficaram dentro da tolerância
        distancias_iguais=("diferenca_distancia", lambda diferencas: int((diferencas <= TOLERANCIA_DISTANCIA).sum())),
        distancia_media=(f"distancia_{NOME_A_ESTRELA}", "mean"),
        expandidos_medio_a_estrela=(f"nos_expandidos_{NOME_A_ESTRELA}", "mean"),
        expandidos_medio_dijkstra=(f"nos_expandidos_{NOME_DIJKSTRA}", "mean"),
        razao_expandidos_media=("razao_expandidos", "mean"),
        tempo_medio_ms_a_estrela=(f"tempo_ms_{NOME_A_ESTRELA}", "mean"),
        tempo_medio_ms_dijkstra=(f"tempo_ms_{NOME_DIJKSTRA}", "mean"),
        razao_tempo_media=("razao_tempo", "mean"),
    )

    # Calcula a redução percentual de nós expandidos do A* em relação ao Dijkstra, a partir das médias
    resumo["reducao_expandidos_pct"] = 100 * (1 - resumo["expandidos_medio_a_estrela"] / resumo["expandidos_medio_dijkstra"])

    return resumo.reset_index()

def salva_grafico(comparacao: pd.DataFrame, caminhoArquivo: Path) -> None:
    """
    Salva o gráfico de nós expandidos em função da distância entre origem e destino.

    Gera um painel por valor de k, lado a lado e com o mesmo eixo y, para que
    os três possam ser comparados diretamente.

    Args:
        comparacao: Tabela com a coluna distancia_euclidiana.
        caminhoArquivo: Caminho do arquivo de imagem a salvar.
    """
    # Cria um painel por k; sharey=True faz os três usarem a mesma escala vertical
    figura, listaEixos = plt.subplots(1, len(VALORES_K), figsize=(15, 5), sharey=True, facecolor=COR_FUNDO)

    for eixo, valorK in zip(listaEixos, VALORES_K):
        # Seleciona os pares deste valor de k
        dadosK = comparacao[comparacao["k"] == valorK]

        # Desenha um ponto por par para cada algoritmo, com contorno da cor do fundo para separar pontos sobrepostos
        eixo.scatter(dadosK["distancia_euclidiana"], dadosK[f"nos_expandidos_{NOME_DIJKSTRA}"], s=40, color=COR_DIJKSTRA, edgecolors=COR_FUNDO, linewidths=1, label=NOME_DIJKSTRA, zorder=3)
        eixo.scatter(dadosK["distancia_euclidiana"], dadosK[f"nos_expandidos_{NOME_A_ESTRELA}"], s=40, color=COR_A_ESTRELA, edgecolors=COR_FUNDO, linewidths=1, label=NOME_A_ESTRELA, zorder=3)

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

    figura.suptitle("Nós expandidos por A* e Dijkstra em redes KNN com 2500 vértices", fontsize=13, color=COR_TEXTO)
    figura.tight_layout()
    figura.savefig(caminhoArquivo, dpi=150, facecolor=COR_FUNDO)
    plt.close(figura)

def main() -> None:
    """
    Executa o experimento iii completo.
    """
    PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)

    comparacao = adiciona_distancia_euclidiana(carrega_comparacao())
    resumo = resume_comparacao(comparacao)

    # Salva a comparação par a par e o resumo por k
    comparacao.to_csv(PASTA_RESULTADOS / "comparacao_por_par.csv", index=False, float_format="%.4f")
    resumo.to_csv(PASTA_RESULTADOS / "resumo.csv", index=False, float_format="%.4f")

    salva_grafico(comparacao, PASTA_RESULTADOS / "expandidos_por_distancia.png")

    # Mostra o resumo no terminal, com todas as colunas e duas casas decimais
    with pd.option_context("display.width", 250, "display.max_columns", None, "display.float_format", "{:.2f}".format):
        print(resumo.to_string(index=False))

    print(f"\nresultados salvos em {PASTA_RESULTADOS}")

if __name__ == "__main__":
    main()