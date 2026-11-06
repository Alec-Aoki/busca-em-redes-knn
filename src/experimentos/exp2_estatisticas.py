"""
Experimento ii: estatísticas dos seis algoritmos de busca em redes KNN grandes.

Gera redes de 2500 vértices com k = 3, 7 e 11 e roda os seis algoritmos em
30 pares (origem, destino). Para cada algoritmo e cada k, calcula a média e
o desvio padrão da distância percorrida e do tempo de execução, a média de
nós expandidos e a taxa de sucesso.

Os três valores de k usam a mesma semente, então os vértices ficam nas
mesmas posições e só as arestas mudam. Como os k vizinhos mais próximos de
um vértice estão sempre entre os k' mais próximos para k' > k, toda aresta
da rede com k = 3 também existe nas redes com k = 7 e k = 11. Por isso os
pares são sorteados uma vez só, na maior componente da rede com k = 3, e
continuam válidos nas outras duas, o que deixa a comparação entre valores de
k feita sobre exatamente os mesmos pares.
"""

import gc
import sys
import time
from pathlib import Path

import pandas as pd

# Caminho da raiz do projeto, que fica uma pasta acima desta
RAIZ_PROJETO = Path(__file__).resolve().parent.parent

# Coloca a pasta src no caminho de busca de módulos, para que os imports abaixo funcionem rodando de qualquer pasta
sys.path.insert(0, str(RAIZ_PROJETO / "src"))

from algoritmos import ALGORITMOS, FuncaoBusca
from rede import Grafo, gera_rede_knn
from resultado import ResultadoBusca
from utils import SEMENTE_PADRAO, sorteia_pares

# Número de vértices das redes, fixado pelo enunciado
QUANTIDADE_VERTICES = 2500

# Valores de k pedidos pelo enunciado
VALORES_K = (3, 7, 11)

# Número de pares (origem, destino); o enunciado pede no mínimo 10
QUANTIDADE_PARES = 30

# Quantas vezes cada busca é repetida para medir o tempo; o tempo do par é a média dessas repetições
# As buscas levam menos de alguns milissegundos, então 20 repetições custam pouco e reduzem bastante o ruído
REPETICOES_TEMPO = 20

# Pasta onde os resultados do experimento são salvos
PASTA_RESULTADOS = RAIZ_PROJETO / "resultados" / "exp2"

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
        grafo = gera_rede_knn(QUANTIDADE_VERTICES, valorK, semente=SEMENTE_PADRAO)
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

def main() -> None:
    """
    Executa o experimento ii completo.
    """
    PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)

    # Gera a rede com o menor k só para sortear os pares, que ficam válidos para os outros k
    grafoMenorK = gera_rede_knn(QUANTIDADE_VERTICES, min(VALORES_K), semente=SEMENTE_PADRAO)
    pares = sorteia_pares(grafoMenorK, QUANTIDADE_PARES, semente=SEMENTE_PADRAO)

    execucoes = coleta_execucoes(pares)
    resumo = resume_execucoes(execucoes)

    # Salva a tabela completa, que também é usada pelo experimento iii
    execucoes.to_csv(PASTA_RESULTADOS / "execucoes.csv", index=False)

    # Salva o resumo com quatro casas decimais
    resumo.to_csv(PASTA_RESULTADOS / "resumo.csv", index=False, float_format="%.4f")

    # Mostra o resumo no terminal, com todas as colunas e duas casas decimais
    with pd.option_context("display.width", 200, "display.max_columns", None, "display.float_format", "{:.2f}".format):
        print()
        print(resumo.to_string(index=False))

    print(f"\nresultados salvos em {PASTA_RESULTADOS}")

if __name__ == "__main__":
    main()