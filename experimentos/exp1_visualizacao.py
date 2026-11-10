"""
Experimento i: visualização dos seis algoritmos de busca em uma rede KNN pequena.

Gera uma rede de 300 vértices e escolhe dois pares (origem, destino): um em
que todos os algoritmos chegam ao destino e outro em que o Hill Climbing fica
preso em um mínimo local. Para cada par, roda os seis algoritmos e salva uma
animação por algoritmo, a figura em grade com os estados finais e um CSV com
caminho, distância e nós expandidos de cada um.
"""

import csv
import sys
from pathlib import Path

import matplotlib

# Usa o backend Agg, que desenha direto em arquivo e funciona sem janela gráfica
matplotlib.use("Agg")

import matplotlib.pyplot as plt

# Caminho da raiz do projeto, que fica uma pasta acima desta
RAIZ_PROJETO = Path(__file__).resolve().parent.parent

# Coloca a pasta src no caminho de busca de módulos, para que os imports abaixo funcionem rodando de qualquer pasta
sys.path.insert(0, str(RAIZ_PROJETO / "src"))

from algoritmos import ALGORITMOS
from rede import Grafo, gera_rede_knn
from resultado import ResultadoBusca
from utils import SEMENTE_PADRAO, sorteia_pares
from visualizacao import COR_FUNDO, cria_animacao, cria_legenda, desenha_estado_final, salva_animacao, salva_grade

# Número de vértices da rede, dentro da faixa de 100 a 500 pedida pelo enunciado
QUANTIDADE_VERTICES = 300

# Número de vizinhos mais próximos de cada vértice
VALOR_K = 5

# Distância mínima em linha reta entre origem e destino, para que a busca atravesse boa parte da rede
DISTANCIA_MINIMA_PAR = 200

# Quantos pares candidatos são sorteados para procurar os dois cenários
QUANTIDADE_CANDIDATOS = 200

# Velocidade das animações
QUADROS_POR_SEGUNDO = 15

# Pasta onde os resultados do experimento são salvos
PASTA_RESULTADOS = RAIZ_PROJETO / "resultados" / "exp1"

# Nome de cada cenário, usado como nome da subpasta de saída
CENARIO_SUCESSO = "sucesso"
CENARIO_FALHA = "falha_hill_climbing"

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
    pastaCenario = PASTA_RESULTADOS / cenario
    pastaCenario.mkdir(parents=True, exist_ok=True)

    # Roda todos os algoritmos gravando eventos, que são necessários para a visualização
    resultados = {
        nomeAlgoritmo: funcaoBusca(grafo, origem, destino, True)
        for nomeAlgoritmo, funcaoBusca in ALGORITMOS.items()
    }

    imprime_tabela(cenario, origem, destino, resultados)
    salva_csv(pastaCenario / "resultados.csv", cenario, origem, destino, resultados)
    salva_grade(grafo, origem, destino, resultados, pastaCenario / "grade.png")

    # Gera a imagem final e a animação de cada algoritmo
    for nomeAlgoritmo, resultado in resultados.items():
        # Monta um nome de arquivo sem espaços nem asterisco, por exemplo "hill_climbing" e "a_estrela"
        nomeArquivo = monta_nome_arquivo(nomeAlgoritmo)

        # Salva o estado final deste algoritmo sozinho, além da grade com todos
        salva_estado_final(grafo, origem, destino, nomeAlgoritmo, resultado, pastaCenario / f"{nomeArquivo}.png")

        figura, animacaoBusca = cria_animacao(grafo, origem, destino, nomeAlgoritmo, resultado)
        salva_animacao(figura, animacaoBusca, pastaCenario / nomeArquivo, QUADROS_POR_SEGUNDO)
        print(f"animação salva: {cenario}/{nomeArquivo}")

def main() -> None:
    """
    Executa o experimento i completo.
    """
    # Gera a rede pequena com a semente padrão
    grafo = gera_rede_knn(QUANTIDADE_VERTICES, VALOR_K, semente=SEMENTE_PADRAO)
    print(f"rede: {grafo.quantidade_vertices()} vértices, {grafo.quantidade_arestas()} arestas, grau médio {grafo.grau_medio():.2f}")

    # Escolhe um par para cada cenário e roda os seis algoritmos em cada um
    for cenario, (origem, destino) in escolhe_pares(grafo).items():
        roda_cenario(grafo, cenario, origem, destino)

    print(f"\nresultados salvos em {PASTA_RESULTADOS}")

if __name__ == "__main__":
    main()