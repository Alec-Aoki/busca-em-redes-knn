# Busca em Redes KNN

Implementação e visualização de algoritmos de busca (BFS, DFS, Best-First, A*, Hill Climbing e Dijkstra) em redes KNN.

Entregáveis: código-fonte, relatório em formato de artigo (máx. 8 páginas) e apresentação de até 15 minutos (não pode ser `.zip`).

---

## 1. Objetivos

1. **Gerar redes KNN**: `n` vértices com coordenadas aleatórias em `[0, n] × [0, n]`. Cada vértice é ligado aos seus `k` vizinhos mais próximos, e o peso da aresta é a distância euclidiana. O grafo é **não-direcionado**: se A está entre os k mais próximos de B, a aresta A–B existe mesmo que B não esteja entre os k mais próximos de A (por causa disso o grau médio real fica um pouco acima de `k`, comentar no relatório).
2. **Implementar** os algoritmos de busca (sem usar `networkx.astar_path`, `shortest_path` ou similares):
   - Busca em profundidade (DFS)
   - Busca em largura (BFS)
   - Best-First (gulosa, ordena por `h(n)`)
   - A* (ordena por `g(n) + h(n)`)
   - Hill Climbing **sem backtracking**: anda sempre para o vizinho com menor `h`. Se nenhum vizinho melhora, para e reporta falha (mínimo local).
   - Dijkstra
3. **Rodar os três experimentos** do enunciado e salva os resultados (CSV, figuras, animações) em `resultados/`.

A heurística `h(n)` é a **distância euclidiana** até o destino. Ela é admissível, porque nenhum caminho pelo grafo é mais curto que a linha reta. Logo, o A* deve encontrar sempre a mesma distância que o Dijkstra.

---

## 2. Experimentos (do enunciado)

### Experimento i: visualização

O enunciado pede: para uma rede, escolher origem e destino, rodar todos os algoritmos e mostrar o **caminho final** (lista de vértices) e a **distância final**. A visualização deve usar uma rede **pequena (100–500 vértices)** e permitir **perceber como cada algoritmo explora o espaço de busca**. A qualidade dessa visualização conta na nota.

Como vamos atender:

- Uma rede KNN de **~300 vértices**, com o **mesmo par origem/destino para todos os algoritmos**, para a comparação ser justa.
- **Uma animação por algoritmo** (GIF ou MP4), passo a passo, com este código de cores:
  - cinza: vértice ainda não visto
  - amarelo: na fronteira (fila/pilha/heap)
  - azul: já expandido
  - vermelho: vértice sendo expandido no passo atual
  - verde grosso: caminho final, desenhado ao término
  - origem e destino com marcadores próprios (estrela/quadrado)
  - legenda com nome do algoritmo, nº do passo, nº de nós expandidos e, ao fim, distância final e nº de vértices no caminho
- **Uma figura estática em grade 2×3** com o estado final dos 6 algoritmos lado a lado. É a melhor imagem para um slide e para o relatório.
- **Saída textual** com, para cada algoritmo: caminho (lista de vértices), distância final, nós expandidos e se encontrou solução. É o que o enunciado pede literalmente ("mostrar o caminho final… e a distância final").

As animações são exportadas para arquivo, para não dependermos de rodar código ao vivo na apresentação.

### Experimento ii: estatísticas em redes grandes

- `n = 2500`, com três configurações: `k = 3`, `k = 7` e `k = 11`.
- Para cada configuração, **no mínimo 10 pares (origem, destino) distintos**. Vamos usar 30 para as médias ficarem mais estáveis. Os pares são sorteados **dentro da mesma componente conexa**, porque com `k = 3` a rede pode ser desconexa.
- Métricas por algoritmo: **distância média**, **desvio padrão da distância**, **tempo médio**, **desvio padrão do tempo**, **nós expandidos (média)** e **taxa de sucesso** (relevante para o Hill Climbing).
- O tempo é medido com `time.perf_counter()`, **sem gravar eventos de visualização**, para não poluir a medição.

### Experimento iii: A* vs Dijkstra

- Nas mesmas redes do experimento ii, comparar **distância** (deve ser igual), **nós expandidos** e **tempo**.
- Extra (se der tempo): gráfico de nós expandidos em função da distância euclidiana entre origem e destino, para mostrar que a vantagem do A* cresce com a distância.

---

## 3. Estrutura planejada

```
busca-em-redes-knn/
├── README.md
├── requirements.txt
├── src/
│   ├── rede.py               # geração da rede KNN e estrutura do grafo
│   ├── heuristicas.py        # heurística euclidiana
│   ├── resultado.py          # dataclass ResultadoBusca (padrão de retorno)
│   ├── algoritmos/
│   │   ├── __init__.py       # expõe ALGORITMOS = {nome: função}
│   │   ├── bfs.py
│   │   ├── dfs.py
│   │   ├── best_first.py
│   │   ├── a_estrela.py
│   │   ├── hill_climbing.py
│   │   └── dijkstra.py
│   ├── visualizacao.py       # animações e figura em grade
│   └── utils.py              # sorteio de pares, componentes conexas, sementes
├── experimentos/
│   ├── exp1_visualizacao.py
│   ├── exp2_estatisticas.py
│   └── exp3_astar_dijkstra.py
├── resultados/               # gerado pelos scripts (CSV, PNG, GIF, MP4)
├── relatorio/                # artigo
└── slides/
```

### Interfaces combinadas

Todo algoritmo tem **a mesma assinatura** e devolve um `ResultadoBusca`. Isso é o que permite os experimentos e a visualização tratarem todos igual.

```python
# src/rede.py
class Grafo:
    coordenadas: list[tuple[float, float]]           # índice = id do vértice
    adjacencias: list[list[tuple[int, float]]]       # (vizinho, peso)

def gera_vertices(quantidade: int, semente: int | None = None) -> list[tuple[float, float]]: ...
def gera_rede_knn(quantidade: int, k: int, semente: int | None = None) -> Grafo: ...
```

```python
# src/resultado.py
@dataclass
class ResultadoBusca:
    encontrouSolucao: bool
    caminho: list[int]               # origem ... destino (vazio se falhou)
    distancia: float                 # soma dos pesos das arestas do caminho (inf se falhou)
    nosExpandidos: int
    eventos: list[tuple[str, int]]   # ("fronteira" | "expandido", vértice); vazio se gravarEventos=False
```

```python
# assinatura de todos os algoritmos
def busca_x(grafo: Grafo, origem: int, destino: int, gravarEventos: bool = False) -> ResultadoBusca: ...
```

Os eventos são a "gravação" da execução: a visualização só reproduz essa lista. Quando `gravarEventos=False`, nada é anotado, e é assim que medimos tempo.

---

## 4. To-do

### Base
- [ ] `requirements.txt` (`numpy`, `scipy`, `matplotlib`, `pandas`; `pillow` para GIF, `ffmpeg` no sistema para MP4)
- [ ] `rede.py`: `gera_vertices`, `gera_rede_knn` (KDTree do scipy ou força bruta com numpy), arestas não-direcionadas sem duplicatas
- [ ] `utils.py`: componentes conexas (BFS simples), sorteio de pares na mesma componente, controle de semente
- [ ] `heuristicas.py`: distância euclidiana
- [ ] `resultado.py`: `ResultadoBusca` + função que reconstrói o caminho a partir do dicionário de pais e soma a distância

### Algoritmos
- [ ] BFS (fila `collections.deque`)
- [ ] DFS (pilha, iterativa para não estourar recursão com n = 2500)
- [ ] Best-First (`heapq` por `h`)
- [ ] A* (`heapq` por `g + h`, com descarte de entradas obsoletas do heap)
- [ ] Hill Climbing sem backtracking (reporta falha em mínimo local)
- [ ] Dijkstra (`heapq` por `g`, parando ao expandir o destino)
- [ ] Teste rápido: A* e Dijkstra devem dar a mesma distância em vários pares

### Experimentos
- [ ] `exp1_visualizacao.py`: rede de ~300 vértices, um par, roda os 6, gera 6 animações + grade 2×3 + CSV com caminhos e distâncias
- [ ] `exp2_estatisticas.py`: n = 2500, k ∈ {3, 7, 11}, 30 pares, CSV com médias/desvios, tabela para o relatório
- [ ] `exp3_astar_dijkstra.py`: tabela comparativa + gráfico (extra)

### Entrega
- [ ] Relatório: Resumo, Introdução, Implementação (rede + algoritmos), Resultados, Discussão (incluindo as visualizações)
- [ ] Slides (≤ 15 min, todos apresentam)
- [ ] Subir código, relatório e slides no e-disciplinas, sem compactar

---

## 5. Observações

- **Algoritmos à mão.** `networkx` pode ser usado no máximo para desenhar, nunca para buscar. A implementação faz parte da avaliação.
- **Respeite as interfaces da seção 3.** Os experimentos e a visualização dependem delas.
- **Sementes fixas** em todos os experimentos, para os resultados serem reproduzíveis.
- **BFS minimiza número de arestas, não distância.** Ainda assim, reportamos a distância (soma dos pesos) do caminho que ela encontrou.
- **DFS e Best-First não garantem caminho ótimo**, e é esperado que deem distâncias maiores. Isso é resultado, não bug.
- **Hill Climbing vai falhar às vezes.** Contar a taxa de sucesso e não incluir as falhas na média de distância (deixar isso explícito no relatório).

### Convenções de código

- Comentários, docstrings e nomes em português.
- Variáveis em `camelCase`, funções em `snake_case`, classes em `PascalCase`, constantes em `MAIUSCULO`.
- Nada de nomes de uma letra, nem em loops (`indice`, `vizinho`, `vertice`…). Exceção: `k` e `n`, por serem os parâmetros da rede no enunciado.
- Type hints em todas as funções; docstrings no estilo Google.
- Comentários no presente ("calcula…", "expande…") e sem emoji pfv :).

---

## 6. Como rodar (a preencher)

```bash
pip install -r requirements.txt
python experimentos/exp1_visualizacao.py
python experimentos/exp2_estatisticas.py
python experimentos/exp3_astar_dijkstra.py
```