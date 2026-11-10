# Busca em Redes KNN

Implementação e visualização de algoritmos de busca (DFS, BFS, Best-First, A*, Hill Climbing e Dijkstra) em redes KNN, para o trabalho de SCC0230 – Inteligência Artificial (Abordagem A).

Entregáveis: código-fonte, relatório em formato de artigo (máx. 8 páginas) e apresentação de até 15 minutos (não pode ser `.zip`).

---

## Como rodar

Requer Python 3.10 ou mais novo. Para salvar as animações em MP4, também é preciso ter o `ffmpeg` instalado no sistema (`sudo apt install ffmpeg` no Linux, `brew install ffmpeg` no macOS). Sem ele, as animações são salvas só em GIF.

```bash
pip install -r requirements.txt

python experimentos/exp1_visualizacao.py     # ~6 min
python experimentos/exp2_estatisticas.py     # alguns segundos
python experimentos/exp3_astar_dijkstra.py   # alguns segundos (precisa do exp2 rodado antes)
```

Os comandos são rodados da raiz do repositório, e os resultados vão para `resultados/`. Todas as sementes são fixas, então os resultados se repetem a cada execução (exceto os tempos, que dependem da máquina).

Cada arquivo de `src/` também tem um teste rápido no final. Para rodá-lo, entre em `src/` e use `python rede.py`, `python visualizacao.py` etc. Para os algoritmos, use `python -m algoritmos.bfs`, `python -m algoritmos.a_estrela` etc.

---

## Estrutura

```
busca-em-redes-knn/
├── README.md
├── requirements.txt
├── src/
│   ├── rede.py
│   ├── utils.py
│   ├── heuristicas.py
│   ├── resultado.py
│   ├── algoritmos/
│   │   ├── __init__.py
│   │   ├── bfs.py
│   │   ├── dfs.py
│   │   ├── best_first.py
│   │   ├── a_estrela.py
│   │   ├── hill_climbing.py
│   │   └── dijkstra.py
│   └── visualizacao.py
├── experimentos/
│   ├── exp1_visualizacao.py
│   ├── exp2_estatisticas.py
│   └── exp3_astar_dijkstra.py
└── resultados/
```

### `src/`

**`rede.py`** define a classe `Grafo` e gera as redes KNN: sorteia as coordenadas dos vértices em `[0, n] × [0, n]` e liga cada vértice aos seus `k` vizinhos mais próximos, com peso igual à distância euclidiana.

**`utils.py`** encontra as componentes conexas da rede e sorteia pares (origem, destino) dentro da maior delas, para garantir que sempre existe caminho. Também define a semente usada em todo o projeto.

**`heuristicas.py`** tem a heurística usada pelo Best-First, pelo A* e pelo Hill Climbing: a distância em linha reta até o destino.

**`resultado.py`** define o `ResultadoBusca`, que é o que todo algoritmo devolve (caminho, distância, nós expandidos e eventos para a visualização), e as funções que reconstroem o caminho e calculam sua distância.

**`algoritmos/`** tem um arquivo por algoritmo, todos com a mesma assinatura: `busca_x(grafo, origem, destino, gravarEventos)`. O `__init__.py` junta todos no dicionário `ALGORITMOS`, que é o que os experimentos percorrem.

**`visualizacao.py`** gera a animação passo a passo de uma busca (GIF e MP4) e a figura em grade com o estado final de todos os algoritmos lado a lado.

### `experimentos/`

**`exp1_visualizacao.py`** gera uma rede pequena (300 vértices) e escolhe dois pares (origem, destino): um em que todos os algoritmos chegam ao destino e outro em que o Hill Climbing fica preso em um mínimo local. Para cada par, roda os seis algoritmos e salva em `resultados/exp1/` uma animação e uma imagem do estado final de cada algoritmo, a figura em grade com todos juntos e um CSV com o caminho e a distância de cada um.

**`exp2_estatisticas.py`** gera redes com 2500 vértices e `k = 3`, `7` e `11`, roda os seis algoritmos em 30 pares e calcula, para cada algoritmo e cada `k`, a distância média, o tempo médio, os desvios padrão, os nós expandidos e a taxa de sucesso. Salva em `resultados/exp2/` uma tabela com todas as execuções e outra com o resumo.

**`exp3_astar_dijkstra.py`** usa as execuções do exp2 para comparar o A* e o Dijkstra: confere se as distâncias são iguais e compara nós expandidos e tempo. Salva em `resultados/exp3/` as tabelas e um gráfico de nós expandidos em função da distância entre origem e destino.

---

## To-do

### Base
- [x] `requirements.txt`
- [x] `rede.py`
- [x] `utils.py`
- [x] `heuristicas.py`
- [x] `resultado.py`

### Algoritmos
- [x] BFS
- [x] DFS
- [x] Best-First
- [x] A*
- [x] Hill Climbing (sem backtracking)
- [x] Dijkstra
- [x] Teste: A* e Dijkstra devem dar a mesma distância

### Experimentos
- [x] `exp1_visualizacao.py`
- [x] `exp2_estatisticas.py`
- [x] `exp3_astar_dijkstra.py`

### Entrega
- [X] Relatório
- [ ] Slides (≤ 15 min, todos apresentam)
- [ ] Subir código, relatório e slides no e-disciplinas, sem compactar

---

## Convenções de código

- Comentários, docstrings e nomes em português.
- Variáveis em `camelCase`, funções em `snake_case`, classes em `PascalCase`, constantes em `MAIUSCULO`.
- Nada de nomes de uma letra, nem em loops (`indice`, `vizinho`, `vertice`…). Exceção: `k` e `n`, por serem os parâmetros da rede no enunciado.
- Type hints em todas as funções e docstrings no estilo Google, com quebra de linha depois do `"""` de abertura e antes do de fechamento.
- Algoritmos implementados à mão: nada de `networkx.astar_path`, `shortest_path` ou similares.