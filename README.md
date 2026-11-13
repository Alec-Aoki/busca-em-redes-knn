# Busca em Redes KNN

Implementação e visualização de algoritmos de busca (DFS, BFS, Best-First, A*, Hill Climbing e Dijkstra) em redes KNN.

**Grupo:** Alec Campos Aoki, Daniel Jorge Manzano, Fernando Valentim Torres, Frederico Scheffel Oliveira e Pedro Henrique Perez Dias.

---

## Como rodar

Requer Python 3.10 ou mais novo.

Instale as dependências:

```bash
pip install "numpy>=1.26" "matplotlib>=3.8" "pandas>=2.1" "pillow>=10.0"
```

Para salvar as animações também em MP4, é preciso ter o `ffmpeg` instalado no sistema (`sudo apt install ffmpeg` no Linux, `brew install ffmpeg` no macOS). Sem ele, as animações são salvas só em GIF.

Depois, de dentro da pasta dos três arquivos, rode:

```bash
python experimentos.py
```

Esse comando roda os três experimentos em sequência, sem precisar de argumentos, e leva cerca de 5 minutos (quase todo o tempo vai na geração das animações do experimento i). Os resultados são salvos em `resultados/`, criada automaticamente ao lado dos scripts. Todas as sementes são fixas, então os resultados se repetem a cada execução (exceto os tempos, que dependem da máquina).

---

## Arquivos

O projeto tem três arquivos, que devem ficar na mesma pasta.

**`rede.py`** define a classe `Grafo` e gera as redes KNN: sorteia as coordenadas dos vértices em `[0, n] × [0, n]` e liga cada vértice aos seus `k` vizinhos mais próximos, por força bruta, com arestas não-direcionadas e peso igual à distância euclidiana. O grafo é guardado como lista de adjacências. Também encontra as componentes conexas da rede e sorteia pares (origem, destino) dentro da maior delas, para garantir que sempre existe caminho, e define a semente usada em todo o projeto.

**`algoritmos.py`** tem as heurísticas (distância euclidiana até o destino, a principal, e distância de Chebyshev, usada no experimento iii), o `ResultadoBusca` que todo algoritmo devolve (caminho, distância, nós expandidos e eventos para a visualização) e os seis algoritmos, todos com a mesma assinatura: `busca_x(grafo, origem, destino, gravarEventos)`. O A* aceita também a heurística como parâmetro opcional (euclidiana por padrão). No final, o dicionário `ALGORITMOS` junta todos, e é ele que os experimentos percorrem.

**`experimentos.py`** gera a animação passo a passo de cada busca (GIF e MP4), com os vértices do caminho final destacados, e a figura em grade com o estado final de todos os algoritmos lado a lado. Em seguida, roda os três experimentos:

- **Experimento i:** gera uma rede pequena (300 vértices) e escolhe dois pares (origem, destino): um em que todos os algoritmos chegam ao destino e outro em que o Hill Climbing fica preso em um mínimo local. Para cada par, roda os seis algoritmos e salva em `resultados/exp1/` uma animação e uma imagem do estado final de cada algoritmo, a figura em grade com todos juntos e um CSV com o caminho e a distância de cada um.
- **Experimento ii:** gera redes com 2500 vértices e `k = 3`, `7` e `11`, roda os seis algoritmos em 30 pares e calcula, para cada algoritmo e cada `k`, a distância média, o tempo médio, os desvios padrão, os nós expandidos e a taxa de sucesso. Salva em `resultados/exp2/` uma tabela com todas as execuções e outra com o resumo.
- **Experimento iii:** usa as execuções do experimento ii para comparar o A* e o Dijkstra, e roda também o A* com a heurística de Chebyshev nos mesmos pares. Confere se as distâncias são iguais e compara nós expandidos e tempo. Salva em `resultados/exp3/` as tabelas e um gráfico de nós expandidos em função da distância entre origem e destino.

As tabelas de cada experimento também são impressas no terminal.
