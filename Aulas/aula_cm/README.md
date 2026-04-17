# Problema dos Missionários e Canibais

Este projeto implementa a solução do problema clássico de IA dos **Missionários e Canibais** utilizando Python. Foram implementados dois algoritmos de busca: **Busca em Profundidade (DFS)** e **A* (A-Estrela)**.

## O Problema
Três missionários e três canibais estão em uma margem de um rio e possuem um barco que comporta até duas pessoas. O objetivo é levar todos para a outra margem seguindo as restrições:
1. O barco nunca pode navegar vazio.
2. Em nenhuma margem o número de canibais pode ser superior ao de missionários (caso existam missionários naquela margem), pois senão os canibais devorarão os missionários.

---

## Estrutura do Código

### 1. Representação do Estado (`class State`)
A classe `State` armazena a configuração do problema em um dado momento:
- `m_left`, `c_left`: Missionários e Canibais na margem esquerda.
- `boat`: Posição do barco (0 para esquerda, 1 para direita).
- `m_right`, `c_right`: Missionários e Canibais na margem direita.
- `parent`: Referência ao estado anterior para reconstruir o caminho da solução.

### 2. Validação e Regras (`is_valid`)
O método `is_valid` garante que o estado gerado é seguro:
- Impede números negativos de pessoas.
- Aplica a regra principal: `if m > 0 then m >= c`.

### 3. Gerador de Sucessores (`get_successors`)
Esta função gera todos os movimentos possíveis a partir de um estado:
- Movimentar 1 Missionário.
- Movimentar 2 Missionários.
- Movimentar 1 Canibal.
- Movimentar 2 Canibais.
- Movimentar 1 Missionário e 1 Canibal.

---

## Algoritmos de Busca

### Busca em Profundidade (DFS)
- **Lógica**: Utiliza uma **Pilha (LIFO)**. Explora um caminho o máximo possível antes de retroceder (backtracking).
- **Características**: 
  - Encontra uma solução rapidamente em problemas de estado finito.
  - **Não garante** o caminho mais curto (solução ótima).
  - Mantém um conjunto `visited` para evitar loops infinitos.

### Busca A* (A-Estrela)
- **Lógica**: Utiliza uma **Fila de Prioridade** baseada na função $f(n) = g(n) + h(n)$.
  - $g(n)$: Custo do caminho do início até o estado atual.
  - $h(n)$: Heurística (estimativa do custo até o objetivo).
- **Heurística adotada**: A soma de missionários e canibais na margem esquerda ($m\_left + c\_left$). 
- **Características**: 
  - É um algoritmo **ótimo** (sempre encontra o menor número de passos) se a heurística for admissível.

---

## Como Executar
Certifique-se de ter o Python 3 instalado e execute:

```bash
python missionaries_cannibals.py
```

O script imprimirá o passo a passo de ambas as soluções, permitindo comparar a eficiência e o caminho tomado por cada algoritmo.
