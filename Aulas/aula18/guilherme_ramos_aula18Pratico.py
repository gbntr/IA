"""
Mundo de Wumpus com SARSA — Aula 18 (Aprendizado por Reforço)
Aluno: Guilherme Ramos
"""

import os
import json
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

# ── Constantes ────────────────────────────────────────────────────────────────
ACTIONS   = ['forward', 'left', 'right', 'shoot', 'grab', 'climb']
N_ACTIONS = len(ACTIONS)

# Deltas (dr, dc) por direção: 0=cima, 1=direita, 2=baixo, 3=esquerda
DIR_DELTA = [(1, 0), (0, 1), (-1, 0), (0, -1)]
DIR_NAME  = ['↑', '→', '↓', '←']

_HERE       = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(_HERE, 'figuras')

# ── Mapas ──────────────────────────────────────────────────────────────────────
# grid[row][col] — row 0 = base, row 3 = topo da grade 4×4.
# Agente começa em (0, 0) voltado para a direita (dir=1).
# '.' = vazio  'P' = abismo  'W' = Wumpus  'G' = ouro

MAP_TRAIN = [
    ['.', '.', '.', 'G'],   # row 0  ← início (0,0)  ouro (0,3)
    ['.', '.', 'P', '.'],   # row 1
    ['W', '.', '.', '.'],   # row 2
    ['.', '.', '.', 'P'],   # row 3
]
# Visualização (row 3 = topo):
#   3: .  .  .  P
#   2: W  .  .  .
#   1: .  .  P  .
#   0: A  .  .  G

# Mapa de teste deliberadamente diferente:
#   - ouro no canto superior esquerdo (3,0)  →  agente deve ir para CIMA, não para a direita
#   - Wumpus em (2,2) bloqueia o centro
#   - abismo em (1,1) força desvio
#   - caminho ótimo: (0,0)→↑×3→(3,0) grab → ↓×3→(0,0) climb  (10 ações)
MAP_TEST = [
    ['.', '.', '.', '.'],   # row 0
    ['.', 'P', '.', '.'],   # row 1  ← abismo (1,1)
    ['.', '.', 'W', '.'],   # row 2  ← Wumpus (2,2)
    ['G', '.', '.', '.'],   # row 3  ← ouro (3,0)
]
# Visualização (row 3 = topo):
#   3: G  .  .  .
#   2: .  .  W  .
#   1: .  P  .  .
#   0: A  .  .  .


# ── Ambiente ──────────────────────────────────────────────────────────────────
class WumpusEnv:
    """Grade 4×4 do Mundo de Wumpus com interface gym-like."""

    SIZE = 4

    def __init__(self, grid):
        self._base        = [row[:] for row in grid]
        self.grid         = None
        self.pos          = (0, 0)
        self.direction    = 1
        self.has_gold     = False
        self.has_arrow    = True
        self.wumpus_alive = True
        self.done         = False
        self.won          = False
        self.reset()

    def reset(self):
        """Reinicia o episódio e devolve o estado inicial."""
        self.grid         = [row[:] for row in self._base]
        self.pos          = (0, 0)
        self.direction    = 1
        self.has_gold     = False
        self.has_arrow    = True
        self.wumpus_alive = True
        self.done         = False
        self.won          = False
        return self._state()

    def step(self, action_idx):
        """
        Executa a ação e devolve (next_state, reward, done).
        Custo base de -1 por ação; bônus/penalidades adicionais por evento.
        """
        if self.done:
            raise RuntimeError("Episódio encerrado — chame reset().")

        # Dispatch por índice evita comparação de strings a cada passo.
        reward = -1
        if action_idx == 0:    # forward
            reward += self._do_forward()
        elif action_idx == 1:  # left
            self.direction = (self.direction - 1) % 4
        elif action_idx == 2:  # right
            self.direction = (self.direction + 1) % 4
        elif action_idx == 3:  # shoot
            reward += self._do_shoot()
        elif action_idx == 4:  # grab
            reward += self._do_grab()
        else:                  # climb
            reward += self._do_climb()

        return self._state(), reward, self.done

    # ── Percepção / estado ─────────────────────────────────────────────────────
    def _state(self):
        """
        Tupla de estado:
        (row, col, dir, has_gold, has_arrow, wumpus_alive, breeze, stench, glitter)

        Faz uma única passagem pelos vizinhos para computar brisa e cheiro,
        em vez de três passagens separadas.
        """
        r, c = self.pos
        breeze = stench = False
        for dr, dc in DIR_DELTA:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.SIZE and 0 <= nc < self.SIZE:
                cell = self.grid[nr][nc]
                if cell == 'P':
                    breeze = True
                elif cell == 'W' and self.wumpus_alive:
                    stench = True
        glitter = self.grid[r][c] == 'G'
        return (r, c, self.direction,
                int(self.has_gold), int(self.has_arrow),
                int(self.wumpus_alive),
                int(breeze), int(stench), int(glitter))

    def _adj_has(self, symbol):
        """Verifica adjacência — usada apenas em render()."""
        r, c = self.pos
        for dr, dc in DIR_DELTA:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.SIZE and 0 <= nc < self.SIZE:
                if self.grid[nr][nc] == symbol:
                    return True
        return False

    def render(self):
        """Exibe o mapa e percepções no terminal (para depuração)."""
        ar, ac = self.pos
        for r in range(self.SIZE - 1, -1, -1):
            row = []
            for c in range(self.SIZE):
                row.append(f"A{DIR_NAME[self.direction]}" if (r, c) == (ar, ac)
                            else f"{self.grid[r][c]:2}")
            print(f"  {r}: {'  '.join(row)}")
        r, c = self.pos
        br = self._adj_has('P')
        st = self.wumpus_alive and self._adj_has('W')
        gl = self.grid[r][c] == 'G'
        print(f"  pos={self.pos} dir={DIR_NAME[self.direction]}"
              f"  gold={self.has_gold}  arrow={self.has_arrow}"
              f"  wumpus_vivo={self.wumpus_alive}")
        print(f"  percep: brisa={br}  cheiro={st}  brilho={gl}")

    # ── Ações internas ──────────────────────────────────────────────────────────
    def _do_forward(self):
        dr, dc = DIR_DELTA[self.direction]
        nr, nc = self.pos[0] + dr, self.pos[1] + dc
        if not (0 <= nr < self.SIZE and 0 <= nc < self.SIZE):
            return -1
        self.pos = (nr, nc)
        if self.grid[nr][nc] in ('P', 'W'):
            self.done = True
            return -1000
        return 0

    def _do_shoot(self):
        if not self.has_arrow:
            return -1
        self.has_arrow = False
        dr, dc = DIR_DELTA[self.direction]
        r, c = self.pos[0] + dr, self.pos[1] + dc
        while 0 <= r < self.SIZE and 0 <= c < self.SIZE:
            if self.grid[r][c] == 'W':
                self.grid[r][c] = '.'
                self.wumpus_alive = False
                return 50
            r += dr
            c += dc
        return -10

    def _do_grab(self):
        r, c = self.pos
        if self.grid[r][c] == 'G':
            self.grid[r][c] = '.'
            self.has_gold = True
            return 100
        return -1

    def _do_climb(self):
        if self.pos == (0, 0):
            self.done = True
            if self.has_gold:
                self.won = True
                return 1000
            return -50
        return -10


# ── Agente SARSA ──────────────────────────────────────────────────────────────
class SARSAAgent:
    """
    Agente on-policy SARSA (TD(0)).

    Atualização:
        Q(s, a) ← Q(s, a) + α · [r + γ · Q(s', a') − Q(s, a)]

    Política ε-greedy com decaimento exponencial de ε.
    """

    def __init__(self, n_actions=N_ACTIONS,
                 alpha=0.1, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.02, epsilon_decay=0.9995):
        self.n_actions     = n_actions
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.Q             = defaultdict(float)

    def choose_action(self, state):
        """Política ε-greedy."""
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        return self._greedy(state)

    def _greedy(self, state):
        qs = [self.Q[(state, a)] for a in range(self.n_actions)]
        m  = max(qs)
        best = [a for a, v in enumerate(qs) if v == m]
        return best[0] if len(best) == 1 else random.choice(best)

    def update(self, s, a, r, s_next, a_next, done):
        """Passo de atualização SARSA — cache de Q(s,a) evita dupla busca."""
        q_sa         = self.Q[(s, a)]
        q_next       = 0.0 if done else self.Q[(s_next, a_next)]
        self.Q[(s, a)] = q_sa + self.alpha * (r + self.gamma * q_next - q_sa)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ── Treinamento ───────────────────────────────────────────────────────────────
def train(env, agent, n_episodes=5000, max_steps=200, verbose=True):
    """
    Loop principal SARSA.
    Devolve lista com a recompensa acumulada de cada episódio.
    """
    rewards = []
    wins    = 0

    for ep in range(1, n_episodes + 1):
        s = env.reset()
        a = agent.choose_action(s)
        total = 0

        for _ in range(max_steps):
            s2, r, done = env.step(a)
            a2 = agent.choose_action(s2)
            agent.update(s, a, r, s2, a2, done)
            s, a = s2, a2
            total += r
            if done:
                if env.won:
                    wins += 1
                break

        agent.decay_epsilon()
        rewards.append(total)

        if verbose and ep % 500 == 0:
            wr  = 100.0 * wins / ep
            avg = float(np.mean(rewards[-500:]))
            print(f"  ep {ep:>5} | ε={agent.epsilon:.4f} | "
                  f"recomp. média (últ 500)={avg:>9.1f} | wins={wr:.1f}%")

    return rewards


# ── Avaliação ─────────────────────────────────────────────────────────────────
def evaluate(env, agent, n_episodes=200, max_steps=200):
    """Avalia o agente sem exploração (ε = 0)."""
    saved_eps     = agent.epsilon
    agent.epsilon = 0.0
    wins = deaths = timeouts = 0
    all_r, all_steps = [], []

    for _ in range(n_episodes):
        s = env.reset()
        total = 0
        for step in range(1, max_steps + 1):
            a = agent.choose_action(s)
            s, r, done = env.step(a)
            total += r
            if done:
                if env.won:
                    wins += 1
                else:
                    deaths += 1
                all_steps.append(step)
                break
        else:
            timeouts += 1
            all_steps.append(max_steps)
        all_r.append(total)

    agent.epsilon = saved_eps
    return {
        'wins':        wins,
        'deaths':      deaths,
        'timeouts':    timeouts,
        'mean_reward': float(np.mean(all_r)),
        'mean_steps':  float(np.mean(all_steps)),
        'n':           n_episodes,
    }


def print_eval(res, label):
    n = res['n']
    print(f"\n  [{label}]")
    print(f"    Vitórias  : {res['wins']:>4}/{n}  ({100*res['wins']/n:.1f}%)")
    print(f"    Mortes    : {res['deaths']:>4}/{n}  ({100*res['deaths']/n:.1f}%)")
    print(f"    Timeouts  : {res['timeouts']:>4}/{n}")
    print(f"    Recomp. média : {res['mean_reward']:.1f}")
    print(f"    Passos médios : {res['mean_steps']:.1f}")


# ── Demonstração de episódio ──────────────────────────────────────────────────
def run_demo(env, agent, label="Mapa", max_steps=60):
    """Executa um episódio greedy passo a passo e imprime tudo."""
    saved_eps     = agent.epsilon
    agent.epsilon = 0.0
    s = env.reset()
    print(f"\n  ── Demonstração: {label} ──")
    env.render()

    for step in range(1, max_steps + 1):
        a = agent.choose_action(s)
        s, r, done = env.step(a)
        print(f"\n  passo {step}: {ACTIONS[a]}  recomp={r}")
        env.render()
        if done:
            print(f"  >> {'VITÓRIA' if env.won else 'DERROTA'}")
            break
    else:
        print("  >> TIMEOUT")

    agent.epsilon = saved_eps


# ── Gráficos ──────────────────────────────────────────────────────────────────
def _ma(data, w):
    return np.convolve(data, np.ones(w) / w, mode='valid')


def plot_learning_curve(rewards, label, filename, window=200):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rewards, alpha=0.2, color='steelblue')
    if len(rewards) >= window:
        ma = _ma(rewards, window)
        ax.plot(range(window - 1, len(rewards)), ma,
                color='darkorange', lw=2, label=f'média móvel ({window} ep.)')
    ax.set_xlabel("Episódio")
    ax.set_ylabel("Recompensa total")
    ax.set_title(f"Curva de aprendizado — {label}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=150)
    plt.close()


def plot_win_rate(rewards_list, labels, filename, window=200):
    colors = ['steelblue', 'darkorange', 'seagreen']
    fig, ax = plt.subplots(figsize=(8, 4))
    for rewards, label, color in zip(rewards_list, labels, colors):
        wins = np.array([1.0 if r > 500 else 0.0 for r in rewards])
        if len(wins) >= window:
            rate = _ma(wins, window) * 100
            ax.plot(range(window - 1, len(wins)), rate,
                    lw=2, label=label, color=color)
    ax.set_xlabel("Episódio")
    ax.set_ylabel("Taxa de vitórias (%)")
    ax.set_title("Taxa de vitórias (média móvel)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=150)
    plt.close()


# ── Ponto de entrada ──────────────────────────────────────────────────────────
def main():
    random.seed(42)
    np.random.seed(42)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    EPISODES_TRAIN = 10_000
    EPISODES_FT    = 5_000
    MAX_STEPS      = 150
    ALPHA          = 0.1
    GAMMA          = 0.95
    EPS_START      = 1.0
    EPS_MIN        = 0.02
    EPS_DECAY      = 0.9995

    # ── 1. Treinamento no mapa fixo ──────────────────────────────────────────
    print("=" * 60)
    print("  FASE 1 — Treinamento (mapa fixo)")
    print("=" * 60)

    env_train = WumpusEnv(MAP_TRAIN)
    agent     = SARSAAgent(alpha=ALPHA, gamma=GAMMA,
                           epsilon=EPS_START, epsilon_min=EPS_MIN,
                           epsilon_decay=EPS_DECAY)

    rewards_train = train(env_train, agent,
                          n_episodes=EPISODES_TRAIN, max_steps=MAX_STEPS)

    res_train = evaluate(env_train, agent, n_episodes=200)
    print_eval(res_train, "Mapa de Treino — pós-treinamento")
    run_demo(env_train, agent, label="Mapa de Treino")

    # ── 2. Transferência zero-shot para mapa de teste ────────────────────────
    print("\n" + "=" * 60)
    print("  FASE 2 — Transferência zero-shot (mapa de teste)")
    print("=" * 60)

    env_test  = WumpusEnv(MAP_TEST)
    res_test0 = evaluate(env_test, agent, n_episodes=200)
    print_eval(res_test0, "Mapa de Teste — zero-shot")
    run_demo(env_test, agent, label="Mapa de Teste (zero-shot)")

    # ── 3. Fine-tuning no mapa de teste ─────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  FASE 3 — Fine-tuning ({EPISODES_FT} episódios no mapa de teste)")
    print("=" * 60)

    agent.epsilon = 0.5   # exploração moderada para aprender novo layout
    rewards_ft = train(env_test, agent,
                       n_episodes=EPISODES_FT, max_steps=MAX_STEPS)

    res_test_ft = evaluate(env_test, agent, n_episodes=200)
    print_eval(res_test_ft, "Mapa de Teste — pós fine-tuning")
    run_demo(env_test, agent, label="Mapa de Teste (pós fine-tuning)")

    # ── 4. Gráficos ──────────────────────────────────────────────────────────
    plot_learning_curve(rewards_train,
                        "Treinamento (Mapa Fixo)",
                        "sarsa_train_rewards.png")

    plot_learning_curve(rewards_ft,
                        "Fine-tuning (Mapa de Teste)",
                        "sarsa_ft_rewards.png",
                        window=200)

    plot_win_rate([rewards_train],
                  ["Treino"],
                  "sarsa_win_rate_train.png")

    plot_win_rate([rewards_ft],
                  ["Fine-tuning"],
                  "sarsa_win_rate_ft.png")

    # ── 5. Salva resultados ───────────────────────────────────────────────────
    results = {
        'train':          res_train,
        'test_zero_shot': res_test0,
        'test_fine_tuned':res_test_ft,
    }
    out_path = os.path.join(_HERE, 'resultados.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\nFiguras salvas em:", FIGURES_DIR)
    print("Resultados em:    ", out_path)
    return results


if __name__ == '__main__':
    main()
