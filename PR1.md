# Reinforcement Learning for 6×6 Checkers: A Tabular Q-Learning and Curriculum Self-Play Approach

## 1. Introduction & Project Overview

This project trains a **tabular Q-learning agent** to play **6×6 Checkers** (English draughts on a reduced board) in a custom **Gymnasium** environment. The goal is to learn a policy that maximizes discounted return against a curriculum of opponents: first random, then a fixed heuristic, and finally self-play with historical and recent Q-table snapshots. The implementation uses a dictionary-based Q-table, state-space reduction via canonical board hashing, backward-pass Q-updates over full episodes, state-dependent exploration, and decoupled evaluation so that reported win rates reflect the learned policy rather than the training mixture. Success is measured by win rate against a fixed heuristic baseline and by the stability of learning across both player roles (Player 1 and Player 2).

---

## 2. Environment Design & Game Mechanics

### 2.1 Custom `Checkers6x6Env`

The environment is implemented in `checkers_env.py` as a Gymnasium `Env` with a 6×6 board. Only **dark squares** are playable, i.e. cells where \((r + c) \equiv 1 \pmod{2}\), giving 18 playable positions.

**Observation space:** A `Dict` with:

- **`board`:** `Box(low=0, high=4, shape=(6,6), dtype=np.int8)`. Encoding: 0 = empty; 1 = Player 1 man; 2 = Player 2 man; 3 = Player 1 king; 4 = Player 2 king.
- **`current_player`:** `Discrete(2)` (0 = Player 1 to move, 1 = Player 2 to move).

**Action space:** `MultiDiscrete([6, 6, 6, 6])`, i.e. \((s_r, s_c, e_r, e_c)\): start row, start column, end row, end column. The nominal space has \(6^4 = 1296\) combinations; legality is enforced by the environment (diagonal moves, piece ownership, forced capture, multi-jump).

### 2.2 Rules Enforced

- **Diagonal moves only;** men move **forward** (Player 1 toward row 0, Player 2 toward row 5); **kings** move one step in any diagonal direction.
- **Forced capture:** If any capture is legal, only capture moves are legal (`_get_legal_moves` returns `capture_moves` when non-empty; otherwise `simple_moves`).
- **Multi-jump:** After a capture, if the same piece can capture again (and did not promote), `active_piece` is set and the turn continues with that piece until no further capture or promotion.
- **Promotion:** A man reaching the opponent's back rank (row 0 for P1, row 5 for P2) becomes a king; the implementation sets `_last_move_promoted` and applies the promotion reward.
- **Termination:** Win (opponent has no pieces or no legal move), loss (agent has no pieces), draw on move limit (default 200 steps) or no-progress (40 consecutive steps without a capture or without a man move).

Invalid moves leave the board and current player unchanged and the same player moves again.

### 2.3 Reward Shaping

Rewards are **normalized** and defined from the **current player's** perspective at each step (see `checkers_env.py` docstring and `step()`):

| Event | Reward |
|--------|--------|
| Win | \(+1.0\) |
| Loss | \(-1.0\) |
| Draw | \(0.0\) (no extra terminal bonus) |
| Capture | \(+0.1\) |
| Promotion to king | \(+0.15\) |
| Per-step (stalling penalty) | \(-0.005\) |

So the agent is shaped toward winning, capturing, and promoting, and mildly discouraged from stalling. The learning agent receives these rewards when it is the current player; when the opponent scores (e.g. capture), the agent's reward is negated in `run_episode` (e.g. `total_reward += -r_opp`).

---

## 3. Core Reinforcement Learning Concepts Used

### 3.1 Tabular Q-Learning and Dictionary-Based Q-Table

The agent maintains \(Q(s,a)\) only for **visited** state–action pairs. The implementation uses a Python dictionary keyed by \((s, a)\):

- **State \(s\):** Hashable canonical representation from `observation_to_state()` (see Section 5): an 18-tuple of piece codes on dark squares, always from "Player 0 to move" perspective.
- **Action \(a\):** Tuple \((s_r, s_c, e_r, e_c)\) in canonical coordinates.

`get_q_value(state, action)` returns `q_table.get((state, action), 0.0)`. Unvisited pairs are treated as 0.0. The Bellman update is applied only for \((s,a)\) that appear in the current episode's memory, so the table grows with experience and remains sparse relative to the full 18-cell state space.

### 3.2 Exploration vs. Exploitation: State-Dependent \(\varepsilon\)-Greedy

The behavior policy is **\(\varepsilon\)-greedy** with a **state-dependent** \(\varepsilon\) so that frequently visited states exploit more:

\[
\varepsilon(s) = \frac{N_0}{N_0 + N(s)}, \quad N_0 = 100,
\]

where \(N(s)\) is the number of times the agent has chosen an action from state \(s\) (`state_visit_counts` in `q_agent.py`). With probability \(\varepsilon(s)\) the agent selects a **random legal action**; otherwise it selects \(\arg\max_{a \in \mathcal{A}(s)} Q(s,a)\) over the current legal moves. Thus exploration decays per state as \(N(s)\) grows, without a global schedule; early in training and in rare states the agent still explores.

### 3.3 Backward-Pass Update and Bellman Equation

Each training episode is run without updating Q. The agent stores a list of **transitions** from its perspective: each entry is \((s, a, r, s', \mathcal{A}(s'))\), where the transition spans one full "agent move plus opponent response" (and any multi-jump), so \(r\) is the cumulative reward over that span and \(s'\) is the next state where the agent acts (or terminal).

After the episode, **backward-pass Q-learning** is applied in **reverse** order over this list:

\[
Q(s,a) \leftarrow Q(s,a) + \alpha(s,a) \left[ r + \gamma \max_{a' \in \mathcal{A}(s')} Q(s', a') - Q(s,a) \right],
\]

with \(\gamma = 0.99\). The maximum is taken only over **legal** next actions \(\mathcal{A}(s')\); for terminal \(s'\), \(\max_{a'} Q(s',a') = 0\). The learning rate is **state–action dependent** with a floor:

\[
\alpha(s,a) = \max\left(0.005,\; \frac{1}{\sqrt{N(s,a)+1}}\right),
\]

where \(N(s,a)\) is the visit count for \((s,a)\). Processing in reverse ensures that when updating \((s,a)\), the values \(Q(s',a')\) for the "next" state \(s'\) (which appears later in the episode) have already been updated, improving the target \(r + \gamma \max_{a'} Q(s',a')\).

---

## 4. Curriculum Learning Pipeline

Training uses three **phases** that change the **mixture of opponents** (random, heuristic, self-play). The mixture is sampled each episode; advancement to the next phase is based on **decoupled** evaluation against the **fixed heuristic** (Section 5).

- **Phase 0 (mostly random):** 80% random, 20% heuristic, 0% self-play. The agent learns legal play and basic tactics (captures, promotion) against weak opposition.
- **Phase 1 (mostly heuristic):** 10% random, 80% heuristic, 10% self-play. The agent must beat the priority-based heuristic; the small self-play fraction starts exposure to its own policy.
- **Phase 2 (mostly self-play):** 0% random, 20% heuristic, 80% self-play. The agent plays mainly against past versions of itself (from the opponent pool), with a 20% heuristic anchor to limit forgetting and retain robustness against the heuristic.

**Advancement condition** (in `train.py`): move to the next phase when **both** heuristic-benchmark win rates exceed thresholds and the current phase is below 2:

- Win rate as Player 1 vs heuristic \(> 0.75\)
- Win rate as Player 2 vs heuristic \(> 0.60\)

So the curriculum is driven by **heuristic** performance, not by the current training mixture, which avoids distorted metrics when self-play dominates.

---

## 5. Evolution of the Agent (Key Improvements)

### 5.1 State-Space Reduction and Canonical (Horizontal) Symmetry

The raw observation is a 6×6 board plus current player. The agent uses a **canonical** state so that "Player 1 to move" and "Player 2 to move" are represented in a single frame:

- If `current_player == 1`, the board is **flipped** (e.g. `np.flip(board_2d)`) and piece IDs are swapped (1↔2, 3↔4). The position is then interpreted as "Player 0 to move" in that flipped view.
- Only the **18 dark squares** are stored, in a fixed order, as a tuple of integers in \(\{0,1,2,3,4\}\).

So \(s \in \{0,1,2,3,4\}^{18}\) and is **hashable**. Benefits: (1) one state covers both colors by symmetry, cutting redundant entries; (2) the table only stores visited \((s,a)\) pairs, so size reflects discovered states; (3) learning generalizes across sides because both map to the same canonical \(s\).

### 5.2 Historical and Recent Opponent Pool (Self-Play Without Forgetting)

In self-play, the opponent is not the **current** Q-table (which would create a moving target and instability). Instead, the opponent is sampled from an **opponent pool** of **saved Q-table snapshots**:

- Every **5,000** episodes, the current `agent.q_table` is deep-copied into the pool.
- **Recent pool:** Up to the **5 most recent** snapshots; when a new one is added, the oldest is dropped.
- **Historical pool:** Milestones at episodes 5,000; 10,000; 25,000; 50,000 are appended and kept indefinitely.

When the opponent type is self-play, `sample_opponent_q_table()` picks a Q-table from the pool: with probability 0.7 from "recent" and 0.3 from "historical" when both are non-empty; otherwise from whichever is non-empty. The opponent plays greedily with that snapshot (no exploration). This provides a **diverse**, slightly lagged set of opponents and reduces **catastrophic forgetting** of earlier strategies.

### 5.3 Alternating Colors and Dynamic Role Assignment

The agent must perform well as **both** Player 1 (id 0) and Player 2 (id 1). Training alternates roles so that the same policy is learned for both sides in canonical form. To correct **asymmetry** (e.g. P1 stronger than P2), role assignment is **dynamic**:

- If the heuristic-benchmark win rate as P1 exceeds that as P2 by **more than 0.15**, the agent is assigned to play as Player 2 with probability **0.75** and as Player 1 with probability **0.25** for that episode.
- Otherwise, **50/50** between the two roles.

So when P2 is lagging, the agent trains more often as P2, balancing learning across sides.

### 5.4 Decoupled Evaluation

Training uses exploration and a **mixture** of opponents, so the **training** win rate (agent wins in the current episode) is noisy and tied to the curriculum. To measure the **learned policy** accurately, evaluation is **decoupled**:

- Every **1,000** episodes, training is **paused**.
- The agent is evaluated with **no exploration** (\(\varepsilon=0\), i.e. greedy_action) and **no Q-updates** (`update_q=False`) in **100** games vs a **fixed random** opponent and **100** games vs a **fixed heuristic** opponent (50 as P1, 50 as P2 each).
- Opponent pools for evaluation are **empty** (no self-play); so the benchmarks are always "random" and "heuristic."
- Results are stored as `eval_win_random`, `eval_win_heuristic`, `eval_win_p1_heuristic`, `eval_win_p2_heuristic` and used for curriculum advancement and role assignment.

Thus the reported curves reflect **true** performance of the current policy against fixed opponents, not the noisy training loop.

### 5.5 Global Seeding and Reproducibility

`set_seed(42)` is called at the start of `train()`, setting `random.seed(42)` and `np.random.seed(42)`. The first episode uses `env.reset(seed=42)`; later episodes use `reset()` without a seed for variety. This keeps runs reproducible while allowing diverse episode sequences. The codebase uses type hints, docstrings, and a clear separation (env, agent, heuristic, train, plots) consistent with maintainable, documentable research code.

---

## 6. Results & Graphical Analysis

The following interprets the **plots** produced by `plots.py` from `training_stats.npz` (after a full training run).

### 6.1 Training Win Rate vs. Evaluation Win Rate

- **Training win rate** (e.g. moving average over 1,000 episodes): Proportion of episodes in which the agent won, **across the current mixture** of random, heuristic, and self-play. This is **noisy** and **biased** by the curriculum (e.g. high when mostly random, variable when heuristic/self-play dominate).
- **Evaluation win rate** (vs random and vs heuristic at each 1,000-episode checkpoint): **Smooth**, **decoupled** curves with no exploration. They show how the **greedy policy** improves against fixed opponents. The **gap** between training and evaluation curves illustrates that training metrics are not reliable measures of true strength; the evaluation curves are the appropriate benchmark.

### 6.2 Game Length Over Training

Episode length (e.g. moving average of environment steps) often **increases** when entering the **self-play phase**: both sides play more carefully, defenses improve, and games last longer. Earlier phases (vs random/heuristic) can show shorter games (quick wins). So the game-length curve is consistent with a shift toward more strategic, defensive play in Phase 2.

### 6.3 Player 1 vs. Player 2 Disparity

The **P1 vs P2 evaluation** plot (win rate as P1 vs heuristic and as P2 vs heuristic) typically shows **asymmetric** learning: e.g. P1 (first mover) may reach high win rates sooner than P2. The **dynamic role assignment** (Section 5.3) is designed to correct this by training more as P2 when P2 lags. Over time, both curves should improve, with P2 catching up as the curriculum and role bias take effect.

### 6.4 Q-Table State-Space Growth

The **Q-table size** (number of state–action entries) over checkpoints usually shows:

- **Rapid growth** early (Phase 0): many new states and actions visited against random play.
- **Slower growth or plateau** in the heuristic phase: fewer new positions once the agent has seen the main heuristic patterns.
- **Further growth** in self-play: new positions and tactics appear as the agent plays against past versions of itself. Canonical state and hashing keep the table from exploding compared to a naive 6×6 full-board representation.

### 6.5 Final Performance Distribution

The **performance distribution** plot (e.g. stacked bar of win/draw/loss for Random, Heuristic, and Q-Learning agents vs a common opponent) shows that the **trained Q-learning agent** achieves a higher win proportion and lower loss proportion than the **heuristic** baseline when both are evaluated under the same conditions, demonstrating that the combination of tabular Q-learning and curriculum self-play yields a policy that **dominates** the fixed heuristic.

---

## 7. Conclusion

This project demonstrates that a **tabular Q-learning** agent can learn to play 6×6 Checkers at a level **above** a priority-based heuristic when trained with:

1. A **canonical, reduced state representation** (18 dark squares, board flip for symmetry) to shrink the effective state space and accelerate learning.
2. **Backward-pass** Q-updates over full episodes with state–action-dependent step sizes and legal-action constraints.
3. **State-dependent \(\varepsilon\)-greedy** exploration so that exploitation increases in frequently visited states.
4. A **three-phase curriculum** (random → heuristic → self-play) with phase advancement based on **decoupled** heuristic-benchmark win rates.
5. A **historical and recent opponent pool** for self-play to avoid catastrophic forgetting and to stabilize training.
6. **Alternating colors** and **dynamic role assignment** so that the agent learns strong play as both Player 1 and Player 2.
7. **Decoupled evaluation** (greedy, fixed opponents, no updates) to report true policy performance.

The result is an agent that surpasses the heuristic baseline and exhibits stable, interpretable learning dynamics, demonstrating that tabular Q-learning combined with curriculum self-play and careful state representation is effective for this 6×6 Checkers domain.
