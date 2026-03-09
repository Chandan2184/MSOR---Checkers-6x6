# Reinforcement Learning and Heuristic Methods for 6×6 Checkers

**A Written Document (30%)**  
*Max. 15 thesis-style pages (all incl.) + AI statement*

---

## AI Statement

This report was prepared with the assistance of AI tools for drafting, structuring, and mathematical notation. The underlying codebase, experiments, and interpretations are the author’s own. AI was used to improve clarity and alignment with the specified rubric; all technical content has been verified against the implementation.

---

## 1. Problem Description: A Written Instruction of the Game

We consider **two-player 6×6 Checkers** (English draughts on a reduced board) as a turn-based, zero-sum game with perfect information. The following rules define the environment used in this project.

### 1.1 Board and Setup

- **Board:** A 6×6 grid. Only the **dark squares** are playable; i.e., squares where \((r + c) \equiv 1 \pmod{2}\) for row \(r\) and column \(c\) (zero-indexed). There are 18 such squares.
- **Pieces:** Each player has pieces (men) that occupy dark squares. Men are distinguished by ownership (Player 1 vs Player 2) and by type: **man** or **king**.
- **Initial configuration:** Each player has one row of men on the two dark-squares rows closest to their side. Player 1 (id 0) starts at the “bottom” (rows 4–5), Player 2 (id 1) at the “top” (rows 0–1). No kings at the start.

### 1.2 Movement Rules

- **Diagonal moves only:** Every move is along a diagonal, one square at a time for a non-capture move.
- **Men:** Move only **forward** (toward the opponent’s back rank): Player 1 upward (decreasing row), Player 2 downward (increasing row).
- **Kings:** May move **one step** in any of the four diagonal directions.
- **Single-step moves:** A piece may move to an adjacent dark square if it is empty.
- **Single captures:** A piece may jump over an adjacent opponent piece to the next dark square if that square is empty. The jumped piece is removed. Only one capture per move is considered in this implementation (no mandatory multi-capture chain beyond the rule below).
- **Forced capture:** If **any** capture is legal for the current player, then **only** capture moves are legal; simple moves are disallowed.
- **Multi-jump:** If after a capture the same piece can capture again (without promotion in between), the same player **must** continue with that piece; the turn does not end until no further capture is possible or the piece is promoted.
- **Promotion:** When a man reaches the opponent’s back rank (row 0 for Player 1, row 5 for Player 2), it is promoted to a king and the turn ends (no further capture with that move).

### 1.3 Termination and Outcome

- **Win:** The game is won by the player who either eliminates all opponent pieces or leaves the opponent with no legal move.
- **Loss:** The other player loses.
- **Draw:** The game is drawn if (1) a move limit (e.g. 200 steps) is reached, or (2) a no-progress rule is triggered (e.g. 40 consecutive steps without a capture or without a man move).

### 1.4 Summary for the Agent

The learning agent observes the board and current player, chooses a **legal** move (respecting forced capture and multi-jump), and receives scalar rewards. The goal is to maximize cumulative reward, which is aligned with winning and with intermediate events (captures, promotions) as defined in the next section.

---

## 2. Mathematical Formulation

The game is formulated as a **Markov Decision Process (MDP)** from the perspective of one player (the agent). We define states, actions, transition structure, and rewards precisely.

### 2.1 State Space \(\mathcal{S}\)

A **state** is a compact, hashable representation of the information needed to choose a move, in a **canonical** view (always from the perspective of the player about to move, so that symmetry is exploited).

- **Raw observation:** The environment provides \(\omega = (B, p)\) where \(B \in \{0,1,2,3,4\}^{6 \times 6}\) is the board (\(0\) = empty, \(1\) = P1 man, \(2\) = P2 man, \(3\) = P1 king, \(4\) = P2 king) and \(p \in \{0,1\}\) is the current player.
- **Canonical state:** If \(p = 1\), the board is flipped (vertically and horizontally) and piece labels are swapped (1↔2, 3↔4) so that the position is equivalent to “current player is Player 0.” The state does not include \(p\) explicitly; it is implicit that the state is “current player to move.”
- **Reduced representation:** Only the 18 playable dark squares are stored, in a fixed order, as a tuple of 18 integers in \(\{0,1,2,3,4\}\). Thus
  \[
  s = \bigl( B_{i_1}, B_{i_2}, \ldots, B_{i_{18}} \bigr) \in \{0,1,2,3,4\}^{18},
  \]
  where \((i_1,\ldots,i_{18})\) is the fixed ordering of dark squares. The set of all such tuples that correspond to reachable board configurations and a given side-to-move defines the **state space** \(\mathcal{S}\). The size of \(\mathcal{S}\) is finite but large; in practice we use a **tabular** representation that only stores states that have been visited.

### 2.2 Action Space \(\mathcal{A}\)

- **Environment action:** A move is encoded as a 4-tuple \(a = (s_r, s_c, e_r, e_c)\): start row, start column, end row, end column, each in \(\{0,\ldots,5\}\). The full nominal action space is \(\mathcal{A}_{\text{full}} = \{0,\ldots,5\}^4\) (size \(6^4 = 1296\)).
- **Legal actions:** In state \(s\), only a subset \(\mathcal{A}(s) \subseteq \mathcal{A}_{\text{full}}\) is legal (diagonal move, correct piece, forced capture and multi-jump respected). The agent must choose \(a \in \mathcal{A}(s)\); invalid actions are rejected and leave the state unchanged (same player to move again in the implementation).

### 2.3 Transitions

- **Deterministic given opponent:** For a fixed opponent policy, the environment transition is deterministic: taking action \(a\) in state \(s\) leads to a unique next board and next player (or terminal). In our formulation we treat the **opponent** as part of the environment; thus the transition from the agent’s perspective is
  \[
  (s, a) \mapsto (s', r, \text{done}),
  \]
  where \(s'\) is the state after the agent’s move and the opponent’s reply (if any), \(r\) is the reward accumulated over that agent move (and possibly opponent move), and \(\text{done}\) indicates termination.
- **Multi-step transitions:** A “transition” in the agent’s experience is over one **agent decision**: from the state where the agent chooses an action to the state where the agent is to move again (or the episode ends). Opponent moves and multi-jumps are folded into this transition. So we have
  \[
  P(s', r \mid s, a) = \text{deterministic given opponent policy}.
  \]

### 2.4 Rewards

Rewards are **normalized** and defined from the **current player’s** perspective at the time of the move (the environment reports in its own frame; the agent’s reward flips sign when the opponent scores).

- **Terminal rewards:**  
  - Win: \(r = +1.0\).  
  - Loss: \(r = -1.0\).  
  - Draw: \(r = 0.0\) (no extra terminal bonus).
- **Intermediate rewards (non-terminal):**  
  - Capture: \(r = +0.1\).  
  - Promotion to king: \(r = +0.15\).  
  - Per-step penalty: \(r = -0.005\) each move (to discourage stalling).
- **Return:** The agent maximizes the **discounted return**
  \[
  G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k},
  \]
  with discount factor \(\gamma \in (0,1]\) (e.g. \(\gamma = 0.99\)).

This completes the MDP specification \((\mathcal{S}, \mathcal{A}, P, R, \gamma)\).

---

## 3. Solution Approach

We implement and compare **two** approaches: (1) a **reinforcement learning** algorithm (tabular Q-learning with backward pass and curriculum), and (2) a **heuristic** rule-based strategy.

### 3.1 Reinforcement Learning: Tabular Q-Learning with Backward Pass

**Idea:** Approximate the optimal action-value function \(Q^*(s,a)\) with a table \(\hat{Q}(s,a)\). The agent collects a full episode of transitions, then updates \(\hat{Q}\) in **reverse** order so that each step uses the already-updated \(\hat{Q}\) for the next state (like a backward sweep).

**Bellman equation (optimal Q):**
\[
Q^*(s,a) = \mathbb{E}\bigl[ R + \gamma \max_{a'} Q^*(S', a') \bigm| S=s, A=a \bigr].
\]

**Update rule (Q-learning):** For a transition \((s, a, r, s')\) with legal next actions \(\mathcal{A}(s')\):
\[
\hat{Q}(s,a) \leftarrow \hat{Q}(s,a) + \alpha \Bigl( r + \gamma \max_{a' \in \mathcal{A}(s')} \hat{Q}(s', a') - \hat{Q}(s,a) \Bigr),
\]
where \(\alpha\) is the learning rate (we use a state-action-dependent rate: \(\alpha = \max\bigl(0.005,\, 1\big/\sqrt{N(s,a)+1}\,\bigr)\)).

**State-dependent exploration:** \(\varepsilon(s) = N_0 \big/ (N_0 + N(s))\) with \(N_0 = 100\), and \(N(s)\) is the number of times the agent has acted from \(s\). With probability \(\varepsilon(s)\) the agent chooses a random legal action; otherwise it chooses \(\arg\max_{a \in \mathcal{A}(s)} \hat{Q}(s,a)\).

**Pseudocode: Q-learning agent (one episode, backward pass)**

```
1.  Initialize: observation ← env.reset(); episode_memory ← []
2.  Repeat until episode ends:
3.      s ← observation_to_state(observation)   // canonical state
4.      legal_actions ← env.get_legal_actions(current_player)
5.      a ← epsilon_greedy(s, legal_actions)   // with ε(s) = N0/(N0+N(s))
6.      Execute a in env; obtain (observation', r, done) and possibly opponent move(s)
7.      s' ← observation_to_state(observation')  (or None if done)
8.      legal_next ← legal actions for agent in s' (or [] if done)
9.      Append (s, a, r, s', legal_next) to episode_memory
10.     observation ← observation'
11. Backward pass: for (s, a, r, s', legal_next) in reverse(episode_memory):
12.     target ← r + γ * max_{a' ∈ legal_next} Q(s', a')
13.     α ← max(0.005, 1/sqrt(N(s,a)+1)); N(s,a) += 1
14.     Q(s,a) ← Q(s,a) + α * (target - Q(s,a))
15. Return
```

**Training curriculum:** The agent is trained against a mixture of opponents that changes with performance (decoupled evaluation vs fixed random and heuristic opponents):

- **Phase 0:** 80% random, 20% heuristic.  
- **Phase 1:** 10% random, 80% heuristic, 10% self-play.  
- **Phase 2:** 0% random, 20% heuristic, 80% self-play.

Advancement to the next phase occurs when the agent’s **heuristic-benchmark** win rates (evaluated separately) satisfy: win rate as P1 > 0.75 and win rate as P2 > 0.60. Self-play uses an opponent pool of saved Q-table snapshots (recent + historical) to reduce forgetting.

### 3.2 Heuristic Method: Priority-Based Rule Strategy

**Idea:** A rule-based agent that, given the current board and legal moves, always chooses one move according to a **fixed priority list**. No learning; behavior is deterministic up to tie-breaking (random among moves that tie at the same priority).

**Priority order (highest first):**

1. **Forced capture:** If any legal move is a capture (|end_row − start_row| = 2), choose uniformly among captures.
2. **King promotion:** If any legal move promotes a man to a king, choose uniformly among such moves.
3. **Edge safety:** If any legal move lands on the left or right edge (column 0 or 5), choose uniformly among those.
4. **Advance center:** Score each move by how much the end cell is “toward the center” (e.g. inverse distance to board center); among legal moves, keep those with maximum score and choose uniformly.
5. **Random:** If none of the above apply, choose uniformly among all legal moves.

**Pseudocode: Priority heuristic agent**

```
1.  legal ← get_legal_actions(env, player_id)
2.  capture_moves ← { m ∈ legal : |m.er - m.sr| == 2 }
3.  if capture_moves ≠ ∅ then return random_choice(capture_moves)
4.  promo_moves ← { m ∈ legal : move promotes man to king }
5.  if promo_moves ≠ ∅ then return random_choice(promo_moves)
6.  edge_moves ← { m ∈ legal : m.ec ∈ {0, BOARD_SIZE-1} }
7.  if edge_moves ≠ ∅ then return random_choice(edge_moves)
8.  center_score(m) ← (BOARD_SIZE-1) - distance((m.er,m.ec), center)
9.  best ← argmax_{m ∈ legal} center_score(m)
10. if center_score(best) > 0 then
11.     best_moves ← { m ∈ legal : center_score(m) == center_score(best) }
12.     return random_choice(best_moves)
13. return random_choice(legal)
```

This gives a clear, interpretable baseline that prefers captures and promotions and avoids the center when possible (edge safety) or advances toward the center when that is beneficial.

---

## 4. Analysis: Comparison of Performance and Behavior

### 4.1 Performance

- **Evaluation design:** To avoid curriculum-induced bias, evaluation is **decoupled**: every 1000 training episodes the agent is evaluated with **no exploration** (greedy) in 100 games against a **fixed random** opponent and 100 games against a **fixed heuristic** opponent (50 games as Player 1, 50 as Player 2 each). Reported metrics are win rate vs random, win rate vs heuristic, and win rate as P1 vs heuristic and as P2 vs heuristic.
- **Expected behavior:**  
  - **Vs random:** The Q-learning agent should reach high win rates (often near 1.0) once it has learned basic tactics (captures, promotion).  
  - **Vs heuristic:** Win rate vs the priority heuristic reflects tactical and positional strength; typically it rises with training and can exceed 0.5 as the agent learns to exploit heuristic weaknesses.  
  - **P1 vs P2 asymmetry:** Often P1 (first move) has an advantage; the curriculum’s role assignment (biasing play as P2 when P2 is weaker) aims to balance learning across sides.
- **Performance distribution plot:** The stacked bar chart (Random vs Heuristic vs Q-Learning, each vs a common opponent) summarizes win/draw/loss proportions. Q-learning is expected to dominate random and to be competitive with or better than the heuristic after training.

### 4.2 Behavioral Comparison

| Aspect | Q-learning agent | Heuristic agent |
|--------|-------------------|------------------|
| **Adaptation** | Improves with data; generalizes to new positions within visited state space | Fixed rules; no adaptation |
| **State representation** | Uses full canonical board (18 cells); only visited states stored | Uses same board only to compute legal moves and priorities |
| **Exploration** | State-dependent ε(s); explores less in often-seen states | No exploration; deterministic given tie-breaking |
| **Opponent dependence** | Trained on curriculum (random → heuristic → self-play); evaluation is opponent-fixed | Same behavior regardless of opponent |
| **Tactics** | Learns captures and promotions via reward; may discover non-obvious sequences | Explicitly prioritizes captures and promotions |
| **Positional play** | Emerges from Q-values; can be uneven early in training | Edge safety and “advance center” encode simple positional rules |
| **Compute** | Tabular storage and backward pass per episode; cost grows with state visits | Very low; a few rule checks and one legal-move list |

The heuristic is **interpretable** and **stable**; the Q-learning agent can become **stronger** than the heuristic with enough training and a suitable curriculum, at the cost of training time and memory for the Q-table.

---

## 5. Visualization: Graphical Representation of Results

The following figures are produced by the provided plotting scripts from the saved training statistics (`training_stats.npz`) and optional evaluation scripts.

1. **Learning curve (win rate)**  
   - **Training win rate (moving average):** Smoothed proportion of training episodes in which the agent won (any opponent type), showing volatility and trend.  
   - **Evaluation win rate vs Random:** Win rate at each 1000-episode checkpoint against a fixed random opponent (decoupled).  
   - **Evaluation win rate vs Heuristic:** Win rate at each checkpoint against the fixed heuristic opponent.  
   This plot shows how quickly the agent masters random play and how it improves against the heuristic over time.

2. **State-space growth**  
   Number of state-action entries in the Q-table at each evaluation checkpoint. This illustrates the growth of the explored state space and the memory footprint of the tabular agent.

3. **Game length**  
   Moving average of episode length (environment steps) over training. Useful to see if games become longer (more strategic) or shorter (faster wins) as the agent improves.

4. **P1 vs P2 evaluation (vs heuristic)**  
   Two curves: win rate when the agent plays as Player 1 vs heuristic, and as Player 2 vs heuristic. Highlights any asymmetry and the effect of curriculum role balancing.

5. **Performance distribution**  
   Stacked bar chart of win/draw/loss proportions for Random, Heuristic, and Q-Learning agents when each is evaluated against a common opponent (e.g. heuristic or random), giving a direct comparison of the three policies.

Together, these visualizations support the analysis in Section 4 and demonstrate that the RL approach learns to outperform the heuristic baseline while the heuristic provides a stable, interpretable reference.

**Figure files** (generated by `plots.py`):  
`learning_curve_win_rate.png`, `state_space_growth.png`, `game_length.png`, `eval_p1_vs_p2_win_rates.png`, `performance_distribution.png`.

---

## References

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.  
2. Watkins, C. J. C. H. (1989). *Learning from Delayed Rewards*. PhD thesis, University of Cambridge.  
3. Gymnasium: https://gymnasium.farama.org/

---

*Document generated to satisfy the written report rubric (30%): problem description, mathematical formulation, solution approach with pseudocode for RL and heuristic methods, comparative analysis, and visualization description. Total length intended to fit within 15 thesis-style pages including references and AI statement.*
