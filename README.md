## Project Overview

This project implements a **custom 6x6 Checkers environment** using **Gymnasium**, along with:

- A **Tabular Q-Learning agent** with:
  - Canonical, symmetry-normalized state representation.
  - Strict action masking over legal moves.
  - State-dependent exploration (epsilon) and per–state-action learning rates.
- A **Priority Heuristic agent** that follows a rule-based strategy (capture, promotion, edge safety, center control, etc.).
- A **curriculum-based training loop** that:
  - Starts against mostly random opponents.
  - Progresses to heuristic opponents.
  - Ends in self-play against historical versions of the trained agent.
- A full plotting pipeline to visualize learning curves, Q-table growth, game length trends, and performance distributions.

### Environment

- **Board**: 6x6 grid.
- **Pieces**:
  - `0` = empty  
  - `1` = player1 man  
  - `2` = player2 man  
  - `3` = player1 king  
  - `4` = player2 king  
- **Observation** (`checkers_env.Checkers6x6Env`):
  - Dict with:
    - `"board"`: `np.ndarray` of shape `(6, 6)` with piece encodings.
    - `"current_player"`: `0` (Player 1) or `1` (Player 2).
- **Action space**:
  - `MultiDiscrete([6, 6, 6, 6])` representing `(start_row, start_col, end_row, end_col)`.
  - Environment internally enforces:
    - Diagonal moves only.
    - Men move forward; kings move both directions.
    - Single captures and mandatory capture rules.
- **Rewards (normalized)**:
  - Win: **+1.0**
  - Loss: **-1.0**
  - Draw: **0.0**
  - Capture: **+0.1**
  - Promotion: **+0.15**
  - Step penalty: **-0.005** (per move)  
  These smaller magnitudes keep Q-values numerically stable.

## Requirements & Installation

### 1. Create and activate a Python virtual environment

Using **Python 3.9+** is recommended.

```bash
python -m venv .venv
```

- On **Windows (PowerShell)**:

```bash
.\.venv\Scripts\Activate.ps1
```

- On **macOS / Linux**:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

A minimal `requirements.txt` is provided. From the project root:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

The key packages are:

- `numpy`
- `gymnasium`
- `matplotlib`

If you prefer manual installation:

```bash
pip install numpy gymnasium matplotlib
```

## File Structure

Below is a glossary of the core files in this project:

- **`checkers_env.py`**
  - Defines the **`Checkers6x6Env`** Gymnasium environment.
  - Handles:
    - Board initialization and legal move generation.
    - Mandatory captures, promotions, and game termination.
    - Normalized reward structure.
  - Provides a `make_env()` helper.

- **`q_agent.py`**
  - Implements the **`QLearningAgent`**:
    - Tabular Q-table (`Dict[(state, action), value]`).
    - Canonical state encoding using board symmetry:
      - If `current_player == 1`, the board is flipped and piece IDs swapped to a Player-0 perspective.
      - Only the 18 playable dark squares are encoded; `current_player` is dropped from the state key.
    - Dynamic learning rate per state-action:
      - `alpha(s,a) = max(0.05, 1 / sqrt(N(s,a) + 1))`.
    - State-dependent exploration:
      - Tracks `state_visit_counts` and uses
        - `epsilon(s) = N0 / (N0 + state_visit_counts.get(s, 0))` with `N0 = 100`.
    - Strict legal-action masking:
      - `epsilon_greedy_policy(state, legal_actions)` and `greedy_action(state, legal_actions)` only consider legal moves.

- **`heuristic_agent.py`**
  - Implements a **Priority-based heuristic agent**:
    - Priorities moves in this order:
      1. Captures.
      2. Promotions.
      3. Edge safety.
      4. Center advancement.
      5. Random fallback among legal moves.
  - Used for curriculum training and as a baseline opponent.

- **`train.py`**
  - Main **training script**:
    - Uses `Checkers6x6Env` and `QLearningAgent`.
    - Runs **100,000 episodes** by default.
    - Implements:
      - **Symmetry-aware action mapping** for both Player 0 and Player 1:
        - Coordinates are flipped when agent plays as Player 2.
      - **State-dependent epsilon-greedy** (epsilon handled inside the agent).
      - **Performance-based curriculum**:
        - Tracks evaluation win rate every 1,000 episodes.
        - Progresses through phases (Random → Heuristic → Self-play) when combined eval win rate > 80%.
      - **Historical opponent pool**:
        - Stores up to 5 snapshot Q-tables for self-play opponents (FIFO).
      - **Evaluation loop**:
        - Records clean eval win rates as Player 1 and Player 2 every 1,000 episodes.
      - **Artifacts**:
        - `q_table.pkl`: pickled Q-table.
        - `training_stats.npz`: training and evaluation metrics.

- **`play.py`**
  - Evaluation / visualization script:
    - Loads `q_table.pkl` and runs games in the terminal.
    - Q-agent can be **Player 1 or Player 2**:
      - Uses the same canonical perspective logic and coordinate flips as training.
    - Opponent is either:
      - Random (`opponent_type="random"`) or
      - Heuristic (`opponent_type="heuristic"`).
    - Provides a simple text-based board render with optional delay between moves.

- **`plots.py`**
  - Offline analysis / plotting utilities:
    - Reads `training_stats.npz`.
    - Generates:
      - Training vs. evaluation win-rate curves.
      - Q-table size growth over time.
      - Moving-average episode lengths.
      - Evaluation win rates as Player 1 vs Player 2.
      - Performance distribution (Random vs Heuristic vs Q-Learning agents).

## How to Run (Reproducibility Steps)

### Step 1: Train the Q-Learning Agent

From the project root (with the virtual environment activated):

```bash
python train.py
```

This will:

- Initialize the environment and Q-agent with global seed 42 for reproducibility.
- Run the training loop with:
  - State-dependent epsilon-greedy exploration.
  - Performance-based curriculum over opponent types.
  - Symmetry-normalized states and actions.
- Periodically evaluate the agent and store metrics.
- Write the following artifacts on completion:
  - `q_table.pkl`
  - `training_stats.npz`

You can adjust hyperparameters by editing `train()` in `train.py` (e.g., `num_episodes`, `gamma`).

### Step 2: Watch the Trained Agent Play

After training completes, you can watch the agent play against a random or heuristic opponent.

Run:

```bash
python play.py
```

By default, this will:

- Load `q_table.pkl`.
- Create a `Checkers6x6Env`.
- Run 500 episodes with the agent as Player 1 vs. a random opponent, rendering each move in the terminal.

To change the opponent type and other arguments, edit the `if __name__ == "__main__":` block in `play.py`, for example:

```python
if __name__ == "__main__":
    # Q-agent vs random opponent
    evaluate(num_episodes=500, opponent_type="random", render=True)

    # Q-agent vs heuristic opponent
    # evaluate(num_episodes=500, opponent_type="heuristic", render=True)
```

### Step 3: Generate Plots

Once `training_stats.npz` exists, generate all analysis plots:

```bash
python plots.py
```

This will save several `.png` files in the project root, including:

- `learning_curve_win_rate.png`
- `state_space_growth.png`
- `game_length.png`
- `eval_p1_vs_p2_win_rates.png`
- `performance_distribution.png`

These images are suitable for inclusion in reports or GitHub README visuals.

## Agent Features & Curriculum

### Tabular Q-Learning Agent

Key features of the Q-agent (`q_agent.py`):

- **Canonical state representation**:
  - Observations are transformed so that all states are viewed from Player 0’s perspective:
    - If it’s Player 2’s turn, the board is flipped vertically and horizontally and piece IDs are swapped:
      - `1 ↔ 2`, `3 ↔ 4`.
  - Only the 18 playable dark squares are encoded.
  - `current_player` is dropped from the state key.
- **Dynamic learning rate**:
  - Each `(state, action)` pair maintains a visit count `N(s,a)`:
    - `alpha(s,a) = max(0.05, 1 / sqrt(N(s,a) + 1))`.
- **State-dependent exploration (epsilon)**:
  - For each state `s`, `state_visit_counts[s]` is maintained.
  - Epsilon for that state is:
    - `epsilon(s) = N0 / (N0 + visit_count(s))` with `N0 = 100`.
  - `epsilon_greedy_policy(state, legal_actions)` uses this `epsilon(s)` to decide between exploration and exploitation.
- **Strict action masking**:
  - The agent is always passed `legal_actions` derived from `env.get_legal_actions(player=...)`, guaranteeing:
    - No invalid actions.
    - Consistency with mandatory capture rules.

### Priority Heuristic Agent

The heuristic agent (`heuristic_agent.py`) uses a **hand-coded priority strategy**:

1. **Captures**: Any move that captures an opponent piece.
2. **Promotions**: Moves that reach the promotion row to gain a king.
3. **Edge safety**: Moves placing a piece on board edges to avoid being captured.
4. **Center control**: Moves that increase presence in the central region.
5. **Random fallback** among remaining legal moves.

This agent provides a strong, interpretable baseline opponent.

### Curriculum & Opponent Types

Training uses a **performance-based auto-curriculum**:

- Opponent types:
  - `"random"`: random legal moves.
  - `"heuristic"`: Priority heuristic agent.
  - `"self_play"`: the Q-agent plays against historical snapshots of itself.

- Curriculum phases:
  - **Phase 0**: 80% Random, 20% Heuristic, 0% Self-play.
  - **Phase 1**: 10% Random, 80% Heuristic, 10% Self-play.
  - **Phase 2**: 0% Random, 10% Heuristic, 90% Self-play.

- Progression rule:
  - Every 1,000 episodes, the agent is evaluated.
  - If the combined evaluation win rate (averaged over playing as Player 1 and Player 2) exceeds **80%**, the curriculum phase increments (up to Phase 2).

### Symmetry-Aware Action Mapping

- During training and evaluation:
  - When the agent is **Player 1**, legal actions and chosen actions are **flipped** relative to the canonical Player-0 perspective before/after passing them to/from the Q-agent.
  - This ensures:
    - A **single Q-table** generalizes over both player roles.
    - Training data is maximally reused via board and action symmetries.

---


