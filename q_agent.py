from __future__ import annotations

import random
from typing import Any, Dict, Hashable, List, Tuple, Optional

import numpy as np
from gymnasium.spaces import MultiDiscrete


State = Hashable  # we will typically use a tuple representation
Action = Tuple[int, int, int, int]  # (sr, sc, er, ec)


def observation_to_state(observation: Dict[str, Any]) -> State:
    """
    Convert an environment observation into a hashable state.

    Observation format (from Checkers6x6Env):
        {
            "board": np.ndarray shape (6,6), dtype=int,
            "current_player": int (0 or 1),
        }

    We normalize the board to always be from Player 0's perspective:
    - If current_player == 1, flip the board vertically and horizontally
      and swap piece IDs (1<->2, 3<->4). This makes the position
      strategically equivalent to a Player 0-to-move view.
    - Then we keep only the 18 playable dark squares where (r + c) % 2 == 1.

    current_player is NOT included in the state tuple; all states are
    represented from the canonical Player 0 perspective.
    """
    board_2d = np.array(observation["board"], dtype=np.int8)
    current_player = int(observation["current_player"])

    # Normalize to Player 0 perspective if current_player == 1
    if current_player == 1:
        flipped = np.flip(board_2d)  # flip vertically and horizontally
        # Map piece IDs: 0->0, 1->2, 2->1, 3->4, 4->3
        mapping = np.array([0, 2, 1, 4, 3], dtype=np.int8)
        board_2d = mapping[flipped]

    playable: List[int] = []
    rows, cols = board_2d.shape
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 1:
                playable.append(int(board_2d[r, c]))
    return tuple(playable)


class QLearningAgent:
    """
    Tabular Q-learning agent with a dictionary-based Q-table and
    a backward-pass (episode-based) update.

    Q-table keys: (state, action) where
        state  = hashable representation (e.g., tuple from observation_to_state)
        action = (sr, sc, er, ec) tuple
    """

    def __init__(self, action_space: MultiDiscrete):
        """
        Parameters
        ----------
        action_space:
            The MultiDiscrete([6,6,6,6]) action space from the environment.
            Used to enumerate all possible actions.
        """
        self.action_space: MultiDiscrete = action_space
        self.q_table: Dict[Tuple[State, Action], float] = {}
        # Visit counts for state-action pairs for dynamic learning rates
        self.visit_counts: Dict[Tuple[State, Action], int] = {}
        # Visit counts per state for state-dependent exploration
        self.state_visit_counts: Dict[State, int] = {}

        # Precompute the full discrete action set (6^4 = 1296 actions)
        nvec = action_space.nvec
        self.all_actions: List[Action] = [
            (sr, sc, er, ec)
            for sr in range(int(nvec[0]))
            for sc in range(int(nvec[1]))
            for er in range(int(nvec[2]))
            for ec in range(int(nvec[3]))
        ]

    # ------------------------------------------------------------------
    # Q-table helpers
    # ------------------------------------------------------------------
    def get_q_value(self, state: State, action: Action) -> float:
        """
        Return Q(s,a) if present, else 0.0.
        """
        return self.q_table.get((state, action), 0.0)

    def _set_q_value(self, state: State, action: Action, value: float) -> None:
        self.q_table[(state, action)] = value

    def _max_q_value(self, state: Optional[State], legal_actions: List[Action]) -> float:
        """
        Maximum Q-value over legal actions for a given state.
        If state is None or there are no legal actions, returns 0.0.
        """
        if state is None or not legal_actions:
            return 0.0
        return max(self.get_q_value(state, a) for a in legal_actions)

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------
    def epsilon_greedy_policy(self, state: State, legal_actions: List[Action]) -> Action:
        """
        Epsilon-greedy action selection.

        Epsilon is state-dependent:
            epsilon(s) = N0 / (N0 + visit_count(s))
        where N0 is a constant (100) and visit_count(s) is how many times
        the agent has acted from this state.

        With probability epsilon(s), choose a random action.
        Otherwise, choose the action with the highest Q-value for this state,
        restricted to the provided legal_actions.
        """
        if not legal_actions:
            raise ValueError("epsilon_greedy_policy called with empty legal_actions.")

        # Update state visit count and compute epsilon(s)
        N0 = 100.0
        n_s = self.state_visit_counts.get(state, 0) + 1
        self.state_visit_counts[state] = n_s
        epsilon = N0 / (N0 + float(n_s))

        if random.random() < epsilon:
            return random.choice(legal_actions)

        # Exploit: pick argmax_a Q(s,a)
        best_action: Optional[Action] = None
        best_value = -float("inf")
        for action in legal_actions:
            q = self.get_q_value(state, action)
            if q > best_value:
                best_value = q
                best_action = action

        # Fallback (should not happen) to random
        if best_action is None:
            best_action = random.choice(legal_actions)
        return best_action

    # ------------------------------------------------------------------
    # Backward-pass update
    # ------------------------------------------------------------------
    def backward_pass_update(
        self,
        episode_memory: List[Tuple[State, Action, float, Optional[State], List[Action]]],
        gamma: float,
    ) -> None:
        """
        Perform Q-learning updates over an episode in reverse (backward pass).

        episode_memory:
            List of (state, action, reward, next_state, legal_next_actions) tuples
            collected in chronological order during the episode.

        We iterate in reverse order:
            Q(s,a) = Q(s,a) + alpha * [reward + gamma * max_a' Q(next_s, a') - Q(s,a)]
        where max is taken only over legal_next_actions.
        """
        for state, action, reward, next_state, legal_next_actions in reversed(episode_memory):
            old_q = self.get_q_value(state, action)
            max_next_q = self._max_q_value(next_state, legal_next_actions)
            target = reward + gamma * max_next_q

            # Dynamic learning rate based on visit count with floor:
            # alpha(s,a) = max(0.005, 1 / sqrt(N(s,a) + 1))
            key = (state, action)
            n = self.visit_counts.get(key, 0) + 1
            self.visit_counts[key] = n
            alpha_sa = max(0.005, 1.0 / np.sqrt(float(n)))

            new_q = old_q + alpha_sa * (target - old_q)
            self._set_q_value(state, action, new_q)

    # ------------------------------------------------------------------
    # Convenience for evaluation
    # ------------------------------------------------------------------
    def greedy_action(self, state: State, legal_actions: List[Action]) -> Action:
        """
        Pure exploitation: return argmax_a Q(s,a).
        """
        if not legal_actions:
            raise ValueError("greedy_action called with empty legal_actions.")

        best_action: Optional[Action] = None
        best_value = -float("inf")
        for action in legal_actions:
            q = self.get_q_value(state, action)
            if q > best_value:
                best_value = q
                best_action = action

        # Fallback to random (should not happen)
        if best_action is None:
            best_action = random.choice(legal_actions)
        return best_action
