from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np

from checkers_env import BOARD_SIZE, Checkers6x6Env


Move = Tuple[int, int, int, int]  # (sr, sc, er, ec)


class PriorityHeuristicAgent:
    """
    Rule-based "Priority-Based Agent" for 6x6 checkers.

    Priority order:
        1. Forced Capture      (any capture move)
        2. King Promotion      (move that promotes to king)
        3. Edge Safety         (move landing on columns 0 or 5)
        4. Advance Center      (move toward the center rows/cols)
        5. Random Move         (fallback)
    """

    def __init__(self, player_id: int):
        """
        player_id: 0 for player1, 1 for player2
        """
        self.player_id = player_id

    def select_move(self, env: Checkers6x6Env) -> Move:
        legal = env.get_legal_actions(player=self.player_id)
        if not legal:
            raise ValueError("No legal moves for heuristic agent.")

        # Precompute capture moves: moves where row distance == 2
        capture_moves = [m for m in legal if abs(m[2] - m[0]) == 2]
        if capture_moves:
            # Priority 1: any capture move (choose randomly among them)
            return random.choice(capture_moves)

        # Priority 2: king promotion
        promo_moves = [m for m in legal if self._is_promotion(env, m)]
        if promo_moves:
            return random.choice(promo_moves)

        # Priority 3: edge safety
        edge_moves = [m for m in legal if m[3] in (0, BOARD_SIZE - 1)]
        if edge_moves:
            return random.choice(edge_moves)

        # Priority 4: advance center
        center_moves = sorted(
            legal,
            key=lambda m: -self._center_score(m),
        )
        best_center_score = self._center_score(center_moves[0])
        if best_center_score > 0:
            # Keep only moves with max center score and pick randomly among them
            best_moves = [m for m in center_moves if self._center_score(m) == best_center_score]
            return random.choice(best_moves)

        # Priority 5: random
        return random.choice(legal)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _is_promotion(self, env: Checkers6x6Env, move: Move) -> bool:
        sr, sc, er, ec = move
        piece = int(env.board[sr, sc])
        # Men only
        if self.player_id == 0 and piece != 1:
            return False
        if self.player_id == 1 and piece != 2:
            return False

        if self.player_id == 0 and er == 0:
            return True
        if self.player_id == 1 and er == BOARD_SIZE - 1:
            return True
        return False

    @staticmethod
    def _center_score(move: Move) -> float:
        """Higher score for landing closer to the center of the board."""
        _, _, er, ec = move
        # Distance from board center (2.5, 2.5) for 6x6
        center_r = (BOARD_SIZE - 1) / 2.0
        center_c = (BOARD_SIZE - 1) / 2.0
        dist = np.sqrt((er - center_r) ** 2 + (ec - center_c) ** 2)
        # Invert distance so that closer to center => larger score
        return float((BOARD_SIZE - 1) - dist)
