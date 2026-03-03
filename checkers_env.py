import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import List, Tuple, Optional, Dict, Any


"""
Custom 6x6 Checkers environment for Gymnasium.

Board encoding (6x6 numpy array):
    0 = empty
    1 = player1 man
    2 = player2 man
    3 = player1 king
    4 = player2 king

Turn:
    0 = player1 to move
    1 = player2 to move

Action space:
    MultiDiscrete([6, 6, 6, 6]) representing
    (start_row, start_col, end_row, end_col)

Rules implemented:
    - Diagonal moves only
    - Men move forward only (toward opponent)
    - Kings move in both directions
    - Single-step moves and single captures
    - Forced capture: if any capture is available, only capture moves are legal
    - Promotion to king on reaching back rank

Rewards (normalized):
    +1.0    winning the game
    -1.0    losing the game
    0.0     drawing the game
    +0.1    capturing an opponent's piece
    -0.1    (implicitly for opponent) when its piece is captured
    +0.15   promotion to king
    -0.005  step penalty per turn

Invalid actions:
- State does not change
- Same player moves again
"""


BOARD_SIZE = 6


class Checkers6x6Env(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self, max_steps: int = 200):
        super().__init__()

        # Observation: board + current player
        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(
                    low=0,
                    high=4,
                    shape=(BOARD_SIZE, BOARD_SIZE),
                    dtype=np.int8,
                ),
                "current_player": spaces.Discrete(2),
            }
        )

        # Action: (start_row, start_col, end_row, end_col)
        self.action_space = spaces.MultiDiscrete([BOARD_SIZE, BOARD_SIZE, BOARD_SIZE, BOARD_SIZE])

        self.board: np.ndarray = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.current_player: int = 0  # 0 = player1, 1 = player2
        self.step_count: int = 0
        self.max_steps: int = max_steps

        # No-progress draw rule: counts consecutive non-capture, non-man moves
        self.no_progress_counter: int = 0
        # Maximum number of such steps before declaring a draw
        self.max_no_progress_steps: int = 40

        # Active piece for enforcing multi-jump sequences
        self.active_piece: Optional[Tuple[int, int]] = None

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self._init_board()
        self.current_player = 0
        self.step_count = 0
        self.no_progress_counter = 0
        self.active_piece = None
        self.no_progress_counter = 0
        observation = self._get_obs()
        info: Dict[str, Any] = {}
        return observation, info

    def step(self, action):
        """
        action: array-like of 4 ints (sr, sc, er, ec)
        """
        sr, sc, er, ec = self._normalize_action(action)

        terminated = False
        truncated = False
        reward = 0.0
        info: Dict[str, Any] = {}

        # Validate and apply move
        legal_moves, capture_moves = self._get_legal_moves(self.current_player)

        move = (sr, sc, er, ec)
        forced_capture = len(capture_moves) > 0

        if forced_capture:
            is_legal = move in capture_moves
        else:
            is_legal = move in legal_moves

        if not is_legal:
            # Invalid move: no state change, same player continues, no extra penalty
            observation = self._get_obs()
            info["invalid_action"] = True
            return observation, reward, terminated, truncated, info

        # Track the type of piece being moved for no-progress rule
        piece_moved = int(self.board[sr, sc])

        # Track the type of piece being moved for no-progress and multi-jump rules
        piece_moved = int(self.board[sr, sc])

        # Execute the move
        captured_piece = self._apply_move(move)
        if captured_piece:
            reward += 0.1

        # Promotion reward (computed inside _apply_move by return flag)
        # _apply_move also returns whether promotion happened.

        # Check promotion
        promoted = self._last_move_promoted
        if promoted:
            reward += 0.15

        # Update no-progress counter:
        # Reset on capture or when a regular man (1 or 2) moves; otherwise increment.
        if captured_piece or piece_moved in (1, 2):
            self.no_progress_counter = 0
        else:
            self.no_progress_counter += 1

        # Determine if additional captures are available for multi-jump
        more_captures = False
        if captured_piece and not promoted:
            # Temporarily restrict legal moves to the moved piece and look for captures
            prev_active = self.active_piece
            self.active_piece = (er, ec)
            _, further_captures = self._get_legal_moves(self.current_player)
            more_captures = len(further_captures) > 0
            self.active_piece = prev_active

        self.step_count += 1

        # Determine terminal condition
        winner: Optional[int] = None  # 0 or 1, or None for ongoing, -1 for draw

        opponent = 1 - self.current_player
        my_pieces = self._count_pieces(self.current_player)
        opp_pieces = self._count_pieces(opponent)

        opp_legal, opp_capture = self._get_legal_moves(opponent)
        opp_has_move = len(opp_legal) > 0 or len(opp_capture) > 0

        if opp_pieces == 0 or not opp_has_move:
            # Current player wins
            terminated = True
            winner = self.current_player
            reward += 1.0
        elif my_pieces == 0:
            # Current player loses (should be rare given capture rules)
            terminated = True
            winner = opponent
            reward += -1.0
        elif self.step_count >= self.max_steps:
            # Draw by move limit
            terminated = True
            winner = -1  # draw
            # No terminal bonus; cumulative reward so far stands
        elif self.no_progress_counter >= self.max_no_progress_steps:
            # Draw by no-progress rule
            terminated = True
            winner = -1  # draw

        # Step penalty every move (small negative to discourage stalling)
        reward += -0.005

        # Determine if the same piece must continue a multi-jump
        continue_turn = captured_piece and (not promoted) and more_captures and not terminated
        if continue_turn:
            # Lock the active piece for subsequent capture-only moves
            self.active_piece = (er, ec)
        else:
            self.active_piece = None
            if not terminated:
                # Switch to opponent if game is not over
                self.current_player = opponent

        observation = self._get_obs()
        info["winner"] = winner
        return observation, reward, terminated, truncated, info

    def render(self, mode: str = "ansi"):
        """
        Simple text-based render to stdout.
        """
        symbols = {
            0: ".",
            1: "x",  # player1 man
            2: "o",  # player2 man
            3: "X",  # player1 king
            4: "O",  # player2 king
        }

        lines = []
        for r in range(BOARD_SIZE):
            row_syms = []
            for c in range(BOARD_SIZE):
                val = int(self.board[r, c])
                row_syms.append(symbols.get(val, "?"))
            lines.append(" ".join(row_syms))
        board_str = "\n".join(lines)
        print(board_str)
        print(f"Current player: {self.current_player + 1}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _init_board(self):
        """
        Standard starting position for 6x6 checkers:
        - Pieces on dark squares only (where (r + c) % 2 == 1)
        - 2 rows of pieces per player

        Player2 (id=1) at the top (rows 0-1)
        Player1 (id=0) at the bottom (rows 4-5)
        """
        self.board.fill(0)
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if (r + c) % 2 == 0:
                    continue
                if r < 2:
                    # Top side: player2 men
                    self.board[r, c] = 2
                elif r > BOARD_SIZE - 3:
                    # Bottom side: player1 men
                    self.board[r, c] = 1
        self._last_move_promoted = False

    def _get_obs(self) -> Dict[str, Any]:
        return {
            "board": self.board.copy(),
            "current_player": int(self.current_player),
        }

    @staticmethod
    def _normalize_action(action) -> Tuple[int, int, int, int]:
        arr = np.array(action, dtype=int).reshape(4,)
        return int(arr[0]), int(arr[1]), int(arr[2]), int(arr[3])

    # ------------------------------------------------------------------
    # Move generation
    # ------------------------------------------------------------------
    def _get_legal_moves(
        self, player: int
    ) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]:
        """
        Returns (simple_moves, capture_moves) for the given player.
        Each move is (sr, sc, er, ec).
        """
        simple_moves: List[Tuple[int, int, int, int]] = []
        capture_moves: List[Tuple[int, int, int, int]] = []

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                piece = int(self.board[r, c])
                if not self._is_player_piece(player, piece):
                    continue

                is_king = self._is_king(piece)
                directions = self._move_directions(player, is_king)

                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    # Simple move
                    if self._on_board(nr, nc) and (nr + nc) % 2 == 1 and self.board[nr, nc] == 0:
                        simple_moves.append((r, c, nr, nc))

                    # Capture move
                    jr, jc = r + 2 * dr, c + 2 * dc
                    mr, mc = r + dr, c + dc
                    if (
                        self._on_board(jr, jc)
                        and self._on_board(mr, mc)
                        and (jr + jc) % 2 == 1
                        and self.board[jr, jc] == 0
                        and self._is_opponent_piece(player, int(self.board[mr, mc]))
                    ):
                        capture_moves.append((r, c, jr, jc))

        return simple_moves, capture_moves

    def get_legal_actions(self, player: Optional[int] = None) -> List[Tuple[int, int, int, int]]:
        """
        Public helper: list of legal actions for the given player (or current player if None).
        Respects forced-capture rule.
        """
        p = self.current_player if player is None else player
        simple, captures = self._get_legal_moves(p)
        return captures if len(captures) > 0 else simple

    # ------------------------------------------------------------------
    # Move application
    # ------------------------------------------------------------------
    def _apply_move(self, move: Tuple[int, int, int, int]) -> bool:
        """
        Apply move (sr, sc, er, ec).
        Returns True if a capture occurred, and sets self._last_move_promoted.
        """
        sr, sc, er, ec = move
        piece = int(self.board[sr, sc])
        assert piece != 0, "No piece on start square"

        self._last_move_promoted = False

        captured = False
        # Capture?
        if abs(er - sr) == 2:
            mr = (sr + er) // 2
            mc = (sc + ec) // 2
            if self.board[mr, mc] != 0:
                captured = True
            self.board[mr, mc] = 0

        # Move the piece
        self.board[sr, sc] = 0

        # Promotion
        player = self._piece_owner(piece)
        if player == 0 and er == 0 and piece == 1:
            piece = 3  # player1 king
            self._last_move_promoted = True
        elif player == 1 and er == BOARD_SIZE - 1 and piece == 2:
            piece = 4  # player2 king
            self._last_move_promoted = True

        self.board[er, ec] = piece
        return captured

    # ------------------------------------------------------------------
    # Board utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _on_board(r: int, c: int) -> bool:
        return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE

    @staticmethod
    def _piece_owner(piece: int) -> Optional[int]:
        """
        Returns player id (0 or 1) for a piece value, or None if empty.
        """
        if piece == 0:
            return None
        if piece in (1, 3):
            return 0
        if piece in (2, 4):
            return 1
        return None

    @staticmethod
    def _is_king(piece: int) -> bool:
        return piece in (3, 4)

    @staticmethod
    def _is_player_piece(player: int, piece: int) -> bool:
        if player == 0:
            return piece in (1, 3)
        else:
            return piece in (2, 4)

    @staticmethod
    def _is_opponent_piece(player: int, piece: int) -> bool:
        if player == 0:
            return piece in (2, 4)
        else:
            return piece in (1, 3)

    @staticmethod
    def _move_directions(player: int, is_king: bool) -> List[Tuple[int, int]]:
        """
        Returns list of (dr, dc) for movement.
        player 0 moves "up" (towards row 0)
        player 1 moves "down" (towards row BOARD_SIZE-1)
        Kings move both directions.
        """
        if is_king:
            return [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        if player == 0:
            return [(-1, -1), (-1, 1)]
        else:
            return [(1, -1), (1, 1)]

    def _count_pieces(self, player: int) -> int:
        if player == 0:
            return int(np.sum((self.board == 1) | (self.board == 3)))
        else:
            return int(np.sum((self.board == 2) | (self.board == 4)))


def make_env(max_steps: int = 200) -> Checkers6x6Env:
    return Checkers6x6Env(max_steps=max_steps)
