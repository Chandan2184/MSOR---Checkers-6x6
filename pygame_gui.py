import argparse
import pickle
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import pygame
import numpy as np

from checkers_env import make_env, Checkers6x6Env, BOARD_SIZE
from q_agent import QLearningAgent, observation_to_state, Action
from heuristic_agent import PriorityHeuristicAgent


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "q_table.pkl"

# UI / game-loop constants
STATUS_HEIGHT = 80
FPS = 60
AI_WAIT_MS = 1300  # delay before AI moves (ms)


def load_q_agent(env: Checkers6x6Env) -> QLearningAgent:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No Q-table found at {MODEL_PATH}. Run train.py first.")
    with MODEL_PATH.open("rb") as f:
        q_table = pickle.load(f)
    agent = QLearningAgent(env.action_space)
    agent.q_table = q_table
    return agent


def draw_board(
    screen: pygame.Surface,
    env: Checkers6x6Env,
    selected: Optional[Tuple[int, int]],
    highlight_destinations: List[Tuple[int, int]],
    status_text: str,
    last_reward: float,
    cumulative_reward: float,
    font: pygame.font.Font,
    offset_x: int,
    offset_y: int,
    square_size: int,
) -> None:
    screen.fill((0, 0, 0))

    width, height = screen.get_size()
    board_size_px = square_size * BOARD_SIZE
    light_color = (222, 184, 135)   # beige
    dark_color = (139, 69, 19)      # brown
    highlight_color = (255, 215, 0) # gold

    # Draw squares
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            x = offset_x + c * square_size
            y = offset_y + r * square_size
            if (r + c) % 2 == 0:
                color = light_color
            else:
                color = dark_color
            pygame.draw.rect(screen, color, (x, y, square_size, square_size))

            # Highlight selected square
            if selected == (r, c):
                pygame.draw.rect(screen, highlight_color, (x, y, square_size, square_size), 4)

    # Draw pieces
    radius = square_size // 2 - 6
    board = env.board
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            piece = int(board[r, c])
            if piece == 0:
                continue
            x_center = offset_x + c * square_size + square_size // 2
            y_center = offset_y + r * square_size + square_size // 2

            if piece in (1, 3):  # Player 1 (black)
                color = (0, 0, 0)
            else:  # Player 2 (red)
                color = (200, 0, 0)

            pygame.draw.circle(screen, color, (x_center, y_center), radius)

            # Kings: draw a gold crown marker
            if piece in (3, 4):
                pygame.draw.circle(screen, (255, 215, 0), (x_center, y_center), radius // 2, 2)

    # Highlight legal destination squares for selected piece (small hollow circles)
    for (r, c) in highlight_destinations:
        x_center = offset_x + c * square_size + square_size // 2
        y_center = offset_y + r * square_size + square_size // 2
        pygame.draw.circle(screen, (255, 215, 0), (x_center, y_center), square_size // 6, 2)

    # Status bar
    bar_y = board_size_px
    pygame.draw.rect(screen, (30, 30, 30), (0, bar_y, width, STATUS_HEIGHT))

    # Current turn text
    player_str = "Black" if env.current_player == 0 else "Red"
    status_surface = font.render(f"Turn: {player_str} | {status_text}", True, (255, 255, 255))
    screen.blit(status_surface, (10, bar_y + 10))

    reward_surface = font.render(
        f"Last reward: {last_reward:.3f} | Cumulative reward: {cumulative_reward:.3f}",
        True,
        (200, 200, 200),
    )
    screen.blit(reward_surface, (10, bar_y + 40))

    pygame.display.flip()


def human_select_move(env: Checkers6x6Env,
                      event: pygame.event.Event,
                      selected: Optional[Tuple[int, int]],
                      legal_moves: List[Action],
                      offset_x: int,
                      offset_y: int,
                      square_size: int) -> Tuple[Optional[Tuple[int, int]], Optional[Action]]:
    """Handle human clicks and return (new_selected, action) if a full move is chosen."""
    if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
        return selected, None

    mx, my = event.pos
    col = (mx - offset_x) // square_size
    row = (my - offset_y) // square_size

    if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
        return None, None

    # First click: select a piece (only if it has legal moves)
    if selected is None:
        piece = int(env.board[row, col])
        if env.current_player == 0 and piece in (1, 3):
            has_moves = any(sr == row and sc == col for (sr, sc, _, _) in legal_moves)
            if has_moves:
                return (row, col), None
        elif env.current_player == 1 and piece in (2, 4):
            has_moves = any(sr == row and sc == col for (sr, sc, _, _) in legal_moves)
            if has_moves:
                return (row, col), None
        return None, None

    # Second click: destination
    sr, sc = selected
    er, ec = row, col
    action = (sr, sc, er, ec)
    return None, action


def main():
    parser = argparse.ArgumentParser(description="Pygame GUI for 6x6 Checkers RL environment")
    parser.add_argument("--p1", choices=["human", "q_agent", "heuristic"], default="human",
                        help="Controller for Player 1 (Black)")
    parser.add_argument("--p2", choices=["human", "q_agent", "heuristic"], default="q_agent",
                        help="Controller for Player 2 (Red)")
    args = parser.parse_args()

    pygame.init()
    pygame.display.set_caption("6x6 Checkers - RL Environment")
    window_size = (640, 720)
    screen = pygame.display.set_mode(window_size)
    font = pygame.font.SysFont("arial", 20)

    env = make_env()
    obs, _ = env.reset()

    # Agents
    player_types = {0: args.p1, 1: args.p2}
    q_agent: Optional[QLearningAgent] = None
    if "q_agent" in (args.p1, args.p2):
        q_agent = load_q_agent(env)

    heuristic_agents: Dict[int, PriorityHeuristicAgent] = {}
    for pid in (0, 1):
        if player_types[pid] == "heuristic":
            heuristic_agents[pid] = PriorityHeuristicAgent(player_id=pid)

    last_reward = 0.0
    cumulative_reward = 0.0
    terminated = False
    truncated = False
    info: Dict[str, Optional[int]] = {"winner": None}

    selected: Optional[Tuple[int, int]] = None
    clock = pygame.time.Clock()
    ai_waiting = False
    ai_wait_start_ms = 0

    running = True
    while running:
        width, height = screen.get_size()
        board_size_px = min(width, height - STATUS_HEIGHT)
        square_size = board_size_px // BOARD_SIZE
        offset_x = (width - board_size_px) // 2
        offset_y = 0

        # Status message
        if terminated or truncated:
            w = info.get("winner", -1)
            if w == 0:
                status_text = "Game Over: Black wins"
            elif w == 1:
                status_text = "Game Over: Red wins"
            else:
                status_text = "Game Over: Draw"
        else:
            p_type = player_types[env.current_player]
            role = "Human" if p_type == "human" else ("Q-Agent" if p_type == "q_agent" else "Heuristic")
            status_text = f"Turn: {role}"

        # Compute legal moves for current player for highlighting and human validation
        if not (terminated or truncated):
            current_legal_moves = env.get_legal_actions(player=env.current_player)
        else:
            current_legal_moves = []

        # Highlight destinations for selected piece, if any
        highlight_destinations: List[Tuple[int, int]] = []
        if selected is not None:
            sr, sc = selected
            for (r0, c0, r1, c1) in current_legal_moves:
                if r0 == sr and c0 == sc:
                    highlight_destinations.append((r1, c1))

        draw_board(
            screen,
            env,
            selected,
            highlight_destinations,
            status_text,
            last_reward,
            cumulative_reward,
            font,
            offset_x,
            offset_y,
            square_size,
        )

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

            if terminated or truncated:
                # Any key or mouse click starts a new game
                if event.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
                    obs, _ = env.reset()
                    last_reward = 0.0
                    cumulative_reward = 0.0
                    terminated = False
                    truncated = False
                    info = {"winner": None}
                    selected = None
                continue

            # Human input
            if player_types[env.current_player] == "human":
                new_selected, action = human_select_move(
                    env, event, selected, current_legal_moves, offset_x, offset_y, square_size
                )
                selected = new_selected

                if action is not None:
                    legal_moves = env.get_legal_actions(player=env.current_player)
                    if action in legal_moves:
                        obs, reward, terminated, truncated, info = env.step(action)
                        last_reward = float(reward)
                        cumulative_reward += float(reward)
                    # Whether valid or not, clear selection; invalid moves are ignored
                    selected = None

        if not running:
            break

        # AI moves (when not terminated and not human), using state-based delay
        if not (terminated or truncated):
            current_player = env.current_player
            p_type = player_types[current_player]

            if p_type in ("q_agent", "heuristic"):
                now = pygame.time.get_ticks()
                if not ai_waiting:
                    ai_waiting = True
                    ai_wait_start_ms = now
                elif now - ai_wait_start_ms >= AI_WAIT_MS:
                    ai_waiting = False
                    if p_type == "heuristic":
                        h_agent = heuristic_agents[current_player]
                        move = h_agent.select_move(env)
                        obs, reward, terminated, truncated, info = env.step(move)
                        last_reward = float(reward)
                        cumulative_reward += float(reward)
                    else:  # q_agent
                        if q_agent is None:
                            raise RuntimeError("Q-agent requested but q_table.pkl not loaded.")

                        state = observation_to_state(obs)
                        legal_moves_env = env.get_legal_actions(player=current_player)
                        if not legal_moves_env:
                            # No legal moves, environment will handle winner on next step
                            terminated = True
                            info = {"winner": 1 - current_player}
                        else:
                            # Map legal moves to canonical perspective if agent is Player 2
                            if current_player == 0:
                                legal_moves_canonical = legal_moves_env
                            else:
                                legal_moves_canonical = [
                                    (
                                        BOARD_SIZE - 1 - sr,
                                        BOARD_SIZE - 1 - sc,
                                        BOARD_SIZE - 1 - er,
                                        BOARD_SIZE - 1 - ec,
                                    )
                                    for (sr, sc, er, ec) in legal_moves_env
                                ]

                            action_canonical = q_agent.greedy_action(state, legal_moves_canonical)

                            # Map canonical action back to env coordinates
                            if current_player == 0:
                                action_env = action_canonical
                            else:
                                sr, sc, er, ec = action_canonical
                                action_env = (
                                    BOARD_SIZE - 1 - sr,
                                    BOARD_SIZE - 1 - sc,
                                    BOARD_SIZE - 1 - er,
                                    BOARD_SIZE - 1 - ec,
                                )

                            obs, reward, terminated, truncated, info = env.step(action_env)
                            last_reward = float(reward)
                            cumulative_reward += float(reward)

        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()

