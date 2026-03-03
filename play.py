import pickle
import time
from pathlib import Path
from typing import Optional
import random

import numpy as np

from checkers_env import make_env, Checkers6x6Env, BOARD_SIZE
from q_agent import QLearningAgent, observation_to_state
from heuristic_agent import PriorityHeuristicAgent


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "q_table.pkl"


def set_seed(seed: int = 42) -> None:
    """Set global random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def load_agent(env: Checkers6x6Env) -> QLearningAgent:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No Q-table found at {MODEL_PATH}. Run train.py first.")
    with MODEL_PATH.open("rb") as f:
        q_table = pickle.load(f)

    agent = QLearningAgent(env.action_space)
    agent.q_table = q_table
    return agent


def random_legal_move(env: Checkers6x6Env, player_id: int):
    import random

    moves = env.get_legal_actions(player=player_id)
    if not moves:
        raise ValueError("No legal moves available.")
    return random.choice(moves)


def run_episode_play(
    env: Checkers6x6Env,
    agent: QLearningAgent,
    opponent_type: str = "random",
    render: bool = True,
    sleep_sec: float = 0.5,
    seed: Optional[int] = None,
    agent_player_id: int = 0,
) -> int:
    """
    Run a single episode:
        - Player 1 (id=0): Q-learning agent (greedy policy, epsilon=0)
        - Player 2 (id=1): random or heuristic opponent

    Returns:
        winner id: 0 (Q-agent), 1 (opponent), -1 (draw/other)
    """
    if opponent_type not in {"random", "heuristic"}:
        raise ValueError("opponent_type must be 'random' or 'heuristic'.")

    opponent_player_id = 1 - agent_player_id

    heuristic_opponent: Optional[PriorityHeuristicAgent] = None
    if opponent_type == "heuristic":
        heuristic_opponent = PriorityHeuristicAgent(player_id=opponent_player_id)

    if seed is not None:
        obs, _ = env.reset(seed=seed)
    else:
        obs, _ = env.reset()
    done = False
    winner = -1

    if render:
        env.render()
        time.sleep(sleep_sec)

    while not done:
        current_player = obs["current_player"]

        if current_player == agent_player_id:
            # Agent's turn
            state = observation_to_state(obs)

            legal_moves_env = env.get_legal_actions(player=agent_player_id)
            if not legal_moves_env:
                # No legal moves -> opponent wins
                winner = opponent_player_id
                break

            # Map env moves to canonical perspective if agent is Player 1
            if agent_player_id == 0:
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

            # Greedy choice in canonical space
            action_canonical = agent.greedy_action(state, legal_moves_canonical)

            # Map back to env coordinates
            if agent_player_id == 0:
                action_env = action_canonical
            else:
                sr, sc, er, ec = action_canonical
                action_env = (
                    BOARD_SIZE - 1 - sr,
                    BOARD_SIZE - 1 - sc,
                    BOARD_SIZE - 1 - er,
                    BOARD_SIZE - 1 - ec,
                )

            obs_next, r_agent, terminated, truncated, info = env.step(action_env)
            done = terminated or truncated
            if render:
                env.render()
                time.sleep(sleep_sec)

            if done:
                winner = info.get("winner", -1)
                break

            obs = obs_next

        else:
            # Opponent's turn
            assert current_player == opponent_player_id

            if opponent_type == "random":
                opp_move = random_legal_move(env, player_id=opponent_player_id)
            else:
                opp_move = heuristic_opponent.select_move(env)  # type: ignore[union-attr]

            obs_next, r_opp, terminated2, truncated2, info2 = env.step(opp_move)
            done = terminated2 or truncated2
            if render:
                env.render()
                time.sleep(sleep_sec)

            if done:
                winner = info2.get("winner", -1)
                break

            obs = obs_next

    return winner


def evaluate(
    num_episodes: int = 500,
    opponent_type: str = "random",
    render: bool = True,
):
    # Global seeding for reproducibility
    set_seed(42)

    env = make_env()
    agent = load_agent(env)

    wins = 0
    losses = 0
    draws = 0

    for i in range(num_episodes):
        # Only seed the very first reset; subsequent episodes rely on global RNG state
        seed = 42 if i == 0 else None
        winner = run_episode_play(
            env,
            agent,
            opponent_type=opponent_type,
            render=render,
            seed=seed,
        )
        if winner == 0:
            wins += 1
        elif winner == 1:
            losses += 1
        else:
            draws += 1

        print(
            f"Episode {i+1}/{num_episodes} "
            f"| winner: {winner} (0=Q-agent, 1={opponent_type}, -1=draw)"
        )

    print(
        f"\nResults vs {opponent_type} over {num_episodes} episodes:\n"
        f"  Wins:   {wins} ({wins/num_episodes:.3f})\n"
        f"  Losses: {losses} ({losses/num_episodes:.3f})\n"
        f"  Draws:  {draws} ({draws/num_episodes:.3f})"
    )


if __name__ == "__main__":
    # Example: run 500 episodes vs random opponent, rendering each game
    evaluate(num_episodes=500, opponent_type="random", render=True)

