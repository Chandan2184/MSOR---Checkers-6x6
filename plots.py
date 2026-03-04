from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pickle

from checkers_env import make_env
from q_agent import QLearningAgent
from heuristic_agent import PriorityHeuristicAgent


ROOT = Path(__file__).resolve().parent
STATS_PATH = ROOT / "training_stats.npz"
MODEL_PATH = ROOT / "q_table.pkl"


def plot_learning_curve(window: int = 1000):
    data = np.load(STATS_PATH)
    rewards = data["rewards"]
    winners = data["winners"]
    num_episodes = int(data["num_episodes"])
    eval_win_random = data["eval_win_random"]
    eval_win_heuristic = data["eval_win_heuristic"]

    # Noisy training win rate (agent wins, regardless of color)
    win_flags = (winners == 0).astype(np.float32)
    win_ma = np.convolve(win_flags, np.ones(window) / window, mode="valid")
    xs_train = np.arange(len(win_ma)) + window

    # Evaluation win rates at each 1000-episode checkpoint
    eval_checkpoints = (np.arange(len(eval_win_random)) + 1) * 1000

    plt.figure(figsize=(12, 7))

    # Noisy training win rate (semi-transparent)
    plt.plot(xs_train, win_ma, label="Training win rate (moving avg)", color="tab:blue", alpha=0.4)

    # Clean evaluation win rates vs fixed benchmarks
    plt.plot(
        eval_checkpoints,
        eval_win_random,
        label="Eval win vs Random",
        color="tab:orange",
        linewidth=2.0,
    )
    plt.plot(
        eval_checkpoints,
        eval_win_heuristic,
        label="Eval win vs Heuristic",
        color="tab:purple",
        linewidth=2.0,
    )

    plt.xlabel("Episode")
    plt.ylabel("Win rate")
    plt.title(f"Training vs Evaluation Win Rate (window={window})")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(ROOT / "learning_curve_win_rate.png", dpi=200)


def plot_state_space_growth():
    data = np.load(STATS_PATH)
    num_episodes = int(data["num_episodes"])
    q_table_sizes = data["q_table_sizes"]

    eval_checkpoints = (np.arange(len(q_table_sizes)) + 1) * (num_episodes // len(q_table_sizes))

    plt.figure(figsize=(10, 6))
    plt.plot(eval_checkpoints, q_table_sizes, marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Q-table size (number of state-action entries)")
    plt.title("Growth of Q-table over Training")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(ROOT / "state_space_growth.png", dpi=200)


def plot_game_length(window: int = 1000):
    data = np.load(STATS_PATH)
    episode_lengths = data["episode_lengths"]
    num_episodes = int(data["num_episodes"])

    length_ma = np.convolve(episode_lengths, np.ones(window) / window, mode="valid")
    xs = np.arange(len(length_ma)) + window

    plt.figure(figsize=(10, 6))
    plt.plot(xs, length_ma, label=f"Episode length (MA, window={window})", color="tab:green")
    plt.xlabel("Episode")
    plt.ylabel("Number of environment steps")
    plt.title("Game Length over Training")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(ROOT / "game_length.png", dpi=200)


def plot_p1_vs_p2_eval():
    data = np.load(STATS_PATH)
    num_episodes = int(data["num_episodes"])
    eval_win_p1_heuristic = data["eval_win_p1_heuristic"]
    eval_win_p2_heuristic = data["eval_win_p2_heuristic"]

    eval_checkpoints = (np.arange(len(eval_win_p1_heuristic)) + 1) * 1000

    plt.figure(figsize=(10, 6))
    plt.plot(
        eval_checkpoints,
        eval_win_p1_heuristic,
        label="Eval win rate as P1 vs Heuristic",
        color="tab:orange",
    )
    plt.plot(
        eval_checkpoints,
        eval_win_p2_heuristic,
        label="Eval win rate as P2 vs Heuristic",
        color="tab:purple",
    )
    plt.xlabel("Episode")
    plt.ylabel("Win rate")
    plt.title("Evaluation Win Rate: Player 1 vs Player 2")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(ROOT / "eval_p1_vs_p2_win_rates.png", dpi=200)


def performance_distribution(num_games: int = 1000):
    """
    Compare Random, Heuristic, and Q-learning agents against a common opponent.
    Produces a stacked bar chart of Win/Loss/Draw percentages.
    """
    env = make_env()
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Train the agent first to create q_table.pkl.")

    with MODEL_PATH.open("rb") as f:
        q_table = pickle.load(f)
    q_agent = QLearningAgent(env.action_space)
    q_agent.q_table = q_table

    from play import run_episode_play, random_legal_move

    def eval_agent(name: str, agent_type: str) -> Dict[str, float]:
        wins = losses = draws = 0
        for _ in range(num_games):
            env = make_env()
            if agent_type == "q":
                winner = run_episode_play(env, q_agent, opponent_type="heuristic", render=False)
            elif agent_type == "heuristic":
                # Heuristic vs Random baseline: follow env.current_player to support multi-jumps.
                env.reset()
                h_agent = PriorityHeuristicAgent(player_id=0)
                done = False
                winner_local = -1
                obs, _ = env.reset()
                while not done:
                    current_player = int(obs["current_player"])
                    if current_player == 0:
                        # P1 heuristic
                        move = h_agent.select_move(env)
                    else:
                        # P2 random
                        move = random_legal_move(env, player_id=1)
                    obs, _, term, trunc, info = env.step(move)
                    done = term or trunc
                    if done:
                        winner_local = info.get("winner", -1)
                        break
                winner = winner_local
            else:  # random vs random
                env.reset()
                done = False
                winner_local = -1
                obs, _ = env.reset()
                import random

                while not done:
                    current_player = int(obs["current_player"])
                    move = random_legal_move(env, player_id=current_player)
                    obs, _, term, trunc, info = env.step(move)
                    done = term or trunc
                    if done:
                        winner_local = info.get("winner", -1)
                        break
                winner = winner_local

            if winner == 0:
                wins += 1
            elif winner == 1:
                losses += 1
            else:
                draws += 1

        return {
            "wins": wins / num_games,
            "losses": losses / num_games,
            "draws": draws / num_games,
        }

    agents = [
        ("Random", "random"),
        ("Heuristic", "heuristic"),
        ("Q-Learning", "q"),
    ]

    results = [eval_agent(name, t) for name, t in agents]

    labels = [name for name, _ in agents]
    wins = [r["wins"] for r in results]
    losses = [r["losses"] for r in results]
    draws = [r["draws"] for r in results]

    x = np.arange(len(labels))

    plt.figure(figsize=(8, 6))
    plt.bar(x, wins, label="Wins")
    plt.bar(x, draws, bottom=wins, label="Draws")
    bottom_ld = np.array(wins) + np.array(draws)
    plt.bar(x, losses, bottom=bottom_ld, label="Losses")
    plt.xticks(x, labels)
    plt.ylabel("Proportion of games")
    plt.title("Performance Distribution vs Common Opponent")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROOT / "performance_distribution.png", dpi=200)


if __name__ == "__main__":
    # Generate all plots for the report
    plot_learning_curve(window=1000)
    plot_state_space_growth()
    plot_game_length(window=1000)
    plot_p1_vs_p2_eval()
    performance_distribution(num_games=1000)
 
