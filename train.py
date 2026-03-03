import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import copy
import random

import numpy as np

from checkers_env import make_env, BOARD_SIZE
from q_agent import QLearningAgent, observation_to_state, Action, State


# region agent log
def _agent_debug_log(
    run_id: str,
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict,
) -> None:
    """
    Lightweight NDJSON logger for debug-b40519.log.
    Does nothing on failure and must never break training.
    """
    try:
        import json
        import time

        payload = {
            "sessionId": "b40519",
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open("debug-b40519.log", "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        # Debug logging must never interfere with training.
        pass


# endregion


def set_seed(seed: int = 42) -> None:
    """Set global random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def evaluate_agent(
    env,
    agent: QLearningAgent,
    num_games: int,
    opponent_type: str,
    opponent_pool: List[Dict[Tuple[State, Action], float]],
) -> Tuple[float, float, float]:
    """
    Run a decoupled evaluation of the agent:
        - num_games total (half as Player 0, half as Player 1)
        - Pure exploitation (epsilon=0.0)
        - No Q-table updates (backward_pass_update is skipped)
    Returns (win_rate_p1, win_rate_p2, avg_reward) from the agent's perspective,
    where win_rate_p1 is when the agent plays as Player 1 (id=0) and
    win_rate_p2 is when the agent plays as Player 2 (id=1).
    """
    total_reward = 0.0
    wins_p0 = wins_p1 = 0
    games_p0 = games_p1 = 0

    games_per_side = max(1, num_games // 2)

    def _play_one(agent_player_id: int):
        nonlocal total_reward, wins_p0, wins_p1, games_p0, games_p1

        # Sample historical opponent for self-play, if available
        opponent_q_table = None
        if opponent_type == "self_play" and opponent_pool:
            opponent_q_table = opponent_pool[np.random.randint(len(opponent_pool))]

        ep_reward, winner, _steps = run_episode(
            env,
            agent,
            gamma=0.0,
            opponent_type=opponent_type,
            opponent_q_table=opponent_q_table,
            agent_player_id=agent_player_id,
            update_q=False,
        )
        total_reward += ep_reward
        if agent_player_id == 0:
            games_p0 += 1
            if winner == 0:
                wins_p0 += 1
        else:
            games_p1 += 1
            if winner == 1:
                wins_p1 += 1

    # Half games as Player 0, half as Player 1 (approx)
    for _ in range(games_per_side):
        _play_one(0)
        _play_one(1)

    n_games = games_p0 + games_p1
    win_rate_p1 = wins_p0 / games_p0 if games_p0 > 0 else 0.0
    win_rate_p2 = wins_p1 / games_p1 if games_p1 > 0 else 0.0
    avg_reward = total_reward / n_games if n_games > 0 else 0.0
    return win_rate_p1, win_rate_p2, avg_reward
from heuristic_agent import PriorityHeuristicAgent


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "q_table.pkl"
STATS_PATH = ROOT / "training_stats.npz"


def linear_decay(start: float, end: float, step: int, total_steps: int) -> float:
    """Linearly decay from start to end over total_steps, then stay at end."""
    if step >= total_steps:
        return end
    frac = step / float(total_steps)
    return start + frac * (end - start)


def run_episode(
    env,
    agent: QLearningAgent,
    gamma: float,
    opponent_type: str,
    opponent_q_table: Optional[Dict[Tuple[State, Action], float]] = None,
    agent_player_id: int = 0,
    update_q: bool = True,
    seed: Optional[int] = None,
    exploit_only: bool = False,
) -> Tuple[float, int, int]:
    """
    Run one episode where the Q-learning agent can be Player 1 (id=0)
    or Player 2 (id=1), against a configurable opponent:
        - opponent_type="random": random legal moves
        - opponent_type="heuristic": PriorityHeuristicAgent
        - opponent_type="self_play": same Q-agent policy as Player 2 (no extra updates)

    We collect transitions from the *agent's* perspective only.
    Each transition spans one full agent+opponent cycle:
        (state_before_agent_move, agent_action, reward_for_agent, next_state_when_agent_moves_again_or_None)
    """
    if seed is not None:
        observation, _ = env.reset(seed=seed)
    else:
        observation, _ = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    episode_memory: List[Tuple[tuple, Action, float, Optional[tuple], List[Action]]] = []
    winner = -1

    import random

    heuristic_opponent: Optional[PriorityHeuristicAgent] = None
    opponent_player_id = 1 - agent_player_id
    if opponent_type == "heuristic":
        heuristic_opponent = PriorityHeuristicAgent(player_id=opponent_player_id)

    while not done:
        # If it's not the agent's turn, let the opponent move first
        while observation["current_player"] != agent_player_id and not done:
            # Opponent move (from opponent's perspective)
            opp_actions = env.get_legal_actions(player=opponent_player_id)
            if not opp_actions:
                # No move for opponent -> agent wins
                winner = agent_player_id
                done = True
                break

            if opponent_type == "random":
                opp_action = random.choice(opp_actions)
            elif opponent_type == "heuristic":
                opp_action = heuristic_opponent.select_move(env)  # type: ignore[union-attr]
            else:  # self_play
                state_opp = observation_to_state(observation)
                best_q = -float("inf")
                best_a: Optional[Action] = None
                if opponent_q_table is not None:
                    for a in opp_actions:
                        q_val = opponent_q_table.get((state_opp, a), 0.0)
                        if q_val > best_q:
                            best_q = q_val
                            best_a = a
                else:
                    for a in opp_actions:
                        q_val = agent.get_q_value(state_opp, a)
                        if q_val > best_q:
                            best_q = q_val
                            best_a = a
                opp_action = best_a if best_a is not None else random.choice(opp_actions)

            obs_after_opp, r_opp, terminated_opp, truncated_opp, info_opp = env.step(opp_action)
            steps += 1
            done = terminated_opp or truncated_opp
            # From the agent's perspective, opponent's positive reward is negative for agent
            total_reward += -r_opp
            observation = obs_after_opp
            if done:
                winner = info_opp.get("winner", -1)
                break

        if done:
            break

        # Now it must be the agent's turn
        assert observation["current_player"] == agent_player_id, "Expected agent to move."

        state_before = observation_to_state(observation)

        # Restrict actions to legal moves only to avoid invalid actions
        legal_moves_env = env.get_legal_actions(player=agent_player_id)
        if not legal_moves_env:
            # No legal moves for agent -> opponent effectively wins
            winner = opponent_player_id
            break

        # Map environment actions to canonical (Player 0) perspective if needed
        if agent_player_id == 0:
            legal_moves_canonical = legal_moves_env
        else:
            # Flip coordinates for Player 1 to match canonical perspective
            legal_moves_canonical = [
                (
                    BOARD_SIZE - 1 - sr,
                    BOARD_SIZE - 1 - sc,
                    BOARD_SIZE - 1 - er,
                    BOARD_SIZE - 1 - ec,
                )
                for (sr, sc, er, ec) in legal_moves_env
            ]

        # Choose canonical action: exploratory during training, greedy during evaluation

        if exploit_only:
            action_canonical = agent.greedy_action(state_before, legal_moves_canonical) 
        else:
            action_canonical = agent.epsilon_greedy_policy(state_before, legal_moves_canonical)

        # Map canonical action back to environment coordinates
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

        # Agent move (now guaranteed legal)
        obs_mid, r_agent, terminated, truncated, info = env.step(action_env)
        steps += 1
        done = terminated or truncated

        # region agent log
        _agent_debug_log(
            run_id="pre-fix",
            hypothesis_id="H1",
            location="train.run_episode:after_agent_step",
            message="After agent move before opponent/multi-jump handling",
            data={
                "obs_mid_current_player": int(obs_mid.get("current_player", -1)),
                "agent_player_id": int(agent_player_id),
                "opponent_player_id": int(opponent_player_id),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            },
        )
        # endregion

        # If game ended on agent move, there is no opponent reply
        if done:
            total_reward += r_agent
            # Terminal state: no next state and no legal next actions
            episode_memory.append((state_before, action_canonical, r_agent, None, []))
            winner = info.get("winner", -1)
            break

        # If after the agent move it is still the agent's turn, this is a multi-jump.
        if obs_mid["current_player"] == agent_player_id:
            # From the agent's perspective, only its own reward applies in this sub-step.
            r_step = r_agent
            total_reward += r_step

            next_state = observation_to_state(obs_mid)
            legal_next_actions = env.get_legal_actions(player=agent_player_id)

            episode_memory.append(
                (state_before, action_canonical, r_step, next_state, legal_next_actions)
            )

            # Continue the episode with the updated observation; opponent has not moved yet.
            observation = obs_mid
            continue

        # Opponent move (standard single-move turn)
        assert obs_mid["current_player"] == opponent_player_id

        opp_actions_env = env.get_legal_actions(player=opponent_player_id)
        if not opp_actions_env:
            # No move for opponent -> agent wins
            # Treat as terminal from the agent's perspective
            r_opp = -100.0
            r_step = r_agent - r_opp
            total_reward += r_step
            episode_memory.append((state_before, action_canonical, r_step, None, []))
            winner = agent_player_id
            break

        # Choose opponent action according to curriculum
        if opponent_type == "random":
            opp_action_env = random.choice(opp_actions_env)
        elif opponent_type == "heuristic":
            opp_action_env = heuristic_opponent.select_move(env)  # type: ignore[union-attr]
        else:  # self_play
            # Self-play with historical opponent pool:
            # If an opponent_q_table is provided, use it for opponent's greedy moves.
            state_opp = observation_to_state(obs_mid)
            best_q = -float("inf")
            best_a_env: Optional[Action] = None

            for a_env in opp_actions_env:
                if opponent_player_id == 0:
                    a_canonical = a_env
                else:
                    sr, sc, er, ec = a_env
                    a_canonical = (
                        BOARD_SIZE - 1 - sr,
                        BOARD_SIZE - 1 - sc,
                        BOARD_SIZE - 1 - er,
                        BOARD_SIZE - 1 - ec,
                    )

                if opponent_q_table is not None:
                    q_val = opponent_q_table.get((state_opp, a_canonical), 0.0)
                else:
                    q_val = agent.get_q_value(state_opp, a_canonical)

                if q_val > best_q:
                    best_q = q_val
                    best_a_env = a_env

            opp_action_env = best_a_env if best_a_env is not None else random.choice(opp_actions_env)

        obs_next, r_opp, terminated2, truncated2, info2 = env.step(opp_action_env)
        steps += 1
        done = terminated2 or truncated2

        # Net reward for the agent for this full cycle:
        # r_agent is from env's perspective when agent moves (positive good for agent),
        # r_opp is from env's perspective when opponent moves (positive good for opponent),
        # so from the agent's perspective we subtract r_opp.
        r_step = r_agent - r_opp
        total_reward += r_step

        if done:
            next_state = None
            legal_next_actions: List[Action] = []
        else:
            next_state = observation_to_state(obs_next)
            # Legal actions available to the agent in the next state
            legal_next_actions = env.get_legal_actions(player=agent_player_id)

        episode_memory.append((state_before, action_canonical, r_step, next_state, legal_next_actions))

        if done:
            winner = info2.get("winner", -1)
            break

        # Continue with next observation (should be agent's turn again)
        observation = obs_next

    # Backward-pass Q-learning update over the episode (optional)
    if update_q:
        agent.backward_pass_update(episode_memory, gamma=gamma)

    return total_reward, winner, steps


def train(
    num_episodes: int = 100_000,
    gamma: float = 0.99,
):
    # Global seeding for reproducibility
    set_seed(42)

    env = make_env()
    agent = QLearningAgent(env.action_space)

    # Historical opponent pool for self-play (to prevent catastrophic forgetting)
    opponent_pool: List[Dict[Tuple[State, Action], float]] = []

    # Stats
    rewards = np.zeros(num_episodes, dtype=np.float32)
    winners = np.full(num_episodes, -1, dtype=np.int8)  # 0=agent, 1=opponent, -1=draw/other
    episode_lengths = np.zeros(num_episodes, dtype=np.int32)

    # Clean evaluation metrics (decoupled from training loop)
    eval_win_rates: List[float] = []
    eval_rewards: List[float] = []
    eval_win_rates_p1: List[float] = []
    eval_win_rates_p2: List[float] = []
    q_table_sizes: List[int] = []

    # Curriculum phase: 0 = mostly random, 1 = mostly heuristic, 2 = mostly self-play
    current_curriculum_phase = 0

    for ep in range(num_episodes):
        # Performance-based auto-curriculum for opponent type.
        # Opponent types: [random, heuristic, self_play]
        if current_curriculum_phase == 0:
            # Phase 0: 80% random, 20% heuristic, 0% self_play
            probs = np.array([0.8, 0.2, 0.0], dtype=np.float32)
        elif current_curriculum_phase == 1:
            # Phase 1: 10% random, 80% heuristic, 10% self_play
            probs = np.array([0.1, 0.8, 0.1], dtype=np.float32)
        else:
            # Phase 2: 0% random, 10% heuristic, 90% self_play
            probs = np.array([0.0, 0.1, 0.9], dtype=np.float32)

        probs = probs / probs.sum()
        opponent_type = np.random.choice(
            ["random", "heuristic", "self_play"],
            p=probs,
        )

        # For self-play, sample a historical opponent Q-table if available
        opponent_q_table = None
        if opponent_type == "self_play" and opponent_pool:
            opponent_q_table = opponent_pool[np.random.randint(len(opponent_pool))]

        # Randomly assign the agent to be Player 0 or Player 1 for this episode
        agent_player_id = int(np.random.choice([0, 1]))

        # Only seed the very first reset to keep episodes deterministic but varied
        seed = 42 if ep == 0 else None

        total_reward, winner, ep_steps = run_episode(
            env,
            agent,
            gamma,
            opponent_type,
            opponent_q_table=opponent_q_table,
            agent_player_id=agent_player_id,
            seed=seed,
        )

        rewards[ep] = total_reward
        winners[ep] = winner
        episode_lengths[ep] = ep_steps

        # Progress logging & evaluation every 1000 episodes
        if (ep + 1) % 1000 == 0:
            # Episode window start index for logging
            start = max(0, ep - 999)

            # Run a decoupled evaluation: 100 games (50 as P1, 50 as P2)
            eval_win_p1, eval_win_p2, eval_avg_reward = evaluate_agent(
                env,
                agent,
                num_games=100,
                opponent_type=opponent_type,
                opponent_pool=opponent_pool,
            )
            # Overall win rate as mean of P1 and P2 win rates
            eval_win_rate = 0.5 * (eval_win_p1 + eval_win_p2)
            eval_win_rates.append(eval_win_rate)
            eval_rewards.append(eval_avg_reward)
            eval_win_rates_p1.append(eval_win_p1)
            eval_win_rates_p2.append(eval_win_p2)
            q_table_sizes.append(len(agent.q_table))

            # Auto-curriculum advancement based on evaluation performance.
            # If combined eval win rate exceeds 80%, advance to the next phase (up to phase 2).
            if eval_win_rate > 0.80 and current_curriculum_phase < 2:
                current_curriculum_phase += 1

            print(
                f"Episodes {start+1:6d}-{ep+1:6d} | "
                f"Eval win P1: {eval_win_p1:6.3f} | "
                f"Eval win P2: {eval_win_p2:6.3f} | "
                f"Eval avg reward: {eval_avg_reward:8.3f} | "
                f"phase_opponent: {opponent_type} | "
                f"curriculum_phase: {current_curriculum_phase}"
            )

        # Every 5000 episodes, snapshot the current Q-table into the historical pool
        if (ep + 1) % 5000 == 0:
            opponent_pool.append(copy.deepcopy(agent.q_table))
            # Cap the historical pool size to avoid excessive memory use
            if len(opponent_pool) > 5:
                opponent_pool.pop(0)

    # Save Q-table, training stats, and clean evaluation metrics
    with MODEL_PATH.open("wb") as f:
        pickle.dump(agent.q_table, f)
    np.savez_compressed(
        STATS_PATH,
        rewards=rewards,
        winners=winners,
        episode_lengths=episode_lengths,
        num_episodes=num_episodes,
        eval_win_rates=np.array(eval_win_rates, dtype=np.float32),
        eval_rewards=np.array(eval_rewards, dtype=np.float32),
        eval_win_rates_p1=np.array(eval_win_rates_p1, dtype=np.float32),
        eval_win_rates_p2=np.array(eval_win_rates_p2, dtype=np.float32),
        q_table_sizes=np.array(q_table_sizes, dtype=np.int32),
    )
    print(f"Saved Q-table to {MODEL_PATH}")
    print(f"Saved training stats to {STATS_PATH}")


if __name__ == "__main__":
    train()

