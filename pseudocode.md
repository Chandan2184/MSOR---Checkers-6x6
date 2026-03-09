# Q-Learning Algorithm (off-policy)

## Input

- **Discount rate \(\gamma\) (gamma):** Importance of future rewards (e.g. \(\gamma = 0.99\)).
- **State-dependent exploration parameter \(N_0\):** Constant used in \(\varepsilon(s) = N_0 / (N_0 + N(s))\); e.g. \(N_0 = 100\).
- **Learning rate \(\alpha(s,a)\):** Dynamic step size,
  \[
  \alpha(s,a) = \max\left(0.005,\; \frac{1}{\sqrt{N(s,a) + 1}}\right),
  \]
  where \(N(s,a)\) is the visit count for \((s,a)\).
- **Number of episodes \(K\):** Total number of training episodes.

## Initialization

- \(i = 0\) (episode counter).
- \(Q(s,a) = 0\) for all \((s,a)\) when first visited; the Q-table is implemented as a dictionary (default 0.0 for unvisited pairs).
- \(N(s,a) = 0\) for all \((s,a)\) (visit count for state–action pairs).
- \(N(s) = 0\) for all \(s\) (visit count for states, used for \(\varepsilon(s)\)).

## Output

- The estimated state-action value function \(Q \approx Q^*\) (optimal Q).

---

## Algorithm

1. **Sample initial state \(s_0\):** Reset environment; obtain observation and convert to canonical state \(s_0\). Set current state \(s_t = s_0\).

2. **Sample action \(a_0\) for \(s_0\)** when following **\(\varepsilon\)-greedy policy** w.r.t. \(Q\):
   - Compute \(\varepsilon(s) = N_0 / (N_0 + N(s))\); increment \(N(s)\).
   - With probability \(\varepsilon(s)\), choose a random **legal** action; otherwise choose \(a = \arg\max_{a' \in \mathcal{A}(s)} Q(s, a')\).

3. **while** \(i < K\) **do**

4. **Take action \(a_t\) in environment and observe \(r_t\), \(s_{t+1}\):**  
   Execute the agent’s move (and opponent’s reply if applicable). Observe cumulative reward \(r_t\) over this cycle and next state \(s_{t+1}\) (or terminal). Restrict to **legal** actions only.

5. **\(i \leftarrow i + 1\)**

6. **(Optional) Update exploration:** \(\varepsilon(s)\) is state-dependent via \(\varepsilon(s) = N_0/(N_0 + N(s))\); no global \(\varepsilon\) decay is required.

7. **Compute TD target \(\delta_t\):**
   - If \(s_{t+1}\) is **terminal:** \(\delta_t = r_t\).
   - **Otherwise (bootstrapping):** \(\delta_t = r_t + \gamma \max_{a'} Q(s_{t+1}, a')\), where the maximum is taken over **legal** actions \(a' \in \mathcal{A}(s_{t+1})\) only.

8. **Update estimated state-action value:**
   \[
   Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha(s_t, a_t) \cdot \bigl( \delta_t - Q(s_t, a_t) \bigr),
   \]
   where \(\alpha(s_t, a_t) = \max\bigl(0.005,\, 1\big/\sqrt{N(s_t,a_t)+1}\,\bigr)\); increment \(N(s_t, a_t)\).

9. **Update state:**
   - If not terminated or truncated: \(s_t \leftarrow s_{t+1}\).
   - Otherwise: sample new initial state \(s_0\) and set \(s_t \leftarrow s_0\).

10. **Sample action \(a_t\) for \(s_t\)** when following **\(\varepsilon\)-greedy policy** w.r.t. \(Q\) (over legal actions only).

**end while**

---

## Backward-pass variant (as implemented)

In the codebase, updates are performed in a **backward pass** after each episode:

- **During episode:** Collect in order the list of transitions \((s, a, r, s', \mathcal{A}(s'))\) (agent move + opponent reply per step; \(r\) is cumulative over that cycle).
- **After episode:** For each transition in **reverse** order, compute \(\delta = r + \gamma \max_{a' \in \mathcal{A}(s')} Q(s', a')\) (or \(\delta = r\) if \(s'\) is terminal), then apply
  \[
  Q(s,a) \leftarrow Q(s,a) + \alpha(s,a) \cdot \bigl( \delta - Q(s,a) \bigr),
  \]
  with \(\alpha(s,a)\) and \(N(s,a)\) as above. Processing in reverse ensures that \(Q(s', \cdot)\) is already updated when used in \(\delta\).

---

## Notes

- **Off-policy:** The update uses \(\max_{a'} Q(s', a')\) (greedy target); the behavior policy is \(\varepsilon\)-greedy.
- **Legal actions:** All maximizations and action selections are over **legal** actions only (e.g. forced capture, multi-jump in checkers).
- **State representation:** States are canonical (e.g. 18-tuple of playable squares); symmetry is used so both sides share the same representation.
