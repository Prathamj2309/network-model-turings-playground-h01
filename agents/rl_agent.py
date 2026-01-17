# agents/rl_agent.py

import random
from agents.base_agent import BaseAgent


class RLAgent(BaseAgent):
    """
    Tabular Q-learning agent for congestion control.

    State:
      - Normalized queueing delay (4 RTT states)
      - Relative loss ratio (3 loss states)
      - Send efficiency / oversend awareness (3 states)

    Actions:
      - Small additive rate changes (-2 to +2)

    Design goals:
      - Robust across different link parameters
      - Scale-invariant reward
      - Stable behavior under stochastic loss
    """

    def __init__(
        self,
        base_rtt: float,
        actions=(-2, -1, 0, 1, 2),
        alpha=0.1,
        gamma=0.9,
        epsilon=0.2,
        epsilon_min=0.02,
        epsilon_decay=0.995,
        osc_penalty=0.3,
        rtt_weight=0.85,
    ):
        self.base_rtt = base_rtt
        self.actions = actions

        self.alpha = alpha
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.osc_penalty = osc_penalty
        self.rtt_weight = rtt_weight

        # Q-table: {(rtt_state, loss_state, send_ratio_state): {action: value}}
        self.Q = {}

        self.prev_state = None
        self.prev_action = None

    # ------------------------------------------------------------------
    # State discretization
    # ------------------------------------------------------------------

    def _discretize_rtt(self, avg_rtt):
        if avg_rtt <= 0:
            return 0  # no RTT sample yet

        q_delay_ratio = (avg_rtt - self.base_rtt) / self.base_rtt

        if q_delay_ratio < 0.1:
            return 0  # free-flow
        elif q_delay_ratio < 0.4:
            return 1  # early queue
        elif q_delay_ratio < 0.8:
            return 2  # queue building (knee)
        else:
            return 3  # heavy congestion

    def _discretize_loss(self, loss, send_rate):
        if send_rate <= 0:
            return 0

        loss_ratio = loss / send_rate

        if loss_ratio == 0:
            return 0  # clean
        elif loss_ratio < 0.3:
            return 1  # tolerable / wireless-like
        else:
            return 2  # severe / congestion-like

    def _discretize_send_ratio(self, send_rate, throughput):
        """
        How wasteful is the current sending rate?
        Dimensionless and environment-agnostic.
        """
        ratio = throughput/max(send_rate, 1)

        if ratio > 0.6:
            return 0   # efficient
        elif ratio > 0.3:
            return 1   # moderate oversend
        else:
            return 2   # severe oversend

    def _get_state(self, observation):
        rtt_state = self._discretize_rtt(observation["avg_rtt"])
        loss_state = self._discretize_loss(
            observation["loss"],
            observation["send_rate"],
        )
        send_ratio_state = self._discretize_send_ratio(
            observation["send_rate"],
            observation["throughput"],
        )

        return (rtt_state, loss_state, send_ratio_state)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self, observation, action):
        throughput = observation["throughput"]
        avg_rtt = observation["avg_rtt"]
        loss = observation["loss"]
        send_rate = observation["send_rate"]

        # Normalized quantities
        rtt_ratio = avg_rtt / self.base_rtt if avg_rtt > 0 else 1.0
        loss_ratio = loss / max(send_rate, 1)
        oversend_ratio = send_rate / max(throughput, 1)

        reward = (
            + throughput
            - self.rtt_weight * (rtt_ratio - 1.0)
            - 1.1 * loss_ratio
            - 0.40 * oversend_ratio
        )

        # Oscillation penalty (discourage thrashing)
        if self.prev_action is not None and avg_rtt > 0:
            reward -= self.osc_penalty * abs(action - self.prev_action)

        return reward

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _ensure_state(self, state):
        if state not in self.Q:
            self.Q[state] = {a: 0.0 for a in self.actions}

    # ------------------------------------------------------------------
    # Core RL logic
    # ------------------------------------------------------------------

    def act(self, observation: dict) -> int:
        state = self._get_state(observation)
        self._ensure_state(state)

        # --- Q-learning update ---
        if self.prev_state is not None:
            reward = self._compute_reward(observation, self.prev_action)
            self._ensure_state(self.prev_state)

            best_next_q = max(self.Q[state].values())
            old_q = self.Q[self.prev_state][self.prev_action]

            self.Q[self.prev_state][self.prev_action] = (
                old_q
                + self.alpha * (reward + self.gamma * best_next_q - old_q)
            )

        # --- Action selection (epsilon-greedy) ---
        if random.random() < self.epsilon:
            action = random.choice([0, 1, 2])  # conservative exploration
        else:
            best_q = max(self.Q[state].values())
            best_actions = [
                a for a, q in self.Q[state].items()
                if q == best_q
            ]
            action = random.choice(best_actions)

        # Epsilon decay
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )

        self.prev_state = state
        self.prev_action = action

        return action