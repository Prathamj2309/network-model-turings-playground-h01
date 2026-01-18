import random
from agents.base_agent import BaseAgent

class RLAgent(BaseAgent):
    def __init__(
        self,
        base_rtt: float,
        actions=(-2, -1, 0, 1, 2),
        alpha=0.1,
        gamma=0.9,
        epsilon=0.2,
        epsilon_min=0.02,
        epsilon_decay=0.995,
        osc_penalty=0.2,
        best_thr_ema_alpha=0.05,
    ):
        self.base_rtt = base_rtt
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.osc_penalty = osc_penalty
        self.best_thr_ema_alpha = best_thr_ema_alpha

        self.Q = {}

        # Sender-side memory
        self.prev_state = None
        self.prev_action = None
        self.prev_obs = None
        self.prev_send_rate = None

        # Throughput anchor
        self.best_thr_ema = 0.0
        
        # [NEW] RTT Smoothing to ignore wireless noise
        self.rtt_ema = base_rtt  

    # --------------------------------------------------
    # State discretization
    # --------------------------------------------------

    def _discretize_rtt(self, avg_rtt):
        # Use smoothed RTT for state to prevent state-flickering
        if avg_rtt <= 0:
            return 0
        ratio = (avg_rtt - self.base_rtt) / self.base_rtt
        
        # [Adjusted] Slightly tighter bins to detect buffer bloat earlier
        if ratio < 0.05: return 0  # Very clean
        elif ratio < 0.2: return 1 # Mild load
        elif ratio < 0.5: return 2 # Heavy load
        elif ratio < 1.0: return 3 # Buffer bloat
        else: return 4             # Congestion collapse

    # ... [Keep other discretizers: _discretize_efficiency, _discretize_rate_trend, etc.] ...
    def _discretize_efficiency(self, thr, rate):
        ratio = thr / max(rate, 1)
        if ratio >= 0.7: return 0
        elif ratio >= 0.4: return 1
        else: return 2

    def _discretize_rate_trend(self, rate):
        if self.prev_send_rate is None: return 0
        if rate > self.prev_send_rate: return 1
        elif rate < self.prev_send_rate: return 2
        else: return 0

    def _discretize_delivery(self, thr, loss):
        delivery = thr / max(thr + loss, 1)
        if delivery >= 0.6: return 0
        elif delivery >= 0.3: return 1
        else: return 2

    def _get_state(self, obs):
        # [NEW] Update RTT EMA
        # Alpha of 0.2 means we trust the history more than the instant spike
        self.rtt_ema = 0.8 * self.rtt_ema + 0.2 * obs["avg_rtt"]

        return (
            self._discretize_rtt(self.rtt_ema), # Use EMA here
            self._discretize_efficiency(obs["throughput"], obs["send_rate"]),
            self._discretize_rate_trend(obs["send_rate"]),
            self._discretize_delivery(obs["throughput"], obs["loss"]),
        )

    # --------------------------------------------------
    # Reward (Improved for Stability & Latency)
    # --------------------------------------------------

    def _compute_reward(self, obs):
        if self.prev_obs is None:
            return 0.0

        p = self.prev_obs
        c = obs

        # 1. Efficiency Delta
        eff_p = p["throughput"] / max(p["send_rate"], 1)
        eff_c = c["throughput"] / max(c["send_rate"], 1)
        d_eff = eff_c - eff_p

        # 2. RTT Delta (Relative Change)
        if self.base_rtt > 0:
            # Normalize by Base RTT to keep scale consistent
            d_rtt = (c["avg_rtt"] - p["avg_rtt"]) / self.base_rtt
            
            # [NEW] Absolute RTT Penalty (Fixes Buffer Bloat)
            # Penalize simply for existing in a high-latency state
            abs_rtt_penalty = (c["avg_rtt"] - self.base_rtt) / self.base_rtt
        else:
            d_rtt = 0.0
            abs_rtt_penalty = 0.0

        # 3. Delivery Delta
        del_p = p["throughput"] / max(p["throughput"] + p["loss"], 1)
        del_c = c["throughput"] / max(c["throughput"] + c["loss"], 1)
        d_del = del_c - del_p

        # [Adjusted] Weights
        reward = (
            + 2.0 * d_eff
            + 3.0 * d_del         # Increased priority on delivery
            - 1.0 * d_rtt         # Penalize sudden spikes
            - 0.5 * abs_rtt_penalty # [NEW] Constant pressure to drain queue
        )

        # Pathological delivery penalty
        if del_c < 0.5: # [Adjusted] Stricter threshold
            reward -= 5.0 * (0.5 - del_c)

        # Oscillation penalty
        if self.prev_action is not None:
            reward -= self.osc_penalty * abs(self.prev_action)

        # -------------------------------------------------------
        # Throughput Anchor (Logic Fix)
        # -------------------------------------------------------
        loss_ratio = c["loss"] / max(c["send_rate"], 1)
        
        # [FIXED] Relaxed "Safe" definition
        # If Loss is low, we tolerate slight RTT jitter (up to 15% delta)
        # This prevents the Env 03 "Panic Drop"
        safe = (loss_ratio < 0.05) and (d_rtt <= 0.15) 

        if safe and self.best_thr_ema > 0:
            thr_ratio = c["throughput"] / max(self.best_thr_ema, 1e-6)

            # Drifting down while safe is bad
            if thr_ratio < 0.85:
                reward -= 1.5 * (0.85 - thr_ratio)

            # Holding near best is good
            elif thr_ratio > 0.90:
                reward += 1.0 * (thr_ratio - 0.90) # Increased incentive

        return reward

    def _ensure_state(self, s):
        if s not in self.Q:
            self.Q[s] = {a: 0.0 for a in self.actions}

    def act(self, observation):
        state = self._get_state(observation)
        self._ensure_state(state)

        # -------- Q-learning update --------
        if self.prev_state is not None:
            r = self._compute_reward(observation)
            best_next = max(self.Q[state].values())
            old = self.Q[self.prev_state][self.prev_action]
            self.Q[self.prev_state][self.prev_action] = (
                old + self.alpha * (r + self.gamma * best_next - old)
            )

        # -------- update throughput anchor --------
        thr = observation["throughput"]
        self.best_thr_ema = max(
            self.best_thr_ema,
            (1 - self.best_thr_ema_alpha) * self.best_thr_ema
            + self.best_thr_ema_alpha * thr,
        )

        # -------- LOSS-REGIME ACTION MASKING --------
        send_rate = observation["send_rate"]
        loss = observation["loss"]

        loss_ratio = loss / max(send_rate, 1)
        delivery = thr / max(thr + loss, 1)

        allowed_actions = list(self.actions)

        if loss_ratio > 0.15 or delivery < 0.3:
            allowed_actions = [a for a in allowed_actions if a <= 0]
        elif loss_ratio > 0.08:
            allowed_actions = [a for a in allowed_actions if a <= 1]

        if send_rate <= 0.1 * self.best_thr_ema and loss_ratio <= 0.08:
            allowed_actions = [a for a in allowed_actions if a >= 0]


        # -------- Action selection --------
        if random.random() < self.epsilon:
            action = random.choice(allowed_actions)
        else:
            best_q = max(self.Q[state][a] for a in allowed_actions)
            action = random.choice(
                [a for a in allowed_actions if self.Q[state][a] == best_q]
            )

        # -------- Bookkeeping --------
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        else:
            self.epsilon = 0

        self.prev_state = state
        self.prev_action = action
        self.prev_obs = observation.copy()
        self.prev_send_rate = send_rate

        return action
