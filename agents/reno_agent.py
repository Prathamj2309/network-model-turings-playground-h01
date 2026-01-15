# agents/reno_agent.py

from agents.base_agent import BaseAgent


class RenoAgent(BaseAgent):
    """
    Simplified TCP Reno-style congestion control agent.

    Behavior:
    - If any loss is detected -> decrease sending rate aggressively
    - Otherwise -> increase sending rate slowly
    """

    def __init__(self, increase_step=1, decrease_factor=0.5):
        self.increase_step = increase_step
        self.decrease_factor = decrease_factor

    def act(self, observation: dict) -> int:
        """
        Decide rate adjustment based on observation.

        observation keys expected:
          - loss
          - send_rate
        """

        loss = observation.get("loss", 0)
        send_rate = observation.get("send_rate", 1)

        # Any loss -> multiplicative decrease
        if loss > 0:
            new_rate = max(1, int(send_rate * self.decrease_factor))
            return new_rate - send_rate

        # No loss -> additive increase
        return self.increase_step