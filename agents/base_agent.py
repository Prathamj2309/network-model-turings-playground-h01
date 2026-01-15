from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """
    Abstract base class for congestion control agents.
    """

    @abstractmethod
    def act(self, observation: dict) -> int:
        """
        Decide how to adjust the sending rate.

        observation: dict with keys like
          - throughput
          - avg_rtt
          - loss
          - send_rate

        returns:
          int delta to apply to send_rate
        """
        pass
