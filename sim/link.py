import random
from collections import deque
from sim.packet import Packet


class Link:
    """
    Simulates a single bottleneck network link with:
    - finite bandwidth
    - finite queue
    - random wireless loss
    - queue-induced delay (RTT increase)
    """

    def __init__(
        self,
        capacity: int,
        queue_limit: int,
        base_rtt: float,
        noise_prob: float,
    ):
        self.capacity = capacity              # packets per timestep
        self.queue_limit = queue_limit        # max packets in queue
        self.base_rtt = base_rtt
        self.noise_prob = noise_prob

        self.queue = deque()                  # FIFO queue

    def enqueue(self, packets):
        """
        Add incoming packets to the queue.
        Returns number of packets dropped due to congestion.
        """
        dropped = 0
        for pkt in packets:
            if len(self.queue) < self.queue_limit:
                self.queue.append(pkt)
            else:
                dropped += 1  # congestion loss
        return dropped

    def step(self):
        """
        Process one timestep.
        Returns:
          delivered_packets: list[Packet]
          current_rtt: float
          wireless_drops: int
        """
        delivered = []
        wireless_drops = 0

        # Compute queueing delay
        queue_delay = len(self.queue) / self.capacity
        current_rtt = self.base_rtt + queue_delay

        # Transmit up to capacity packets
        for _ in range(min(self.capacity, len(self.queue))):
            pkt = self.queue.popleft()

            # Wireless loss
            if random.random() < self.noise_prob:
                wireless_drops += 1
                continue

            delivered.append(pkt)

        return delivered, current_rtt, wireless_drops