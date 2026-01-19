from sim.sender import Sender
from sim.link import Link
from sim.receiver import Receiver


class Environment:
    """
    Coordinates Sender, Link, and Receiver.
    Advances time and exposes step-based interaction.
    """

    def __init__(
        self,
        sender: Sender,
        link: Link,
        receiver: Receiver,
    ):
        self.sender = sender
        self.link = link
        self.receiver = receiver

        self.time = 0

    def step(self):
        """
        Advance the simulation by one timestep.
        Returns observable metrics.
        """

        # 1. Sender sends packets
        outgoing_packets = self.sender.send(self.time)

        # 2. Enqueue packets into the link
        congestion_drops = self.link.enqueue(outgoing_packets)

        # 3. Link processes packets
        delivered_packets, rtt, wireless_drops = self.link.step()

        # 4. Receiver schedules ACKs
        self.receiver.receive(
            delivered_packets,
            current_time=self.time,
            rtt=rtt
        )

        # 5. Receiver delivers ACKs whose time has arrived
        acked_packets = self.receiver.get_acks(self.time)
        acks_received = len(acked_packets)

        # 6. Sender processes ACKs
        self.sender.receive_acks(acked_packets, self.time)

        # 7. Sender infers loss
        inferred_loss = self.sender.detect_loss(self.time)

        # 8. Collect sender metrics
        metrics = self.sender.get_metrics()

        # --- SEMANTIC FIX ---
        metrics["throughput"] = acks_received
        metrics["delivered_packets"] = len(delivered_packets)  

        # Optional extra info (for debugging / analysis)
        metrics.update({
            "time": self.time,
            "congestion_drops": congestion_drops,
            "wireless_drops": wireless_drops,
            "inferred_loss": inferred_loss,
        })

        # Advance time
        self.time += 1

        return metrics
