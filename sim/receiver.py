import random

class Receiver:
    """
    Receiver generates ACKs for delivered packets.
    ACKs are delayed by RTT before reaching the sender.
    ACKs themselves may be lost or jittered (wireless realism).
    """

    def __init__(self, ack_loss_prob=0.01, ack_jitter=0.5):
        self.pending_acks = []  # list of (ack_time, packet)
        self.ack_loss_prob = ack_loss_prob
        self.ack_jitter = ack_jitter

    def receive(self, packets, current_time, rtt):
        """
        Called when packets arrive from the link.
        """
        for pkt in packets:
            jitter = random.uniform(-self.ack_jitter, self.ack_jitter)
            ack_time = current_time + max(1, int(round(rtt + jitter)))
            self.pending_acks.append((ack_time, pkt))

    def get_acks(self, current_time):
        """
        Return ACKed packets whose ACK time has arrived.
        Some ACKs may be lost due to wireless noise.
        """
        arrived = []
        remaining = []

        for ack_time, pkt in self.pending_acks:
            if ack_time <= current_time:
                if random.random() >= self.ack_loss_prob:
                    arrived.append(pkt)
                # else: ACK lost
            else:
                remaining.append((ack_time, pkt))

        self.pending_acks = remaining
        return arrived
