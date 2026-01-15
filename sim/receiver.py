class Receiver:
    """
    Receiver generates ACKs for delivered packets.
    ACKs are delayed by RTT before reaching the sender.
    """

    def __init__(self):
        # list of (ack_time, packet)
        self.pending_acks = []

    def receive(self, packets, current_time, rtt):
        """
        Called when packets arrive from the link.

        packets: list[Packet]
        current_time: int
        rtt: float
        """
        ack_time = current_time + int(round(rtt))
        for pkt in packets:
            self.pending_acks.append((ack_time, pkt))

    def get_acks(self, current_time):
        """
        Return ACKed packets whose ACK time has arrived.
        """
        arrived = []
        remaining = []

        for ack_time, pkt in self.pending_acks:
            if ack_time <= current_time:
                arrived.append(pkt)
            else:
                remaining.append((ack_time, pkt))

        self.pending_acks = remaining
        return arrived
