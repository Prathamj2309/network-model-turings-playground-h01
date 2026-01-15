from sim.packet import Packet


class Sender:
    """
    Sender maintains sending rate, tracks ACKs,
    and computes observable network metrics.
    """

    def __init__(self, initial_rate: int):
        self.send_rate = initial_rate

        # Tracking packets in flight
        self.in_flight = []

        # Metrics
        self.acked_packets = 0
        self.lost_packets = 0
        self.rtt_samples = []

    def send(self, current_time):
        """
        Create packets to send this timestep.
        """
        packets = []
        for _ in range(self.send_rate):
            pkt = Packet(send_time=current_time)
            packets.append(pkt)
            self.in_flight.append(pkt)
        return packets

    def receive_acks(self, acked_packets, current_time):
        """
        Process ACKed packets.
        """
        for pkt in acked_packets:
            if pkt in self.in_flight:
                self.in_flight.remove(pkt)
                self.acked_packets += 1
                rtt = current_time - pkt.send_time
                self.rtt_samples.append(rtt)

    def detect_loss(self, current_time, timeout=10):
        """
        Infer packet loss using a simple timeout.
        """
        still_in_flight = []
        lost = 0

        for pkt in self.in_flight:
            if current_time - pkt.send_time > timeout:
                lost += 1
            else:
                still_in_flight.append(pkt)

        self.in_flight = still_in_flight
        self.lost_packets += lost
        return lost


    def get_metrics(self):
        """
        Return observable metrics for this timestep.
        """
        avg_rtt = (
            sum(self.rtt_samples) / len(self.rtt_samples)
            if self.rtt_samples else 0
        )

        metrics = {
            "throughput": self.acked_packets,
            "avg_rtt": avg_rtt,
            "loss": self.lost_packets,
            "send_rate": self.send_rate,
        }

        # Reset timestep metrics
        self.acked_packets = 0
        self.lost_packets = 0
        self.rtt_samples = []

        return metrics

    def adjust_rate(self, delta: int):
        """
        Update sending rate.
        """
        self.send_rate = max(1, self.send_rate + delta)
