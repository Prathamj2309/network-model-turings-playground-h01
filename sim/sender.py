from sim.packet import Packet


class Sender:
    """
    Sender maintains sending rate, tracks packets in flight,
    estimates RTT, and infers loss using an adaptive timeout.
    """

    def __init__(self, initial_rate: int):
        self.send_rate = initial_rate

        # Packets currently in flight
        self.in_flight = []

        # Per-timestep metrics
        self.acked_packets = 0
        self.lost_packets = 0
        self.rtt_samples = []

        # -------- RTT / RTO estimation (TCP-like) --------
        self.srtt = None
        self.rttvar = None
        self.rto = 10  # conservative initial timeout

    # --------------------------------------------------
    # Sending
    # --------------------------------------------------

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

    # --------------------------------------------------
    # ACK processing + RTT estimation
    # --------------------------------------------------

    def receive_acks(self, acked_packets, current_time):
        """
        Process ACKed packets and update RTT estimates.
        """
        for pkt in acked_packets:
            if pkt in self.in_flight:
                self.in_flight.remove(pkt)
                self.acked_packets += 1

                rtt = current_time - pkt.send_time
                self.rtt_samples.append(rtt)

                # ----- TCP-style RTT estimation -----
                if self.srtt is None:
                    self.srtt = rtt
                    self.rttvar = rtt / 2
                else:
                    alpha = 0.125
                    beta = 0.25

                    self.rttvar = (
                        (1 - beta) * self.rttvar
                        + beta * abs(self.srtt - rtt)
                    )
                    self.srtt = (1 - alpha) * self.srtt + alpha * rtt

                # ----- Adaptive RTO with bounds -----
                self.rto = self.srtt + 4 * self.rttvar
                self.rto = min(max(self.rto, 2), 50)

    # --------------------------------------------------
    # Loss detection via adaptive timeout
    # --------------------------------------------------

    def detect_loss(self, current_time):
        """
        Infer packet loss using adaptive RTO.
        """
        still_in_flight = []
        lost = 0

        timeout = self.rto

        for pkt in self.in_flight:
            if current_time - pkt.send_time > timeout:
                lost += 1
            else:
                still_in_flight.append(pkt)

        self.in_flight = still_in_flight
        self.lost_packets += lost
        return lost

    # --------------------------------------------------
    # Observable metrics
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Rate control
    # --------------------------------------------------

    def adjust_rate(self, delta: int):
        """
        Update sending rate.
        """
        self.send_rate = max(1, self.send_rate + delta)
