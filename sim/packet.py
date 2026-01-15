class Packet:
    """
    Minimal packet abstraction for the simulator.

    A packet only knows when it was sent.
    RTT is inferred when an ACK is received.
    """

    def __init__(self, send_time: int):
        self.send_time = send_time
