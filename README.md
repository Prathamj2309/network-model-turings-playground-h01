# Antar‑Drishti TCP – Reinforcement Learning Congestion Control for Wireless Networks

*See the real obstacles. Ignore the illusions.*

Antar‑Drishti TCP is a lightweight **reinforcement‑learning–based TCP congestion control agent** designed for noisy wireless networks such as 5G and Wi‑Fi. Unlike traditional TCP variants (Reno, Cubic, etc.), it learns to distinguish between **random packet loss caused by wireless interference** and **true congestion**, allowing it to maintain high throughput while keeping latency low.

---

## Problem Overview

Legacy TCP protocols assume:

> Packet loss = network congestion

In wireless networks:

> Packet loss = congestion *or* signal noise

This incorrect assumption leads to:

* Unnecessary reduction in sending rate
* Poor throughput
* Increased latency
* Underutilized bandwidth

Antar‑Drishti TCP replaces the congestion‑window update logic with an AI agent that learns optimal rate control behavior under uncertainty.

---

## Objectives

* Maximize throughput
* Minimize latency
* Remain stable under 1–5% random packet corruption
* Operate using sender‑side information only
* Stay within strict resource limits

---

## Adversary (Simulation Environment)

* Wireless link randomly corrupts 1–5% of packets
* Congestion may occur due to competing traffic
* TCP Reno and Cubic degrade significantly under these conditions

---

## Key Constraints

| Constraint          | Description                                |
| ------------------- | ------------------------------------------ |
| Sender‑side only    | No router queue size or receiver internals |
| No hidden variables | Must infer only from TCP metrics           |
| Model size          | ≤ 5 MB                                     |
| Inference time      | ≤ 5 ms per step                            |
| Deployment          | IoT‑friendly                               |

---

## Solution Architecture

```
Application
    |
    v
[ Antar‑Drishti TCP Agent ]  <- Reinforcement Learning Policy
    |
    v
TCP Socket Layer
    |
    v
Wireless Network (loss + congestion)
```

The agent replaces the traditional congestion‑window (cwnd) update logic.

---

## Observations (State Space)

The agent only uses standard TCP sender metrics:

* Smoothed RTT
* RTT variance
* Recent packet loss rate
* ACK inter‑arrival time
* Current congestion window (cwnd)
* Throughput estimate
* In‑flight packets

No privileged network information is used.

---

## Actions (Control Space)

Discrete actions:

* Increase cwnd (small / medium / aggressive)
* Decrease cwnd (small / medium)
* Keep cwnd unchanged

Or numerically:

```
Δcwnd ∈ { -4, -2, -1, 0, +1, +2, +4 }
```

---

## Training Method

* Environment: manual simulator (python)
* Episodes include:

  * Random noise levels
  * Variable bandwidth
  * Cross traffic
  * RTT changes

---


## Project Structure

```
antar-drishti/
│
├── README.md
│
├── sim/                    # Simulation core (no RL here)
│   ├── __init__.py
│   ├── packet.py           # Packet data structure
│   ├── sender.py           # Sender logic (rate, cwnd)
│   ├── link.py             # Bandwidth, queue, noise model
│   ├── receiver.py         # ACK generation
│   └── environment.py      # Ties sender, link, receiver
│
├── agents/                 # Control logic (pluggable)
│   ├── __init__.py
│   ├── base_agent.py       # Abstract interface
│   ├── reno_agent.py       # Baseline implementation
│   └── rl_agent.py         # Antar‑Drishti agent
│
├── metrics/
│   ├── logger.py           # Throughput, RTT, loss tracking
│   └── plots.py            # Visualization
│
├── experiments/
│   ├── run_baseline.py     # Reno vs noise
│   └── run_rl.py           # RL experiments
│
├── configs/
│   ├── default.yaml        # Link params, noise, RTT
│   └── stress.yaml
│
└── requirements.txt        # Minimal dependencies
```

## Benchmarking

Compared against:

* TCP Reno

Metrics:

* Average throughput
* latency
* Packet loss rate
* Fairness

---

## Deployment

Supported targets:

* Linux TCP module integration
* User‑space TCP stacks
* QUIC sender adaptation
* ARM‑based IoT devices

---

## Philosophy

Just as *Antar‑Drishti* refers to inner clarity, the agent focuses on distinguishing real congestion from random wireless loss and responding only when necessary.

---

## License

MIT License

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss the proposal.

---

## Contact

Built for experimentation in next‑generation network congestion control.

---