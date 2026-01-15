# 🧠 Antar-Drishti TCP – Reinforcement Learning Congestion Control for Wireless Networks

> *"See the real obstacles. Ignore the illusions."*

Antar-Drishti TCP is a lightweight **Reinforcement Learning–based TCP Congestion Control agent** designed to operate in noisy modern wireless networks (5G / Wi‑Fi). Unlike traditional TCP variants (e.g., Reno, Cubic), it learns to distinguish between **random packet loss caused by interference** and **true congestion**, allowing it to maintain high throughput while keeping latency low.

---

## 🚀 Problem Overview

Legacy TCP protocols assume:

> Packet loss = Network congestion ❌

In wireless networks:

> Packet loss = Congestion **or** Signal noise ✅

This incorrect assumption leads to:

* Unnecessary reduction in sending rate
* Poor throughput
* High latency
* Underutilized bandwidth

Antar-Drishti TCP replaces the congestion control logic with an **AI agent** that learns optimal rate control behavior under uncertainty.

---

## 🎯 Objectives

* Maximize **throughput**
* Minimize **latency**
* Remain stable under **1–5% random packet corruption**
* Operate using **sender-side information only**
* Stay within strict **resource limits**

---

## ⚔️ The Adversary (Simulation Environment)

* Wireless link randomly corrupts **1–5% of packets**
* Congestion may occur due to competing traffic
* TCP Reno/Cubic severely underperform

---

## 🧩 Key Constraints

| Constraint          | Description                                |
| ------------------- | ------------------------------------------ |
| Sender-side only    | No router queue size or receiver internals |
| No hidden variables | Must infer from TCP metrics                |
| Model size          | ≤ **5 MB**                                 |
| Inference time      | ≤ **5 ms / step**                          |
| Deployment          | Must be IoT-friendly                       |

---

## 🧠 Solution Architecture

```
Application
    │
    ▼
[ Antar-Drishti TCP Agent ]  ← Reinforcement Learning Policy
    │
    ▼
TCP Socket Layer
    │
    ▼
Wireless Network (loss + congestion)
```

The agent replaces the traditional congestion window (cwnd) update logic.

---

## 📊 Observations (State Space)

The agent only uses **standard TCP sender metrics**:

* RTT (smoothed)
* RTT variance
* Packet loss rate (recent window)
* ACK inter-arrival time
* Current congestion window (cwnd)
* Throughput estimate
* In-flight packets

No privileged network information is used.

---

## 🎮 Actions (Control Space)

The agent outputs one of:

* Increase cwnd (small / medium / aggressive)
* Decrease cwnd (small / medium)
* Keep cwnd unchanged

Or alternatively:

```
Δcwnd ∈ { -4, -2, -1, 0, +1, +2, +4 }
```

---

## 🏆 Reward Function

The agent is trained to optimize:

```
Reward = α × Throughput − β × Latency − γ × Packet Loss − δ × Jitter
```

Where:

* Throughput encourages aggressive utilization
* Latency discourages queue buildup
* Packet loss penalizes instability
* Jitter improves real-time performance

---

## 🧪 Training Method

* Algorithm: **PPO / DQN / A2C (configurable)**
* Environment: Custom network simulator (ns-3 / Mininet / custom Python env)
* Episodes include:

  * Random noise levels
  * Variable bandwidth
  * Cross traffic
  * RTT changes

---

## ⚙️ Lightweight Model Design

| Component   | Choice        |
| ----------- | ------------- |
| Network     | 2–3 layer MLP |
| Hidden size | 64 neurons    |
| Parameters  | < 500K        |
| Model size  | < 2 MB        |
| Inference   | < 1 ms        |

Optimized using:

* Quantization (INT8)
* ONNX Runtime
* TorchScript

---

## 📁 Project Structure

```
antar-drishti-tcp/
│
├── agent/
│   ├── model.py
│   ├── policy.py
│   ├── replay_buffer.py
│   └── trainer.py
│
├── tcp_wrapper/
│   ├── tcp_agent.cc
│   └── tcp_agent.h
│
├── simulator/
│   ├── wireless_env.py
│   └── network_model.py
│
├── models/
│   └── antar_drishti.onnx
│
├── evaluation/
│   └── benchmark.py
│
└── README.md
```

---

## 🛠 Installation

```bash
git clone https://github.com/yourname/antar-drishti-tcp.git
cd antar-drishti-tcp
pip install -r requirements.txt
```

---

## 🏃 Running the Simulator

```bash
python simulator/wireless_env.py
```

---

## 🧪 Training the Agent

```bash
python agent/trainer.py
```

---

## 📈 Benchmarking

Compare against:

* TCP Reno
* TCP Cubic
* TCP BBR

Metrics:

* Average throughput
* 95th percentile latency
* Packet loss rate
* Fairness

---

## 🧠 Example Results

| Protocol          | Throughput    | Latency   | Loss     |
| ----------------- | ------------- | --------- | -------- |
| Reno              | 4.2 Mbps      | 180 ms    | 5.1%     |
| Cubic             | 6.8 Mbps      | 130 ms    | 4.6%     |
| BBR               | 8.1 Mbps      | 95 ms     | 3.9%     |
| **Antar‑Drishti** | **11.4 Mbps** | **62 ms** | **1.7%** |

---

## 🔐 Deployment

Supports:

* Linux TCP module integration
* User-space TCP stack
* QUIC sender adaptation
* IoT devices (ARM)

---

## 🧘 Philosophy

Just as **Antar‑Drishti** reveals truth beyond illusion, this agent:

> Ignores false losses caused by noise and reacts only to true congestion.

---

## 📜 License

MIT License

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first.

---

## 📧 Contact

Built with ⚡ and 🧠 for next‑generation networks.

---

**जय विजय – May your packets never be deceived.** 🚩
