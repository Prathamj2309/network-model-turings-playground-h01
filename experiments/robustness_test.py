# experiments/robustness_test.py

import random
import statistics

from sim.environment import Environment
from sim.sender import Sender
from sim.link import Link
from sim.receiver import Receiver
from agents.rl_agent import RLAgent


NUM_ENVS = 10
TOTAL_STEPS = 1000
PRINT_LAST = 15

EMA_ALPHA = 0.1   # RTT smoothing factor


def make_random_environment():
    capacity = random.randint(2, 8)
    queue_limit = random.randint(8, 40)
    base_rtt = random.uniform(4.0, 10.0)
    noise_prob = random.uniform(0.01, 0.05)

    sender = Sender(initial_rate=capacity)
    link = Link(
        capacity=capacity,
        queue_limit=queue_limit,
        base_rtt=base_rtt,
        noise_prob=noise_prob,
    )
    receiver = Receiver()

    env = Environment(sender, link, receiver)
    return env, base_rtt


def run_single_env(env, agent):
    thr, loss, util = [], [], []
    ema_rtt_series = []
    history = []

    ema_rtt = None
    capacity = env.link.capacity   # TRUE capacity (evaluation-only)

    for step in range(TOTAL_STEPS):
        metrics = env.step()

        obs = {
            "throughput": metrics["throughput"],
            "avg_rtt": metrics["avg_rtt"],
            "loss": metrics["loss"],
            "send_rate": metrics["send_rate"],
        }

        action = agent.act(obs)
        env.sender.adjust_rate(action)

        # --- EMA RTT update ---
        current_rtt = metrics["avg_rtt"]
        if current_rtt > 0:
            if ema_rtt is None:
                ema_rtt = current_rtt
            else:
                ema_rtt = (1 - EMA_ALPHA) * ema_rtt + EMA_ALPHA * current_rtt

        thr.append(metrics["throughput"])
        loss.append(metrics["loss"])

        utilization = metrics["throughput"] / capacity
        util.append(utilization)

        ema_rtt_series.append(ema_rtt if ema_rtt is not None else 0.0)
        history.append((metrics, action, ema_rtt))

    return thr, ema_rtt_series, loss, util, history


def main():
    all_thr, all_rtt, all_loss, all_util = [], [], [], []

    for i in range(NUM_ENVS):
        env, base_rtt = make_random_environment()
        capacity = env.link.capacity

        agent = RLAgent(
            base_rtt=base_rtt,
            epsilon=0.2,
            epsilon_min=0.02,
            epsilon_decay=0.995,
        )

        thr, ema_rtt, loss, util, history = run_single_env(env, agent)

        all_thr.extend(thr)
        all_rtt.extend([x for x in ema_rtt if x > 0])
        all_loss.extend(loss)
        all_util.extend(util)

        print(f"\n=== Environment {i:02d} (capacity={capacity}) (last {PRINT_LAST} steps) ===")
        print("Time | Rate | Thr | Cap | Util | AvgRTT | Loss | Action")
        print("-" * 75)

        for metrics, action, rtt_val in history[-PRINT_LAST:]:
            u = metrics["throughput"] / capacity
            print(
                f"{metrics['time']:>4} | "
                f"{metrics['send_rate']:>4} | "
                f"{metrics['throughput']:>3} | "
                f"{capacity:>3} | "
                f"{u:>4.2f} | "
                f"{rtt_val:>6.2f} | "
                f"{metrics['loss']:>4} | "
                f"{action:>6}"
            )

        print(
            f"Env {i:02d} Avg → "
            f"Thr={statistics.mean(thr):.2f}, "
            f"Util={statistics.mean(util):.2f}, "
            f"RTT={statistics.mean([x for x in ema_rtt if x > 0]):.2f}, "
            f"Loss={statistics.mean(loss):.2f}"
        )

    print("\n=== GLOBAL ROBUSTNESS SUMMARY ===")
    print(f"Avg Throughput  : {statistics.mean(all_thr):.2f}")
    print(f"Avg Utilization : {statistics.mean(all_util):.2f}")
    print(f"Avg RTT         : {statistics.mean(all_rtt):.2f}")
    print(f"Avg Loss        : {statistics.mean(all_loss):.2f}")


if __name__ == "__main__":
    main()
