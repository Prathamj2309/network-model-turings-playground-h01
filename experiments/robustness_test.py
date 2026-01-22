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

    # --- NEW: post-stabilization buffers ---
    stab_thr, stab_loss, stab_util, stab_rtt = [], [], [], []

    ema_rtt = None
    capacity = env.link.capacity

    stabilized = False  # NEW

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

        utilization = metrics["throughput"] / capacity

        # --- NEW: detect stabilization ---
        if not stabilized and agent.epsilon == 0.0:
            stabilized = True

        # --- Collect stabilized-only stats ---
        if stabilized:
            stab_thr.append(metrics["throughput"])
            stab_loss.append(metrics["loss"])
            stab_util.append(utilization)
            if ema_rtt is not None:
                stab_rtt.append(ema_rtt)

        # --- Full history (unchanged, for printing) ---
        thr.append(metrics["throughput"])
        loss.append(metrics["loss"])
        util.append(utilization)
        ema_rtt_series.append(ema_rtt if ema_rtt is not None else 0.0)
        history.append((metrics, action, ema_rtt))

    return (
        thr,
        ema_rtt_series,
        loss,
        util,
        history,
        stab_thr,
        stab_loss,
        stab_util,
        stab_rtt,
    )


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

        (
            thr,
            ema_rtt,
            loss,
            util,
            history,
            stab_thr,
            stab_loss,
            stab_util,
            stab_rtt,
        ) = run_single_env(env, agent)

        # --- GLOBAL (stabilized only) ---
        if stab_thr:
            all_thr.extend(stab_thr)
            all_loss.extend(stab_loss)
            all_util.extend(stab_util)
            all_rtt.extend(stab_rtt)

        print(f"\n=== Environment {i:02d} (capacity={capacity}) (last {PRINT_LAST} steps) ===")
        print("Time | Rate | Thr | Cap | Util | AvgRTT | Loss | Action")
        print("-" * 75)

        for metrics, action, rtt_val in history[-PRINT_LAST:]:
            u = metrics["delivered_packets"] / capacity
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

        if stab_thr:
            print(
                f"Env {i:02d} Stabilized Avg → "
                f"Thr={statistics.mean(stab_thr):.2f}, "
                f"Util={statistics.mean(stab_util):.2f}, "
                f"RTT={statistics.mean(stab_rtt):.2f}, "
                f"Loss={statistics.mean(stab_loss):.2f}"
            )
        else:
            print(f"Env {i:02d} Stabilized Avg → (no stabilized steps)")


    print("\n=== GLOBAL ROBUSTNESS SUMMARY (POST-STABILIZATION) ===")
    print(f"Avg Throughput  : {statistics.mean(all_thr):.2f}")
    print(f"Avg Utilization : {statistics.mean(all_util):.2f}")
    print(f"Avg RTT         : {statistics.mean(all_rtt):.2f}")
    print(f"Avg Loss        : {statistics.mean(all_loss):.2f}")


if __name__ == "__main__":
    main()