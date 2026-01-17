# experiments/run_rl.py

from sim.environment import Environment
from sim.sender import Sender
from sim.link import Link
from sim.receiver import Receiver

from agents.rl_agent import RLAgent


TOTAL_STEPS = 450
EVAL_START = 420   # last 30 steps


def main():
    # --- Network setup (same as Reno for fair comparison) ---
    sender = Sender(initial_rate=5)

    link = Link(
        capacity=4,
        queue_limit=10,
        base_rtt=5,
        noise_prob=0.2,
    )

    receiver = Receiver()
    env = Environment(sender, link, receiver)

    # --- RL Agent ---
    agent = RLAgent(
        base_rtt=5,
        actions=(-2, -1, 0, 1, 2),
        alpha=0.1,
        gamma=0.9,
        epsilon=0.2,
        epsilon_min=0.02,
        epsilon_decay=0.995,
        osc_penalty=0.3,
    )

    # --- Metrics for evaluation window ---
    eval_thr = []
    eval_rtt = []
    eval_loss = []

    print("Time | Rate | Thr | RTT    | Loss | Action")
    print("-" * 50)

    for step in range(TOTAL_STEPS):
        metrics = env.step()

        observation = {
            "throughput": metrics["throughput"],
            "avg_rtt": metrics["avg_rtt"],
            "loss": metrics["loss"],
            "send_rate": metrics["send_rate"],
        }

        action = agent.act(observation)
        sender.adjust_rate(action)

        # --- Only print + record evaluation window ---
        if step >= EVAL_START:
            eval_thr.append(metrics["throughput"])
            eval_rtt.append(metrics["avg_rtt"])
            eval_loss.append(metrics["loss"])

            print(
                f"{metrics['time']:>4} | "
                f"{metrics['send_rate']:>4} | "
                f"{metrics['throughput']:>3} | "
                f"{metrics['avg_rtt']:.2f} | "
                f"{metrics['loss']:>4} | "
                f"{action:>6}"
            )

    # --- Summary ---
    print("\n=== Evaluation Summary (last 30 steps) ===")
    print(f"Avg Throughput : {sum(eval_thr) / len(eval_thr):.2f}")
    print(f"Avg RTT        : {sum(eval_rtt) / len(eval_rtt):.2f}")
    print(f"Avg Loss       : {sum(eval_loss) / len(eval_loss):.2f}")


if __name__ == "__main__":
    main()