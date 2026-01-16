# experiments/run_rl.py

from sim.environment import Environment
from sim.sender import Sender
from sim.link import Link
from sim.receiver import Receiver

from agents.rl_agent import RLAgent


def main():
    # --- Network setup (same as Reno for fair comparison) ---
    sender = Sender(initial_rate=5)

    link = Link(
        capacity=4,
        queue_limit=10,
        base_rtt=5,
        noise_prob=0.2,   # wireless noise
    )

    receiver = Receiver()
    env = Environment(sender, link, receiver)

    # --- RL Agent ---
    agent = RLAgent(
        base_rtt=5,
        actions=(-2, -1, 0, 1, 2),
        alpha=0.1,
        gamma=0.9,
        epsilon=0.1,
        osc_penalty=0.3,
    )

    print("Time | Rate | Thr | RTT    | Loss | Action")
    print("-" * 50)

    # --- Simulation loop ---
    for step in range(400):
        metrics = env.step()

        observation = {
            "throughput": metrics["throughput"],
            "avg_rtt": metrics["avg_rtt"],
            "loss": metrics["loss"],
            "send_rate": metrics["send_rate"],
        }

        action = agent.act(observation)
        sender.adjust_rate(action)

        print(
            f"{metrics['time']:>4} | "
            f"{metrics['send_rate']:>4} | "
            f"{metrics['throughput']:>3} | "
            f"{metrics['avg_rtt']:.2f} | "
            f"{metrics['loss']:>4} | "
            f"{action:>6}"
        )


if __name__ == "__main__":
    main()