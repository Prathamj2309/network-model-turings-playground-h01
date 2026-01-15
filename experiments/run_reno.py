# run_reno.py

from sim.environment import Environment
from sim.sender import Sender
from sim.link import Link
from sim.receiver import Receiver
from agents.reno_agent import RenoAgent

def main():
    # --- 1. Initialize Components ---
    # initial_rate is the starting cwnd
    sender = Sender(initial_rate=5)

    # Link parameters: capacity, queue_limit, base_rtt, noise_prob
    # noise_prob=0.2 means 20% random wireless loss
    link = Link(
        capacity=4,
        queue_limit=10,
        base_rtt=5.0,
        noise_prob=0.2
    )

    receiver = Receiver()
    
    # Environment ties everything together
    # Note: We do NOT pass the agent here, matching your current environment.py
    env = Environment(sender, link, receiver)

    # Initialize Reno Agent with AIMD parameters
    agent = RenoAgent(increase_step=1, decrease_factor=0.5)

    print(f"{'Time':<5} | {'Rate':<5} | {'Thr':<5} | {'RTT':<8} | {'Loss':<5} | {'Delta':<5}")
    print("-" * 55)

    # --- 2. Run Simulation ---
    # Running for 30 timesteps
    for _ in range(30):
        # Step advances time and processes packet movement
        metrics = env.step()

        # Extract observation for the agent
        observation = {
            "throughput": metrics["throughput"],
            "avg_rtt": metrics["avg_rtt"],
            "loss": metrics["loss"],
            "send_rate": metrics["send_rate"],
        }

        # Agent decides adjustment based on observed metrics
        delta = agent.act(observation)

        # Apply the delta to the sender's rate for the NEXT timestep
        sender.adjust_rate(delta)

        # Logging the results in a table row
        print(
            f"{metrics['time']:<5} | "
            f"{metrics['send_rate']:<5} | "
            f"{metrics['throughput']:<5} | "
            f"{metrics['avg_rtt']:<8.2f} | "
            f"{metrics['loss']:<5} | "
            f"{delta:<5}"
        )

if __name__ == "__main__":
    main()