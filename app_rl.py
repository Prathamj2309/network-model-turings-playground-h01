# Final frontend - Aesthetic & Interactive Edition

import streamlit as st
import pandas as pd
import time
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import your simulation components
from sim.environment import Environment
from sim.sender import Sender
from sim.link import Link
from sim.receiver import Receiver
from agents.reno_agent import RenoAgent
from agents.rl_agent import RLAgent

# 1. Page Configuration & Dark Theme Style
st.set_page_config(page_title="ANTAR-DRISHTI --> INNER VISION", layout="wide")

# Custom CSS for high-tech "Cyberpunk" aesthetics
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    
    /* Neon Metric Cards */
    .metric-container {
        display: flex;
        justify-content: space-around;
        gap: 10px;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #30363d;
        text-align: center;
        flex: 1;
    }
    .legacy-card { border-top: 4px solid #ff4b4b; box-shadow: 0 4px 10px rgba(255, 75, 75, 0.1); }
    .ai-card { border-top: 4px solid #00d488; box-shadow: 0 4px 10px rgba(0, 212, 136, 0.1); }
    
    .card-label { font-size: 12px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }
    .card-value { font-size: 24px; font-weight: bold; font-family: 'Courier New', monospace; margin-top: 5px; }
    </style>
    """, unsafe_allow_html=True)

# 2. Sidebar Configuration
with st.sidebar:
    st.markdown("<h2 style='color: #00d488; font-family: monospace;'>● NETWORK ARGS</h2>", unsafe_allow_html=True)
    noise = st.slider("Wireless Noise (%)", 0.0, 0.5, 0.20, help="Random packet loss probability.")
    capacity = st.slider("Link Capacity", 1, 20, 4)
    queue_limit = st.slider("Queue Length (pkts)", 5, 50, 15, help="Max packets in router queue.")
    
    st.markdown("---")
    st.markdown("<h2 style='color: #00d488; font-family: monospace;'>● AGENT ARGS</h2>", unsafe_allow_html=True)
    sim_steps = st.slider("Simulation Length", 50, 5000, 300)
    ema_alpha = st.slider("RTT Smoothing (Alpha)", 0.01, 0.5, 0.1)

# 3. Header
c1, c2 = st.columns([2, 1])
with c1:
    st.markdown("<h2 style='color: #00d488; font-family: monospace;'>● ANTAR-DRISHTI // <span style='color: white;'>INNER VISION</span></h2>", unsafe_allow_html=True)
with c2:
    st.markdown("<div style='text-align: right; color: #8b949e; font-family: monospace; padding-top: 10px;'>SYSTEM STATUS: <span style='color: #00d488;'>ACTIVE</span></div>", unsafe_allow_html=True)

# 4. Simulation Initialization
def init_sims():
    env_r = Environment(Sender(5), Link(capacity, queue_limit, 5.0, noise), Receiver())
    ag_r = RenoAgent()
    env_a = Environment(Sender(5), Link(capacity, queue_limit, 5.0, noise), Receiver())
    ag_a = RLAgent(base_rtt=5.0)
    return env_r, ag_r, env_a, ag_a

# 5. Dashboard Setup
if 'run' not in st.session_state: st.session_state.run = False

def start_sim(): st.session_state.run = True

st.button("INITIATE NEURAL COMPARISON", on_click=start_sim, use_container_width=True)

if st.session_state.run:
    env_r, ag_r, env_a, ag_a = init_sims()
    
    # Static row for metrics (avoids the 1000-column bug)
    st.markdown("<div class='card-label'>Live Performance Averages</div>", unsafe_allow_html=True)
    m_cols = st.columns(6)
    m_placeholders = [col.empty() for col in m_cols]

    # Chart Area
    plot_time = st.empty()
    st.markdown("---")
    plot_corr = st.empty()

    history = []
    ema_r, ema_a = 5.0, 5.0
    totals = {"L_Thr": 0, "L_Rate": 0, "L_EMA": 0, "A_Thr": 0, "A_Rate": 0, "A_EMA": 0}

    for t in range(1, sim_steps + 1):
        m_r = env_r.step()
        m_a = env_a.step()

        if m_r['avg_rtt'] > 0: ema_r = (1 - ema_alpha) * ema_r + ema_alpha * m_r['avg_rtt']
        if m_a['avg_rtt'] > 0: ema_a = (1 - ema_alpha) * ema_a + ema_alpha * m_a['avg_rtt']

        env_r.sender.adjust_rate(ag_r.act(m_r))
        env_a.sender.adjust_rate(ag_a.act(m_a))

        # Log & Sum for Averages
        totals["L_Thr"] += m_r['throughput']; totals["L_Rate"] += m_r['send_rate']; totals["L_EMA"] += ema_r
        totals["A_Thr"] += m_a['throughput']; totals["A_Rate"] += m_a['send_rate']; totals["A_EMA"] += ema_a

        history.append({
            "t": t, "L_Thr": m_r['throughput'], "A_Thr": m_a['throughput'],
            "L_Rate": m_r['send_rate'], "A_Rate": m_a['send_rate'],
            "L_EMA": ema_r, "A_EMA": ema_a,
            "L_Loss": m_r['loss'], "A_Loss": m_a['loss']
        })

        # Update Metric Cards In-Place
        card_contents = [
            ("Legacy Thr", totals["L_Thr"]/t, "#ff4b4b", "legacy"),
            ("Legacy CWND", totals["L_Rate"]/t, "#ff4b4b", "legacy"),
            ("Legacy RTT", totals["L_EMA"]/t, "#ff4b4b", "legacy"),
            ("AI Thr", totals["A_Thr"]/t, "#00d488", "ai"),
            ("AI CWND", totals["A_Rate"]/t, "#00d488", "ai"),
            ("AI RTT", totals["A_EMA"]/t, "#00d488", "ai"),
        ]
        
        for i, (label, val, color, style) in enumerate(card_contents):
            m_placeholders[i].markdown(f"""
                <div class="metric-card {style}-card">
                    <div class="card-label">{label}</div>
                    <div class="card-value" style="color: {color};">{val:.1f}</div>
                </div>
            """, unsafe_allow_html=True)

        if t % 10 == 0 or t == sim_steps:
            df = pd.DataFrame(history)
            
            # --- Subplots Logic (Same as your request but styled) ---
            fig_time = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                                     subplot_titles=("Throughput", "Send Rate", "EMA RTT", "Packet Loss"))
            
            colors = {'L': '#ff4b4b', 'A': '#00d488'}
            
            # Throughput
            fig_time.add_trace(go.Scatter(x=df['t'], y=df['L_Thr'], name='Legacy', line=dict(color=colors['L'])), row=1, col=1)
            fig_time.add_trace(go.Scatter(x=df['t'], y=df['A_Thr'], name='AI', line=dict(color=colors['A'])), row=1, col=1)
            # Send Rate
            fig_time.add_trace(go.Scatter(x=df['t'], y=df['L_Rate'], name='Legacy', line=dict(color=colors['L'], dash='dot')), row=2, col=1)
            fig_time.add_trace(go.Scatter(x=df['t'], y=df['A_Rate'], name='AI', line=dict(color=colors['A'], dash='dot')), row=2, col=1)
            # RTT
            fig_time.add_trace(go.Scatter(x=df['t'], y=df['L_EMA'], name='Legacy', line=dict(color=colors['L'], width=1)), row=3, col=1)
            fig_time.add_trace(go.Scatter(x=df['t'], y=df['A_EMA'], name='AI', line=dict(color=colors['A'], width=1)), row=3, col=1)
            # Loss
            fig_time.add_trace(go.Scatter(x=df['t'], y=df['L_Loss'], name='Legacy', line=dict(color=colors['L'], width=2)), row=4, col=1)
            fig_time.add_trace(go.Scatter(x=df['t'], y=df['A_Loss'], name='AI', line=dict(color=colors['A'], width=2)), row=4, col=1)
            
            fig_time.update_layout(height=900, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                                  font_color="#8b949e", showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
            plot_time.plotly_chart(fig_time, use_container_width=True)

            # Correlation Plots
            fig_corr = make_subplots(rows=1, cols=2, subplot_titles=("Legacy Analysis", "AI Analysis"))
            fig_corr.add_trace(go.Scatter(x=df['L_EMA'], y=df['L_Rate'], mode='markers', marker=dict(color=colors['L'], opacity=0.4)), row=1, col=1)
            fig_corr.add_trace(go.Scatter(x=df['A_EMA'], y=df['A_Rate'], mode='markers', marker=dict(color=colors['A'], opacity=0.4)), row=1, col=2)
            fig_corr.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#8b949e", showlegend=False)
            plot_corr.plotly_chart(fig_corr, use_container_width=True)

        time.sleep(0.01)