import time

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from model import TrainModel
from training_simulation import TrafficEnv

st.set_page_config(page_title="Traffic RL App", layout="wide")
st.title("🚦 Traffic Light Optimization – DQN Controller")

st.write("🧪 Loading trained model...")


@st.cache_resource(show_spinner=True)
def load_agent():
    try:
        env = TrafficEnv()
        agent = TrainModel(env.state_dim, env.action_dim)
        agent.load("models/traffic_dqn_weights.h5", env.state_dim, env.action_dim)
        return agent, True, ""
    except Exception as e:
        return None, False, str(e)


agent, ok, err = load_agent()
if not ok:
    st.error(f"❌ Failed to load trained model\n\n**Error:** `{err}`")
    st.stop()
else:
    st.success("✅ Model Loaded Successfully")


# --------- Helper: draw intersection frame ---------
def draw_intersection(queues, phase):
    """
    Draw a simple top-down intersection with cars as points.
    queues: [N, E, S, W]
    phase: 0 -> NS green, 1 -> EW green
    """
    qN, qE, qS, qW = queues

    fig, ax = plt.subplots(figsize=(4, 4))

    # Background (roads)
    ax.axhspan(-0.1, 0.1, xmin=0.2, xmax=0.8, color="dimgray")  # horizontal road
    ax.axvspan(-0.1, 0.1, ymin=0.2, ymax=0.8, color="dimgray")  # vertical road

    # Draw cars as points along each approach
    # Limit to max 10 visually so plot doesn't explode
    max_show = 10

    # North (coming from top, moving down)
    n_cars = int(min(qN, max_show))
    for i in range(n_cars):
        y = 0.6 + 0.05 * i
        ax.scatter(0, y, s=60, color="cyan")

    # South (coming from bottom, moving up)
    s_cars = int(min(qS, max_show))
    for i in range(s_cars):
        y = -0.6 - 0.05 * i
        ax.scatter(0, y, s=60, color="cyan")

    # East (coming from right, moving left)
    e_cars = int(min(qE, max_show))
    for i in range(e_cars):
        x = 0.6 + 0.05 * i
        ax.scatter(x, 0, s=60, color="orange")

    # West (coming from left, moving right)
    w_cars = int(min(qW, max_show))
    for i in range(w_cars):
        x = -0.6 - 0.05 * i
        ax.scatter(x, 0, s=60, color="orange")

    # Traffic light indicators
    if phase == 0:
        # NS green, EW red
        ax.text(-0.9, 0.9, "NS: GREEN", color="lime", fontsize=9, weight="bold")
        ax.text(-0.9, 0.8, "EW: RED", color="red", fontsize=9, weight="bold")
    else:
        ax.text(-0.9, 0.9, "NS: RED", color="red", fontsize=9, weight="bold")
        ax.text(-0.9, 0.8, "EW: GREEN", color="lime", fontsize=9, weight="bold")

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.axis("off")
    fig.tight_layout()
    return fig


# --------- UI Controls ---------
st.sidebar.header("Simulation Controls")

max_steps = st.sidebar.slider("Episode Length", 50, 400, 200, step=25)
live_anim = st.sidebar.checkbox("Show live traffic animation (slower)", value=True)
step_delay = st.sidebar.slider("Animation speed (seconds per step)", 0.0, 0.5, 0.1, 0.05)

run = st.button("▶ Run Simulation")

# placeholders
anim_placeholder = st.empty()
summary_placeholder = st.empty()
charts_container = st.container()

if run:
    env = TrafficEnv(max_episode_steps=max_steps)
    state = env.reset()

    waiting_data = []
    reward_data = []
    phase_data = []
    queue_log = []

    for _ in range(max_steps):
        action = agent.act(state, epsilon=0.0)
        next_state, reward, done, info = env.step(action)

        queues = state[:4]
        phase = int(state[4])

        queue_log.append(queues.copy())
        waiting_data.append(info["total_waiting"])
        reward_data.append(reward)
        phase_data.append(phase)

        # ----- live animation -----
        if live_anim:
            fig = draw_intersection(queues, phase)
            anim_placeholder.pyplot(fig)
            plt.close(fig)
            if step_delay > 0:
                time.sleep(step_delay)

        state = next_state
        if done:
            break

    # -------- Summary metrics --------
    total_reward = float(sum(reward_data))
    avg_waiting = float(sum(waiting_data) / len(waiting_data))
    steps_taken = len(waiting_data)

    with summary_placeholder:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Reward", f"{total_reward:.2f}")
        c2.metric("Average Waiting (cars)", f"{avg_waiting:.2f}")
        c3.metric("Steps", f"{steps_taken}")

    # -------- Charts --------
    with charts_container:
        st.markdown("---")
        st.subheader("📉 Total Waiting per Step")
        st.line_chart(waiting_data)

        st.subheader("🚗 Queue Length per Lane")
        q = np.array(queue_log)
        st.line_chart(
            {
                "North": q[:, 0],
                "East": q[:, 1],
                "South": q[:, 2],
                "West": q[:, 3],
            }
        )

        st.subheader("🏆 Reward per Step")
        st.line_chart(reward_data)

        st.subheader("🚦 Phase Pattern (0 = NS green, 1 = EW green)")
        st.line_chart(phase_data)

        st.markdown(
            """
**Interpretation for user / teacher**  

- Animated view shows cars accumulating on each approach.  
- Green direction drains cars; red direction accumulates them.  
- RL agent chooses **when to switch** to keep queues balanced and waiting low.
"""
        )
else:
    st.info("Set parameters in the sidebar and click **Run Simulation** to see the live traffic animation.")
