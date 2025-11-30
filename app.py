# import time
# import numpy as np
# import streamlit as st
# import matplotlib.pyplot as plt

# from model import TrainModel
# from training_simulation import TrafficEnv


# # ----------------------- UI SECTION -----------------------
# st.set_page_config(page_title="Traffic RL App", layout="wide")
# st.title("🚦 Traffic Light Optimization – DQN Controller")

# st.sidebar.header("Simulation Controls")
# max_steps = st.sidebar.slider("Episode Length", 50, 400, 200, step=25)
# live_anim = st.sidebar.checkbox("Show Live Visualization (slower)", value=True)
# step_delay = st.sidebar.slider("Speed (sec/step)", 0.00, 0.50, 0.10, 0.05)


# # ---- CAR INPUT BOXES ----
# st.sidebar.subheader("🚗 Set Initial Car Count")
# north_cars = st.sidebar.number_input("North Lane Cars", 0, 30, 5)
# east_cars  = st.sidebar.number_input("East Lane Cars", 0, 30, 5)
# south_cars = st.sidebar.number_input("South Lane Cars", 0, 30, 5)
# west_cars  = st.sidebar.number_input("West Lane Cars", 0, 30, 5)


# # -------- LOAD TRAINED MODEL --------
# @st.cache_resource
# def load_agent():
#     env = TrafficEnv()
#     agent = TrainModel(env.state_dim, env.action_dim)
#     agent.load("models/traffic_dqn_weights.h5", env.state_dim, env.action_dim)
#     return agent

# agent = load_agent()
# st.success("🟢 Model Loaded Successfully")


# # -------- DRAW INTERSECTION --------
# def draw_intersection(queues, phase):
#     qN, qE, qS, qW = queues
#     fig, ax = plt.subplots(figsize=(4,4))

#     # Roads
#     ax.axhspan(-0.1, 0.1, xmin=0.2, xmax=0.8, color="grey")
#     ax.axvspan(-0.1, 0.1, ymin=0.2, ymax=0.8, color="grey")

#     max_show = 10

#     for i in range(min(int(qN),max_show)): ax.scatter(0,  0.6+0.05*i, s=60,color="cyan")
#     for i in range(min(int(qS),max_show)): ax.scatter(0, -0.6-0.05*i, s=60,color="cyan")
#     for i in range(min(int(qE),max_show)): ax.scatter(0.6+0.05*i,0, s=60,color="orange")
#     for i in range(min(int(qW),max_show)): ax.scatter(-0.6-0.05*i,0, s=60,color="orange")

#     ax.text(-1.0,1.0,f"Phase: {'NS Green' if phase==0 else 'EW Green'}",color="lime" if phase==0 else "red")
#     ax.set_xlim(-1.2,1.2); ax.set_ylim(-1.2,1.2); ax.axis("off")
#     return fig


# # -------- RUN BUTTON --------
# run = st.button("▶ Run Simulation", use_container_width=True)

# anim_placeholder = st.empty()
# summary_placeholder = st.empty()
# charts_container = st.container()
# # -----------------------------------------------------------


# # ======================= RL SIMULATION ======================
# if run:

#     env = TrafficEnv(
#         max_episode_steps=max_steps,
#         initial_cars=[north_cars, east_cars, south_cars, west_cars]
#     )

#     state = env.reset()

#     queue_log = []
#     reward_data = []
#     waiting_data = []
#     phase_data = []

#     for step in range(max_steps):
#         action = agent.act(state, epsilon=0.0)
#         next_state, reward, done, info = env.step(action)

#         qN,qE,qS,qW,phase = map(int,next_state[:5])

#         queue_log.append([qN,qE,qS,qW])
#         reward_data.append(float(reward))
#         waiting_data.append(float(info["total_waiting"]))
#         phase_data.append(phase)

#         if live_anim:
#             fig = draw_intersection([qN,qE,qS,qW], phase)
#             anim_placeholder.pyplot(fig)
#             plt.close()
#             time.sleep(step_delay)

#         state = next_state
#         if done: break


#     # -------- Results --------
#     avg_wait = np.mean(waiting_data)
#     total_reward = np.sum(reward_data)

#     summary_placeholder.success("Simulation Finished 🎉")
#     col1,col2,col3 = st.columns(3)
#     col1.metric("Avg Waiting", f"{avg_wait:.2f} cars")
#     col2.metric("Total Reward", f"{total_reward:.2f}")
#     col3.metric("Steps", len(waiting_data))


#     # ---- Graphs ----
#     q = np.array(queue_log)

#     st.subheader("📊 Queue Per Lane")
#     st.line_chart({"North":q[:,0],"East":q[:,1],"South":q[:,2],"West":q[:,3]})

#     st.subheader("🏆 Reward Trend")
#     st.line_chart(reward_data)

#     st.subheader("🚦 Phase Pattern (0=NS ,1=EW)")
#     st.line_chart(phase_data)

# else:
#     st.info("Configure settings and press ▶ Run Simulation.")






import time
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from model import TrainModel
from training_simulation import TrafficEnv


# ----------------------- UI SECTION -----------------------
st.set_page_config(page_title="Traffic RL App", layout="wide")
st.title("🚦 Traffic Light Optimization – DQN Controller")

st.sidebar.header("Simulation Controls")
max_steps = st.sidebar.slider("Episode Length", 50, 400, 200, step=25)
live_anim = st.sidebar.checkbox("Show Live Visualization (slower)", value=True)
step_delay = st.sidebar.slider("Speed (sec/step)", 0.00, 0.50, 0.10, 0.05)

st.sidebar.markdown("---")
compare_baseline = st.sidebar.checkbox(
    "Compare with fixed-time (non-RL) controller", value=True
)

# ---- CAR INPUT BOXES ----
st.sidebar.subheader("🚗 Set Initial Car Count")
north_cars = st.sidebar.number_input("North Lane Cars", 0, 50, 5)
east_cars  = st.sidebar.number_input("East Lane Cars",  0, 50, 5)
south_cars = st.sidebar.number_input("South Lane Cars", 0, 50, 5)
west_cars  = st.sidebar.number_input("West Lane Cars",  0, 50, 5)


# -------- LOAD TRAINED MODEL --------
@st.cache_resource
def load_agent():
    env = TrafficEnv()
    agent = TrainModel(env.state_dim, env.action_dim)
    agent.load("models/traffic_dqn_weights.h5", env.state_dim, env.action_dim)
    return agent

with st.spinner("Loading trained DQN agent..."):
    agent = load_agent()
st.success("🟢 Model Loaded Successfully")


# -------- DRAW INTERSECTION --------
def draw_intersection(queues, phase):
    qN, qE, qS, qW = queues
    fig, ax = plt.subplots(figsize=(4, 4))

    # Roads
    ax.axhspan(-0.1, 0.1, xmin=0.2, xmax=0.8, color="grey")
    ax.axvspan(-0.1, 0.1, ymin=0.2, ymax=0.8, color="grey")

    max_show = 10

    # North
    for i in range(min(int(qN), max_show)):
        ax.scatter(0, 0.6 + 0.05 * i, s=60, color="cyan")
    # South
    for i in range(min(int(qS), max_show)):
        ax.scatter(0, -0.6 - 0.05 * i, s=60, color="cyan")
    # East
    for i in range(min(int(qE), max_show)):
        ax.scatter(0.6 + 0.05 * i, 0, s=60, color="orange")
    # West
    for i in range(min(int(qW), max_show)):
        ax.scatter(-0.6 - 0.05 * i, 0, s=60, color="orange")

    ax.text(
        -1.1,
        1.0,
        f"Phase: {'NS GREEN' if phase == 0 else 'EW GREEN'}",
        color="lime" if phase == 0 else "red",
        fontsize=9,
        weight="bold",
    )
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.axis("off")
    fig.tight_layout()
    return fig


# -------- FIXED-TIME BASELINE (non-RL) --------
def run_fixed_baseline(max_steps, initial_cars, switch_every=15):
    """
    Classic signal controller:
    - Switch phase every `switch_every` steps, ignoring traffic.
    """
    env = TrafficEnv(max_episode_steps=max_steps, initial_cars=initial_cars)
    state = env.reset()

    waiting_data = []
    reward_data = []

    for t in range(max_steps):
        # every switch_every steps → force phase change
        if t % switch_every == 0:
            action = 1  # switch
        else:
            action = 0  # keep

        next_state, reward, done, info = env.step(action)

        waiting_data.append(float(info["total_waiting"]))
        reward_data.append(float(reward))

        state = next_state
        if done:
            break

    return waiting_data, reward_data


# -------- RUN BUTTON --------
run = st.button("▶ Run Simulation", use_container_width=True)

anim_placeholder = st.empty()
summary_placeholder = st.empty()
charts_container = st.container()


# ======================= RL SIMULATION ======================
if run:

    initial_cars = [north_cars, east_cars, south_cars, west_cars]

    # ------------ RL agent environment ------------
    env = TrafficEnv(
        max_episode_steps=max_steps,
        initial_cars=initial_cars,
    )

    state = env.reset()

    queue_log = []
    reward_data = []
    waiting_data = []
    phase_data = []

    for step in range(max_steps):
        action = agent.act(state, epsilon=0.0)  # greedy policy
        next_state, reward, done, info = env.step(action)

        # next_state = [qN, qE, qS, qW, phase]
        qN, qE, qS, qW, phase = next_state[:5]
        qN, qE, qS, qW, phase = int(qN), int(qE), int(qS), int(qW), int(phase)

        queue_log.append([qN, qE, qS, qW])
        reward_data.append(float(reward))
        waiting_data.append(float(info["total_waiting"]))
        phase_data.append(phase)

        if live_anim:
            fig = draw_intersection([qN, qE, qS, qW], phase)
            anim_placeholder.pyplot(fig)
            plt.close(fig)
            if step_delay > 0:
                time.sleep(step_delay)

        state = next_state
        if done:
            break

    # -------- RL Results --------
    avg_wait = float(np.mean(waiting_data))
    total_reward = float(np.sum(reward_data))

    summary_placeholder.success("✅ RL Simulation Finished")
    col1, col2, col3 = st.columns(3)
    col1.metric("RL Avg Waiting", f"{avg_wait:.2f} cars")
    col2.metric("RL Total Reward", f"{total_reward:.2f}")
    col3.metric("Steps (RL)", len(waiting_data))

    q = np.array(queue_log)

    st.subheader("📊 RL – Queue Per Lane")
    st.line_chart(
        {
            "North": q[:, 0],
            "East": q[:, 1],
            "South": q[:, 2],
            "West": q[:, 3],
        }
    )

    st.subheader("🏆 RL – Reward Trend")
    st.line_chart(reward_data)

    st.subheader("🚦 RL – Phase Pattern (0 = NS green, 1 = EW green)")
    st.line_chart(phase_data)

    # ======================= BASELINE COMPARISON ======================
    if compare_baseline:
        st.markdown("---")
        st.subheader("⚖ RL vs Fixed-Time Controller (Baseline)")

        base_wait, base_reward = run_fixed_baseline(
            max_steps=max_steps, initial_cars=initial_cars, switch_every=15
        )

        base_avg_wait = float(np.mean(base_wait))
        base_total_reward = float(np.sum(base_reward))

        b1, b2 = st.columns(2)
        b1.metric("Baseline Avg Waiting", f"{base_avg_wait:.2f} cars")
        b2.metric("Baseline Total Reward", f"{base_total_reward:.2f}")

        st.markdown("#### ⏱ Waiting Time Comparison")
        st.line_chart(
            {
                "RL Waiting": waiting_data,
                "Baseline Waiting": base_wait,
            }
        )

        st.markdown("#### 💰 Reward Comparison")
        st.line_chart(
            {
                "RL Reward": reward_data,
                "Baseline Reward": base_reward,
            }
        )

        st.info(
            "- RL should show **lower average waiting** and **higher total reward** than the fixed-time baseline.\n"
            "- Baseline uses a dumb strategy: it just switches every fixed number of steps, without looking at queues."
        )

else:
    st.info("Configure settings in the sidebar and press ▶ **Run Simulation**.")
