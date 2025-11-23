Reinforcement Learning Snake — Q-Learning, SARSA & DQN

An interactive browser-based Reinforcement Learning playground that demonstrates how different RL algorithms learn to play Snake.
This project visualizes real-time agent decisions, learning curves, exploration strategies, neural network training, and final performance comparison.

 Demo Showcase

(Insert your demo GIF or video here)
![Demo](demo.gif)

 Project Overview

This project implements and compares three RL agents:

Model	Type	Characteristics
Q-Learning	Off-policy Tabular RL	Fast, aggressive, converges to optimal but riskier policies
SARSA	On-policy Tabular RL	Safer, stable, avoids walls naturally
DQN	Deep Reinforcement Learning	Neural network, scalable, uses experience replay

Each model interacts with the same environment, enabling clean comparison of learning behaviour.

 Visual Components
1️. Environment (Game World)

![Environment](assets/environment.png)

Grid Specifications

Grid Size: 
20
×
20
20×20 (400 cells)

Action Space: Discrete(4) → [Up, Down, Left, Right]

Coordinate System: (0,0) top-left → (19,19) bottom-right

3D Perspective

Uses CSS transforms:
perspective(1000px) rotateX(25deg) scale(0.85)

Gives the game a retro neon-arcade aesthetic.

2️. State Representation (7-Bit Vector)

![State](assets/state.png)

Each agent receives a 7-binary-feature state vector:

Danger Straight

Danger Left

Danger Right

Food Up

Food Down

Food Left

Food Right

State Space Efficiency

Input Vector Size: 7

Total Possible State Combinations:

2
7
=
128
2
7
=128

Compared to raw pixel input (
20
×
20
=
400
20×20=400 values), this is 300× smaller, enabling fast, stable training in the browser.

3️. Reward Mechanism (Exact Values Used)

![Rewards](assets/rewards.png)

Event	Reward
Eat Food	+50
Collision (Wall/Self)	−50
Living Penalty	−0.1 per step
Move Closer to Food	+0.5
Move Away from Food	−0.6

This reward shaping encourages:

Efficient movement

Strategic navigation

Avoidance of loops

Rapid convergence

Food-directed behaviour

 Reinforcement Learning Models
1. Q-Learning (Off-Policy Tabular RL)

![QLearning](assets/qlearning.png)

Q-Learning updates using the Bellman Optimality Equation:

𝑄
(
𝑆
,
𝐴
)
←
𝑄
(
𝑆
,
𝐴
)
+
𝛼
[
𝑅
+
𝛾
max
⁡
𝑎
𝑄
(
𝑆
′
,
𝑎
)
−
𝑄
(
𝑆
,
𝐴
)
]
Q(S,A)←Q(S,A)+α[R+γ
a
max
	​

Q(S
′
,a)−Q(S,A)]

Hyperparameters

Learning Rate 
𝛼
=
0.1
α=0.1

Discount Factor 
𝛾
=
0.9
γ=0.9

Behaviour

Fast learning

Greedy & aggressive

Sometimes takes risky paths close to walls

Converges quickly due to small state space (128 states)

2. SARSA (On-Policy Tabular RL)

![SARSA](assets/sarsa.png)

SARSA updates using the actual next action taken:

𝑄
(
𝑆
,
𝐴
)
←
𝑄
(
𝑆
,
𝐴
)
+
𝛼
[
𝑅
+
𝛾
𝑄
(
𝑆
′
,
𝐴
′
)
−
𝑄
(
𝑆
,
𝐴
)
]
Q(S,A)←Q(S,A)+α[R+γQ(S
′
,A
′
)−Q(S,A)]

Key Insight
SARSA learns the value of its real behavior, not the optimal hypothetical one.

Risk Profile

Naturally safer

Avoids edge-clinging policies

More stable early gameplay

3. Deep Q-Network (DQN)

![DQN](assets/dqn.png)

Neural Network Architecture
Layer	Units	Activation
Input	7	—
Hidden 1	24	ReLU
Hidden 2	24	ReLU
Output	4	Linear
Loss Function (Mean Squared Error)
𝐿
=
(
𝑦
−
𝑄
(
𝑠
,
𝑎
)
)
2
L=(y−Q(s,a))
2

Where:

𝑦
=
𝑅
+
𝛾
max
⁡
𝑄
(
𝑠
′
,
𝑎
′
)
y=R+γmaxQ(s
′
,a
′
)
Experience Replay

Replay Buffer: 2000 transitions

Batch Size: 32 samples

Randomized training breaks correlation between consecutive states → stabilizing learning.

Behaviour

Learns slower initially

Becomes smoother over time

More scalable for larger grids (e.g., 100×100)

 Training Workflow

![Training Pipeline](assets/pipeline.png)

Each episode follows:

Step 1 — Observe State

Extract 7-bit state vector.

Step 2 — Choose Action (Epsilon-Greedy)

Exploration Strategy

Initial Epsilon 
𝜖
=
1.0
ϵ=1.0 (100% random)

Decay each episode:

𝜖
←
𝜖
×
0.995
ϵ←ϵ×0.995

Minimum 
𝜖
=
0.01
ϵ=0.01

Step 3 — Environment Responds

Reward + new state + done flag.

Step 4 — Update Model

Q-Learning uses max future Q

SARSA uses actual future action

DQN uses neural network optimization

Step 5 — Chart Update

Graphs update for score, rewards, and (for DQN) loss.

 Performance Visualization
1. Score Progression

![Scores](assets/scores.png)

Shows learning speed and stability.

2. Average Reward Comparison

![Reward](assets/reward_graph.png)

Usually:

Q-Learning → fastest rise

SARSA → smooth & safe

DQN → slow but powerful

3. DQN Loss Curve

![Loss](assets/loss.png)

Demonstrates convergence of the neural network.

 User Interface Preview

![UI](assets/ui.png)

Includes:

Model selector

Speed modes (Normal, Fast, Hyper)

Episode counter

Live charts

Lock & Compare mode

 Use Cases

Perfect for:

RL education and demonstrations

Comparing on-policy vs. off-policy learning

Understanding exploration decay

Visualising neural network training in the browser

🧭 Project Structure Overview

![Folders](assets/folders.png)

Conceptual components:

Game Engine

Agents (Q-Learning, SARSA, DQN)

Replay Memory

UI / Controls

Chart Manager

Visual Renderer

 Future Enhancements

Double DQN

Dueling Networks

Prioritized Replay

Convolutional input states

Multi-agent competition
