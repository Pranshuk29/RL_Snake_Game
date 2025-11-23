# 🐍 Neon Snake RL
A browser-based Reinforcement Learning (RL) playground that visualizes how different AI agents learn to play the classic Snake game in real time. The project compares **Q-Learning**, **SARSA**, and **Deep Q-Network (DQN)** using an interactive training environment built entirely with web technologies.

---

## 🚀 Features
- **Real-Time Training Visualization**  
  Watch agents learn from random behavior to strategic gameplay.
  
- **Multiple Algorithms**
  - Q-Learning (Off-Policy)
  - SARSA (On-Policy)
  - DQN (TensorFlow.js)

- **3D Neon Game Board**  
  Custom CSS with 3D transforms (`perspective`, `rotateX`) for a retro arcade look.

- **Dynamic Learning Charts**  
  Performance tracked live using Chart.js.

- **Speed Modes**
  - Normal
  - Fast
  - Hyper (skips rendering for maximum RL speed)

- **Comparison Mode**
  Lock completed runs and compare multiple algorithms visually.

---

## 🧠 State Representation
Each agent receives a **7-dimensional binary state vector** instead of the full grid:

1. Danger Straight  
2. Danger Left  
3. Danger Right  
4. Food Up  
5. Food Down  
6. Food Left  
7. Food Right

This compact state space allows tabular RL to learn efficiently.

---

## 🎯 Reward Shaping
| Event                     | Reward |
|--------------------------|--------|
| Eat Food                 | +50    |
| Crash (wall/self)        | -50    |
| Step Penalty             | -0.1   |
| Move towards food        | +0.5   |
| Move away from food      | -0.6   |

---

## 🧪 Algorithms

### **Q-Learning**
- Off-policy  
- Greedy updates using:  
  `max(Q(s', a'))`  
- Learns fast but risk-seeking.

### **SARSA**
- On-policy  
- Updates using actual next action:  
  `Q(s', a')`  
- More conservative and safer.

### **Deep Q-Network (DQN)**
- TensorFlow.js neural network  
- Architecture: `7 → 24 → 24 → 4`  
- Uses Experience Replay for stable learning.

---

## 📊 How to Run Experiments
1. Select an algorithm using the UI.  
2. Set speed (Hyper recommended).  
3. Click **Start** and let it train for 100–200 episodes.  
4. Click **Lock Run** to freeze the line.  
5. Repeat for other algorithms.  
6. Export comparison chart (PNG) using the download button.

---

## 🛠️ Tech Stack
- **HTML5**
- **Tailwind CSS** + custom CSS for 3D effects
- **Vanilla JavaScript (ES6+)**
- **TensorFlow.js** (for DQN)
- **Chart.js** (learning visualization)
- **Web Audio API** (sound effects)

---

## 📁 Project Structure
/index.html
/style.css
/game/
SnakeGame.js
Renderer.js
Controls.js
agents/
BaseAgent.js
QLearningAgent.js
SarsaAgent.js
DQNAgent.js
charts/
ChartManager.js
assets/
icons/
README.md


---

## ▶️ Running the Project
Just open **index.html** in any modern browser—no backend or setup required.

If TensorFlow.js fails to load, ensure you are connected to the internet (CDN dependency).

---

## 📸 Screenshots
_(Insert images such as gameplay screenshot, training charts, or comparison plots here.)_

---

## 📜 License
This project is released under the MIT License.

---

## ✨ Author
**Pranshu Kaushik**

---

