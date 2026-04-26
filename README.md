# 🐍 Snake AI — Deep Q-Learning Agent

A self-learning Snake game powered by a Deep Q-Network (DQN). The agent starts with zero knowledge and teaches itself to play Snake through trial, reward, and experience replay — improving in real time as you watch.

---

## 📸 Overview

The project combines a live Pygame Snake environment with a PyTorch neural network agent. The snake learns by exploring the game, collecting rewards for eating food, and being penalized for dying. Over hundreds of games, it develops a strategy that consistently beats human-level play.

---

## 🧠 How It Works

### Reinforcement Learning Loop

At every step, the agent:
1. Observes an **11-dimensional state** of the game
2. Chooses an **action** (go straight, turn left, turn right)
3. Receives a **reward** (+10 for food, -10 for dying)
4. Stores the experience in **replay memory**
5. Trains on a random batch from memory (**experience replay**)

### State Space (11 features)

| Feature | Description |
|---|---|
| Danger straight | Collision risk directly ahead |
| Danger right | Collision risk to the right |
| Danger left | Collision risk to the left |
| Direction (×4) | Current heading: L / R / U / D |
| Food left/right | Whether food is to the left or right of head |
| Food up/down | Whether food is above or below head |

### Neural Network

```
Input (11) → Linear → ReLU → Linear → Output (3)
              256 units                [straight, right, left]
```

A simple two-layer fully connected network trained with the **Adam optimizer** and **MSE loss**.

### Q-Learning Update

```
Q_new = reward + γ · max(Q(next_state))
```

Where `γ = 0.9` is the discount factor. The best model weights are saved automatically when a new high score is achieved (`snake_model.pth`).

---

## 🗂️ Project Structure

```
snake-ai/
├── snake_ai.py          # Full source: game, model, agent, training loop
├── snake_model.pth      # Saved model weights (auto-generated on new record)
└── README.md
```

---

## ⚙️ Requirements

- Python 3.8+
- PyTorch
- Pygame
- NumPy
- Matplotlib

Install dependencies:

```bash
pip install torch pygame numpy matplotlib
```

---

## 🚀 Running the Project

```bash
python snake_ai.py
```

The game window and a live training plot will open simultaneously. The plot tracks per-game score and rolling mean score across all games.

---

## 📈 Training Behavior

| Phase | Games | Behavior |
|---|---|---|
| Exploration | 0–80 | High randomness (`ε`-greedy), agent experiments |
| Learning | 80–200 | Epsilon decays, model predictions take over |
| Convergence | 200+ | Agent reliably scores 20–50+ per game |

Training progress is printed to the console:

```
Game 1   | Score: 0  | Record: 0
Game 42  | Score: 7  | Record: 12
Game 150 | Score: 31 | Record: 38
```

---

## 🎨 Visual Style

The game uses a pink color palette:

- **Background**: Soft white-pink `#FFF0F5`
- **Snake body**: Hot pink `#FF69B4` with a lighter inner block
- **Food**: Deep magenta `#C7158E`

---

## 🔧 Tuning & Customization

| Parameter | Location | Default | Effect |
|---|---|---|---|
| `SPEED` | Constant | `40` | Game speed (FPS) |
| `hidden_size` | `LinearQNet` | `256` | Network capacity |
| `lr` | `QTrainer` | `0.001` | Learning rate |
| `gamma` | `Agent` | `0.9` | Future reward discount |
| `maxlen` | `Agent.memory` | `100000` | Replay buffer size |
| `epsilon` formula | `get_action` | `80 - n_games` | Exploration decay rate |

---

## 💾 Saving & Loading

The best model is automatically saved to `snake_model.pth` whenever the agent achieves a new record score. To load and resume:

```python
agent.model.load_state_dict(torch.load('snake_model.pth'))
```

---

## 📚 References

- [Playing Atari with Deep Reinforcement Learning — DeepMind (2013)](https://arxiv.org/abs/1312.5602)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Pygame Documentation](https://www.pygame.org/docs/)

---

## Demo

https://github.com/meow-coding/snakeAi/assets/main/2026-02-28%2017-04-01.mp4
