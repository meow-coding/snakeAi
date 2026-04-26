import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt

# ─── CONSTANTS ───
BLOCK_SIZE = 20
SPEED = 40
WHITE = (255, 240, 245)
PINK1 = (255, 105, 180)
PINK2 = (255, 182, 193)
DARK_PINK = (199, 21, 133)
BLACK = (0, 0, 0)

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

# ─── SNAKE GAME ───
class SnakeGame:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake AI')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = RIGHT
        self.head = [self.w//2, self.h//2]
        self.snake = [
            self.head[:],
            [self.head[0]-BLOCK_SIZE, self.head[1]],
            [self.head[0]-2*BLOCK_SIZE, self.head[1]]
        ]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
        self.food = [x, y]
        if self.food in self.snake:
            self._place_food()

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt[0] >= self.w or pt[0] < 0 or pt[1] >= self.h or pt[1] < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False

    def _move(self, action):
        clock_wise = [RIGHT, DOWN, LEFT, UP]
        idx = clock_wise.index(self.direction)
        if action == 0:
            new_dir = clock_wise[idx]
        elif action == 1:
            new_dir = clock_wise[(idx+1) % 4]
        else:
            new_dir = clock_wise[(idx-1) % 4]
        self.direction = new_dir

        x, y = self.head
        if self.direction == RIGHT: x += BLOCK_SIZE
        elif self.direction == LEFT: x -= BLOCK_SIZE
        elif self.direction == DOWN: y += BLOCK_SIZE
        elif self.direction == UP:   y -= BLOCK_SIZE
        self.head = [x, y]

    def play_step(self, action):
        self.frame_iteration += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        move_idx = action.index(1) if isinstance(action, list) else int(action)
        self._move(move_idx)
        self.snake.insert(0, self.head[:])

        reward = 0
        game_over = False

        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        self._update_ui()
        self.clock.tick(SPEED)
        return reward, game_over, self.score

    def _update_ui(self):
        self.display.fill(WHITE)
        for pt in self.snake:
            pygame.draw.rect(self.display, PINK1, pygame.Rect(pt[0], pt[1], BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, PINK2, pygame.Rect(pt[0]+4, pt[1]+4, 12, 12))
        pygame.draw.rect(self.display, DARK_PINK, pygame.Rect(self.food[0], self.food[1], BLOCK_SIZE, BLOCK_SIZE))
        font = pygame.font.SysFont('arial', 25)
        text = font.render(f'Score: {self.score}', True, BLACK)
        self.display.blit(text, [0, 0])
        pygame.display.flip()


# ─── NEURAL NETWORK ───
class LinearQNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ─── TRAINER ───
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)

        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done,)

        pred = self.model(state)
        target = pred.clone()

        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx].unsqueeze(0)))
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()


# ─── AGENT ───
class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=100000)
        self.model = LinearQNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=0.001, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        pt_l = [head[0] - BLOCK_SIZE, head[1]]
        pt_r = [head[0] + BLOCK_SIZE, head[1]]
        pt_u = [head[0], head[1] - BLOCK_SIZE]
        pt_d = [head[0], head[1] + BLOCK_SIZE]

        dir_l = game.direction == LEFT
        dir_r = game.direction == RIGHT
        dir_u = game.direction == UP
        dir_d = game.direction == DOWN

        state = [
            (dir_r and game.is_collision(pt_r)) or
            (dir_l and game.is_collision(pt_l)) or
            (dir_u and game.is_collision(pt_u)) or
            (dir_d and game.is_collision(pt_d)),

            (dir_u and game.is_collision(pt_r)) or
            (dir_d and game.is_collision(pt_l)) or
            (dir_l and game.is_collision(pt_u)) or
            (dir_r and game.is_collision(pt_d)),

            (dir_d and game.is_collision(pt_r)) or
            (dir_u and game.is_collision(pt_l)) or
            (dir_r and game.is_collision(pt_u)) or
            (dir_l and game.is_collision(pt_d)),

            dir_l, dir_r, dir_u, dir_d,

            game.food[0] < game.head[0],
            game.food[0] > game.head[0],
            game.food[1] < game.head[1],
            game.food[1] > game.head[1]
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > 1000:
            mini_sample = random.sample(self.memory, 1000)
        else:
            mini_sample = list(self.memory)
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move


# ─── TRAINING LOOP ───
def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGame()

    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                torch.save(agent.model.state_dict(), 'snake_model.pth')

            print(f'Game {agent.n_games} | Score: {score} | Record: {record}')

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)

            plt.clf()
            plt.title('Snake AI Training Progress')
            plt.xlabel('Number of Games')
            plt.ylabel('Score')
            plt.plot(plot_scores, label='Score', color='hotpink')
            plt.plot(plot_mean_scores, label='Mean Score', color='deeppink')
            plt.legend()
            plt.pause(0.1)


if __name__ == '__main__':
    pygame.init()
    plt.ion()
    train()
    