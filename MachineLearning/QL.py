import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation
import time

class QLearning:
    def __init__(self, grid_size=5, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.grid_size = grid_size
        self.actions = ['up', 'right', 'down', 'left']
        self.Q = np.zeros((grid_size, grid_size, len(self.actions)))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.goal_state = (grid_size-1, grid_size-1)
        self.goal_reward = 1
        self.step_reward = -0.01
        self.obstacle_reward = -1
        self.maze = self.create_maze()

    def create_maze(self):
        maze = np.zeros((self.grid_size, self.grid_size))
        maze[0, 0] = 2  # 起点
        maze[self.grid_size-1, self.grid_size-1] = 3  # 终点
        
        # 添加一些障碍物
        obstacles = [(np.random.randint(2, self.grid_size), np.random.randint(2, self.grid_size)), (2, 1), (3, 3), (np.random.randint(2, self.grid_size), np.random.randint(2, self.grid_size))]
        for obs in obstacles:
            maze[obs] = 1
        
        return maze

    def plot_maze(self):
        cmap = ListedColormap(['white', 'black', 'green', 'red'])
        plt.figure(figsize=(8, 8))
        plt.imshow(self.maze, cmap=cmap)
        plt.title("Maze Configuration")
        plt.colorbar(ticks=[0, 1, 2, 3], label='0: Path, 1: Obstacle, 2: Start, 3: Goal')
        plt.show()

    def get_next_state(self, state, action):
        x, y = state
        if action == 'up':
            next_state = (max(x-1, 0), y)
        elif action == 'right':
            next_state = (x, min(y+1, self.grid_size-1))
        elif action == 'down':
            next_state = (min(x+1, self.grid_size-1), y)
        elif action == 'left':
            next_state = (x, max(y-1, 0))
        
        # 如果下一个状态是障碍物,保持在当前状态
        if self.maze[next_state] == 1:
            return state
        return next_state

    def epsilon_greedy(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            return self.actions[np.argmax(self.Q[state])]

    def train(self, episodes):
        for episode in range(episodes):
            state = (0, 0)  # 起始状态
            steps = 0
            total_reward = 0
            
            while state != self.goal_state:
                action = self.epsilon_greedy(state)
                next_state = self.get_next_state(state, action)
                
                if next_state == state:  # 撞到障碍物
                    reward = self.obstacle_reward
                elif next_state == self.goal_state:
                    reward = self.goal_reward
                else:
                    reward = self.step_reward
                
                # Q-learning更新公式
                self.Q[state][self.actions.index(action)] += self.alpha * (reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state][self.actions.index(action)])
                
                state = next_state
                steps += 1
                total_reward += reward
            
            if episode % 100 == 0:
                print(f"Episode {episode}: Steps = {steps}, Total Reward = {total_reward:.2f}")

    def plot_q_values(self):
        plt.figure(figsize=(10, 8))
        plt.imshow(np.max(self.Q, axis=2), cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title('Q-values')
        plt.show()

    def animate_optimal_path(self):
        fig, ax = plt.subplots(figsize=(8, 8))
        
        maze_with_path = self.maze.copy()
        state = (0, 0)
        path = [state]
        
        def update(frame):
            nonlocal state
            if state != self.goal_state:
                action = self.actions[np.argmax(self.Q[state])]
                next_state = self.get_next_state(state, action)
                if next_state != state:  # 只有在成功移动时才标记路径
                    maze_with_path[next_state] = 4  # 标记路径
                state = next_state
                path.append(state)
            
            ax.clear()
            cmap = ListedColormap(['white', 'black', 'green', 'red', 'yellow'])
            ax.imshow(maze_with_path, cmap=cmap)
            ax.set_title(f"Step {frame+1}")
            return ax,
        
        anim = FuncAnimation(fig, update, frames=self.grid_size*self.grid_size, interval=500, repeat=False)
        plt.show()
        
        return path

if __name__ == "__main__":
    ql = QLearning(10)
    
    # 显示迷宫配置
    ql.plot_maze()
    
    # 训练模型
    ql.train(1000)
    
    # 显示Q值
    ql.plot_q_values()
    
    # 显示最优路径动画
    optimal_path = ql.animate_optimal_path()
    print("Optimal path:", optimal_path)