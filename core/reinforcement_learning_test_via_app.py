import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import parse_game_state as pgs
import pyautogui
import time
from gymnasium import spaces
from tianshou.data import Batch
from tianshou.policy import DQNPolicy
from core.reinforcement_learning_train_9x9 import MinesweeperAgent


class MinesweeperTestEnv():

    def __init__(self, templates, grid_size=9):
        self.grid_size = grid_size
        self.state_size = grid_size * grid_size

        _screen_width, _screen_height = pyautogui.size()
        self.region = {'left': 0, 'top': 0, 'width': _screen_width, 'height': _screen_height}
        self.templates = templates
        _init_img = pgs.capture_screen(self.region)
        self.grid_structure = pgs.parse_grid_structure(_init_img, self.templates, grid_size=self.grid_size)
        self.grid_pos = pgs.parse_subgrid_pos(self.grid_structure[0], self.grid_structure[1], self.grid_structure[2], self.grid_structure[3])
        self.reset_pos = pgs.parse_reset_pos(_init_img, self.templates['reset_button_template'])
        
        self.reset()

    def step(self, action):

        # 动作转换为网格坐标
        row = action // self.grid_size
        col = action % self.grid_size

        click_type = 'left' if self.state[action] == 7 else 'right'

        # 点击指定位置
        print("click_position ... ", row, col, click_type)
        pgs.click_position(*self.grid_pos[(row, col)], num_click=2, button=click_type)

        # 获取新的屏幕截图并解析网格状态
        img = pgs.capture_screen(self.region)
        
        new_grid_state = pgs.parse_grid_state(img, *self.grid_structure)
        # print(new_grid_state, "\n == \n")

        # 更新状态
        self.state = np.array(new_grid_state).reshape(-1)

        # 检查游戏是否结束
        done = self.check_done(self.state)

        if done:

            time.sleep(3)
            self.reset()
            return 9

        if 7 not in self.state:
            time.sleep(3)
            self.reset()
            return 100

    def reset(self):

        # 重置游戏状态
        self.state = np.full(self.state_size, 7, dtype=np.int32)
        
        # 点击重置按钮
        pgs.click_position(*self.reset_pos, num_click=2)

        return 
    
    def check_done(self, grid_state):
        # 检查是否有雷被点击
        if 9 in grid_state:  # 9表示失败
            return True
        return False

def main():

    base_path = os.path.dirname(os.path.abspath(__file__))

    templates = {

        # numbers of mines
        0: cv2.imread(os.path.join(base_path,'../config/screenshot_template/0_template.png'), cv2.IMREAD_GRAYSCALE),
        1: cv2.imread(os.path.join(base_path,'../config/screenshot_template/1_template.png'), cv2.IMREAD_GRAYSCALE),
        2: cv2.imread(os.path.join(base_path,'../config/screenshot_template/2_template.png'), cv2.IMREAD_GRAYSCALE),
        3: cv2.imread(os.path.join(base_path,'../config/screenshot_template/3_template.png'), cv2.IMREAD_GRAYSCALE),
        4: cv2.imread(os.path.join(base_path,'../config/screenshot_template/4_template.png'), cv2.IMREAD_GRAYSCALE),
        5: cv2.imread(os.path.join(base_path,'../config/screenshot_template/5_template.png'), cv2.IMREAD_GRAYSCALE),

        # mine
        9: cv2.imread(os.path.join(base_path,'../config/screenshot_template/mine_template.png'), cv2.IMREAD_GRAYSCALE),
        # flagged mine
        8: cv2.imread(os.path.join(base_path,'../config/screenshot_template/flag_template.png'), cv2.IMREAD_GRAYSCALE),
        # unemploded cell
        7: cv2.imread(os.path.join(base_path,'../config/screenshot_template/empty_template.png'), cv2.IMREAD_GRAYSCALE),

        # grid
        'grid_template': cv2.imread(os.path.join(base_path,'../config/screenshot_template/grid_template.png'), cv2.IMREAD_GRAYSCALE),
        # reset button
        'reset_button_template': cv2.imread(os.path.join(base_path,'../config/screenshot_template/reset_button_template.png'), cv2.IMREAD_GRAYSCALE)
    }

    # 初始化环境和智能体
    grid_size = 9
    state_size = grid_size * grid_size
    action_size = grid_size * grid_size

    game_env = MinesweeperTestEnv(templates, grid_size=grid_size)
    agent = MinesweeperAgent(state_size, action_size)

    # 加载模型
    model_path = os.path.join(base_path, '../log/minesweeper_9x9/policy.pth')
    if os.path.exists(model_path):
        try:
            agent.policy.load_state_dict(torch.load(model_path))
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print("Model not found in ", model_path)

    # 游戏主循环
    for _ in range(30):  # 玩10局游戏
        state = game_env.state
        # print("state: ", state, "\n")

        # 将状态转换为Batch对象
        batch = Batch(
            obs=torch.FloatTensor([state]),
            info={}
        )
        
        # 使用策略选择动作
        with torch.no_grad():
            result = agent.policy(batch)
            # print("result: ", result, "\n")
            action = result.act[0].item()
            # print("action: ", action, "\n")
        
        # 执行动作
        res = game_env.step(action)
        
        if res == 9:  # 游戏结束
            print("Game Over")
            time.sleep(3)  # 等待一秒后开始新游戏
            break

        if res == 100:  # 游戏胜利
            print("Game Win")
            time.sleep(3)
            break

if __name__ == '__main__':
    main()