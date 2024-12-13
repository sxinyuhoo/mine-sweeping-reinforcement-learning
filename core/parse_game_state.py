import os
import cv2
import numpy as np
import random
import time
import pyautogui
from mss import mss

def capture_screen(region=None):
    """
    获取屏幕截图
    """
    with mss() as sct:
        screenshot = sct.grab(region)
        img = np.array(screenshot)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
def parse_grid_structure(img, templates, grid_size=9):
    """
    解析网格结构
    """

    # 以 9x9 的格子为例，每个格子是一个数字，或者一个雷，或者一个空白，或者一个旗子，或者一个问号，或者一个未知的状态
    # 0: 未知的状态 1: 旗子 2: 空白 3: 数字 4: 雷 
    # 0. Define the game state
    grid = [[0 for _ in range(grid_size)] for _ in range(grid_size)]

    grid_template = templates['grid_template']
    # 除grid_template外，其他的模板都是单个数字或者雷的模板 
    grid_state_template = {k: v for k, v in templates.items() if k != 'grid_template' and k != 'reset_button_template'}

    # 匹配格子grid_size x grid_size的区域
    grid_top_left, grid_bottom_right, _grid_match_val = match_template(img, grid_template)
    grid_width = grid_bottom_right[0] - grid_top_left[0]
    grid_height = grid_bottom_right[1] - grid_top_left[1]

    cell_width = grid_width // grid_size
    cell_height = grid_height // grid_size

    return grid, grid_top_left, cell_width, cell_height, grid_state_template
    
def parse_grid_state(img, grid, grid_top_left, cell_width, cell_height, grid_state_template):
    """
    解析每个格子的状态
    """

    grid_size = len(grid)

    for i in range(grid_size):
        for j in range(grid_size):
            # 计算每个格子的坐标
            x = grid_top_left[0] + j * cell_width
            y = grid_top_left[1] + i * cell_height
            cell = img[y:y+cell_height, x:x+cell_width]

            grid[i][j] = recognize_cell(cell, grid_state_template)

            # 画出每个格子的边界
            # cv2.rectangle(img, (x, y), (x + cell_width, y + cell_height), (0, 255, 0), 1)

    return grid

def parse_subgrid_pos(grid, grid_top_left, cell_width, cell_height):
    """
    解析每个格子的中心坐标，构建一个字典，key是格子的坐标，value是格子的中心坐标
    """
    grid_size = len(grid)

    centers = {}

    for i in range(grid_size):
        for j in range(grid_size):
            x = grid_top_left[0] + j * cell_width + cell_width // 2
            y = grid_top_left[1] + i * cell_height + cell_height // 2
            centers[(i,j)] = (x, y)

    return centers

def parse_reset_pos(img, reset_button_template):
    """
    解析重置按钮的中心位置
    """

    top_left, bottom_right, _ = match_template(img, reset_button_template)

    center_pos_x = bottom_right[0] - (bottom_right[0] - top_left[0]) // 2
    center_pos_y = bottom_right[1] - (bottom_right[1] - top_left[1]) // 2
    return (center_pos_x, center_pos_y)

def recognize_cell(cell, templates, threshold=0.8):
    """
    识别每个格子的状态
    """

    max_match_val = 0
    best_match_value = None

    for value, template in templates.items():
        top_left, bottom_right, match_val = match_template(cell, template)
        
        # if match_val > max_match_val:
        #     max_match_val = match_val
        #     best_match_value = value
        if match_val > threshold:
            best_match_value = value
            break
    
    # # 如果最佳匹配值是0或者77，那么需要进一步判断
    if best_match_value in (0, 77):
        
        # 取cell的左上角像素点的颜色，如果是白色，则为0，否则是77
        if cell[5, 5][0] > 200 and cell[5, 5][1] > 200 and cell[5, 5][2] > 200:
            best_match_value = 0
        else:
            best_match_value = 77

    return best_match_value # 已探索过的空白区域

def match_template(img, template):
    """
    模版匹配
    """
    
    # 1. 将图像转换为灰度图像
    if len(img.shape) == 2:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. 进行模版匹配
    res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)

    # 3. 获取最佳匹配的位置，以及最大值和最小值，最大值和最小值的位置
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # 4. 获取匹配的左上角和右下角的坐标，原点在左上角
    top_left = max_loc
    h, w = template.shape
    bottom_right = (top_left[0] + w, top_left[1] + h)

    return top_left, bottom_right, max_val

def locate_game_elements(img, reset_button_template, game_area_template):
    
    game_area_top_left, game_area_bottom_right, _game_area_val = match_template(img, game_area_template)
    reset_button_top_left, reset_button_bottom_right, _reset_button_val = match_template(img, reset_button_template)

    print("game_area_top_left: ", game_area_top_left, "game_area_bottom_right: ", game_area_bottom_right)
    print("reset_button_top_left: ", reset_button_top_left, "reset_button_bottom_right: ", reset_button_bottom_right)

    cv2.rectangle(img, game_area_top_left, game_area_bottom_right, (0, 0, 255), 2)
    cv2.rectangle(img, reset_button_top_left, reset_button_bottom_right, (0, 0, 255), 2)

    return {
        'game_area': (game_area_top_left, game_area_bottom_right),
        'reset_button': (reset_button_top_left, reset_button_bottom_right)
    }

def click_random_in_area(area, num_click=1):
    """
    在指定区域内随机点击
    """

    top_left, bottom_right = area
    print("top_left: ", top_left, "bottom_right: ", bottom_right)
    x = random.randint(top_left[0]+10, bottom_right[0]-10) # Avoid clicking on the border
    y = random.randint(top_left[1]+10, bottom_right[1]-10)

    pyautogui.moveTo(x//2, y//2)
    pyautogui.click(clicks=num_click, interval=0.5)

def click_position(x, y, num_click=1, interval=0.5, button='left'): 
    """
    点击指定位置
    """
    pyautogui.moveTo(x//2, y//2)
    pyautogui.click(clicks=num_click, interval=interval, button=button)

def test_click():

    reset_button_template_path = "../docs/screenshot/reset_button_template.png"
    game_area_template_path = "../docs/screenshot/game_area_template.png"

    # img_ms_template = cv2.imread(ms_template_path, cv2.IMREAD_GRAYSCALE)
    img_reset_button_template = cv2.imread(reset_button_template_path, cv2.IMREAD_GRAYSCALE)
    img_game_area_template = cv2.imread(game_area_template_path, cv2.IMREAD_GRAYSCALE)

    screen_width, screen_height = pyautogui.size()
    # Define the region of the screen to capture, start from (left, top) and the width and height of the region
    region = {'left': 0, 'top': 0, 'width': screen_width, 'height': screen_height}
    img = capture_screen(region)

    # img_area = match_template(img_catch_screen, img_ms_template) 

    elements = locate_game_elements(img, img_reset_button_template, img_game_area_template)
    game_area = elements['game_area'] 
    reset_button = elements['reset_button']

    print("click_random_in_game_area ... ")
    click_random_in_area(game_area, num_click=2)
    time.sleep(1)
    print("click_reset_button ... ")
    click_random_in_area(reset_button)

def test_parser():

    base_path = os.path.dirname(os.path.abspath(__file__))

    templates = {
        # unexploded cell
        0: cv2.imread(os.path.join(base_path,'../config/screenshot_template/0_template.png'), cv2.IMREAD_GRAYSCALE),

        # numbers of mines
        1: cv2.imread(os.path.join(base_path,'../config/screenshot_template/1_template.png'), cv2.IMREAD_GRAYSCALE),
        2: cv2.imread(os.path.join(base_path,'../config/screenshot_template/2_template.png'), cv2.IMREAD_GRAYSCALE),
        3: cv2.imread(os.path.join(base_path,'../config/screenshot_template/3_template.png'), cv2.IMREAD_GRAYSCALE),
        4: cv2.imread(os.path.join(base_path,'../config/screenshot_template/4_template.png'), cv2.IMREAD_GRAYSCALE),
        5: cv2.imread(os.path.join(base_path,'../config/screenshot_template/5_template.png'), cv2.IMREAD_GRAYSCALE),

        # failed
        99: cv2.imread(os.path.join(base_path,'../config/screenshot_template/failed_button_template.png'), cv2.IMREAD_GRAYSCALE),
        # flagged mine
        88: cv2.imread(os.path.join(base_path,'../config/screenshot_template/flag_template.png'), cv2.IMREAD_GRAYSCALE),
        # exploded but empty cell
        77: cv2.imread(os.path.join(base_path,'../config/screenshot_template/empty_template.png'), cv2.IMREAD_GRAYSCALE),

        # grid
        'grid_template': cv2.imread(os.path.join(base_path,'../config/screenshot_template/grid_template.png'), cv2.IMREAD_GRAYSCALE),
        # reset button
        'reset_button_template': cv2.imread(os.path.join(base_path,'../config/screenshot_template/reset_button_template.png'), cv2.IMREAD_GRAYSCALE)
    }

    screen_width, screen_height = pyautogui.size()
    region = {'left': 0, 'top': 0, 'width': screen_width, 'height': screen_height}
    img = capture_screen(region)

    time.sleep(3)
    grid_structure = parse_grid_structure(img, templates)
    grid_state = parse_grid_state(img, *grid_structure)
    grid_pos = parse_subgrid_pos(grid_structure[0], grid_structure[1], grid_structure[2], grid_structure[3])
    reset_pos = parse_reset_pos(img, templates['reset_button_template'])

    for row in grid_state:
        print(row) 
    print("\n === \n")

    # click_position(*grid_pos[(8,1)], num_click=2)
    # click_position(*reset_pos, num_click=2)

    # new_img = capture_screen(region)
    # new_grid_state = parse_grid_state(new_img, *grid_structure)

    # for row in new_grid_state:
    #     print(row)

# Example usage:
if __name__ == "__main__":

    test_parser()


