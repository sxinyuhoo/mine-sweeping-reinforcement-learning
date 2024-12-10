
import cv2
import numpy as np
import random
import time
import pyautogui
from mss import mss

def capture_screen(region=None):
    with mss() as sct:
        screenshot = sct.grab(region)
        img = np.array(screenshot)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
def parse_game_state(img):

    # 以 9x9 的格子为例，每个格子是一个数字，或者一个雷，或者一个空白，或者一个旗子，或者一个问号，或者一个未知的状态
    # 0: 未知的状态 1: 旗子 2: 问号 3: 空白 4: 数字 5: 雷 
    # 0. Define the game state
    game_state = []

    # 1. Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Apply Gaussian blur to the image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Apply adaptive thresholding to the image
    thresholded = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # 4. Find contours in the image
    contours, _ = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 5. Loop over the contours
    game_state = []
    for contour in contours:
        # 6. Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)

        # 7. Extract the region of interest (ROI) from the image
        roi = thresholded[y:y+h, x:x+w]

        # 8. Resize the ROI to a fixed size (e.g., 28x28)
        roi_resized = cv2.resize(roi, (28, 28))

        # 9. Flatten the ROI and append it to the game state
        game_state.append(roi_resized.flatten())

    # 10. Return the game state

    return game_state

def match_template(img, template):
    # 1. Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Apply template matching
    res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)

    # 3. Get the location of the best match
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # 4. Get the top-left and bottom-right coordinates of the bounding box
    top_left = max_loc
    h, w = template.shape
    bottom_right = (top_left[0] + w, top_left[1] + h)

    return top_left, bottom_right

def locate_game_elements(img, reset_button_template, game_area_template):
    game_area_top_left, game_area_bottom_right = match_template(img, game_area_template)
    reset_button_top_left, reset_button_bottom_right = match_template(img, reset_button_template)

    print("game_area_top_left: ", game_area_top_left, "game_area_bottom_right: ", game_area_bottom_right)
    print("reset_button_top_left: ", reset_button_top_left, "reset_button_bottom_right: ", reset_button_bottom_right)

    cv2.rectangle(img, game_area_top_left, game_area_bottom_right, (0, 0, 255), 2)
    cv2.rectangle(img, reset_button_top_left, reset_button_bottom_right, (0, 0, 255), 2)

    return {
        'game_area': (game_area_top_left, game_area_bottom_right),
        'reset_button': (reset_button_top_left, reset_button_bottom_right)
    }

def click_random_in_area(area, num_click=1):
    top_left, bottom_right = area
    print("top_left: ", top_left, "bottom_right: ", bottom_right)
    x = random.randint(top_left[0]+10, bottom_right[0]-10) # Avoid clicking on the border
    y = random.randint(top_left[1]+10, bottom_right[1]-10)

    pyautogui.moveTo(x//2, y//2)
    pyautogui.click(clicks=num_click, interval=0.5)

# Example usage:
if __name__ == "__main__":

    reset_button_template_path = "/Users/sean/Documents/study/project/RL/mine-sweeping-reinforcement-learning/docs/screenshot/reset_button_template.png"
    game_area_template_path = "/Users/sean/Documents/study/project/RL/mine-sweeping-reinforcement-learning/docs/screenshot/game_area_template.png"

    # img_ms_template = cv2.imread(ms_template_path, cv2.IMREAD_GRAYSCALE)
    img_reset_button_template = cv2.imread(reset_button_template_path, cv2.IMREAD_GRAYSCALE)
    img_game_area_template = cv2.imread(game_area_template_path, cv2.IMREAD_GRAYSCALE)

    screen_width, screen_height = pyautogui.size()
    print("screen_width: ", screen_width, "screen_height: ", screen_height)
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
    
    # cv2.imshow('Matched img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


