"""A command line version of Minesweeper"""
import random
import numpy as np
import time
from string import ascii_lowercase

class MineSweeper:
    def __init__(self, grid_size = 9, num_mines = 10):
        """
        初始化扫雷游戏的网格大小和地雷数量。
        """
        self.grid_size = grid_size
        self.num_mines = num_mines

        self.cur_grid = [[7 for i in range(grid_size)] for i in range(grid_size)]

        self.grid = []
        self.starttime = 0

    def setup_grid(self, grid_size, start, num_mines):
        """
        设置游戏网格并放置地雷。

        """
        empty_grid = [[0 for i in range(grid_size)] for i in range(grid_size)]

        mines = self.get_mines(empty_grid, start, num_mines) # avoid mines in the starting cell

        for i, j in mines:
            empty_grid[i][j] = 9

        grid = self.update_grid_cell(empty_grid)

        return (grid, mines)

    def get_random_cell(self, grid):
        """
        随机获取一个网格单元的位置 
        """
        grid_size = len(grid)

        a = random.randint(0, grid_size - 1)
        b = random.randint(0, grid_size - 1)

        return (a, b)


    def get_neighbors(self, grid, row_indx, col_indx):
        """
        获取指定单元的邻居单元。

        (-1, -1), (-1, 0), (-1, 1)
        (0, -1), (0, 0), (0, 1)
        (1, -1), (1, 0), (1, 1)
        """
        grid_size = len(grid)
        neighbors = []

        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0: # Skip the cell itself
                    continue
                elif -1 < (row_indx + i) < grid_size and -1 < (col_indx + j) < grid_size:
                    neighbors.append((row_indx + i, col_indx + j))

        return neighbors


    def get_mines(self, grid, start, num_mines):
        """
        随机放置地雷，确保起始单元和其邻居单元没有地雷。
        """
        mines = []
        neighbors = self.get_neighbors(grid, *start)

        for i in range(num_mines):
            cell = self.get_random_cell(grid)
            while cell == start or cell in mines or cell in neighbors:
                cell = self.get_random_cell(grid)
            mines.append(cell)

        return mines


    def update_grid_cell(self, grid):
        """
        计算每个单元周围的地雷数量并更新网格。
        """
        for row_indx, row in enumerate(grid):
            for col_indx, cell in enumerate(row):
                if cell != 9:
                    # Gets the values of the neighbors
                    values = [grid[r][c] for r, c in self.get_neighbors(grid, row_indx, col_indx)]

                    # Counts how many are mines
                    grid[row_indx][col_indx] = values.count(9)

        return grid


    def show_cells(self, grid, cur_grid, row_indx, col_indx):
        """
        显示当前单元及其邻居单元, 递归显示空单元。
        """
        # Exit function if the cell was already shown
        if cur_grid[row_indx][col_indx] != 7:
            return

        # Show current cell
        cur_grid[row_indx][col_indx] = grid[row_indx][col_indx]

        # Get the neighbors if the cell is empty
        if grid[row_indx][col_indx] == 0:
            for r, c in self.get_neighbors(grid, row_indx, col_indx):
                # Repeat function for each neighbor that doesn't have a flag
                self.show_cells(grid, cur_grid, r, c)


    def play_game(self, pos):
        """
        处理游戏逻辑，包括检查游戏结束条件。
        """
        if pos:
            # print('\n\n')
            row_indx, col_indx = pos
            cur_cell = self.cur_grid[row_indx][col_indx]

            if not self.grid:
                self.grid, mines = self.setup_grid(self.grid_size, pos, self.num_mines)
            if not self.starttime:
                self.starttime = time.time()

            # if game over, return 0
            if self.grid[row_indx][col_indx] == 9:
                print('Game Over\n')
                # self.show_grid(self.grid)
                return -1

            elif cur_cell == 7:
                print("Cell is empty")
                self.show_cells(self.grid, self.cur_grid, row_indx, col_indx)
                # self.show_grid(self.cur_grid)
            else:
                # print(pos)
                print("That cell is already shown")
                return -2

            cur_grid_array = np.array(self.cur_grid)
            mines_left = self.num_mines - len(cur_grid_array[cur_grid_array==7])

            if mines_left == 0:
                minutes, seconds = divmod(int(time.time() - self.starttime), 60)
                print(
                    'You Win. '
                    'It took you {} minutes and {} seconds.\n'.format(minutes, seconds))
                # self.show_grid(self.grid)
                return 1
            
        return 0
    
    def show_grid(self, grid):
        """
        显示当前游戏网格。
        """
        grid_size = len(grid)

        horizontal = '   ' + (4 * grid_size * '-') + '-'

        # Print top column letters
        top_label = '     '

        for i in ascii_lowercase[:grid_size]:
            top_label = top_label + i + '   '

        print(top_label + '\n' + horizontal)

        # Print left row numbers
        for idx, i in enumerate(grid):
            row = '{0:2} |'.format(idx + 1)

            for j in i:
                if j == 7:
                    j = ' '
                row = row + ' ' + str(j) + ' |'

            print(row + '\n' + horizontal)

        print('')

if __name__ == "__main__":
    game = MineSweeper(9, 10)
    game.show_grid(game.cur_grid)

    while True:
        try:
            row = int(input('Please enter the row: '))
            col = int(input('Please enter the column: '))
            res = game.play_game((row, col))
            if res == -1:
                break
        except:
            print("Invalid input")
            break
