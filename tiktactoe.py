import torch
import math

class TicTacToeGame():
    board_state = torch.tensor([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ])

    perspective = 1

    history = []

    def move_valid(self, position):
        return self.board_state[position[1], position[0]] == 0

    def move(self, position):
        self.board_state[position[1], position[0]] = 1

        self.history.append(position)

        self.board_state *= -1
        self.perspective *= -1

    def undo(self):
        last_move = self.history.pop()

        self.board_state[last_move[1], last_move[0]] = 0

    def result(self):
        if self.board_state[0, 0] == 1 and self.board_state[0, 1] == 1 and self.board_state[0, 2] == 1:
            return 1
        if self.board_state[1, 0] == 1 and self.board_state[1, 1] == 1 and self.board_state[1, 2] == 1:
            return 1
        if self.board_state[2, 0] == 1 and self.board_state[2, 1] == 1 and self.board_state[2, 2] == 1:
            return 1
        if self.board_state[0, 0] == 1 and self.board_state[1, 0] == 1 and self.board_state[2, 0] == 1:
            return 1
        if self.board_state[0, 1] == 1 and self.board_state[1, 1] == 1 and self.board_state[2, 1] == 1:
            return 1
        if self.board_state[0, 2] == 1 and self.board_state[1, 2] == 1 and self.board_state[2, 2] == 1:
            return 1
        if self.board_state[0, 0] == 1 and self.board_state[1, 1] == 1 and self.board_state[2, 2] == 1:
            return 1
        if self.board_state[0, 2] == 1 and self.board_state[1, 1] == 1 and self.board_state[2, 0] == 1:
            return 1 * self.perspective
        
        if self.board_state[0, 0] == -1 and self.board_state[0, 1] == -1 and self.board_state[0, 2] == -1:
            return -1
        if self.board_state[1, 0] == -1 and self.board_state[1, 1] == -1 and self.board_state[1, 2] == -1:
            return -1
        if self.board_state[2, 0] == -1 and self.board_state[2, 1] == -1 and self.board_state[2, 2] == -1:
            return -1
        if self.board_state[0, 0] == -1 and self.board_state[1, 0] == -1 and self.board_state[2, 0] == -1:
            return -1
        if self.board_state[0, 1] == -1 and self.board_state[1, 1] == -1 and self.board_state[2, 1] == -1:
            return -1
        if self.board_state[0, 2] == -1 and self.board_state[1, 2] == -1 and self.board_state[2, 2] == -1:
            return -1
        if self.board_state[0, 0] == -1 and self.board_state[1, 1] == -1 and self.board_state[2, 2] == -1:
            return -1
        if self.board_state[0, 2] == -1 and self.board_state[1, 1] == -1 and self.board_state[2, 0] == -1:
            return -1 * self.perspective
        
        return 0
    
    def display(self):
        print("-----------")
        for y in range(3):
            line = ""

            for x in range(3):
                if len(line) > 0:
                    line += "|"

                if self.board_state[y, x] == 0:
                    line += "   "
                
                if self.board_state[y, x] * self.perspective == 1:
                    line += " O "

                if self.board_state[y, x] * self.perspective == -1:
                    line += " X "

            print(line)
            print("-----------")