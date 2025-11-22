import torch
import math

class ConnectFourGame():
    board_state = torch.zeros(6, 7)

    perspective = 1

    history = []

    def move_valid(self, position):
        return self.board_state[0, position] == 0

    def move(self, position):
        for i in range(6):
            if self.board_state[5 - i, position] == 0:
                self.board_state[5 - i, position] = 1

                break

        self.history.append(position)

        self.board_state *= -1
        self.perspective *= -1

    def undo(self):
        last_move = self.history.pop()

        for i in range(6):
            if self.board_state[i, last_move] != 0:
                self.board_state[i, last_move] = 0

                break
        
        self.board_state *= -1
        self.perspective *= -1
    
    def valid_moves(self):
        moves = []

        for x in range(7):
            if self.move_valid(x):
                moves.append(x)

        return moves

    def result(self):
        for start_y in range(6):
            for start_x in range(4):
                looking_for = self.board_state[start_y, start_x]

                if looking_for == 0: continue

                found = True

                for i in range(3):
                    if self.board_state[start_y, start_x + 1 + i] != looking_for:
                        found = False
                        break

                if found: return looking_for.item() * self.perspective

        for start_y in range(3):
            for start_x in range(7):
                looking_for = self.board_state[start_y, start_x]

                if looking_for == 0: continue

                found = True

                for i in range(3):
                    if self.board_state[start_y + 1 + i, start_x] != looking_for:
                        found = False
                        break

                if found: return looking_for.item() * self.perspective

        for start_y in range(3):
            for start_x in range(4):
                looking_for = self.board_state[start_y, start_x]

                if looking_for != 0:
                    found = True

                    for i in range(3):
                        if self.board_state[start_y + 1 + i, start_x + 1 + i] != looking_for:
                            found = False
                            break

                    if found: return looking_for.item() * self.perspective

                looking_for = self.board_state[start_y + 3, start_x]

                if looking_for != 0:
                    found = True

                    for i in range(3):
                        if self.board_state[start_y + 2 - i, start_x + 1 + i] != looking_for:
                            found = False
                            break

                    if found: return looking_for.item() * self.perspective
        
        return 0
    
    def display(self):
        print("-------")
        for y in range(6):
            line = ""

            for x in range(7):
                if self.board_state[y, x] == 0:
                    line += " "
                
                if self.board_state[y, x] * self.perspective == 1:
                    line += "O"

                if self.board_state[y, x] * self.perspective == -1:
                    line += "X"

            print(line)
        print("-------")