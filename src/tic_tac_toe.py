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
        
        self.board_state *= -1
        self.perspective *= -1

    def hash(self):
        hash = ""

        for y in range(3):
            for x in range(3):
               hash += self.board_state[y, x]

        return hash
    
    def valid_moves(self):
        moves = []

        for y in range(3):
            for x in range(3):
               if self.move_valid([x, y]):
                   moves.append([x, y])

        return moves

    def result(self):
        if self.board_state[0, 0] == 1 and self.board_state[0, 1] == 1 and self.board_state[0, 2] == 1:
            return 1 * self.perspective
        if self.board_state[1, 0] == 1 and self.board_state[1, 1] == 1 and self.board_state[1, 2] == 1:
            return 1 * self.perspective
        if self.board_state[2, 0] == 1 and self.board_state[2, 1] == 1 and self.board_state[2, 2] == 1:
            return 1 * self.perspective
        if self.board_state[0, 0] == 1 and self.board_state[1, 0] == 1 and self.board_state[2, 0] == 1:
            return 1 * self.perspective
        if self.board_state[0, 1] == 1 and self.board_state[1, 1] == 1 and self.board_state[2, 1] == 1:
            return 1 * self.perspective
        if self.board_state[0, 2] == 1 and self.board_state[1, 2] == 1 and self.board_state[2, 2] == 1:
            return 1 * self.perspective
        if self.board_state[0, 0] == 1 and self.board_state[1, 1] == 1 and self.board_state[2, 2] == 1:
            return 1 * self.perspective
        if self.board_state[0, 2] == 1 and self.board_state[1, 1] == 1 and self.board_state[2, 0] == 1:
            return 1 * self.perspective
        
        if self.board_state[0, 0] == -1 and self.board_state[0, 1] == -1 and self.board_state[0, 2] == -1:
            return -1 * self.perspective
        if self.board_state[1, 0] == -1 and self.board_state[1, 1] == -1 and self.board_state[1, 2] == -1:
            return -1 * self.perspective
        if self.board_state[2, 0] == -1 and self.board_state[2, 1] == -1 and self.board_state[2, 2] == -1:
            return -1 * self.perspective
        if self.board_state[0, 0] == -1 and self.board_state[1, 0] == -1 and self.board_state[2, 0] == -1:
            return -1 * self.perspective
        if self.board_state[0, 1] == -1 and self.board_state[1, 1] == -1 and self.board_state[2, 1] == -1:
            return -1 * self.perspective
        if self.board_state[0, 2] == -1 and self.board_state[1, 2] == -1 and self.board_state[2, 2] == -1:
            return -1 * self.perspective
        if self.board_state[0, 0] == -1 and self.board_state[1, 1] == -1 and self.board_state[2, 2] == -1:
            return -1 * self.perspective
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

class TicTacToeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = torch.nn.Linear(9, 20)
        self.layer2 = torch.nn.Linear(20, 20)
        self.layer3 = torch.nn.Linear(20, 20)
        self.layer4 = torch.nn.Linear(20, 10)

    def forward(self, game):
        value = game.board_state.flatten().float()

        value = torch.relu(self.layer1(value))
        value = torch.relu(self.layer2(value))
        value = torch.relu(self.layer3(value))

        value = self.layer4(value)

        policy = value[:9]
        score = value[9]

        policy = torch.nn.functional.softmax(policy, dim=0)
        score = torch.tanh(score)

        return (policy, score)
    
    def loss(self, policy, score, target_policy, target_score):
        policy_loss = -torch.sum(target_policy * policy)
        
        value_loss = (score - target_score) ** 2
        
        loss = policy_loss + value_loss
        
        return loss