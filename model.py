import torch
import math

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

        