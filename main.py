import torch

torch.set_default_device( torch.device("cuda:0"))

import math
import random
from tic_tac_toe import TicTacToeGame
from monte_carlo import MonteCarloNode
from model import TicTacToeModel

game = TicTacToeGame()

model = TicTacToeModel()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def monte_carlo_tree_search(node, display = False):
    if display: game.display()

    node.visits += 1

    if node.moves == None:
        valid_moves = game.valid_moves()

        policy, score = model(game)

        node.prediction = (policy, score)

        node.moves = []

        for move in valid_moves:
            childNode = MonteCarloNode()
            childNode.move = move
            childNode.parent = node
            childNode.policy_score = policy[move[1] * 3 + move[0]].item()

            node.moves.append(childNode)

    result = game.result()

    if result == 0 and len(node.moves) > 0:
        best_move = None
        best_score = 0

        for move in node.moves:
            score = move.get_score()

            if best_move == None or score > best_score:
                best_move = move
                best_score = score

            if display: print(move.move, score, move.visits)

        if display: print("Exploring ", best_move.move)

        game.move(best_move.move)

        result = monte_carlo_tree_search(best_move, display)

        game.undo()

        node.score_total += -result

        return -result
    else:
        if display: print("Ended with result! ", result)

        node.score_total += result * game.perspective

        return result

def train_model(node):
    if node.prediction == None:
        return

    policy, score = node.prediction

    target_policy = torch.zeros(9)
    target_score = torch.tensor(1.0)

    loss = model.loss(policy, score, target_policy, target_score)

    print(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    result = game.result()

    if result == 0 and len(node.moves) > 0:
        for move in node.moves:
            game.move(move.move)

            train_model(move)

            game.undo()

def iteration():
    monte_carlo_tree = MonteCarloNode()

    for i in range(10):
        monte_carlo_tree_search(monte_carlo_tree)

    train_model(monte_carlo_tree)

# x = torch.linspace(-math.pi, math.pi, 2000)
# y = torch.sin(x)

# p = torch.tensor([1, 2, 3])
# xx = x.unsqueeze(-1).pow(p)

# model = torch.nn.Sequential(
#     torch.nn.Linear(3, 1),
#     torch.nn.Flatten(0, 1)
# )

# loss_fn = torch.nn.MSELoss(reduction='sum')

# learning_rate = 1e-3
# optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

# for t in range(2000):
#     model.zero_grad()

#     y_pred = model(xx)

#     loss = loss_fn(y_pred, y)
#     if t % 100 == 99:
#         print(t, loss.item())

#     loss.backward()

#     optimizer.step()

# linear_layer = model[0]

# print(f'Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:, 0].item()} x + {linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:, 2].item()} x^3')