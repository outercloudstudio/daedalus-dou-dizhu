import torch
import math
from tic_tac_toe import TicTacToeGame
from monte_carlo import MonteCarloNode
torch.set_default_device( torch.device("cuda:0"))

game = TicTacToeGame()

monte_carlo_tree = MonteCarloNode()

def explore_node(node):
    game.display()

    node.visits += 1

    if node.moves == None:
        valid_moves = game.valid_moves()

        node.prediction = []
        node.moves = []

        for move in valid_moves:
            childNode = MonteCarloNode()
            childNode.move = move
            childNode.parent = node
            childNode.policy_score = 0

            node.moves.append(childNode)

    result = game.result()

    if result == 0:
        best_move = None
        best_score = 0

        for move in node.moves:
            score = move.get_score()

            if best_move == None or score > best_score:
                best_move = move
                best_score = score

            print(move.move, score)

        print("Exploring ", best_move.move)

        game.move(best_move.move)

        result = explore_node(best_move)

        game.undo()

        node.score_total += -result

        return -result
    else:
        print("Ended with result! ", result)

        node.score_total += result * game.perspective

        return result

explore_node(monte_carlo_tree)
explore_node(monte_carlo_tree)
explore_node(monte_carlo_tree)
explore_node(monte_carlo_tree)

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