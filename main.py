import torch

torch.set_default_device( torch.device("cuda:0"))
torch.autograd.set_detect_anomaly(True)

import math
import random
from tic_tac_toe import TicTacToeGame
from connect_four import ConnectFourGame
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

            if display:
                if move.visits > 0:
                    print(move.move, math.floor(move.get_score() * 100) / 100, math.floor(move.score_total / move.visits * 100) / 100, move.visits, math.floor(move.policy_score * 100) / 100)
                else:
                    print(move.move, math.floor(move.get_score() * 100) / 100, "no visits", move.visits,  math.floor(move.policy_score * 100) / 100)

        if display: print("Exploring ", best_move.move)

        game.move(best_move.move)

        result = monte_carlo_tree_search(best_move, display)

        game.undo()

        node.score_total += result * -game.perspective

        return result
    else:
        if display: print("Ended with result! ", result)

        node.score_total += result * -game.perspective

        return result

def play_game(node, history):
    for i in range(20):
        monte_carlo_tree_search(node)

    result = game.result()

    if result == 0 and len(node.moves) > 0:
        history.append(node)

        best_move = None
        best_score = 0

        for move in node.moves:
            score = move.get_score()

            if best_move == None or score > best_score:
                best_move = move
                best_score = score

        game.move(best_move.move)

        result = play_game(best_move, history)

        game.undo()

        return result
    else:
        return result

def train(result, history, display=False):
    total_loss = 0

    for history_node in history:
        if history_node.move: game.move(history_node.move)

        if display: 
            game.display()

            for move in history_node.moves:
                if move.visits > 0:
                    print(move.move, math.floor(move.get_score() * 100) / 100, math.floor(move.score_total / move.visits * 100) / 100, move.visits, math.floor(move.policy_score * 100) / 100)
                else:
                    print(move.move, math.floor(move.get_score() * 100) / 100, "no visits", move.visits,  math.floor(move.policy_score * 100) / 100)

        if len(history_node.moves) == 0:
            continue

        (policy, score) = model.forward(game)

        target_policy = torch.zeros(9)

        for move in history_node.moves:
            target_policy[move.move[1] * 3 + move.move[0]] = move.visits


        target_policy = target_policy / torch.sum(target_policy)

        target_score = torch.tensor(result)

        loss = model.loss(policy, score, target_policy, target_score)

        total_loss += abs(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for history_node in history:
        if history_node.move: game.undo()

    print(total_loss / len(history))

# for i in range(1000):
#     history = []
#     result = play_game(MonteCarloNode(), history)
#     train(result, history)

#     if i % 10 == 0:
#         history = []
#         result = play_game(MonteCarloNode(), history)
#         train(result, history, True)

game = ConnectFourGame()
game.move(0)
game.move(1)
game.move(1)
game.move(0)

game.move(2)
game.move(2)
game.move(2)

game.move(3)
game.move(3)
game.move(3)
game.move(3)

game.display()

print(game.result())