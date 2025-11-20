import math

exploration = 1.5

class MonteCarloNode():
    move = None
    parent = None
    visits = 0
    score_total = 0
    policy_score = 0

    prediction = None
    moves = None

    def get_score(self):
        return self.score_total / self.visits + exploration * self.policy_score * math.sqrt(self.parent.visits) / (1 + self.visits)
