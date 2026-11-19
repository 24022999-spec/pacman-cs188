import unittest

import layout
from pacman import GameState
from multiAgents import (
    MinimaxAgent, ExpectimaxAgent, BetterMinimaxAgent,
    BetterExpectimaxAgent, betterEvaluationFunction,
)
from game import Directions


class TreeState:
    def __init__(self, tree, position, agents, counts, terminal=False):
        self.tree = tree
        self.position = position
        self.agents = agents
        self.counts = counts
        self.terminal = terminal

    def getNumAgents(self):
        return self.agents

    def isWin(self):
        return False

    def isLose(self):
        return self.terminal

    def getScore(self):
        node = self.tree[self.position]
        return float(node) if isinstance(node, (int, float)) else 0.0

    def getLegalActions(self, agent):
        node = self.tree[self.position]
        if isinstance(node, (int, float)):
            return []
        return list(node.keys())

    def generateSuccessor(self, agent, action):
        self.counts.append((self.position, agent, action))
        nxt = self.tree[self.position][action]
        return TreeState(self.tree, nxt, self.agents, self.counts)


class MultiagentTests(unittest.TestCase):
    def test_minimax_and_expectimax_choose_different_actions(self):
        tree = {
            'root': {'risky': 'a', 'safe': 'b'},
            'a': {'good': 'v1', 'bad': 'v2'},
            'b': {'first': 'v3', 'second': 'v4'},
            'v1': 12, 'v2': -4, 'v3': 3, 'v4': 3,
        }
        counts = []
        state = TreeState(tree, 'root', 2, counts)
        self.assertEqual(MinimaxAgent(depth='1').getAction(state), 'safe')
        self.assertEqual(len(counts), 6)
        counts[:] = []
        self.assertEqual(ExpectimaxAgent(depth='1').getAction(state), 'risky')
        self.assertEqual(len(counts), 6)

    def test_multiple_ghosts(self):
        tree = {
            'root': {'left': 'l', 'right': 'r'},
            'l': {'a': 'la', 'b': 'lb'},
            'r': {'a': 'ra', 'b': 'rb'},
            'la': {'x': 'a1', 'y': 'a2'},
            'lb': {'x': 'a3', 'y': 'a4'},
            'ra': {'x': 'b1', 'y': 'b2'},
            'rb': {'x': 'b3', 'y': 'b4'},
            'a1': 5, 'a2': 7, 'a3': 2, 'a4': 8,
            'b1': 4, 'b2': 6, 'b3': 3, 'b4': 5,
        }
        counts = []
        state = TreeState(tree, 'root', 3, counts)
        self.assertEqual(MinimaxAgent(depth='1').getAction(state), 'right')
        self.assertEqual(len(counts), 14)
        self.assertEqual([agent for _, agent, _ in counts[:5]], [0, 1, 2, 2, 1])
        counts[:] = []
        self.assertEqual(ExpectimaxAgent(depth='1').getAction(state), 'left')
        self.assertEqual(len(counts), 14)

    def test_second_pacman_turn_counts_as_new_depth(self):
        tree = {
            'root': {'left': 'l', 'right': 'r'},
            'l': {'ghost': 'l2'}, 'r': {'ghost': 'r2'},
            'l2': {'take': 'la', 'skip': 'lb'},
            'r2': {'take': 'ra', 'skip': 'rb'},
            'la': {'g': 'lv1'}, 'lb': {'g': 'lv2'},
            'ra': {'g': 'rv1'}, 'rb': {'g': 'rv2'},
            'lv1': 10, 'lv2': 8, 'rv1': 7, 'rv2': 6,
        }
        counts = []
        state = TreeState(tree, 'root', 2, counts)
        self.assertEqual(MinimaxAgent(depth='2').getAction(state), 'left')
        self.assertEqual(len(counts), 12)
        counts[:] = []
        self.assertEqual(ExpectimaxAgent(depth='2').getAction(state), 'left')
        self.assertEqual(len(counts), 12)

    def test_terminal_and_no_moves(self):
        for cls in (MinimaxAgent, ExpectimaxAgent):
            state = TreeState({'root': {}}, 'root', 2, [])
            self.assertEqual(cls(depth='1').getAction(state), Directions.STOP)
            state = TreeState({'root': {'a': 'leaf'}, 'leaf': 5}, 'root', 2, [])
            self.assertEqual(cls(depth='1').getAction(state), 'a')

    def test_actual_pacman_game_state(self):
        board = layout.getLayout('minimaxClassic')
        self.assertIsNotNone(board)
        game_state = GameState()
        game_state.initialize(board, 2)
        legal = game_state.getLegalActions(0)
        for agent_class in (MinimaxAgent, ExpectimaxAgent, BetterMinimaxAgent, BetterExpectimaxAgent):
            action = agent_class(depth='1').getAction(game_state)
            self.assertIn(action, legal)
        self.assertIsInstance(betterEvaluationFunction(game_state), (float, int))

    def test_evaluation_terminal_scores(self):
        # Terminal states use finite values (score +/- 1000) instead of +/-inf,
        # because Expectimax averages child values and inf - inf is undefined.
        class Finished:
            def __init__(self, win, score=100): self.win, self.score = win, score
            def isWin(self): return self.win
            def isLose(self): return not self.win
            def getScore(self): return self.score
        win_value = betterEvaluationFunction(Finished(True))
        lose_value = betterEvaluationFunction(Finished(False))
        self.assertEqual(win_value, 100 + 1000)
        self.assertEqual(lose_value, 100 - 1000)
        self.assertTrue(abs(win_value) != float('inf') and abs(lose_value) != float('inf'))


if __name__ == '__main__':
    unittest.main()
