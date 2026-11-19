"""Independent checks for Q1 (Reflex) and Q3 (Alpha-Beta).
These are not the official Berkeley/lecturer autograder tests.
"""
import random
import unittest

import layout
from game import Directions
from pacman import GameState
from multiAgents import AlphaBetaAgent, MinimaxAgent, ReflexAgent, ExpectimaxAgent
from test_multiagent import TreeState


class NewQuestionsTests(unittest.TestCase):
    def test_q1_reflex_returns_legal_action_and_action_scores(self):
        board = layout.getLayout('smallClassic')
        self.assertIsNotNone(board)
        game = GameState()
        game.initialize(board, 2)
        agent = ReflexAgent()
        legal = game.getLegalActions(0)
        self.assertTrue(legal)
        self.assertIn(agent.getAction(game), legal)
        for action in legal:
            self.assertIsInstance(agent.evaluationFunction(game, action), (int, float))

    def test_q1_reflex_no_moves(self):
        class Empty:
            def getLegalActions(self, idx):
                return []
        self.assertEqual(ReflexAgent().getAction(Empty()), Directions.STOP)

    def test_q3_alphabeta_matches_minimax_and_prunes(self):
        tree = {'root': {'A': 'a', 'B': 'b'},
                'a': {'x': 'a1', 'y': 'a2'},
                'b': {'x': 'b1', 'y': 'b2'},
                'a1': 5, 'a2': 6, 'b1': 2, 'b2': 100}
        visits = []
        root = TreeState(tree, 'root', 2, visits)
        self.assertEqual(MinimaxAgent(depth='1').getAction(root), 'A')
        self.assertEqual(len(visits), 6)
        visits[:] = []
        self.assertEqual(AlphaBetaAgent(depth='1').getAction(root), 'A')
        self.assertEqual(len(visits), 5, 'Should prune b->b2 after visiting b->b1')
        self.assertNotIn(('b', 1, 'y'), visits)

    def test_q3_alphabeta_does_not_prune_on_equality(self):
        tree = {'root': {'A': 'a', 'B': 'b'},
                'a': {'x': 'a1', 'y': 'a2'},
                'b': {'x': 'b1', 'y': 'b2'},
                'a1': 5, 'a2': 6, 'b1': 5, 'b2': 4}
        visits = []
        root = TreeState(tree, 'root', 2, visits)
        self.assertEqual(AlphaBetaAgent(depth='1').getAction(root), 'A')
        self.assertEqual(len(visits), 6)
        self.assertIn(('b', 1, 'y'), visits)

    def test_q3_alphabeta_game_state_multiple_ghosts(self):
        board = layout.getLayout('minimaxClassic')
        game = GameState()
        game.initialize(board, 3)
        legal = game.getLegalActions(0)
        self.assertTrue(legal)
        for depth in (1, 2):
            minimax = MinimaxAgent(depth=str(depth)).getAction(game)
            alphabeta = AlphaBetaAgent(depth=str(depth)).getAction(game)
            self.assertIn(alphabeta, legal)
            self.assertEqual(alphabeta, minimax)

    def test_q3_random_trees_against_full_minimax(self):
        rng = random.Random(8021)
        for count_agents in (2, 3, 4):
            for depth in (1, 2):
                for tree_index in range(12):
                    tree = {}
                    node_counter = [0]
                    def make_tree(level, maxlevel):
                        name = 'n{}'.format(node_counter[0])
                        node_counter[0] += 1
                        if level == maxlevel:
                            tree[name] = rng.randint(-8, 8)
                        else:
                            size = rng.choice((2, 3))
                            tree[name] = {'m{}'.format(i): make_tree(level+1, maxlevel)
                                          for i in range(size)}
                        return name
                    root_name = make_tree(0, count_agents * depth)
                    root = TreeState(tree, root_name, count_agents, [])
                    full = MinimaxAgent(depth=str(depth)).getAction(root)
                    full_count = len(root.counts)
                    root.counts[:] = []
                    pruned = AlphaBetaAgent(depth=str(depth)).getAction(root)
                    prune_count = len(root.counts)
                    self.assertEqual(full, pruned, (count_agents, depth, tree_index))
                    self.assertLessEqual(prune_count, full_count)


if __name__ == '__main__':
    unittest.main()
