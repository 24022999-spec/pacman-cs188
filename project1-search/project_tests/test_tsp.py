import itertools
import unittest

import layout
import pacman
import search
from game import Actions
from tspAgents import TSPAgent, TSPProblem, tspHeuristic


def new_game(layout_name):
    state = pacman.GameState()
    state.initialize(layout.getLayout(layout_name), 0)
    return state


def simulate(problem, actions):
    position, visited = problem.getStartState()
    for action in actions:
        dx, dy = Actions.directionToVector(action)
        position = int(position[0] + dx), int(position[1] + dy)
        assert not problem.walls[position[0]][position[1]]
        visited |= problem.goal_bits.get(position, 0)
    return position, visited


def exact_cost(problem):
    costs = []
    for order in itertools.permutations(problem.goals):
        current = problem.start
        length = 0
        for next_pos in order:
            length += problem.distance(current, next_pos)
            current = next_pos
        if problem.return_to_start:
            length += problem.distance(current, problem.start)
        costs.append(length)
    if not costs:
        return 0
    return min(costs)


class TSPTests(unittest.TestCase):
    def test_optimal_routes(self):
        for board in ('tinyCorners', 'mediumCorners'):
            game = new_game(board)
            walls = game.getWalls()
            goals = ((1, 1), (1, walls.height - 2),
                     (walls.width - 2, 1), (walls.width - 2, walls.height - 2))
            for closed in (False, True):
                reference = TSPProblem(game, goals, closed)
                optimum = exact_cost(reference)
                for algorithm in ('bfs', 'ucs', 'astar'):
                    problem = TSPProblem(game, goals, closed)
                    if algorithm == 'bfs':
                        path = search.breadthFirstSearch(problem)
                    elif algorithm == 'ucs':
                        path = search.uniformCostSearch(problem)
                    else:
                        path = search.aStarSearch(problem, tspHeuristic)
                    self.assertTrue(problem.isGoalState(simulate(problem, path)),
                                    (board, closed, algorithm))
                    self.assertEqual(len(path), optimum, (board, closed, algorithm))

    def test_heuristic_consistency(self):
        game = new_game('tinyCorners')
        walls = game.getWalls()
        goals = ((1, 1), (1, walls.height - 2),
                 (walls.width - 2, 1), (walls.width - 2, walls.height - 2))
        for closed in (False, True):
            problem = TSPProblem(game, goals, closed)
            frontier = [problem.getStartState()]
            visited = set()
            while frontier:
                state = frontier.pop()
                if state in visited:
                    continue
                visited.add(state)
                h = tspHeuristic(state, problem)
                self.assertGreaterEqual(h, 0)
                if problem.isGoalState(state):
                    self.assertEqual(h, 0)
                for next_state, _, cost in problem.getSuccessors(state):
                    self.assertLessEqual(h, cost + tspHeuristic(next_state, problem))
                    if next_state not in visited:
                        frontier.append(next_state)

    def test_custom_goals_and_invalid_goals(self):
        game = new_game('tinyCorners')
        goal = (1, 1)
        problem = TSPProblem(game, [goal])
        actions = search.aStarSearch(problem, tspHeuristic)
        self.assertTrue(problem.isGoalState(simulate(problem, actions)))
        self.assertEqual(len(actions), problem.distance(game.getPacmanPosition(), goal))
        with self.assertRaises(ValueError):
            TSPProblem(game, [(0, 0)])

    def test_agent_food_and_custom_parsing(self):
        game = new_game('tinyCorners')
        agent = TSPAgent(fn='astar', goals='food', mode='open')
        self.assertEqual(len(agent._get_goals(game)), 4)
        agent = TSPAgent(goals='1:1|6:1|6:6')
        self.assertEqual(agent._get_goals(game), [(1, 1), (6, 1), (6, 6)])


if __name__ == '__main__':
    unittest.main()
