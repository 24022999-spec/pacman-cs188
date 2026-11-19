# tspAgents.py
# Educational Pacman extension based on the UC Berkeley Pacman project.
# The original Pacman files retain their existing Berkeley license notices.

from collections import deque
from functools import lru_cache
from time import perf_counter

from game import Agent, Actions, Directions
import search


MOVES = (Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST)


class TSPProblem(search.SearchProblem):
    def __init__(self, gameState, goals=None, return_to_start=False):
        self.walls = gameState.getWalls()
        self.start = gameState.getPacmanPosition()
        self.goals = tuple(dict.fromkeys(goals or ()))
        self.return_to_start = return_to_start
        self._expanded = 0
        self.goal_bits = {point: 1 << i for i, point in enumerate(self.goals)}
        self.all_visited = (1 << len(self.goals)) - 1
        self.start_mask = self.goal_bits.get(self.start, 0)
        self.distances = {}

        for point in self.goals:
            x, y = point
            if not (0 <= x < self.walls.width and 0 <= y < self.walls.height):
                raise ValueError(f'Mục tiêu {point} nằm ngoài bản đồ')
            if self.walls[x][y]:
                raise ValueError(f'Mục tiêu {point} nằm trên tường')

    def getStartState(self):
        return self.start, self.start_mask

    def isGoalState(self, state):
        position, visited = state
        if visited != self.all_visited:
            return False
        return not self.return_to_start or position == self.start

    def getSuccessors(self, state):
        position, visited = state
        x, y = position
        successors = []
        self._expanded += 1

        for direction in MOVES:
            dx, dy = Actions.directionToVector(direction)
            next_pos = int(x + dx), int(y + dy)
            if not self.walls[next_pos[0]][next_pos[1]]:
                next_mask = visited | self.goal_bits.get(next_pos, 0)
                successors.append(((next_pos, next_mask), direction, 1))

        return successors

    def getCostOfActions(self, actions):
        if actions is None:
            return 999999
        x, y = self.start
        for direction in actions:
            dx, dy = Actions.directionToVector(direction)
            x, y = int(x + dx), int(y + dy)
            if not (0 <= x < self.walls.width and 0 <= y < self.walls.height):
                return 999999
            if self.walls[x][y]:
                return 999999
        return len(actions)

    def distance(self, a, b):
        if a == b:
            return 0
        if b not in self.distances:
            steps = {b: 0}
            queue = deque([b])
            while queue:
                x, y = queue.popleft()
                for direction in MOVES:
                    dx, dy = Actions.directionToVector(direction)
                    point = int(x + dx), int(y + dy)
                    if not self.walls[point[0]][point[1]] and point not in steps:
                        steps[point] = steps[(x, y)] + 1
                        queue.append(point)
            self.distances[b] = steps
        return self.distances[b].get(a, float('inf'))

    @lru_cache(maxsize=None)
    def mst_cost(self, mask):
        remaining = [goal for i, goal in enumerate(self.goals) if not mask & (1 << i)]
        if len(remaining) < 2:
            return 0
        connected = [remaining.pop()]
        cost = 0
        while remaining:
            distance, point = min(
                (self.distance(a, b), b)
                for a in connected for b in remaining
            )
            cost += distance
            connected.append(point)
            remaining.remove(point)
        return cost


def tspHeuristic(state, problem):
    position, visited = state
    remaining = [goal for i, goal in enumerate(problem.goals) if not visited & (1 << i)]

    if not remaining:
        return problem.distance(position, problem.start) if problem.return_to_start else 0

    estimate = min(problem.distance(position, goal) for goal in remaining)
    estimate += problem.mst_cost(visited)
    if problem.return_to_start:
        estimate += min(problem.distance(problem.start, goal) for goal in remaining)
    return estimate


class TSPAgent(Agent):
    def __init__(self, fn='astar', goals='corners', mode='open'):
        self.fn = fn.lower()
        self.goals_setting = goals
        self.return_to_start = mode.lower() in ('closed', 'return', 'tour')
        if self.fn not in ('dfs', 'bfs', 'ucs', 'astar'):
            raise ValueError('fn phải là dfs, bfs, ucs hoặc astar')
        if mode.lower() not in ('open', 'closed', 'return', 'tour'):
            raise ValueError('mode phải là open hoặc closed')
        self.actions = []
        self.next_action = 0

    def _get_goals(self, gameState):
        setting = self.goals_setting.strip().lower()
        walls = gameState.getWalls()
        if setting == 'corners':
            x, y = walls.width - 2, walls.height - 2
            return [(1, 1), (1, y), (x, 1), (x, y)]
        if setting == 'food':
            food = gameState.getFood().asList()
            if len(food) > 14:
                raise ValueError('Chế độ food chỉ nên dùng tối đa 14 mục tiêu; hãy chọn bản đồ nhỏ hoặc danh sách tọa độ')
            return food
        try:
            return [tuple(map(int, item.strip().split(':'))) for item in setting.split('|')]
        except ValueError as exc:
            raise ValueError('Định dạng mục tiêu: goals=1:1|5:1|5:5') from exc

    def registerInitialState(self, gameState):
        goals = self._get_goals(gameState)
        problem = TSPProblem(gameState, goals, self.return_to_start)
        algorithms = {
            'dfs': search.depthFirstSearch,
            'bfs': search.breadthFirstSearch,
            'ucs': search.uniformCostSearch,
            'astar': lambda p: search.aStarSearch(p, tspHeuristic),
        }
        started = perf_counter()
        self.actions = algorithms[self.fn](problem)
        elapsed = perf_counter() - started
        self.next_action = 0

        position, mask = problem.getStartState()
        for action in self.actions:
            dx, dy = Actions.directionToVector(action)
            position = int(position[0] + dx), int(position[1] + dy)
            mask |= problem.goal_bits.get(position, 0)

        self.solved = problem.isGoalState((position, mask))
        print(f'[TSP] Algorithm: {self.fn.upper()}, goals: {len(problem.goals)}, closed: {problem.return_to_start}')
        print(f'[TSP] Steps: {len(self.actions)}, expanded: {problem._expanded}, time: {elapsed:.4f}s')
        print(f'[TSP] Goal reached: {self.solved}')

    def getAction(self, state):
        if self.next_action >= len(self.actions):
            return Directions.STOP
        action = self.actions[self.next_action]
        self.next_action += 1
        return action
