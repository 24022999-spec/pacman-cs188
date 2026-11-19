# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """

    # Khoi tao Stack va tap cac trang thai da duyet
    stack = util.Stack()
    visited = set()

    # Lay trang thai bat dau
    start = problem.getStartState()
    stack.push((start, []))

    # Thuc hien tim kiem DFS
    while not stack.isEmpty():
        state, path = stack.pop()

        # Bo qua trang thai da duyet
        if state in visited:
            continue

        # Kiem tra trang thai dich
        if problem.isGoalState(state):
            return path

        # Danh dau trang thai da duyet
        visited.add(state)

        # Lay cac trang thai ke tiep
        for nextState, action, cost in problem.getSuccessors(state):
            if nextState not in visited:
                newPath = path + [action]
                stack.push((nextState, newPath))

    # Khong tim thay duong di
    return []

def breadthFirstSearch(problem: SearchProblem):
    """Find a minimum-number-of-actions path using graph-search BFS."""
    start = problem.getStartState()
    queue = util.Queue()
    queue.push((start, []))
    visited = {start}
    while not queue.isEmpty():
        state, path = queue.pop()
        if problem.isGoalState(state):
            return path
        for next_state, action, cost in problem.getSuccessors(state):
            if next_state not in visited:
                visited.add(next_state)
                queue.push((next_state, path + [action]))
    return []

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first.

    UCS xep hang doi uu tien theo f(n) = g(n) (khong dung heuristic),
    tuc la A* voi h(n) = 0.
    """
    start = problem.getStartState()
    frontier = util.PriorityQueue()
    frontier.push((start, [], 0), 0)
    best_g = {start: 0}

    while not frontier.isEmpty():
        state, path, g_cost = frontier.pop()
        if g_cost > best_g.get(state, float("inf")):
            continue  # ban ghi cu (da co duong re hon toi state nay)
        if problem.isGoalState(state):
            return path
        for next_state, action, step_cost in problem.getSuccessors(state):
            new_g = g_cost + step_cost
            if new_g < best_g.get(next_state, float("inf")):
                best_g[next_state] = new_g
                frontier.push((next_state, path + [action], new_g), new_g)
    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """
    A* Search:
    f(n) = g(n) + h(n)
    """
    start = problem.getStartState()

    frontier = util.PriorityQueue()
    frontier.push((start, [], 0), heuristic(start, problem))

    best_g = {start: 0}

    while not frontier.isEmpty():
        state, path, g_cost = frontier.pop()

        if g_cost > best_g.get(state, float("inf")):
            continue

        if problem.isGoalState(state):
            return path

        for next_state, action, step_cost in problem.getSuccessors(state):
            new_g = g_cost + step_cost

            if new_g < best_g.get(next_state, float("inf")):
                best_g[next_state] = new_g
                new_path = path + [action]

                # f(n) = g(n) + h(n)
                f_cost = new_g + heuristic(next_state, problem)

                frontier.push(
                    (next_state, new_path, new_g),
                    f_cost
                )

    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch