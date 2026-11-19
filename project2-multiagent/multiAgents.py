# multiAgents.py
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
from game import Agent, Directions, Actions
from util import manhattanDistance
import util
import random


class ReflexAgent(Agent):
    """Q1: Pick the best immediate successor using an action evaluation.

    Unlike the tree-search agents, this agent scores only one Pacman move
    at a time. It re-evaluates the game after every real move.
    """

    def registerInitialState(self, gameState):
        self._visits = {}

    def getAction(self, gameState):
        legal = gameState.getLegalActions(0)
        if not legal:
            return Directions.STOP

        if not hasattr(self, '_visits'):
            self._visits = {}
        current = gameState.getPacmanPosition()
        self._visits[current] = self._visits.get(current, 0) + 1
        scores = []
        for action in legal:
            score = self.evaluationFunction(gameState, action)
            new_position = Actions.getSuccessor(current, action)
            score -= 2.0 * self._visits.get(new_position, 0)
            scores.append(score)
        best_score = max(scores)
        best_actions = [a for a, score in zip(legal, scores) if score == best_score]
        return random.choice(best_actions)

    def evaluationFunction(self, currentGameState, action):
        successor = currentGameState.generatePacmanSuccessor(action)
        if successor.isWin():
            return float('inf')
        if successor.isLose():
            return float('-inf')

        position = successor.getPacmanPosition()
        food = successor.getFood().asList()
        capsules = successor.getCapsules()
        ghosts = successor.getGhostStates()
        result = float(successor.getScore())

        # Reward progress towards the closest remaining food, and actually
        # removing dots and capsules from the board.
        result -= 3.5 * len(food)
        if food:
            nearest_food = min(manhattanDistance(position, dot) for dot in food)
            result += 10.0 / (nearest_food + 1.0)
            result -= 1.2 * nearest_food
        result -= 12.0 * len(capsules)

        for ghost in ghosts:
            ghost_pos = ghost.getPosition()
            if ghost_pos is None:
                continue
            distance = manhattanDistance(position, ghost_pos)
            if ghost.scaredTimer > 0:
                if distance < ghost.scaredTimer:
                    result += 20.0 / (distance + 1.0)
            else:
                if distance <= 1:
                    result -= 400.0
                elif distance < 3:
                    result -= 30.0 / distance
                elif distance < 5:
                    result -= 6.0 / distance

        if action == Directions.STOP:
            result -= 7.0
        return result


def scoreEvaluationFunction(currentGameState):
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        actions = gameState.getLegalActions(0)
        if not actions:
            return Directions.STOP

        agents = gameState.getNumAgents()

        def value(state, agent, depth):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            moves = state.getLegalActions(agent)
            if not moves:
                return self.evaluationFunction(state)

            next_agent = (agent + 1) % agents
            next_depth = depth + 1 if next_agent == 0 else depth

            if agent == 0:
                result = float('-inf')
                for move in moves: 
                    child = state.generateSuccessor(agent, move)
                    result = max(result, value(child, next_agent, next_depth))
                return result

            result = float('inf')
            for move in moves:
                child = state.generateSuccessor(agent, move)
                result = min(result, value(child, next_agent, next_depth))
            return result

        next_agent = 1 % agents
        next_depth = 1 if next_agent == 0 else 0
        best_action = actions[0]
        best_value = float('-inf')

        for action in actions:
            child = gameState.generateSuccessor(0, action)
            result = value(child, next_agent, next_depth)
            if result > best_value:
                best_value = result
                best_action = action
        return best_action



class AlphaBetaAgent(MultiAgentSearchAgent):
    """Q3: Minimax with alpha-beta pruning across any number of ghosts.

    Important for the CS188 grader: expand successors in getLegalActions
    order, and prune on strict inequalities only (not on equality).
    """

    def getAction(self, gameState):
        actions = gameState.getLegalActions(0)
        if not actions:
            return Directions.STOP

        agents = gameState.getNumAgents()
        negative_inf = float('-inf')
        positive_inf = float('inf')

        def value(state, agent, depth, alpha, beta):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            
            moves = state.getLegalActions(agent)
            if not moves:
                return self.evaluationFunction(state)

            next_agent = (agent + 1) % agents
            next_depth = depth + (1 if next_agent == 0 else 0)

            if agent == 0:
                best = negative_inf
                for move in moves:
                    child = state.generateSuccessor(agent, move)
                    best = max(best, value(child, next_agent, next_depth, alpha, beta))
                    if best > beta:  # Strict: do not prune on equality.
                        return best
                    alpha = max(alpha, best)
                return best

            best = positive_inf
            for move in moves:
                child = state.generateSuccessor(agent, move)
                best = min(best, value(child, next_agent, next_depth, alpha, beta))
                if best < alpha:  # Strict: do not prune on equality.
                    return best
                beta = min(beta, best)
            return best

        next_agent = 1 % agents
        next_depth = 1 if next_agent == 0 else 0
        best_action = actions[0]
        best_value = negative_inf
        alpha, beta = negative_inf, positive_inf

        # Share root alpha among actions; don't restart bounds per child.
        for action in actions:
            child = gameState.generateSuccessor(0, action)
            result = value(child, next_agent, next_depth, alpha, beta)
            if result > best_value:
                best_value = result
                best_action = action
            alpha = max(alpha, best_value)
        return best_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        actions = gameState.getLegalActions(0)
        if not actions:
            return Directions.STOP

        agents = gameState.getNumAgents()

        def value(state, agent, depth):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            moves = state.getLegalActions(agent)
            if not moves:
                return self.evaluationFunction(state)

            next_agent = (agent + 1) % agents
            next_depth = depth + 1 if next_agent == 0 else depth

            if agent == 0:
                best = float('-inf')
                for move in moves:
                    child = state.generateSuccessor(agent, move)
                    best = max(best, value(child, next_agent, next_depth))
                return best

            total = 0.0
            for move in moves:
                child = state.generateSuccessor(agent, move)
                total += value(child, next_agent, next_depth)
            return total / len(moves)

        next_agent = 1 % agents
        next_depth = 1 if next_agent == 0 else 0
        best_action = actions[0]
        best_value = float('-inf')

        for action in actions:
            child = gameState.generateSuccessor(0, action)
            result = value(child, next_agent, next_depth)
            if result > best_value:
                best_value = result
                best_action = action
        return best_action


def betterEvaluationFunction(currentGameState):
    if currentGameState.isWin():
        return currentGameState.getScore() + 1000
    if currentGameState.isLose():
        return currentGameState.getScore() - 1000

    position = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    ghosts = currentGameState.getGhostStates()

    result = currentGameState.getScore()
    result -= 2.0 * len(food)

    if food:
        result -= min(manhattanDistance(position, dot) for dot in food)

    for ghost in ghosts:
        distance = manhattanDistance(position, ghost.getPosition())
        if ghost.scaredTimer > 0:
            if distance <= ghost.scaredTimer:
                result += 20.0 / (distance + 1.0)
        else:
            if distance <= 1:
                result -= 500.0
            elif distance < 4:
                result -= 40.0 / distance

    return result


better = betterEvaluationFunction


# Optional helpers for running the required search agents with the Q5 function.
class BetterMinimaxAgent(MinimaxAgent):
    def __init__(self, depth='2'):
        MinimaxAgent.__init__(self, evalFn='betterEvaluationFunction', depth=depth)


class BetterExpectimaxAgent(ExpectimaxAgent):
    def __init__(self, depth='2'):
        ExpectimaxAgent.__init__(self, evalFn='betterEvaluationFunction', depth=depth)