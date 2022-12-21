"""
Introduction to Artificial Intelligence, 89570, Bar Ilan University, ISRAEL

Student name: Liad Brettler
Student ID: 318517182

"""

# multiAgents.py
# --------------
# Attribution Information: part of the code were created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# http://ai.berkeley.edu.
# We thank them for that! :)


import random, util, math

import numpy as np

from connect4 import Agent
from ex2_connect4.gameUtil import PLAYER, AI, AI_PIECE, PLAYER_PIECE


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 1 # agent is always index 1
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class BestRandom(MultiAgentSearchAgent):

    def getAction(self, gameState):

        return gameState.pick_best_move()


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 1)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.isWin():
        Returns whether or not the game state is a winning state for the current turn player

        gameState.isLose():
        Returns whether or not the game state is a losing state for the current turn player

        gameState.is_terminal()
        Return whether or not that state is terminal

        *** YOUR CODE HERE ***

                Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        # Call the minimax function with the initial game state, the desired depth, and a boolean indicating
        # whether the current player is the maximizing player or the minimizing player

        *** YOUR CODE HERE ***
        """
        gameState.switch_turn(gameState.turn)
        _, action = self.minimax(gameState, self.depth, True)
        return action

    def minimax(self, gameState, depth, maximizing_player):
        """
        The minimax function that implements the minimax algorithm.
        """
        gameState.switch_turn(gameState.turn)
        # If the game state is terminal or the depth has been reached, return the value of the terminal state
        if gameState.is_terminal() or depth == 0:
            return self.evaluationFunction(gameState), None

        best_action = None
        if maximizing_player:
            # we need to inizailed the varaible before using it
            value = np.NINF
            for action in gameState.getLegalActions(self.index):
                successor_game_state = gameState.generateSuccessor(self.index, action)
                # we want only the score, so we need the column zero
                new_score = self.minimax(successor_game_state, depth - 1, False)[0]
                if new_score > value:
                    value = new_score
                    # best action is which column we will drop our piece
                    best_action = action
                # first colum will be the score, the second is the column we will put the piece
            return value, best_action

        else: # we are minimzing player
            value = np.inf
            for action in gameState.getLegalActions(self.index):
                successor_game_state = gameState.generateSuccessor(self.index, action)
                new_score = self.minimax(successor_game_state, depth - 1, True)[0]
                if new_score < value:
                    value = new_score
                    best_action = action
            return value, best_action

        """
        # If the game state is terminal or the depth has been reached, return the value of the terminal state
        if gameState.is_terminal() or depth == 0:
            return self.evaluationFunction(gameState), None
        # Initialize variables to store the best value and action for the current player
        bestValue = np.NINF if maximizingPlayer else np.inf
        best_action = None

        # Iterate through the legal actions for the current player
        for action in gameState.getLegalActions(self.index):
            # Generate the successor game state and get its value
            # gameState.switch_turn(AI_PIECE)
            successorGameState = gameState.generateSuccessor(self.index, action)
            # gameState.switch_turn(PLAYER_PIECE)
            value = self.minimax(successorGameState, depth - 1, not maximizingPlayer)[0]

            # Update the best value and action if necessary
            if maximizingPlayer:
                if value > bestValue:
                    bestValue = value
                    best_action = action
            else:
                if value < bestValue:
                    bestValue = value
                    best_action = action

        # Return the best value and action for the current player
        return bestValue, best_action
        """


class AlphaBetaAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        """
            Your minimax agent with alpha-beta pruning (question 2)
        """
        "*** YOUR CODE HERE ***"
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.
        """
        # Call the alphabeta function with the initial game state, the desired depth, and the initial alpha and beta values
        gameState.switch_turn(gameState.turn)
        _, action = self.alphabeta(gameState, self.depth, np.NINF, np.inf, True)
        return action

    def alphabeta(self, gameState, depth, alpha, beta, maximizing_player):
        """
        The alphabeta function that implements the alpha-beta search algorithm.
        """
        # If the game state is terminal or the depth has been reached, return the value of the terminal state
        gameState.switch_turn(gameState.turn)
        if gameState.is_terminal() or depth == 0:
            return self.evaluationFunction(gameState), None
        """
        # Initialize variables to store the best value and action for the current player
        best_value = np.NINF if maximizing_player else np.inf
        best_action = None
        

        # Iterate through the legal actions for the current player
        for action in gameState.getLegalActions(self.index):
            # Generate the successor game state and get its value
            successorGameState = gameState.generateSuccessor(self.index, action)
            value = self.alphabeta(successorGameState, depth - 1, alpha, beta, not maximizing_player)[0]

            # Update the best value and action if necessary
            if maximizing_player:
                if value > best_value:
                    best_value = value
                    best_action = action
                alpha = max(alpha, best_value)
                if beta <= alpha:
                    break
            else:
                if value < best_value:
                    best_value = value
                    best_action = action
                beta = min(beta, best_value)
                if beta <= alpha:
                    break

        # Return the best value and action for the current player
        return best_value, best_action
"""

        # Initialize variables to store the best value and action for the current player
        best_action = None


        if maximizing_player:
            best_value = np.NINF
            # Iterate through the legal actions for the current player
            for action in gameState.getLegalActions(self.index):
                # Generate the successor game state and get its value
                successorGameState = gameState.generateSuccessor(self.index, action)
                value = self.alphabeta(successorGameState, depth - 1, alpha, beta, not maximizing_player)[0]
                if value > best_value:
                    best_value = value
                    best_action = action
                alpha = max(alpha, best_value)
                if beta <= alpha:
                    break
                    # Return the best value and action for the current player
            return best_value, best_action


        else:
            best_value = np.inf
            # Iterate through the legal actions for the current player
            for action in gameState.getLegalActions(self.index):
                # Generate the successor game state and get its value
                successorGameState = gameState.generateSuccessor(self.index, action)
                value = self.alphabeta(successorGameState, depth - 1, alpha, beta, not maximizing_player)[0]
                if value < best_value:
                    best_value = value
                    best_action = action
                beta = min(beta, best_value)
                if beta <= alpha:
                    break
                    # Return the best value and action for the current player
            return best_value, best_action















class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # Call the expectimax function with the initial game state, the desired depth, and a boolean indicating
        # whether the current player is the maximizing player or the minimizing player
        gameState.switch_turn(gameState.turn)
        _, action = self.expectimax(gameState, self.depth, True)
        return action


    def expectimax(self, gameState, depth, maximizingPlayer):
        """
        The expectimax function that implements the expectimax search algorithm.
        """
        gameState.switch_turn(gameState.turn)
        best_action = None
        # If the game state is terminal or the depth has been reached, return the value of the terminal state
        if gameState.is_terminal() or depth == 0:
            return self.evaluationFunction(gameState), None

        # Initialize variables to store the best value and action for the current player
        #best_value =np.NINF if maximizingPlayer else np.inf


        # Get the legal actions for the current player
        #legalActions = gameState.getLegalActions(self.index)

        # If the current player is the maximizing player, iterate through the legal actions and update the best value and action if necessary
        if maximizingPlayer:
            best_value = np.NINF
            for action in gameState.getLegalActions(self.index):
                # Generate the successor game state and get its value
                successorGameState = gameState.generateSuccessor(self.index, action)
                value = self.expectimax(successorGameState, depth - 1, False)[0]
                if value > best_value:
                    best_value = value
                    best_action = action
            return best_value, best_action

        # If the current player is the minimizing player, compute the expected value of the legal actions
        else:
            # Initialize a variable to store the sum of the values of the legal actions
            value_sum = 0
            # Iterate through the legal actions and add their values to the value sum
            for action in gameState.getLegalActions(self.index):
                # Generate the successor game state and get its value
                successorGameState = gameState.generateSuccessor(self.index, action)
                value = self.expectimax(successorGameState, depth - 1, True)[0]
                value_sum += value
            # Compute the expected value by dividing the value sum by the number of legal actions
            expectedValue = value_sum / len(gameState.getLegalActions(self.index))
            best_value = expectedValue
            # Return the best value and action for the current player
            return best_value, best_action
