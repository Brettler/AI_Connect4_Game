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
from connect4 import Agent

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth='2'):
        self.index = 1  # agent is always index 1
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
        """
        gameState.switch_turn(gameState.turn)
        # Calling to the minimax function and getting the preferred column for our move.
        # Initialized the variables ( True for the AI, False for the Player)
        _, best_action = self.minimax(gameState, self.depth, True)
        return best_action

    # Implementing the minimax algorithm
    def minimax(self, gameState, depth, maximizing_player):
        gameState.switch_turn(gameState.turn)
        # Initialized the variable
        best_action = None
        # Iterate through the actions and taking just the possible legal actions
        # Meaning taking the columns we can drop the piece (legal_actions will be list of possible columns)
        # Will make the code more readable
        legal_actions = gameState.getLegalActions(self.index)

        # Terminal state is when someone won\lost or when there is no legal actions to take
        # depth 0 we finished the calls
        # return the state ( we also need to return none cuz we are dealing with 2d structure)
        if depth == 0 or gameState.is_terminal():
            return self.evaluationFunction(gameState), None
        # If we want to maximize the player we will Initialize best score to be minus infinity s.t every score
        # we will be available will be better than the Initialized score.
        if maximizing_player:
            best_score = -math.inf
            for action in legal_actions:
                # game_state_successor will be what state we will be after taking an action
                game_state_successor = gameState.generateSuccessor(self.index, action)
                # We want only the score variable, so we need the column zero
                # False for switching turns while we're doing the recursive calls
                new_score = self.minimax(game_state_successor, depth - 1, False)[0]
                # If the new state ( aka game_state_successor) gave us new score that re better than the current score
                # we take the new score to be the current score s.t it will become the best score
                # we want max score
                if new_score > best_score:
                    best_score = new_score
                    # Best action is which column we will drop our piece
                    best_action = action

            # First colum will be the score, the second is the column we will put the piece
            return best_score, best_action

        # The same as explained in "if maximizing_player:" just the opposite
        # we want to minimize the player, so we will want the new score to be lower than the current score extra...
        else:
            best_value = math.inf
            for action in legal_actions:
                game_state_successor = gameState.generateSuccessor(self.index, action)
                new_score = self.minimax(game_state_successor, depth - 1, True)[0]
                if new_score < best_value:
                    best_value = new_score
                    best_action = action

            return best_value, best_action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 2)
    """
    def getAction(self, gameState):

        gameState.switch_turn(gameState.turn)
        # Calling to the alphabeta function and getting the preferred column for our move.
        # Initialized the variables ( True for the AI, False for the Player)
        # We need also to initialized -math.inf, math.inf for alpha, beta accordingly for this algorithm
        _, best_action = self.alphabeta(gameState, self.depth, -math.inf, math.inf, True)
        return best_action

    # Implementing the alphabeta algorithm
    def alphabeta(self, gameState, depth, alpha, beta, maximizing_player):
        gameState.switch_turn(gameState.turn)
        # Initialized the variable
        best_action = None
        # Iterate through the actions and taking just the possible legal actions
        # Meaning taking the columns we can drop the piece (legal_actions will be list of possible columns)
        # Will make the code more readable
        legal_actions = gameState.getLegalActions(self.index)

        # Terminal state is when someone won\lost or when there is no legal actions to take
        # depth 0 we finished the calls
        # return the state ( we also need to return none cuz we are dealing with 2d structure)
        if depth == 0 or gameState.is_terminal():
            return self.evaluationFunction(gameState), None

        # The same as we explain in the minmax algorithm however this time we save time by checking not develop
        # forwards in the tree if we want to maximize the player and beta <= alpha.
        # Also, we will take the maximum score every time and set it for alpha
        if maximizing_player:
            best_score = -math.inf
            for action in legal_actions:
                # game_state_successor will be what state we will be after taking an action
                game_state_successor = gameState.generateSuccessor(self.index, action)
                # We want only the score variable, so we need the column zero
                # False for switching turns while we're doing the recursive calls
                new_score = self.alphabeta(game_state_successor, depth - 1, alpha, beta, False)[0]

                if new_score > best_score:
                    best_score = new_score
                    best_action = action
                # Changes need to be done for this algorithm
                # We are checking the son of a current node s.t if beta <= alpha there
                # were a better option for the node to chose, so we don't need to keep doing the recursive calls (break)
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break

            # First colum will be the score, the second is the column we will put the piece
            return best_score, best_action

        # The same as explained in "if maximizing_player:" just the opposite
        else:
            best_score = math.inf
            for action in legal_actions:
                game_state_successor = gameState.generateSuccessor(self.index, action)
                value = self.alphabeta(game_state_successor, depth - 1, alpha, beta, True)[0]
                if value < best_score:
                    best_score = value
                    best_action = action
                # Changes need to be done for this algorithm
                beta = min(beta, best_score)
                if beta <= alpha:
                    break

            return best_score, best_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 3)
    """

    def getAction(self, gameState):
        gameState.switch_turn(gameState.turn)
        # Calling to the expectimax function and getting the preferred column for our move.
        # Initialized the variables ( True for the AI, False for the Player)
        _, best_action = self.expectimax(gameState, self.depth, True)
        return best_action

    # Implementing the expectimax algorithm
    def expectimax(self, gameState, depth, maximizing_player):

        gameState.switch_turn(gameState.turn)
        # Initialized the variable
        best_action = None
        # Iterate through the actions and taking just the possible legal actions
        # Meaning taking the columns we can drop the piece (legal_actions will be list of possible columns)
        # Will make the code more readable
        legal_actions = gameState.getLegalActions(self.index)

        # Terminal state is when someone won\lost or when there is no legal actions to take
        # depth 0 we finished the calls
        # return the state ( we also need to return none cuz we are dealing with 2d structure)
        if depth == 0 or gameState.is_terminal():
            return self.evaluationFunction(gameState), None

        if maximizing_player:
            best_score = -math.inf
            for action in legal_actions:
                # game_state_successor will be what state we will be after taking an action
                game_state_successor = gameState.generateSuccessor(self.index, action)
                # We want only the score variable, so we need the column zero
                # False for switching turns while we're doing the recursive calls
                score = self.expectimax(game_state_successor, depth - 1, False)[0]
                # If the new state ( aka game_state_successor) gave us new score that re better than the current score
                # we take the new score to be the current score s.t it will become the best score
                # The best action will be the one that gave us the best score
                if score > best_score:
                    best_score = score
                    best_action = action
            return best_score, best_action

        # The changes to make this algorithm are here
        # We now compute the best score by the average of all the sons
        # we will do the best action depending on the score.
        else:
            # Initialize a variable to save the sum of the scores of the legal actions
            sum_score = 0
            for action in legal_actions:
                game_state_successor = gameState.generateSuccessor(self.index, action)
                score = self.expectimax(game_state_successor, depth - 1, True)[0]
                sum_score += score
            # Compute the average score by dividing the sum of the scores by the number of legal actions
            avg_score = sum_score / len(legal_actions)
            best_score = avg_score
            return best_score, best_action
