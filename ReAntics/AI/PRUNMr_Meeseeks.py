import random
import sys
import time

sys.path.append("..")  # so other modules can be found in parent dir
from Player import *
from Constants import *
from Construction import CONSTR_STATS
from Ant import UNIT_STATS
from Move import Move
from GameState import *
from AIPlayerUtils import *
import numpy as np

##
# AI Homework 5
# Authors: Chris Lytle and Reeca Bardon
# Date: 10/8/18
##


##
# AIPlayer
# Description: The responsbility of this class is to interact with the game by
# deciding a valid move based on a given game state. This class has methods that
# will be implemented by students in Dr. Nuxoll's AI course.
#
# Variables:
#   playerId - The id of the player.
##
class AIPlayer(Player):

    # __init__
    # Description: Creates a new Player
    #
    # Parameters:
    #   inputPlayerId - The id to give the new player (int)
    #   cpy           - whether the player is a copy (when playing itself)
    ##
    def __init__(self, inputPlayerId):
        super(AIPlayer, self).__init__(inputPlayerId, "Mr. Net")  # TODO: What should our name be?
        self.depthLimit = 3
        self.foods = []
        self.homes = []
        self.playerIndex = None
        self.enemyHomes = []
        self.inputs = {'bias': 0.0,
                       'foodCount': 0.0,
                       'myWorkerCount': 0.0,  # This includes num of workers, if they are carrying food, and distance
                       'numSoldersDistToQueen': 0.0,
                       'enQueenHealth': 0.0,
                       'enAnthillHealth': 0.0,
                       'enWorkerCount': 0.0}
        self.weights = None
        self.bias = 1
        self.alpha = 0.5
        self.states = []

    def initWeights(self):
        self.weights = []
        # Create list of randomized weights
        inputCount = len(self.inputs) - 1
        for n in range(0, inputCount):
            self.weights.append([])
            for j in range(0, inputCount):
                self.weights[n].append(random.uniform(-1.0, 1.0))
        # TODO: Should we round these values to a number of significant digits?
        print("===== Initial Weights =====")
        print(self.weights)

    ##
    # getPlacement
    #
    # Description: called during setup phase for each Construction that
    #   must be placed by the player.  These items are: 1 Anthill on
    #   the player's side; 1 tunnel on player's side; 9 grass on the
    #   player's side; and 2 food on the enemy's side.
    #
    # Parameters:
    #   construction - the Construction to be placed.
    #   currentState - the state of the game at this point in time.
    #
    # Return: The coordinates of where the construction is to be placed
    ##
    def getPlacement(self, currentState):
        # If weights haven't been set, create it with random values
        if self.weights is None:
            self.initWeights()
        # implemented by students to return their next move
        if currentState.phase == SETUP_PHASE_1:  # stuff on my side
            num_to_place = 11
            moves = []
            for i in range(0, num_to_place):
                move = None
                while move is None:
                    # Choose any x location
                    x = random.randint(0, 9)
                    # Choose any y location on your side of the board
                    y = random.randint(0, 3)
                    # Set the move if this space is empty
                    if currentState.board[x][y].constr is None and (x, y) not in moves:
                        move = (x, y)
                        # Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        elif currentState.phase == SETUP_PHASE_2:  # stuff on foe's side
            num_to_place = 2
            moves = []
            for i in range(0, num_to_place):
                move = None
                while move is None:
                    # Choose any x location
                    x = random.randint(0, 9)
                    # Choose any y location on enemy side of the board
                    y = random.randint(6, 9)
                    # Set the move if this space is empty
                    if currentState.board[x][y].constr is None and (x, y) not in moves:
                        move = (x, y)
                        # Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr is True
                moves.append(move)
            return moves
        else:
            return [(0, 0)]

    ##
    # getMove
    # Description: Gets the next move from the Player.
    #
    # Parameters:
    #   currentState - The state of the current game waiting for the player's move (GameState)
    #
    # Return: The Move to be made
    ##
    def getMove(self, currentState):
        # set global variable playerIndex
        cpy_state = currentState.fastclone()
        self.playerIndex = cpy_state.whoseTurn
        self.foods = getConstrList(currentState, None, (FOOD,))
        self.homes = getConstrList(currentState, currentState.whoseTurn, (ANTHILL, TUNNEL,))
        self.enemyHomes = getConstrList(currentState, 1 - currentState.whoseTurn, (ANTHILL, TUNNEL,))
        # Find best move method that uses recursive calls
        move = self.startBestMoveSearch(cpy_state, cpy_state.whoseTurn)
        print(self.inputs)
        return move

    ##
    # getAttack
    # Description: Gets the attack to be made from the Player
    #
    # Parameters:
    #   currentState - A clone of the current state (GameState)
    #   attackingAnt - The ant currently making the attack (Ant)
    #   enemyLocation - The Locations of the Enemies that can be attacked (Location[])
    ##
    def getAttack(self, currentState, attackingAnt, enemyLocations):
        # Attack a random enemy.
        return enemyLocations[random.randint(0, len(enemyLocations) - 1)]

    ##
    # startBestMoveSearch
    # Description: Finds the best move by looking at a search tree
    # and selecting the best node witch contains the a move
    #
    # Parameters:
    #   state - A clone of the current state (GameState)
    #   me - reference to who's turn it is
    ##
    def startBestMoveSearch(self, state, me):
        currScore = self.scoreState(state, me)  # Get current score of state
        moves = listAllLegalMoves(state)  #
        thisNode = {"move": None, "state": state, "score": None, "parentNode": None,
                     "alpha": -1000, "beta": 1000}  # Create node by creating dictionary
        nextStates = [self.getNextStateAdversarial(state, move) for move in moves]
        nextNodes = []
        # Start recursion method by calling on all possible moves,  Start at 1 since this is depth 0
        for i in range(0, len(moves)):
            nextNodes.append(
                self.GetBestMove(moves[i], nextStates[i], 1, me, thisNode, thisNode["alpha"], thisNode["beta"]))
        selectMove = Move(END, None, None)
        for node in nextNodes:  # Find best move, if none better then current just end turn
            if node["score"] >= currScore:
                selectMove = node["move"]
                currScore = node["score"]
        return selectMove

    ##
    # GetBestMove
    # Description: Finds the best move by looking at a search tree
    # and selecting the best node witch contains the a move
    #
    # Parameters:
    #   move - a move from a list of legal moves
    #   state - A clone of the current state (GameState)
    #   depth - the current depth of the tree
    #   me - reference to who's turn it is
    #   parent - reference to parent node
    ##

    # TODO change name of this function and delete getBestMove
    def GetBestMove(self, move, state, depth, me, parent, alpha, beta):
        # Create a new node
        thisNode = {"move": move, "state": state, "score": None, "parentNode": parent,
                    "alpha": alpha, "beta": beta}
        # If depth limit reach, then just return this node
        if depth == self.depthLimit:
            thisNode["score"] = self.scoreState(state, me)
            return thisNode
        else:
            moves = listAllLegalMoves(state)  # Get all legal moves
            if len(moves) == 0:
                print("Should never happen")
                return thisNode  # This should never happen
            # Min max evaluations
            # If this AI's turn, then it is a max evaluation (alpha)
            # Otherwise it is a min evaluation (beta)
            # Using alpha beta, if beta <= alpha then don't look at any more branches.
            nextStates = [self.getNextStateAdversarial(state, move) for move in moves]
            # If at depth limit, sort nodes from lowest to highest.
            # Increase efficiency in our alpha beta pruning
            if depth == self.depthLimit - 1:
                stateScores = [self.scoreState(state, me) for state in nextStates]
                lowToHighIndices = sorted(range(len(stateScores)), key=lambda k: stateScores[k])
            else:
                # If depth limit not reached then just go through every node
                lowToHighIndices = [i for i in range(0, len(moves))]
            nodeLength = len(lowToHighIndices)
            if thisNode["state"].whoseTurn == self.playerIndex:
                best = -1000
                for i in reversed(lowToHighIndices):  # starts at the max values
                    next_node = self.GetBestMove(moves[i], nextStates[i], depth + 1, me, thisNode, alpha, beta)
                    best = max(best, next_node["score"])
                    alpha = max(alpha, best)
                    if beta <= alpha:  # No need to look at other branches
                        break
                thisNode["score"] = alpha
                thisNode["alpha"] = alpha

                return thisNode
            # Min Evaluations
            else:
                worst = 1000
                for i in range(0, int(nodeLength / 2)):
                    next_node = self.GetBestMove(moves[i], nextStates[i], depth + 1, me, thisNode, alpha, beta)
                    worst = min(worst, next_node["score"])
                    beta = min(beta, worst)
                    if beta <= alpha:  # No need to look at other branches
                        break
                thisNode["score"] = beta
                thisNode["beta"] = beta
                return thisNode

    ##
    #scoreState
    #Description: scores the advantage of the current state form 1.0 to -1.0,
    #higher numbers advantaging the 'me' player
    #
    #Parameters:
    #   gameState - gameState to analyze
    ##
    def scoreState(self, gameState, me):
        # Set instance variable for ant score
        antScore = 0.0
        enemy = 1 - me
        myInv = gameState.inventories[me]
        enemyInv = gameState.inventories[enemy]
        myQueen = myInv.getQueen()
        enemyQueen = enemyInv.getQueen()
        playerFoodGross = myInv.foodCount
        foodScore = playerFoodGross / 11.0
        self.inputs['foodCount'] = foodScore

        # tests for final win/lose states
        if myQueen is None:
            return -1
        if enemyQueen is None:
            return 1

        # TODO: Maybe delete this because enemyQueen health should never be greater then 10
        if enemyQueen.health > 10:
            antScore = antScore + (1 / enemyQueen.health)
            self.inputs['enQueenHealth'] = antScore  # Update input list

        if foodScore == 1.0:
            return playerFoodGross

        # checks health of enemy queen
        if enemyQueen is not None:
            healthScore = 1.0 - enemyQueen.health / 10.0
            self.inputs['enQueenHealth'] = healthScore
        else:
            return 1.0
        # health of capture anthill
        capturehealthScore = 1.0 - self.enemyHomes[0].captureHealth / 3.0
        self.inputs['enAnthillHealth'] = capturehealthScore
        if capturehealthScore == 1.0:
            return capturehealthScore

        # Ants in the game
        workers = getAntList(gameState, me, (WORKER,))
        fighters = getAntList(gameState, me, (R_SOLDIER,))
        enemyWorkers = getAntList(gameState, enemy, (WORKER,))

        myWorkerCount = 0
        # Keeps one worker, and makes sure it gets food
        if len(workers) < 1:
            antScore = antScore - .2
            myWorkerCount += .2
        for worker in workers:
            (x, y) = worker.coords
            if worker.carrying:
                antScore = antScore + .01
                myWorkerCount += .2
                stepsToHomes = \
                    (approxDist(worker.coords, self.homes[0].coords), approxDist((x, y), self.homes[1].coords))
                minSteps = min(stepsToHomes)
                antScore = antScore + .01 / (1.0 + minSteps)
                myWorkerCount += .01 / (1.0 + minSteps)
            else:
                stepsToFoods = (approxDist((x, y), self.foods[0].coords), approxDist((x, y), self.foods[1].coords),
                                approxDist((x, y), self.foods[2].coords), approxDist((x, y), self.foods[3].coords))
                minSteps = min(stepsToFoods)
                antScore = antScore + .01 / (1.0 + minSteps)
                myWorkerCount += .01 / (1.0 + minSteps)
        self.inputs['myWorkerCount'] = myWorkerCount

        # Only one range solider is created,
        # it first goes and kills the worker AnT and then moves towards the Anthill to kill the Queen
        for fighter in fighters:
            (x, y) = fighter.coords
            if len(fighters) <= 1:
                antScore = antScore + (.2 * len(fighters))
                antScore = antScore + .1 * fighter.health
            if len(enemyWorkers) >= 1:
                stepsToEnemyTunnel = approxDist((x, y), self.enemyHomes[1].coords)
                antScore = antScore + .1 / (1.0 + stepsToEnemyTunnel)
                self.inputs['enWorkerCount'] = .1 / (1.0 + stepsToEnemyTunnel)

            elif len(enemyWorkers) <= 0:
                antScore = antScore + .1
                self.inputs['enWorkerCount'] = .1
                stepsToEnemyQueen = approxDist((x, y), enemyQueen.coords)
                if stepsToEnemyQueen > 2:
                    antScore = antScore + .2 / (1.0 + stepsToEnemyQueen)
                    self.inputs['enWorkerCount'] += .2 / (1.0 + stepsToEnemyQueen)

        sumScore = foodScore + healthScore + capturehealthScore + antScore
        return sumScore / 4.0


    ##
    # registerWin
    #
    # This agent doens't learn
    #
    def registerWin(self, hasWon):
        # method templaste, not implemented
        pass

    ##
    # This is an updated method that was written by Jacob Apenes and sent to everyone by Nuxoll
    # We put this in our agent's class to make sure that the updated version is used instead of the one in the
    # AIPlayerUtils class.
    #
    # Thank You Jacob!!!!!
    #
    # getNextStateAdversarial
    #
    # Description: This is the same as getNextState (above) except that it properly
    # updates the hasMoved property on ants and the END move is processed correctly.
    #
    # Parameters:
    #   currentState - A clone of the current state (GameState)
    #   move - The move that the agent would take (Move)
    #
    # Return: A clone of what the state would look like if the move was made
    ##
    def getNextStateAdversarial(self, currentState, move):
        # variables I will need
        nextState = getNextState(currentState, move)
        myInv = getCurrPlayerInventory(nextState)
        myAnts = myInv.ants

        # If an ant is moved update their coordinates and has moved
        if move.moveType == MOVE_ANT:
            # startingCoord = move.coordList[0]
            startingCoord = move.coordList[len(move.coordList) - 1]
            for ant in myAnts:
                if ant.coords == startingCoord:
                    ant.hasMoved = True
        elif move.moveType == END:
            for ant in myAnts:
                ant.hasMoved = False
            nextState.whoseTurn = 1 - currentState.whoseTurn
        return nextState

    def propagate(self, inputs, weights):
        for i in inputs:
            sum += inputs[i]*weights[i]

        if sum > 1:
            return 1
        elif sum < -1:
            return -1
        else:
            return sum


    def adjustWeights (self, initWeights, learningRate, errorTerm, inputs):
        for i in inputs:
            newWeight = initWeights[i] - (learningRate*errorTerm*inputs[i])
        return newWeight

    def backPropogation(self, weights, inputs, output, g, state):
        actualVal = self.propogate(inputs, weights)
        expectedVal = self.scoreState(self, state, state.whoseTurn)
        error = expectedVal-actualVal
        while error > 0.03 | error < -0.03:
            errorTerm = error*g*(1-g)
            newWeights = self.adjustWeights(self, weights, learningRate, errorTerm, inputs)
            actualVal = self.propagate(self, inputs, newWeights)
            error = expectedVal - actualVal
        return actualVal

def testMeeseek():
    gameState = GameState.getBasicState()
    gameState.inventories[0].foodCount = 11
    AI = AIPlayer(PLAYER_ONE)
    winstate = AI.scoreState(gameState, 0)
    if winstate != 1:
        print('You have an error in your scoreState, not recording 11 food as victory')
    else:
        print("passed unit test")




testMeeseek()