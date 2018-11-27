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
import math
import random

##
# AI Homework 5
# Authors: Chris Lytle and Reeca Bardon
# Date: 11/26/18
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
        super(AIPlayer, self).__init__(inputPlayerId, "Mr. Net")
        # Variables for project 4 (Min-Max)
        self.depthLimit = 3
        self.foods = []
        self.homes = []
        self.playerIndex = None
        self.enemyHomes = []

        # Variables for Project 5
        self.inputs = {'foodCount': 0.0,
                       'myWorkerCount': 0.0,  # This includes num of workers, if they are carrying food, and distance
                       'numSoldersDistToQueen': 0.0,
                       'enQueenHealth': 0.0,
                       'enAnthillHealth': 0.0,
                       'enWorkerCount': 0.0}
        self.inputWeights = None  # Weights from the inputs to each hidden layer node
        self.hiddenNodeWeights = None  # Weights from the hidden layer to the output layer
        self.biasWeights = None  # Weight on the bias' on the hidden layer
        self.bias = 1  # Bias weight for entire net
        self.weightOnOutputBias = 0.0  # Weight on the bias going to the node before the output
        self.alpha = 0.7  # The learning Rate
        self.numOfHiddenNodes = 6  # Number of hidden nodes in this case, We played around with this number a little
        self.xInput = []  # Value of hidden nodes so we can adjust the weights later
        self.finalNodeValue = 0  # Value of final node
        self.correctCount = 0  # Number of correct predictions
        self.moves = 0  # Number of moves or states that the network was trained on
        self.prevStates = []  # List of states to train the agent on

    ##
    # initWeights
    # Description: Creates random values to initiate the values
    # of the weights on the Neural Network
    ##
    def initWeights(self):
        self.inputWeights = []
        # Create list of randomized weights
        inputCount = len(self.inputs)
        for n in range(0, inputCount):
            self.inputWeights.append([])
            for j in range(0, self.numOfHiddenNodes):
                self.inputWeights[n].append(round(random.uniform(-1.0, 1.0), 5))

        # Create hidden node weights and biasWeights
        self.hiddenNodeWeights = []
        self.biasWeights = []
        for n in range(0, self.numOfHiddenNodes):
            self.hiddenNodeWeights.append(round(random.uniform(-1.0, 1.0), 5))
            self.biasWeights.append(round(random.uniform(-1.0, 1.0), 5))
            self.xInput.append(0)

        self.weightOnOutputBias = round(random.uniform(-1.0, 1.0), 5)

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
        self.moves = 0  # Reset moves and correct count numbers
        self.correctCount = 0
        # If weights haven't been set, create it with random values
        if self.inputWeights is None:
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
        # self.moves += 1  # add to move counter
        cpy_state = currentState.fastclone()
        self.playerIndex = cpy_state.whoseTurn
        self.foods = getConstrList(currentState, None, (FOOD,))
        self.homes = getConstrList(currentState, currentState.whoseTurn, (ANTHILL, TUNNEL,))
        self.enemyHomes = getConstrList(currentState, 1 - currentState.whoseTurn, (ANTHILL, TUNNEL,))
        # Find best move method that uses recursive calls
        move = self.startBestMoveSearch(cpy_state, cpy_state.whoseTurn)
        self.prevStates.append(currentState)
        print(str(move))
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
        # Get current score of state
        currScore = self.scoreStateFromNet(state, me)  # Using new method to get score
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
            thisNode["score"] = self.scoreStateFromNet(state, me)  # Using new method to get score
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
                # Using new method to get score
                stateScores = [self.scoreStateFromNet(state, me) for state in nextStates]
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
    # ScoreStateFromNet
    # Description: Gets the score of current state based on weights that were trained on.
    #
    # Parameters:
    #   GameState state - Instance of the current state
    #   Int me - Reference to this AI player's index in game state, needed to get inputs
    #
    # Return:
    #   float val - the value of the current state based on propagate function and given weights
    ##
    def scoreStateFromNet(self, state, me):
        # Instance variables of the trained weights
        inputWieghts = [[-3.657254638284446, 0.5316707141038243, -3.37440586851363,
                         3.481189161618746, 1.911812225361641, -0.7617141134979972],
                        [-0.1419948628500165, 0.5408133459573163, 0.30732462218556644,
                         0.6282180369685747, -0.623221808237557, -0.40542541575131275],
                        [-2.9039221700126734, 2.3214477309727974, -4.1229483511642515,
                         4.385355885614808, 1.446467997275358, -0.6178269904201446],
                        [-3.002540165264651, 0.8274066531995593, -4.120012418654349,
                         3.0265894891576015, 2.591871112775095, -1.290227630952384],
                        [-0.7081027248689823, 0.4358020709106849, -0.7160712298391342,
                         -0.30748940964166055, 0.6340418446947687, -0.11067959251406172],
                        [0.2677833319025874, -0.4643719898011794, -0.9562896052878249,
                         0.7597785803211642, -0.1527834612583077, -0.36381517207034897]]

        hiddenNodeWieghts = [-1.9081998289314128, 0.9584860148444749, -2.849651219982444, 2.2680280743882144,
                             1.188555169512328, -0.6569314486414466]

        biasWeights = [-1.1252804099405211, -0.8905612507341796, -0.3507730897105387, -1.908624837158301,
                       -1.4219125700687236, -1.1401836142819237]

        biasWeightOnOutput = -0.982014739415461

        # Call ScoreState,  This is only to update the inputs to the state not to get the value.
        self.scoreState(state, me)

        inputList = list(self.inputs.values())  # make inputs into a list
        # Call propagate with the trained wights to get the value of the state
        val = self.propagate(inputList, inputWieghts, hiddenNodeWieghts, biasWeights, biasWeightOnOutput)
        return val

    ##
    # scoreState
    # Description: scores the advantage of the current state form 1.0 to -1.0,
    # higher numbers advantaging the 'me' player
    #
    # Old way to get the score of the current state
    #
    # Parameters:
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

        # tests for final win/lose states
        if myQueen is None:
            return -1
        if enemyQueen is None:
            return 1

        # TODO: Maybe delete this because enemyQueen health should never be greater then 10
        if enemyQueen.health > 10:
            antScore = antScore + (1 / enemyQueen.health)

        if foodScore == 1.0:
            return 1

        # checks health of enemy queen
        if enemyQueen is not None:
            healthScore = 1.0 - enemyQueen.health / 10.0
        else:
            return 1.0
        # health of capture anthill
        capturehealthScore = 1.0 - self.enemyHomes[0].captureHealth / 3.0
        if capturehealthScore == 1.0:
            return capturehealthScore

        # Ants in the game
        workers = getAntList(gameState, me, (WORKER,))
        fighters = getAntList(gameState, me, (R_SOLDIER,))
        enemyWorkers = getAntList(gameState, enemy, (WORKER,))

        myWorkerCount = 0
        # Keeps one worker, and makes sure it gets food
        if len(workers) < 1:
            myWorkerCount -= .2
        for worker in workers:
            (x, y) = worker.coords
            if worker.carrying:
                myWorkerCount += .01
                stepsToHomes = \
                    (approxDist(worker.coords, self.homes[0].coords), approxDist((x, y), self.homes[1].coords))
                minSteps = min(stepsToHomes)
                myWorkerCount += .01 / (1.0 + minSteps)
            else:
                stepsToFoods = (approxDist((x, y), self.foods[0].coords), approxDist((x, y), self.foods[1].coords),
                                approxDist((x, y), self.foods[2].coords), approxDist((x, y), self.foods[3].coords))
                minSteps = min(stepsToFoods)
                # antScore = antScore + .01 / (1.0 + minSteps)
                myWorkerCount += .01 / (1.0 + minSteps)

        soldersToQueen = 0.0
        enWorkerCount = 0.0
        # Only one range solider is created,
        # it first goes and kills the worker AnT and then moves towards the Anthill to kill the Queen
        for fighter in fighters:
            (x, y) = fighter.coords
            if len(fighters) <= 1:
                soldersToQueen = soldersToQueen + (.2 * len(fighters))
                soldersToQueen = soldersToQueen + .1 * fighter.health
            if len(enemyWorkers) >= 1:
                stepsToEnemyTunnel = approxDist((x, y), self.enemyHomes[1].coords)
                # antScore = antScore + .1 / (1.0 + stepsToEnemyTunnel)
                enWorkerCount = enWorkerCount + .1 / (1.0 + stepsToEnemyTunnel)
                # self.inputs['enWorkerCount'] = .1 / (1.0 + stepsToEnemyTunnel)
            elif len(enemyWorkers) <= 0:
                enWorkerCount = enWorkerCount + .1
                # antScore = antScore + .1
                stepsToEnemyQueen = approxDist((x, y), enemyQueen.coords)
                if stepsToEnemyQueen > 2:
                    soldersToQueen = soldersToQueen + .2 / (1.0 + stepsToEnemyQueen)

        # Update the inputs to the network.
        self.inputs['foodCount'] = foodScore / 4
        self.inputs['myWorkerCount'] = myWorkerCount / 4
        self.inputs['numSoldersDistToQueen'] = soldersToQueen / 4
        self.inputs['enQueenHealth'] = healthScore / 4
        self.inputs['enAnthillHealth'] = capturehealthScore / 4
        self.inputs['enWorkerCount'] = enWorkerCount / 4
        sumScore = foodScore + healthScore + capturehealthScore + myWorkerCount + soldersToQueen + enWorkerCount
        return sumScore / 4.0

    ##
    # registerWin
    #
    # This agent doens't learn
    #
    def registerWin(self, hasWon):

        # Shuffle the states to train the network on random data.
        random.shuffle(self.prevStates)
        # Call backPropogation on the shuffled list to train the weights on the network
        for state in self.prevStates:
            self.backPropogation(state, self.playerIndex)

        # Used to see the percentage of correct predictions made when training
        # If this value got over about 80% then we can say that the network is trained
        correctPercentage = self.correctCount / self.moves

        # Prints the current weights to a file so that they can be copied later and
        # hardcoded into the program to ge the results.
        file = open("weights.txt", "w")
        file.write("Input Weights: \n")
        file.write(str(self.inputWeights))
        file.write("\nHidden Node weights\n")
        file.write(str(self.hiddenNodeWeights))
        file.write("\nBias Weights\n")
        file.write(str(self.biasWeights))
        file.write("\nweight on output bias\n")
        file.write(str(self.weightOnOutputBias))
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

    ##
    # Propagate
    # Description: Calculates the output of the network given all the weights.
    #
    # Parameters:
    #   [float] inputs - list of inputs
    #   [[float]] weights - weights on the edges from the inputs to the hidden nodes
    #   [float] hiddenWeights - weights from the hidden nodes to the output node
    #   [float] biasWeights - weights on bias' on each hidden node
    #   float finalBiasWeight - weight on the bias on the final node
    #
    # Return:
    #   float g - the final value of the network given all the weights and inputs
    ##
    def propagate(self, inputs, weights, hiddenWeights, biasWeights, finalBiasWeight):
        hiddenNodeValues = []
        bias = 1
        # Calculate value for hidden layer
        for i in range(0, len(weights[0])):
            sum = 0.0
            for j in range(0, len(inputs)):
                sum += weights[j][i] * inputs[j]
            sum += bias * biasWeights[i]
            if math.isnan(sum):  # make sure that the number is not a nan.  If so, then make 0
                sum = 0.0
            g = 1/(1+math.exp(-1*sum))
            self.xInput[i] = g
            hiddenNodeValues.append(g)

        # Calculate the value on the final node
        sum = 0.0
        for j in range(0, len(weights[0])):
            sum += hiddenWeights[j] * hiddenNodeValues[j]
        sum += bias * finalBiasWeight
        if math.isnan(sum):
            sum = 0.0
        self.finalNodeValue = sum
        g = 1 / (1 + math.exp(-1 * sum))
        return g

    ##
    # adjustWeights
    # Description: This function adjust the weights when needed so that the weights
    # cna produce a more accurate outcome.
    #
    # Parameters:
    #   [[float]] weights - the weights on the inputs to the hidden layer
    #   float error - the difference between the expected value and last calculated value
    #   [float] inputs - list of input values used on last calculation
    ##
    def adjustWeights(self, weights, error, inputs):
        deltas = []
        # Calculate delta
        for n in range(0, len(self.hiddenNodeWeights)):
            deltas.append(error*self.hiddenNodeWeights[n])

        # Adjust weights on input node to output node
        for r in range(0, len(weights)):
            for c in range(0, len(weights[0])):
                self.inputWeights[r][c] = weights[r][c] + (self.alpha * deltas[c] * inputs[r])
        # Adjust weights on bias' on the hidden nodes
        for n in range(0, self.numOfHiddenNodes):
            self.biasWeights[n] = self.biasWeights[n] + (self.alpha * deltas[n] * self.bias)
        # Adjust weights on hidden nodes to output node
        for n in range(0, self.numOfHiddenNodes):
            self.hiddenNodeWeights[n] = self.hiddenNodeWeights[n] + (self.alpha * error * self.xInput[n])
        # Adjust weights on the output bias
        self.weightOnOutputBias = self.weightOnOutputBias + (self.alpha * error * self.bias)

    ##
    # backPropogation
    # Description: This function calculates the outcome of the neural network based on the inputs and weights.  This
    # method also calls adjustWeights to correct the weights in the network.  The method adjust the weights until the
    # network in within a certain error range.  Then returns the value of the propagation.
    #
    # Parameters:
    #   GameState state - the current state to be evaluated
    #   int me - index of this player in the GameState
    #
    # Return:
    #   float actualVal - the value calculated thought the network
    ##
    def backPropogation(self, state, me):
        self.moves += 1  # Increase the move counter
        # Get the expected value of the network on the state and update the inputs.
        expectedVal = self.scoreState(state, me)
        # Get weights and create instance variables
        weights = self.inputWeights
        inputs = list(self.inputs.values())
        hiddenWeights = self.hiddenNodeWeights
        biasWeights = self.biasWeights
        weightOnOutputBias = self.weightOnOutputBias
        # Call propagate to find the calculated value on the network
        actualVal = self.propagate(inputs, weights, hiddenWeights, biasWeights, weightOnOutputBias)
        # find the error value or difference between correct and calculated values
        error = float(expectedVal-actualVal)
        # If error value between -0.03 and 0.03 then the weights are correct
        if -0.03 < error < 0.03:
            self.correctCount += 1  # Increase correct counter
        # Train the weights until the values are within the error margin
        while -0.03 > error or error > 0.03:
            self.adjustWeights(self.inputWeights, error, inputs)
            actualVal = self.propagate(inputs, self.inputWeights, self.hiddenNodeWeights, self.biasWeights,
                                       self.weightOnOutputBias)
            error = expectedVal - actualVal
        return actualVal

# Unit tests used to test different methods used in the program.  They are not proper unit tests, but print a value
# to a network that we know the answer to.
import unittest
class testMethods(unittest.TestCase):

    def getInputWeights(self):
        weights = [[-0.6, 0.8, 0.5, 0.5],
                   [-0.1, -0.2, -0.6, -0.1],
                   [-0.9, 0.2, 0.3, -0.5],
                   [-0.9, 0.5, -0.3, 0.4],
                   [0.4, 0.3, -0.3, 0.2],
                   [-0.1, -0.7, 0.8, 0.8]]
        return weights

    def getInputs(self):
        input = [0.09, 0.01, 0.0, 0.0, 0.3, 0.0]
        return input

    def getHiddenLayerWeights(self):
        hiddenLayerWeights = [0.2, -0.6, 0.8, 0.9]
        return hiddenLayerWeights

    def getBiasOnHiddenLayer(self):
        list = [0.7, 0.9, -0.8, 0.1]

        return list

    def test_propagate(self):
        gameState = GameState.getBasicState()
        gameState.inventories[0].foodCount = 11
        AI = AIPlayer(PLAYER_ONE)
        AI.xInput = [0, 0, 0, 0, 0, 0]
        val = AIPlayer.propagate(AI, self.getInputs(), self.getInputWeights(),
                                   self.getHiddenLayerWeights(), self.getBiasOnHiddenLayer(), 0.3)
        print(val)
        pass

if __name__ == '__name__':
    unittest.main()
