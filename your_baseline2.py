# your_baseline2.py
#
# added various features to offensiveReflexAgent so that
# 1) it tries to avoid getting stuck
# 2) it tries to avoid ghosts
# 3) it recognises usefulness of capsules
# However, it was difficult to win baseline1 comfortably
# In fact, this lost more to baseline1 in various layouts

# baselineTeam.py
# ---------------
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


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import distanceCalculator
import random
import time
import util
import sys
from game import Directions
import game
from util import nearestPoint

#################
# Team creation #
#################


def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveReflexAgent', second='DefensiveReflexAgent'):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########


class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions
    """

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        # self.foodEaten keeps track of food eaten by pacman
        self.foodEaten = 0

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = gameState.getLegalActions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(gameState, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        foodLeft = len(self.getFood(gameState).asList())
        # added stuff
        foodList = self.getFood(gameState).asList()

        if foodLeft <= 2:
            bestDist = 9999
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start, pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction

        choice = random.choice(bestActions)

        # icrements self.foodEaten or resets to 0 appropirately
        if self.getSuccessor(gameState, choice).getAgentPosition(self.index) in self.getFood(gameState).asList():
            # you bout to eat a food
            self.foodEaten += 1
            # print("foodEaten: ", self.foodEaten)
        if not gameState.getAgentState(self.index).isPacman:
            self.foodEaten = 0
            # print("foodEaten Reset to 0")

        return choice

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getFeatures(self, gameState, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)
        return features

    def getWeights(self, gameState, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return {'successorScore': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that seeks food. This is an agent
    we give you to get an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()
        features['numFoodLeft'] = len(foodList)
        features['succScore'] = self.getScore(successor)
        # some added values used for Features and Weights
        myPos = successor.getAgentState(self.index).getPosition()
        # print("myPos", myPos)
        succCapsule = successor.getCapsules()  # 동작하나?
        # enemyAgents are divided to defenders and invaders
        enemyAgents = [gameState.getAgentState(
            e) for e in self.getOpponents(successor)]
        invaders = [
            i for i in enemyAgents if not i.isPacman and i.getPosition() != None]
        defenders = [
            d for d in enemyAgents if d.isPacman and d.getPosition() != None]
        # print("enemyAgents: ", enemyAgents)

        # checks if the agent is stuck
        if action == Directions.STOP:
            features["stuck"] = 1.0

        # Compute distance to the nearest food
        if len(foodList) > 0:  # This should always be True,  but better safe than sorry
            minDistance = min([self.getMazeDistance(myPos, food)
                               for food in foodList])
            features['distanceToFood'] = minDistance

        # distance to starting point: used to return
        distToStart = self.getMazeDistance(myPos, self.start)
        features['distToStart'] = distToStart

        # distance to nearest enemy defender
        # if agent is pacman and defender is present
        # check closest defender position, else default is 0
        closestDefender = float("inf")
        if len(defenders) > 0 and gameState.getAgentState(self.index).isPacman:
            for defender in defenders:
                pos = defender.getPosition()
                # print("pos", pos)
                if defender.scaredTimer == 0:
                    defenderDist = self.getMazeDistance(myPos, pos)
                    if defenderDist < closestDefender:
                        closestDefender = defenderDist
        else:
            closestDefender = 0
        features['closestDefender'] = closestDefender

        # distance to nearest Capsule, if there is any
        if len(self.getCapsules(gameState)) > 0 and gameState.getAgentState(self.index).isPacman:
            capsuleList = self.getCapsules(gameState)
            capDistList = [self.getMazeDistance(
                myPos, capsule) for capsule in capsuleList]
            features['closestCapDist'] = min(capDistList)
        else:
            features['closestCapDist'] = 0

        return features

    def getWeights(self, gameState, action):
        # if foodEaten >= 1, the agent tries to move closer to starting point
        if self.foodEaten >= 1:
            return {'succScore': 0, 'numFoodLeft': -100, 'distanceToFood': -1, 'distToStart': -50, 'stuck': -25,
                    'closestDefender': -1000, 'closestCapDist': -1.3}
        else:
            return {'succScore': 0, 'numFoodLeft': -100, 'distanceToFood': -1, 'distToStart': 0, 'stuck': 0,
                    'closestDefender': -1.5, 'closestCapDist': -0.7}


""" ORIGINAL weights
    def getWeights(self, gameState, action):
        # if foodEaten >= 1, the agent tries to move closer to starting point
        if self.foodEaten >= 1:
            return {'succScore': 0, 'numFoodLeft': -10, 'distanceToFood': -1, 'distToStart': -15, 'stuck': -25,
                    'closestDefender': -40, 'closestCapDist': -1.3}
        else:
            return {'succScore': 0, 'numFoodLeft': -10, 'distanceToFood': -1, 'distToStart': 0, 'stuck': 0,
                    'closestDefender': -30, 'closestCapDist': -0.7}
                    """


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman:
            features['onDefense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i)
                   for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition()
                    != None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(
                myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(
            self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features

    def getWeights(self, gameState, action):
        return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}
