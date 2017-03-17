"""
Cody Shepherd
qlearn.py
CS 545: Machine Learning
Homework 6

INSTRUCTIONS ON RUNNING
In command-line: python qlearn.py
Or open in PyCharm and run qlearn.py

REQUIREMENTS
no external files are required for this program to run
"""

from enum import Enum
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


class Cell(Enum):
    """
    This class describes the state of a given cell or square in the grid environment where
    this AI will learn. Also included are some static methods for generating string representations
    of cells and generating random cells.

    As an enumerated type, objects of this class will have only one of the values declared below.

    "ERob" and "CRob" represent Rob, the AI, on Cells where a Can does not and does also reside,
    respectively.

    """

    Empty = 0
    Wall = 1
    Can = 2
    ERob = 3
    CRob = 4

    def __str__(self):
        """
        This function converts a given cell to the string required to place it in the pictoral
        representation of the grid space

        :return: three-character string

        """
        if self.value == 0:
            return "[ ]"
        elif self.value == 1:
            return " = "
        elif self.value == 2:
            return "[c]"
        elif self.value == 3:
            return "[o]"
        else:
            return "[8]"

    @staticmethod
    def toString(c):
        """
        This function converts an integer argument to the string value appropriate for
        placement in the grid space.

        :param c: int : 0 <= c < 5
        :return: three-character string

        """
        if c == 0:
            return "[ ]"
        elif c == 1:
            return " = "
        elif c == 2:
            return "[c]"
        elif c == 3:
            return "[o]"
        else:
            return "[8]"

    @staticmethod
    def canOrEmpty():
        """
        This function returns either an empty cell or a can cell, at random.

        :return: Cell

        """
        i = random.randint(0, 100)
        if i < 50:
            return Cell.Empty
        else:
            return Cell.Can


class Board:
    """
    This class represents the grid space / game board and the "pieces" on it. It maintains the
    location of the "Rob" figure, and determines rewards given to the AI for different actions.

    dirs :: [String] : static field enumerating the actions available to the AI
    tax :: Boolean : flag indicating whether an "action tax" is enabled (True) or not (False)
    base_penalty :: Float : value determining the base penalty (without tax) returned when the
                            AI attempts to pick up a can where there is no can
    grid :: [[Cell]] : a 2-d list of cells that comprises the grid space
    location :: (Int, Int) : a tuple representing the location of the AI. For (i, j), i represents
                            the "x-coordinate," i.e. the index within the nested list, and j
                            represents the "y-coordinate," i.e. which nested list to look for. This
                            means you will see Rob's location as grid[j][i].

    """

    dirs = ["up", "down", "right", "left", "pickup"]

    def __init__(self, dim=10, tax=False, rw=1.0):
        """
        Constructor.

        :param dim: Int : determines the height and width of the grid space, including walls
        :param tax: Boolean : whether or not an "action tax" (reward of -0.5 for taking any
                                action) is enabled
        :param rw: Float : the base penalty for an incorrect pickup action

        """
        self.tax=tax
        self.base_penalty = rw
        self.grid = []
        self.grid.append([Cell.Wall for i in range(dim)])
        for i in range(dim-2):
            self.grid.append([Cell.Wall] + [Cell.canOrEmpty() for j in range(dim-2)] + [Cell.Wall])
        self.grid.append([Cell.Wall for i in range(dim)])
        self.location = (random.randint(1, 8), random.randint(1, 8))
        self.addRob(self.location)

    def __str__(self):
        """
        String conversion method.

        :return: string representing grid space for pretty printing

        """
        st = ""
        for line in self.grid:
            for item in line:
                st += Cell.toString(item)
            st += '\n'
        return st

    @staticmethod
    def actToInt(act):
        """
        This method maps an action to its corresponding integer value, i.e. which column in the
        qtable defines this action.

        :param act: String : the action to be mapped
        :return: Int : the index of the action

        """
        if act not in Board.dirs:
            raise BaseException("actToInt given illegal action")

        if act == "up":
            return 0
        elif act == "down":
            return 1
        elif act == "right":
            return 2
        elif act == "left":
            return 3
        elif act == "pickup":
            return 4
        else:
            raise Exception("Illegal action in actToInt.")

    @staticmethod
    def stateToKey(state):
        """
        This method maps a list of Cells to a numeric string that will represent it in the
        qtable.

        :param state: [Cell] : a list of five Cells
        :return: String : numeric string representing state's "key"

        """
        if len(state) != 5:
            raise Exception("State passed to stateToKey is wrong length!")

        st = ""
        for cell in state:
            if cell == Cell.Empty:
                st += "0"
            elif cell == Cell.Wall:
                st += "1"
            elif cell == Cell.Can:
                st += "2"
            elif cell == Cell.ERob:
                st += "3"
            elif cell == Cell.CRob:
                st += "4"
            else:
                raise Exception("Unknown State encountered")
        return st

    def getState(self):
        """
        Returns the state of Rob's current location, i.e. what lives at north, south, east, west, and
        "here," in that order.

        A state is represented by a dict of Cells with keys 'n' 's' 'e' 'w' and 'h'.

        :return: {String:Cell} : a dict representing the state at self.location

        """
        i = self.location[0]
        j = self.location[1]
        lst = []
        state = {}
        state['n'] = self.grid[j-1][i]
        lst.append(state['n'])
        state['s'] = self.grid[j+1][i]
        lst.append(state['s'])
        state['e'] = self.grid[j][i+1]
        lst.append(state['e'])
        state['w'] = self.grid[j][i-1]
        lst.append(state['w'])
        state['h'] = self.grid[j][i]
        lst.append(state['h'])
        state['k'] = self.stateToKey(lst)
        return state

    def addRob(self, loc):
        """
        Adds Rob to the given location. This is an in-place update.

        Note that this method should probably only be called after calling self.removeRob(), otherwise
        there will be more than one Rob on the board.

        :param loc: (Int, Int) : tuple representing the location at which to add Rob
        :return: None

        """
        i = loc[0]
        j = loc[1]
        lc = self.grid[j][i]
        if lc == Cell.Empty:
            self.grid[j][i] = Cell.ERob
            self.location = loc
        elif lc == Cell.Can:
            self.grid[j][i] = Cell.CRob
            self.location = loc
        else:
            raise Exception("Trying to add Rob in a Wall")

    def removeRob(self, loc):
        """
        Removes Rob from the given location. This is an in-place update.

        Note that this method also voids self.location.

        :param loc: (Int, Int) : the location of Rob, from which xhe will be removed.
        :return: None

        """
        i = loc[0]
        j = loc[1]
        lc = self.grid[j][i]
        if lc == Cell.ERob:
            self.grid[j][i] = Cell.Empty
            self.location = None
        elif lc == Cell.CRob:
            self.grid[j][i] = Cell.Can
            self.location = None
        else:
            raise Exception("Trying to remove Rob and he's not there")

    def moveRob(self, dir):
        """
        Updates the grid to reflect the specified movement, and returns the reward for said
        movement. This is update-in-place.

        :param dir: String : one of self.dirs
        :return: Float : reward for action

        """
        if dir not in self.dirs:
            raise Exception("Illegal move")

        i = self.location[0]
        j = self.location[1]

        if dir == "up":
            nj = max(j-1, 1)
            self.removeRob(self.location)
            self.addRob((i, nj))
            if nj == j:
                return -5.0 if self.tax is False else -5.5
            else:
                return 0.0 if self.tax is False else -0.5
        elif dir == "down":
            nj = min(j+1, 8)
            self.removeRob(self.location)
            self.addRob((i, nj))
            if nj == j:
                return -5.0 if self.tax is False else -5.5
            else:
                return 0.0 if self.tax is False else -0.5
        elif dir == "left":
            ni = max(i-1, 1)
            self.removeRob(self.location)
            self.addRob((ni, j))
            if ni == i:
                return -5.0 if self.tax is False else -5.5
            else:
                return 0.0 if self.tax is False else -0.5
        elif dir == "right":
            ni = min(i+1, 8)
            self.removeRob(self.location)
            self.addRob((ni, j))
            if ni == i:
                return -5.0 if self.tax is False else -5.5
            else:
                return 0.0 if self.tax is False else -0.5
        elif dir == "pickup":
            if self.grid[j][i] == Cell.CRob:
                self.grid[j][i] = Cell.ERob
                return 10.0 if self.tax is False else 9.5
            elif self.grid[j][i] == Cell.ERob:
                return -self.base_penalty if self.tax is False else -(self.base_penalty+0.5)
            else:
                raise Exception("Trying to pickup and Rob is not there")


class Qtable:
    """
    This class stores and manages the q-table or q-matrix used in the q-learning algorithm
    directing Rob's AI.

    table :: {String : [Float]} : a dict implementing the Q-matrix. Keys represent the state
                                    label "column", and values represent the learned reward
                                    values for each possible action, indexed the same as
                                    the output of Board.actToInt()

    """

    def __init__(self):
        """
        Constructor. Generates all possible 'nsewh' strings, with 'nsew' over [0-2], and
        'h' over [0-4]. This is a bit overkill, as not all possible state combinations
        will be encountered, but it is easy to accomplish programmatically.

        It is also true that 'h' (here) does not need 5 states, only two, but I am going
        to overlook that for now.

        :return: None

        """
        strings = [chr(n) + chr(s) + chr(e) + chr(w) + chr(h) for
                   n in xrange(ord('0'), ord('3')) for
                   s in xrange(ord('0'), ord('3')) for
                   e in xrange(ord('0'), ord('3')) for
                   w in xrange(ord('0'), ord('3')) for
                   h in xrange(ord('0'), ord('5'))]
        self.table = {n: [0.0, 0.0, 0.0, 0.0, 0.0] for n in strings}

    def __str__(self):
        """
        String conversion method. This outputs the contents of the entire table, so beware,
        especially if the table dimensions are high.

        :return: None

        """
        st = ""
        for s in sorted(self.table.keys()):
            st += s + ' ' + str(self.table[s]) + '\n'
        return st

    def update(self, state, act, state_p, act_p, eta, gamma, reward):
        """
        Updates the q-matrix according to the q-function. This is an in-place update.

        :param state: Dict : dict representing state when action 'act' was taken
        :param act: Int : number representing action taken, corresponding to Board.actToInt()
        :param state_p: Dict : dict representing s', or the state after action 'act was taken.
        :param act_p: Int : number representing best possible action from s'
        :param eta: Float : hyperparameter of Learner, representing the learning rate
        :param gamma: Float : hyperparameter of Learner, representing the "discount"
        :param reward: Float : reward obtained from action 'act' in state 'state'
        :return: None

        """
        q_sa = self.table[state['k']][act]
        q_sap = self.table[state_p['k']][act_p]
        self.table[state['k']][act] = q_sa + eta*(reward + (gamma*q_sap) - q_sa)

    def sum(self):
        """
        Returns the sum of all values held in self.table.

        :return: Float

        """
        ls = np.array(self.table.values())
        s = np.sum(ls)
        return s


class Learner:
    """
    This class manages and executes the q-learning algorithm.

    init_eps :: Float : initial epsilon value, representing degree of randomness in actions taken
    t :: Boolean : whether or not action tax is enabled
    rw :: Float : base "reward" for incorrect pickups. Note that this value should be positive, even
                    though it will eventually be given a sign change.
    qt :: Qtable : the q-matrix used by this Learner
    board :: Board : the board used by this learner (will be reinitialized between episodes)
    step_rewards :: List : keeps track of rewards produced in each step. reinitialized often
    reward_sums :: List : keeps track of rewards produced every 50 episodes
    eps_dec :: Boolean : whether or not annealing is enabled for the epsilon value
    N :: Int : the number of episodes to run
    M :: Int : the number of steps per episode
    eta :: Float : the learning rate
    gamma :: Float : the discount factor
    epsilon :: Float : the "randomness" factor

    """

    def __init__(self, eps_dec=True, N=5000, M=200, eta=0.2, gamma=0.9, eps=1.0, tax=False, rw=1.0):
        """
        Constructor.

        See class docstring for descriptions of parameters.

        """
        self.init_eps = eps
        self.t = tax
        self.rw = rw
        self.qt = Qtable()
        self.board = Board(tax=self.t, rw=self.rw)
        self.step_rewards = []
        self.reward_sums = []
        self.eps_dec = eps_dec
        self.N = N
        self.M = M
        self.eta = eta
        self.gamma = gamma
        self.epsilon = eps

    def getRandomAction(self):
        """
        Returns a random action from Board.dirs

        :return: String

        """
        r = random.randint(0, 4)
        a = self.board.dirs[r]
        return a

    def getBestAction(self):
        """
        Returns the action with the highest value in the q-matrix row of the current state of
        the grid. If all values are the same, returns a random action.

        :return: String

        """
        k = self.board.getState()
        k = k['k']
        vals = self.qt.table[k]

        if all(vals) == vals[0]:
            return self.getRandomAction()

        i = int(np.argmax(vals))
        a = self.board.dirs[i]
        return a

    def newAction(self):
        """
        returns either a random action or the "best" action (which may be random), depending
        on the value of epsilon and RNG.

        :return: String

        """
        t = random.random()
        if t > self.epsilon:
            return self.getBestAction()
        else:
            return self.getRandomAction()

    def step(self):
        """
        This method represents one step of computation in an episode, or one "move" made by
        Rob's AI. This method follows the Q-learning algorithm, and does in-place updating
        of its data structures.

        :return: None

        """
        state = self.board.getState()
        act = self.newAction()
        a = self.board.actToInt(act)
        reward = self.board.moveRob(act)

        state_p = self.board.getState()
        act_p = self.getBestAction()
        a_p = self.board.actToInt(act_p)
        self.qt.update(state, a, state_p, a_p, self.eta, self.gamma, reward)

    def episode(self):
        """
        Executes M steps, or one episode, re-initializing self.Board beforehand.

        :return: None

        """
        self.board = Board(tax=self.t, rw=self.rw)
        for i in range(self.M):
            self.step()

    def train(self):
        """
        Executes training, comprised of N episodes of M steps, wherein the algorithm is permitted
        to update its q-matrix.

        This method also outputs some results in the form of a pdf and a text file.

        :return: None

        """
        for i in range(self.N):
            if i % 50 == 0 and self.epsilon > 0.1 and self.eps_dec is True:
                self.epsilon -= 0.01
            self.episode()
            self.reward_sums.append(self.qt.sum())
            # print str(self.board)
            # print "Total reward: ", self.reward_sums[-1]
            # print "End of Episode: ", i
        # print "Final total reward: ", self.reward_sums[-1]
        pp = PdfPages("PU_Pen_" + str(self.rw) + "_tax_" + str(self.t) + "_epsdec_" + str(self.eps_dec) +
                      "_Gamma_" + str(self.gamma) + "_Eta_" + str(self.eta) +
                      "_N_" + str(self.N) + "_M_" + str(self.M) + "_ieps_" + str(self.init_eps) + ".pdf")
        plt.figure()
        labels = [str(x) for x in range(0, self.N, 100)]
        parts = [self.reward_sums[i] for i in range(0, len(self.reward_sums), 100)]
        plt.plot(range(self.N), self.reward_sums)
        plt.plot(labels, parts)
        plt.xlabel("Episode")
        plt.ylabel("Sum of Reward in Q Table")
        plt.title("PU Pen: " + str(self.rw) + " Tax: " + str(self.t) + " Eps Dec: " + str(self.eps_dec) +
                  " G: " + str(self.gamma) + " Eta: " + str(self.eta) +
                  " N: " + str(self.N) + " M: " + str(self.M) + " i_eps: " + str(self.init_eps))

        pp.savefig()
        pp.close()
        plt.close()

    def test_step(self):
        """
        A step in testing, during which the algorithm is not updating its matrix.

        This method returns the reward it obtains from each move, for measurement purposes.

        :return: Float

        """
        act = self.newAction()
        reward = self.board.moveRob(act)
        # print self.board
        # raw_input()
        return reward

    def test(self):
        """
        A testing iteration, consisting of N testing episodes of M testing steps.

        This method tracks the rewards obtained at each step, and at each episode, and
        reports the average and standard deviation of rewards at each episode in a text file.

        :return: None

        """
        self.reward_sums = []
        self.epsilon = 0.1
        for i in range(self.N):
            self.step_rewards = []
            self.board = Board(tax=self.t, rw=self.rw)
            for j in range(self.M):
                self.step_rewards.append(self.test_step())
            self.reward_sums.append(np.sum(self.step_rewards))

        avg = np.mean(self.reward_sums)
        stdev = np.std(self.reward_sums)

        with open("PU_pen_" + str(self.rw) + "_tax_" + str(self.t) + "_epsdec_" + str(self.eps_dec) +
                  "_Gamma_" + str(self.gamma) + "_Eta_" + str(self.eta) +
                  "_N_" + str(self.N) + "_M_" + str(self.M) + "_Ieps_ " +
                  str(self.init_eps) + ".txt", 'w+') as fh:
            fh.write("Test Average: " + str(avg) + '\n')
            fh.write("Test Standard Dev: " + str(stdev) + '\n')

# Experiment 1
l = Learner()
l.train()
l.test()

# Experiment 2
l = Learner(eta=0.1)
l.train()
l.test()
l = Learner(eta=0.4)
l.train()
l.test()
l = Learner(eta=0.7)
l.train()
l.test()
l = Learner(eta=0.9)
l.train()
l.test()

# Experiment 3
l = Learner(eps=0.3, eps_dec=False)
l.train()
l.test()
l = Learner(eps=0.7, eps_dec=False)
l.train()
l.test()

# Experiment 4
l = Learner(tax=True)
l.train()
l.test()

# Experiment 5
l = Learner(rw=3.0)
l.train()
l.test()
l = Learner(rw=5.0)
l.train()
l.test()
l = Learner(rw=7.0)
l.train()
l.test()
l = Learner(rw=9.0)
l.train()
l.test()
