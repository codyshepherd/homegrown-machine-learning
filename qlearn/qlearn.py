from enum import Enum
import random
import numpy as np

class Cell(Enum):
    Empty = 0
    Wall = 1
    Can = 2
    ERob = 3
    CRob = 4

    def __str__(self):
        if self.value == 0:
            return "[ ]"
        elif self.value == 1:
            return " = "
        elif self.value == 2:
            return "[c]"
        elif self.value == 3:
            return "[o]"
        else: return "[8]"

    @staticmethod
    def canOrEmpty():
        i = random.randint(0,100)
        if i < 50:
            return Cell.Empty
        else: return Cell.Can

class Board:

    dirs = ["up", "down", "left", "right", "pickup"]

    def __init__(self, dim=10):
        self.grid = []
        self.grid.append([Cell.Wall for i in range(dim)])
        for i in range(dim-2):
            self.grid.append([Cell.Wall] + [Cell.canOrEmpty() for j in range(dim-2)] + [Cell.Wall])
        self.grid.append([Cell.Wall for i in range(dim)])
        self.location = (random.randint(1,8), random.randint(1,8))
        #self.location = (1,1)
        self.addRob(self.location)

    def __str__(self):
        st = ""
        for line in self.grid:
            for item in line:
                st += str(item)
            st += '\n'
        return st

    @staticmethod
    def actToInt(act):
        if act not in Board.dirs:
            raise BaseException("actToInt given illegal action")

        if act == "up": return 0
        elif act == "down": return 1
        elif act == "left": return 2
        elif act == "right": return 3
        else: return 4

    @staticmethod
    def stateToKey(state):
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
            else: raise Exception("Unknown State encountered")
        return st

    def getState(self):
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
        i = loc[0]
        j = loc[1]
        l = self.grid[j][i]
        if l == Cell.Empty:
            self.grid[j][i] = Cell.ERob
            self.location = loc
        elif l == Cell.Can:
            self.grid[j][i] = Cell.CRob
            self.location = loc
        else: raise Exception("Trying to add Rob in a Wall")

    def removeRob(self, loc):
        i = loc[0]
        j = loc[1]
        l = self.grid[j][i]
        if l == Cell.ERob:
            self.grid[j][i] = Cell.Empty
            self.location = None
        elif l == Cell.CRob:
            self.grid[j][i] = Cell.Can
            self.location = None
        else: raise Exception("Trying to remove Rob and he's not there")


    def moveRob(self, dir):
        if dir not in self.dirs:
            raise Exception("Illegal move")

        i = self.location[0]
        j = self.location[1]

        if dir == "up":
            nj = max(j-1, 1)
            self.removeRob(self.location)
            self.addRob((i,nj))
            if nj == j:
                return -5.0
            else: return 0.0
        elif dir == "down":
            nj = min(j+1, 8)
            self.removeRob(self.location)
            self.addRob((i,nj))
            if nj == j:
                return -5.0
            else: return 0.0
        elif dir == "left":
            ni = max(i-1, 1)
            self.removeRob(self.location)
            self.addRob((ni,j))
            if ni == i:
                return -5.0
            else: return 0.0
        elif dir == "right":
            ni = min(i+1, 8)
            self.removeRob(self.location)
            self.addRob((ni,j))
            if ni == i:
                return -5.0
            else: return 0.0
        elif dir == "pickup":
            if self.grid[j][i] == Cell.CRob:
                self.grid[j][i] = Cell.ERob
                return 10.0
            elif self.grid[j][i] == Cell.ERob:
                return -1.0
            else: raise Exception("Trying to pickup and Rob is not there")

class Qtable:

    def __init__(self):
        strings = [ chr(n) + chr(s) + chr(e) + chr(w) + chr(h) for
                    n in xrange(ord('0'), ord('3')) for\
                    s in xrange(ord('0'),ord('3')) for\
                    e in xrange(ord('0'), ord('3')) for\
                    w in xrange(ord('0'), ord('3')) for\
                    h in xrange(ord('0'), ord('5'))]
        self.table = {n:[0.0,0.0,0.0,0.0,0.0,] for n in strings}

    def __str__(self):
        st = ""
        for s in sorted(self.table.keys()):
            st += s + ' ' + str(self.table[s]) + '\n'
        return st

    def update(self, state, act, state_p, act_p, eta, gamma, reward):
        q_sa = self.table[state['k']][act]
        q_sap = np.amax(self.table[state_p['k']])
        self.table[state['k']][act] = q_sa + eta*(reward + (gamma*q_sap) - q_sa)

    def sum(self):
        s = 0.0
        for key in self.table.keys():
            s += np.sum(self.table[key])
        return s

class Learner:
    N = 5000
    M = 200
    eta = 0.2
    gamma = 0.9
    init_epsilon = 1

    def __init__(self,):
        self.qt = Qtable()
        self.board = Board()
        self.epsilon = self.init_epsilon
        self.reward_sums = []

    def getRandomAction(self):
        r = random.randint(0,4)
        return self.board.dirs[r]

    def getBestAction(self):
        k = self.board.getState()
        k = k['k']
        vals = self.qt.table[k]
        ind = np.argmax(vals)
        return self.board.dirs[ ind ]

    def newAction(self):
        threshold = random.random()
        if threshold < (1-self.epsilon):
            #choose learned action
            return self.getBestAction()
        else: return self.getRandomAction()

    def step(self):
        state = self.board.getState()
        act = self.newAction()
        a = self.board.actToInt(act)
        reward = self.board.moveRob(act)
        state_p = self.board.getState()
        act_p = self.getBestAction()
        a_p = self.board.actToInt(act_p)
        self.qt.update(state, a, state_p, a_p, self.eta, self.gamma, reward)

    def episode(self):
        self.board = Board()
        print self.board
        for i in range(self.M):
            self.step()

    def train(self):
        for i in range(self.N):
            if i%50 == 0 and self.epsilon > 0.1:
                self.epsilon -= 0.01
            self.episode()
            self.reward_sums.append(self.qt.sum())
            print self.board
            print "Total reward: ", self.reward_sums[-1]
            print "End of Episode: ", i
        print "Final total reward: ", self.reward_sums[-1]

    def test_step(self):
        act = self.newAction()
        reward = self.board.moveRob(act)
        return reward

    def test(self):
        self.reward_sums = []
        self.epsilon = 0.1
        for i in range(self.N):
            self.step_rewards = []
            self.board = Board()
            for j in range(self.M):
                self.step_rewards.append(self.test_step())
            self.reward_sums.append(np.sum(self.step_rewards))

        avg = np.mean(self.reward_sums)
        stdev = np.std(self.reward_sums)

        return avg, stdev


l = Learner()
l.train()
a, s = l.test()
print "test average sum-over-episodes: ", a
print "test stdev: ", s
