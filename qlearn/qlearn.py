from enum import Enum
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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
    def toString(c):
        if c == 0:
            return "[ ]"
        elif c == 1:
            return " = "
        elif c == 2:
            return "[c]"
        elif c == 3:
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

    def __init__(self, dim=10, tax=False):
        self.tax=tax
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
                #st += str(item)
                st += Cell.toString(item)
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
                return -5.0 if self.tax==False else -5.5
            else: return 0.0 if self.tax==False else -0.5
        elif dir == "down":
            nj = min(j+1, 8)
            self.removeRob(self.location)
            self.addRob((i,nj))
            if nj == j:
                return -5.0 if self.tax==False else -5.5
            else: return 0.0 if self.tax==False else -0.5
        elif dir == "left":
            ni = max(i-1, 1)
            self.removeRob(self.location)
            self.addRob((ni,j))
            if ni == i:
                return -5.0 if self.tax==False else -5.5
            else: return 0.0 if self.tax==False else -0.5
        elif dir == "right":
            ni = min(i+1, 8)
            self.removeRob(self.location)
            self.addRob((ni,j))
            if ni == i:
                return -5.0 if self.tax==False else -5.5
            else: return 0.0 if self.tax==False else -0.5
        elif dir == "pickup":
            if self.grid[j][i] == Cell.CRob:
                self.grid[j][i] = Cell.ERob
                return 10.0 if self.tax==False else 9.5
            elif self.grid[j][i] == Cell.ERob:
                return -1.0 if self.tax==False else -1.5
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
        q_sap = self.table[state_p['k']][act_p]
        self.table[state['k']][act] = q_sa + eta*(reward + (gamma*q_sap) - q_sa)

    def sum(self):
        s = 0.0
        for key in self.table.keys():
            s += np.sum(self.table[key])
        return s

class Learner:

    def __init__(self,eps_dec=True, N=5000, M=200, eta=0.2, gamma=0.9, eps=1.0, tax=False):
        self.init_eps = eps
        self.t=tax
        self.qt = Qtable()
        self.board = Board(tax=self.t)
        self.reward_sums = []
        self.eps_dec = eps_dec
        self.N = N
        self.M = M
        self.eta = eta
        self.gamma = gamma
        self.epsilon = eps

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
        self.board = Board(tax=self.t)
        for i in range(self.M):
            self.step()

    def train(self):
        for i in range(self.N):
            if i%50 == 0 and self.epsilon > 0.1 and self.eps_dec == True:
                self.epsilon -= 0.01
            self.episode()
            self.reward_sums.append(self.qt.sum())
            #print str(self.board)
            #print "Total reward: ", self.reward_sums[-1]
            #print "End of Episode: ", i
        #print "Final total reward: ", self.reward_sums[-1]
        pp = PdfPages("Tax_" + str(self.t) + "_epsdec_" + str(self.eps_dec) +\
                      "_Gamma_" + str(self.gamma) + "_Eta_" + str(self.eta) + \
                      "_N_" + str(self.N) + "_M_" + str(self.M) + "_ieps_" + str(self.init_eps) + ".pdf")
        plt.figure()
        labels = [str(x) for x in range(0,self.N, 100)]
        parts = [self.reward_sums[i] for i in range(0,len(self.reward_sums), 100)]
        plt.plot(range(self.N), self.reward_sums)
        plt.plot(labels, parts)
        plt.xlabel("Episode")
        plt.ylabel("Sum of Reward in Q Table")
        plt.title("Tax: " + str(self.t) + " Eps Dec: " + str(self.eps_dec) + \
                  " Gamma: " + str(self.gamma) + " Eta: " + str(self.eta) + \
                  " N: " + str(self.N) + " M: " + str(self.M) + " i_eps: " + str(self.init_eps))

        pp.savefig()
        pp.close()
        plt.close()

    def test_step(self):
        act = self.newAction()
        reward = self.board.moveRob(act)
        #print self.board
        #raw_input()
        return reward

    def test(self):
        self.reward_sums = []
        self.epsilon = 0.1
        for i in range(self.N):
            self.step_rewards = []
            self.board = Board(tax=self.t)
            for j in range(self.M):
                self.step_rewards.append(self.test_step())
            self.reward_sums.append(np.sum(self.step_rewards))

        avg = np.mean(self.reward_sums)
        stdev = np.std(self.reward_sums)

        with open("Tax_" + str(self.t) + "_epsdec_" + str(self.eps_dec) + \
                  "_Gamma_" + str(self.gamma) + "_Eta_" + str(self.eta) + \
                  "_N_" + str(self.N) + "_M_" + str(self.M) + "_Ieps_ " + \
                  str(self.init_eps) + ".txt", 'w+') as fh:
            fh.write("Test Average: " + str(avg) + '\n')
            fh.write("Test Standard Dev: " + str(stdev) + '\n')


l = Learner()
l.train()
l.test()


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


l = Learner(eps=0.3, eps_dec=False)
l.train()
l.test()
l = Learner(eps=0.7, eps_dec=False)
l.train()
l.test()


l = Learner(tax=True)
l.train()
l.test()
