from solveMDP import *
from multiprocessing import Pool

AgentType = ["poorHigh", "poorLow", "richHigh", "richLow"]
Beta_r = [0.02]
Gamma = [4.0]


def mapOverType(agentType):
    for beta_r in Beta_r:
         for gamma in Gamma:
            # discounting factor
            beta = 1/(1+beta_r)
            # utility function parameters 
            print("agentType: ", agentType)
            print("beta: ", beta)
            print("gamma: ", gamma)
            solveMDP(beta_r, agentType, gamma)

p = Pool(4)
p.map(mapOverType, AgentType)