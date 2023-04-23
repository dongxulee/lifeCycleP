from simulationParallel import *
import os.path
from multiprocessing import Pool 

AgentType = ["poorHigh", "poorLow", "richHigh", "richLow"]
Beta_r = [0.01,0.02,0.04,0.06,0.08,0.10]
Gamma = [2.0, 2.5]

def mapOverBeta(beta_r):
    for gamma in Gamma:
        for agentType in AgentType:
            print(agentType)
            fileName = agentType + "_" + str(beta_r) + "_" + str(gamma)
            if os.path.exists("solution/parallel_waseozcbkhm_" + fileName + ".npy"):
                break 
            if not os.path.exists("solution/" + fileName + ".npy"):
                break
            '''
                Constants 
            '''
            # discounting factor
            beta = 1/(1+beta_r)
            # utility function parameter 
            print("agentType: ", agentType)
            print("beta: ", beta)
            print("gamma: ", gamma)
            simulation(beta_r, agentType, gamma, fileName)

p = Pool(5)
p.map(mapOverBeta, Beta_r)