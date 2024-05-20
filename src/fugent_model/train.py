from .environment import *
from .model import *
import time
import argparse
import os

def envClass(classList, imgDir, ratingsFile):
    if not os.path.exists(imgDir):
        if os.path.exists('/data/data/'):
            imgDir = '/data/data/'
        elif os.path.exists('/mnt/seenas2/data/poincare/runData'):
            imgDir = '/mnt/seenas2/data/poincare/runData'
        else:
            print("Could not find image directory")
            exit(1)
        
    if not os.path.exists(ratingsFile) or not os.path.isfile(ratingsFile):
        if os.path.exists('./logs/ratings_pugmire.csv'):
            ratingsFile = './logs/ratings_pugmire.csv'
        else:
            print("Could not find user log directory")
            exit(1)
        
    env = make_environment(classList)(ratingsFile, imgDir)
    return env

def train(imgDir, modelPath, ratingsFile, configuration, numThreads=10, numEpisodes=1000):
    t0 = time.time()
    num_episodes = numEpisodes
    threads = numThreads
    
    # image classes can be:
    #   Source_Image
    #   Embedded_Image
    # state classes can be:
    #   State_Normalized
    #   State_OneHot
    # Action classes can be:
    #   Action_Normalized
    #   Action_OneHot
    # step classes can be:
    #   Log_Directed_LSTM
    #   Log_Directed_LSTM_Jitter
    #   Random
    
    # must be specified in step class, state class, image class order
    
    classConfig = { classOption.__name__:  classOption for classOption in classOptions }
    
    classList = list(map(lambda x: classConfig[x], configuration))
    env = lambda: envClass(classList, imgDir, ratingsFile)
    
    agent = A3CAgent(modelPath, env, 0.00025, num_episodes)
    agent.train(n_threads=threads)
    t1 = time.time()
    
    tot = t1-t0
    print('time: ', tot)

    with open(os.path.join(agent.model_name, 'time.txt'), 'w+') as f:
        f.write(f'time: {tot}')