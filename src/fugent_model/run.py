from .environment import *
from . import model
import os

agent = None

def envClass(classList, imgDir):
    if not os.path.exists(imgDir):
        if os.path.exists('/data/data/'):
            imgDir = '/data/data/'
        elif os.path.exists('/mnt/seenas2/data/poincare/runData'):
            imgDir = '/mnt/seenas2/data/poincare/runData'
        else:
            print("Could not find image directory")
            exit(1)
        
    env = make_environment(classList)(None, imgDir)
    return env

def run(imgDir, configuration):
    global agent
    
    classConfig = { classOption.__name__:  classOption for classOption in classOptions }
    classList = list(map(lambda x: classConfig[x], configuration))
    env = lambda: envClass(classList, imgDir)
    
    agent = model.A3CAgent(env)