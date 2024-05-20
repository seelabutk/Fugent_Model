import argparse
from .environment import *
from .train import train
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgDir')
    parser.add_argument('--modelPath')
    parser.add_argument('--ratingsFile', default='./logs/ratings_pugmire.csv')
    parser.add_argument('--config', type=str, nargs='*')
    parser.add_argument('--numThreads', type=int, default=10)
    parser.add_argument('--numEpisodes', type=int, default=1000)
    
    args = parser.parse_args()
    imgDir = args.imgDir
    modelPath = args.modelPath
    ratingsFile  = args.ratingsFile
    configuration = args.config
    numThreads = args.numThreads
    numEpisodes = args.numEpisodes
    
    classLists = {
        'largest': ['Log_Directed_LSTM', 'State_OneHot', 'Action_OneHot', 'Source_Image'],
        'smallest': ['Random', 'State_Normalized', 'Action_Normalized', 'Embedded_Image'],
        'middle': ['Log_Directed_LSTM', 'State_OneHot', 'Action_OneHot', 'Embedded_Image'],
        'fun': ['Log_Directed_LSTM', 'State_Normalized', 'Action_Normalized', 'Embedded_Image'],
    }
    
    if len(configuration) == 0:
        configuration = classLists['largest']
    elif len(configuration) == 1:
        configuration = classLists[configuration[0]]
    elif len(configuration == 3):
        pass
    else:
        print('Error, config must contian 0, 1, or 4 items')
        
    
    train(imgDir, modelPath, ratingsFile, configuration, numThreads, numEpisodes)