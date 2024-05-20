import os
import numpy as np
from PIL import Image
import math
import pandas as pd
import math
from scipy.ndimage import gaussian_filter
import copy
import functools
import random
import zipfile


defaultValue = -0.1

def clamp(minimum, maximum, x):
    return max(minimum, min(x, maximum))

def interpolate(x0, x1, y0, y1, x):
    return (y0*(x1-x) + y1*(x-x0))/(x1-x0)

def linear_falloff(radius, value, distance):
    interpolator = lambda d: interpolate(0, radius, value, defaultValue, d)
    if value >= 0:
        return max(min(interpolator(distance), value), 0)
    else:
        return max(min(interpolator(distance), defaultValue), value)

def average(lst):
    return sum(lst)/len(lst) if len(lst) > 0 else defaultValue

def mmax(lst):
    return max(lst, key=lambda x: abs(x)) if len(lst) > 0 else defaultValue

def load_interaction_history(fileName):
    #return pd.read_csv(fileName).iloc[:,1:].values.tolist()
    al = pd.read_csv(fileName).iloc[:,1:]
    al['timestep'] = al['timestep'].astype(str)
    timesteps = al['timestep'].unique()
    return al, timesteps
        
def normalize_interaction_history(actionlist):
    l = {}
    al = actionlist.groupby(['dataset', 'timestep'])
    for i in al:
        l[i[0]] = i[1].values.tolist()
    return l

def generate_region(gridSize, grid, X, Y, radius, value, falloff_function):
    for i in np.arange(-radius, radius):
        for j in np.arange(-radius, radius):
            x = X + i
            y = Y + j
            if x < 0 or x >= gridSize[0]:
                continue
            if y < 0 or y >= gridSize[1]:
                continue
            dist = math.dist((x, y), (X, Y))
            v = falloff_function(radius, value, dist)
            if v != 0:
                grid[x, y].append(v)

def generate_regions_of_interest(actionlist, gridSize):
    falloff_function = linear_falloff
    combination_function = mmax
    
    valueGrid = np.empty(gridSize, dtype=object)
    for i in range(gridSize[0]):
        for j in range(gridSize[1]):
            valueGrid[i,j] = []
    
    for action in actionlist:
        dataset, timestep, X, Y, rating, time = action
        effect_region = 0
        if rating == 0:
            value = 0
        elif rating == -1:
            value = -1
            effect_region = 10
        elif rating == 1:
            effect_region = 25
            value = 1
            
        if value != 0:
            generate_region(gridSize, valueGrid, X, Y, effect_region, value, falloff_function)
            
    roi = np.zeros(gridSize)
    for x, y in np.ndindex(gridSize):
        roi[x, y] = combination_function(valueGrid[x, y])
    roi = gaussian_filter(roi, sigma=1.5)
        
    return roi
        
        
        
        
        
class ActionList():
    def __init__(self, dataset, timestep, actionlist, env):
        self.dataset = dataset
        self.timestep = timestep
        self.actionlist = actionlist
        self.gridSize = env.gridSize
        self.env = env
        self.current_action = 0
        
        self.active_actionlist = None
        self.action_history = None
        self.expired_regions = None
        self.ROI = None
        
        self.reset()
    
    def reset(self):
        self.active_actionlist = copy.deepcopy(self.actionlist)
        self.ROI = generate_regions_of_interest(self.active_actionlist, self.gridSize)
        self.expired_regions = []
        self.action_history =  []
        self.current_action = 0
        return self.get_current_location()
    
    def step(self):
        self.current_action += 1
        return self.current_action == (len(self.actionlist)-self.env.history_length)
        
    def get_current_location(self, offset=0):
        d = self.actionlist[self.current_action + offset]
        return d[0:4]
    
    def get_ROI(self, index):
        return self.ROI[index]
    
    def recalculate_ROI(self):
        self.ROI = generate_regions_of_interest(self.active_actionlist, self.gridSize)
        
    def unexpire(self):
        recompute=False
        for region in self.expired_regions:
            region['expire'] -= 1
            # if our region is unexpired, re-add it to the active action list
            if region['expire'] == 0:
                self.active_actionlist.append(region['roi'])
                recompute=True
        if recompute:
            # reset expired regions and recalculate regions of interest
            self.expired_regions = [region for region in self.expired_regions if region['expire'] != 0]
            self.recalculate_ROI()
            
    def expire(self, roi_proximity, recLoc):
        # if we are too close to the center of a recommendation region
        if roi_proximity > .75:
            # find nearest upvote point
            upvotes = [action for action in self.active_actionlist if action[2] > 0]
            nearest = min(upvotes, key=lambda x: math.dist(recLoc, (x[2], x[3])))
            index = self.active_actionlist.index(nearest)
            # delete it
            del self.active_actionlist[index]
            # add it to expiration list
            self.expired_regions.append({
                'expire': self.env.reward_history,
                'roi': nearest
            })
            self.recalculate_ROI()
            
    def punish_distance_recent(self, recLoc, falloff):
        punishment = 0
        for i in range(len(self.action_history)):
            paction = self.action_history[i]
            recency = 1/(i+1)*12/25 # computed such that max distance for each paction will add up to 1, i.e 1*(1/(0+1))*12/25 + 1*(1/(1+1))*12/25 + 1*(1/(2+1))*12/25 + 1*(1/(3+1))*12/25 = 1
            dist = math.dist(paction, recLoc)
            value = falloff(dist)*recency*0.5
            punishment += value
        
        # only remember last 4 states
        self.action_history.append(recLoc)
        if len(self.action_history) > self.env.reward_history:
            self.action_history.pop(0) 
        return punishment
    
    def punish_distance(self, recLoc, last_state, falloff):
        dist = math.dist(last_state[2:4], recLoc)
        value = falloff(dist)*0.5
        punishment = value
        return punishment
    
    
    
    
    
    
class Poincare_Base_Environment():
    def __init__(self, userLogsFileName, imgDir, load_actionlist=True):    
        self.gridSize = (100,100)
        self.userLogsFileName = userLogsFileName
        self.imgDir = imgDir
        self.timestep_size = 0 #updated in load_interaction_history
        self.timesteps = []
        self.history_length = 4
        self.reward_history = 4
        self.IH = None
        
        self.model_extras = {}
        
        self.punish_for_nearby_rerecommendations = True
        self.punish_for_distant_recommendations = True
        
        self.currentActionListIndex = -1
        self.actionlist = None
        self.stepcount = 0
        
        if load_actionlist:
            IH, timesteps = load_interaction_history(userLogsFileName)
            self.IH = IH
            self.timesteps = timesteps
            self.timestep_size = len(timesteps)
            self.actionlists = [ActionList(AL[0][0], AL[0][1], AL[1], self) for AL in normalize_interaction_history(IH).items()]
    
        self.inputSize = self.timestep_size + self.state_size + self.img_size 
        
    def get_image(self, state):
        pass
    
    def has_data(self, state):
        pass
    
    def make_input_state(self, loc):
        pass
    
    def get_next_state(self):
        pass
    
    def prediction_to_action(self, action):
        pass
    
    def convert_actions(self, action):
        pass
    
    @functools.cache
    def get_state_dtXY(self, loc):
        img = self.get_image(loc)
        img = np.squeeze(np.reshape(img, (-1, self.img_size)))
        
        stateInput = self.make_input_state(loc)
        state = np.squeeze(np.concatenate((stateInput, img), axis=None))
        return state
    
    def convert_states(self, states):
        convertedStates = []
        for state in states:
            dataset, timestep, X, Y = state
            loc = [dataset, timestep, X, Y]
            state = self.get_state_dtXY(tuple(loc))
            convertedStates.append(state)
            
        convertedStates = np.array(convertedStates)
        convertedStates = np.reshape(convertedStates, (1, self.history_length, self.inputSize))
        return convertedStates
    
    def round_location(self, record):
        x = record[0]
        y = record[1]
        xGrid = math.floor(x*self.gridSize[0])
        yGrid = math.floor(y*self.gridSize[1])
        return np.asarray([xGrid, yGrid])

    def action_to_coords(self, action):
        action = math.floor(action)
        recLoc = (math.floor(action/self.gridSize[0]), action%self.gridSize[1])
        return recLoc
    
    def reward(self, action, last_state):
        reward = 0
        punishment = 0
        
        # these functions have been calculated such that the "worst" distance is 0, 
        # "best" is 1, and have characteristics such that
        # there is a large ideal middle-ground
        # domain and range of all functions is [0,1]
        close_falloff = lambda x: (math.e**(-0.2*x))
        far_falloff = lambda x: (x/math.sqrt(2*(100**2)))**2
        proximity_reward = lambda x: (x**3)
        
        recLoc = action
        roi_proximity = self.actionlist.get_ROI(recLoc)
        
        #re-add un-expired regions of interest
        self.actionlist.unexpire()
        
        # if we are too close to the center of a recommendation region
        self.actionlist.expire(roi_proximity, recLoc)
            
        # lower reward if we have recommended regions here recently
        punishment += self.actionlist.punish_distance_recent(recLoc, close_falloff)
        
        # lower reward for further-away recommendations to state
        punishment += self.actionlist.punish_distance(recLoc, last_state, far_falloff)
        
        reward = proximity_reward(roi_proximity)
        tot = reward - punishment
        tot = max(-1, min(tot, 1))
            
        return tot
    
class Embedded_Image(Poincare_Base_Environment):
    def __init__(self, *args, **kwargs):
        self.img_size = 1024
        super().__init__(*args, **kwargs)
        
    def get_image(self, state):
        dataset, timestep, X, Y = state
        fileName = f'{100*Y+X}_{X}_{Y}.npy'
        file = os.path.join(self.imgDir, dataset, timestep, 'embed.zip')
        root = zipfile.Path(file)
        path = root / fileName
        embed = None
        with path.open('rb') as f:
            embed = np.load(f)
        return embed
    
    @functools.cache
    def has_data(self, state):
        dataset, timestep, X, Y = state
        fileName = f'{100*Y+X}_{X}_{Y}.npy'
        file = os.path.join(self.imgDir, dataset, timestep, 'embed.zip')
        root = zipfile.Path(file)
        path = root / fileName
        return path.exists()
    
class Source_Image(Poincare_Base_Environment):
    def __init__(self, *args, **kwargs):
        self.img_size = 84*84
        super().__init__(*args, **kwargs)
        
    def get_image(self, state):
        dataset, timestep, X, Y = state
        fileName = f'{Y*100+X}_{X}_{Y}_density.png'
        file = os.path.join(self.imgDir, dataset, timestep, 'png', fileName)        
        if os.path.exists(file):
            image = Image.open(file)
            image = image.resize((84,84)).convert('L')
            image = np.array(image)
            image = np.reshape(image, (84,84,1))
        else:
            image = np.random.rand(84,84,1)
        return image
    
    @functools.cache
    def has_data(self, state):
        dataset, timestep, X, Y = state
        fileName = f'{Y*100+X}_{X}_{Y}_density.png'
        file = os.path.join(self.imgDir, dataset, timestep, 'png', fileName)
        return os.path.exists(file)

    


class State_OneHot(): # timestep onehot encoded, state one-hot encoded
    def __init__(self, *args, **kwargs):
        self.state_size = 100*100
        super().__init__(*args, **kwargs)
        
    def make_input_state(self, loc):
        dataset, timestep, psi, theta = loc
        thetaPsiOneHot = np.zeros([self.action_size])
        timestepOneHot = np.zeros([self.timestep_size])
        thetaPsiOneHot[theta*100+psi] = 1
        tswhere = np.where(self.timesteps == timestep)
        if len(tswhere[0]) != 0:
            tsi = tswhere[0][0]
            timestepOneHot[tsi] = 1
        return np.concatenate((timestepOneHot, thetaPsiOneHot), axis=None)

class State_Normalized(): # timestep onehot encoded, state is [0 < theta < 1, 0 < psi < 1]
    def __init__(self, *args, **kwargs):
        self.state_size = 2
        super().__init__(*args, **kwargs)
        
    def make_input_state(self, loc):
        dataset, timestep, psi, theta = loc
        theta = theta / self.gridSize[0]
        psi =   psi   / self.gridSize[1]
        thetaPsi = np.array([theta, psi])
        timestepOneHot = np.zeros([self.timestep_size])
        tsi = np.where(self.timesteps == timestep)[0]
        if len(tsi):
            timestepOneHot[tsi[0]] = 1
        return np.concatenate((timestepOneHot, thetaPsi), axis=None)
        
        
class Action_OneHot():
    def __init__(self, *args, **kwargs):
        self.action_size = 100*100
        super().__init__(*args, **kwargs)
        
    def convert_actions(self, action):
        theta, psi = action
        actionID = theta*100+psi
        action_onehot = np.zeros([self.action_size])
        action_onehot[actionID] = 1
        return action_onehot
    
    def prediction_to_action(self, prediction):
        action = np.random.choice(self.action_size, p=prediction.flatten())
        return self.action_to_coords(action)

class Action_Normalized():
    def __init__(self, *args, **kwargs):
        self.action_size = 2
        super().__init__(*args, **kwargs)
        self.model_extras = {
            'actor_loss': 'mean_squared_error',
            'action_activation': 'linear',
        }
        
    def convert_actions(self, action):
        theta, psi = action
        theta = theta / self.gridSize[0]
        psi =   psi   / self.gridSize[1]
        return [theta, psi]
    
    def prediction_to_action(self, action):
        theta, psi = action.flatten()
        theta = clamp(0, self.gridSize[0]-1, round(theta*self.gridSize[0]))
        psi =   clamp(0, self.gridSize[0]-1, round(psi  *self.gridSize[1]))
        return theta, psi


class Log_Directed_LSTM():
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_next_state(self):
        states = []
        i = 0
        found = 0
        while found != self.history_length:
            currLoc = self.actionlist.get_current_location(i)
            if self.has_data(tuple(currLoc)):
                found += 1
                dataset, timestep, X, Y = currLoc
                loc = [dataset, timestep, X, Y]
                states.append(loc)
            i += 1
        return states
        
    def reset(self):
        self.currentActionListIndex = -1
        self.stepcount = 0
        return self.subreset()
    
    def subreset(self):
        self.currentActionListIndex = (self.currentActionListIndex + 1) % len(self.actionlists)
        self.actionlist = self.actionlists[self.currentActionListIndex]
        self.actionlist.reset()
        return self.get_next_state()
    
    def step(self, action, last_state):
        done = False
        subdone = self.actionlist.step()

        next_state = self.get_next_state()
        
        reward = self.reward(action, last_state[-1])
        if subdone:
            if self.currentActionListIndex == len(self.actionlists)-1:
                done = True
            else:
                self.subreset()
        
        return next_state, reward, done
    
    
class Log_Directed_LSTM_Jitter(Log_Directed_LSTM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.jitter_amount = 5
        
    def jitter(self, x,y,amount):
        xJitter = random.randint(-amount,amount)
        yJitter = random.randint(-amount,amount)
        xj = x+xJitter
        yj = y+yJitter
        xj = clamp(0, self.gridSize[0]-1, xj)
        yj = clamp(0, self.gridSize[1]-1, yj)
        return xj, yj
    
    def get_next_state(self):
        states = []
        for i in range(self.history_length):
            currLoc = self.actionlist.get_current_location(i)
            dataset, timestep, X, Y = currLoc
            X0, Y0 = X, Y
            X,Y = self.jitter(X, Y, self.jitter_amount)
            loc = [dataset, timestep, X, Y]
            
            while not self.has_data(tuple(loc)):
                X,Y = self.jitter(X0, Y0, self.jitter_amount)
                loc = [dataset, timestep, X, Y]
                
            states.append(loc)
        return states


class Random():
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history_length = 1
        
    def reset(self):
        self.stepcount = 0
        return self.get_next_state()
    
    def get_next_state(self):
        next_state = None
        while next_state == None or not self.has_data(tuple(next_state[0])):
            #get random actionlist (holds ROI map)
            self.actionlist = np.random.choice(self.actionlists)
            #generate random theta/psi
            theta = np.random.randint(self.gridSize[0])
            psi = np.random.randint(self.gridSize[1])
            next_state = [[self.actionlist.dataset, self.actionlist.timestep, theta, psi]]
        return next_state
    
    #assumes no lstm and history_length of 1
    def step(self, action, last_state):
        self.stepcount += 1
        
        #calculate reward
        reward = self.reward(action, last_state[-1])   
        next_state = self.get_next_state()
        
        #determine if episode is over
        done = self.stepcount == 10
        
        return next_state, reward, done
    

def make_environment(classList):
    return type('_'.join([x.__name__ for x in classList]), (*classList,), {})
    
    
    
    
classOptions = [Source_Image, Embedded_Image, State_Normalized, State_OneHot, Action_Normalized, Action_OneHot, Log_Directed_LSTM, Log_Directed_LSTM_Jitter, Random]
    
    
    
    
    
    
if __name__ == "__main__":
    imgDir = '/mnt/seenas2/data/poincare/runData'
    ratingsFile = './logs/ratings_pugmire.csv'
    env = make_environment(Log_Directed_LSTM, Source_Image)(ratingsFile, imgDir)
    
    done = False
    state = env.reset()
    while not done:
        action = np.random.randint(env.action_size)
        state, _, done = env.step(action, state)