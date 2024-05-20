import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, LSTM, LSTMCell, RNN, Reshape
from tensorflow.keras.optimizers import RMSprop
import numpy as np
from threading import Lock, Thread
import os
import time
import math
from matplotlib import pyplot as plt
from . import environment

import sys
import random

#TODO: merge model and model_embedded functions into one

'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    print(f'GPUs {gpus}')
    try: tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError: pass
'''

SEED = 0

def set_seeds(seed=SEED):
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

def set_global_determinism(seed=SEED):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
set_global_determinism(seed=SEED)


class __Random(random.Random):
    def __int__(self):
        return int.from_bytes(self.randbytes(4), 'little')


def RANDOM(*, seed: int):
    def Random(n: int, /):
        random = random.Random(seed)
        for _ in range(n):
            random.randbytes(1)
        n = random.randbytes(4)
        m = int.from_bytes(n, 'little')
        random = random.Random(n)
        random.np = np.random.default_rng(m)
        random.__class__ = __Random
        return random
    return Random


def model(embedded, lstm_history_size, lstm_size, state_size, img_size, action_size, extras, learning_rate=0.00025):
    inputs = Input((lstm_history_size, state_size + img_size))
    
    state_input, image_input = tf.split(inputs, [state_size, img_size], -1)
    
    
    hidden_state = Dense(
        256, 
        kernel_initializer='he_uniform',
        activation=tf.nn.elu,
    )(state_input)
    
    hidden_state = Reshape((lstm_history_size, 256))(hidden_state)
    
    if not embedded:
        image_input = Reshape((lstm_history_size, 84,84, 1))(image_input)
        conv1 = Conv2D(
            activation=tf.nn.elu,
            filters=16,
            kernel_size=(8,8),
            strides=(4,4),
            padding='valid',
        )(image_input)
        
        conv2 = Conv2D(
            activation=tf.nn.elu,
            filters=32,
            kernel_size=(4,4),
            strides=(2,2),
            padding='valid',
        )(conv1)
        
        image_input = Reshape((lstm_history_size, 9*9*32))(conv2)
    else:
        image_input = Reshape((lstm_history_size, img_size))(image_input)
    
    hidden_img = Dense(
        256, 
        kernel_initializer='he_uniform',
        activation=tf.nn.elu,
    )(image_input)
    
    hidden_img = Reshape((lstm_history_size, 256))(hidden_img)
    
    hidden_combined = tf.concat([hidden_state, hidden_img], -1)
    next_outputs = None
    
    if lstm_size != 1:
        next_outputs = LSTM(lstm_size)(hidden_combined)
    else:
        next_outputs = Reshape((1,-1))(hidden_combined)

    action_activation = extras.get('action_activation', 'softmax')
    action = Dense(action_size, activation=action_activation, kernel_initializer='he_uniform')(next_outputs)
    value = Dense(1, kernel_initializer='he_uniform')(next_outputs)
    
    Actor = Model(inputs = inputs, outputs = action, name='actor')
    actor_loss = extras.get('actor_loss', 'categorical_crossentropy')
    Actor.compile(loss=actor_loss, optimizer=RMSprop(learning_rate=learning_rate))
    
    Critic = Model(inputs = inputs, outputs = value, name='critic')
    Critic.compile(loss='mse', optimizer=RMSprop(learning_rate=learning_rate))
    
    return Actor, Critic
    
class A3CAgent:
    def __init__(self, model_path, envClass, learning_rate=0.00025, episodes=100, name=None, load_only=False):
        self.environment = envClass
        self.env = self.environment()
        self.lock = Lock()
        self.action_size = self.env.action_size
        self.max_average = -1000000
        
        self.name = name if name is not None else type(self.env).__name__
        
        self.lstm_size = 1 if environment.Random in type(self.env).__bases__ else 16
        self.use_embedding = environment.Embedded_Image in type(self.env).__bases__
        
        self.imgSize = self.env.img_size
        self.stateSize = self.env.state_size + self.env.timestep_size
        self.inputSize = self.env.inputSize
        
        #HYPERPARAMETERS
        self.history_length = self.env.history_length
        self.episodes = episodes
        self.learning_rate = learning_rate
        
        self.save_path = model_path or os.path.join(os.path.dirname(__file__), "Models")
        self.save_name = self.name
        self.model_name = os.path.join(self.save_path, self.save_name)
        
        if load_only:
            if self.exists():
                self.load(self.model_name)
            else: 
                print('Cannot find model')
        else:
            self.plot_dir = os.path.join(self.model_name, 'plots')
            # Instantiate filesystem
            self.init_filesystem()
            
            # Create Actor Critic network            
            self.actor, self.critic = model(self.use_embedding, self.history_length, self.lstm_size, self.stateSize, self.imgSize, self.action_size, self.env.model_extras, self.learning_rate)
        
    def init_filesystem(self):
        if not os.path.exists(self.model_name): 
            os.makedirs(self.model_name)
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)
        
    def load(self, model_name):
        self.actor  = load_model(os.path.join(self.model_name, 'Actor.keras') )
        self.critic = load_model(os.path.join(self.model_name, 'Critic.keras'))
        with open(os.path.join(self.model_name, 'score.txt'), 'r') as f:
            self.max_average = float(f.read())
        
    def exists(self):
        return os.path.exists(os.path.join(self.model_name, 'Actor.keras')) \
            and os.path.exists(os.path.join(self.model_name, 'Critic.keras'))
        
    def save(self):
        self.actor.save(os.path.join(self.model_name, 'Actor.keras'))
        self.critic.save(os.path.join(self.model_name, 'Critic.keras'))
        with open(os.path.join(self.model_name, 'score.txt'), 'w') as f:
            f.write(str(self.max_average))
        
    def train(self, n_threads, load_model=True):
        if load_model and self.exists():
            self.load(self.model_name)        
        
        envs = [self.environment() for i in range(n_threads)]
        if n_threads == 1:
            self.train_thread(self, envs[0], 0)
        else:
            threads = [Thread(
                target=self.train_thread,
                daemon=True,
                args=(self, envs[i], i) 
            ) for i in range(n_threads)]
            
            for t in threads:
                time.sleep(2)
                t.start()
                
            for t in threads:
                time.sleep(10)
                t.join()
                
    def plot_performance(self, curr_episode, historic_scores, historic_average, figName):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        
        episodes = np.arange(curr_episode+1)
        
        ax.plot(episodes, historic_scores, color='blue')
        ax.plot(episodes, historic_average, color='red')
        ax.set_title('Reward and Average Reward')
    
        plt.savefig(os.path.join(self.plot_dir, figName))
    
    def train_thread(self, agent, env, threadID):

        historic_scores = []
        historic_average = []
        for e in range(self.episodes):
            score, done, = 0, False
            
            states, actions, rewards = [], [], []
            
            state = self.reset(env)
            
            i = 0
            while not done:
                converted_state = env.convert_states(state)
                action = agent.act(env, converted_state)
                converted_action = env.convert_actions(action)
                
                next_state, reward, done = self.step(action, env, state)
                
                states.append(converted_state)
                actions.append(converted_action)
                rewards.append(reward)
                
                score += reward
                state = next_state
                i += 1
            
            self.lock.acquire()
            self.replay(states, actions, rewards)
            self.lock.release()
            
            with self.lock:
                save = False
                historic_scores.append(score)
                historic_average.append(sum(historic_scores[-self.history_length:]) / len(historic_scores[-self.history_length:]))
                average = historic_average[-1]
                # saving best models
                if average >= self.max_average:
                    self.max_average = average
                    self.save()
                    save = True
                else:
                    save = False
                    
                if e != 0 and (e%100 == 0 or save):
                    self.plot_performance(e, historic_scores, historic_average, f'{math.floor(average)}_{threadID}_{e}.png')
                
                print(f"episode: {e+1}/{self.episodes}, thread: {threadID}, score: {score}, average: {average:.2f} {'saving' if save else ''}")

            
               
    def reset(self, env):
        state = env.reset()
        return state
            
    def step(self, action, env, state):
        next_state, reward, done = env.step(action, state)
        return next_state, reward, done
    
    def act(self, env, state):
        state = np.reshape(state, (1, self.history_length, self.stateSize+self.imgSize))
        prediction = self.actor.predict(state)[0]
        return env.prediction_to_action(prediction)
    
    def discount_rewards(self, reward):
        # Compute the gamma-discounted rewards over an episode
        gamma = 0.99    # discount rate
        running_add = 0
        discounted_r = np.zeros_like(reward)
        for i in reversed(range(0,len(reward))):
            running_add = running_add * gamma + reward[i]
            discounted_r[i] = running_add

        discounted_r -= np.mean(discounted_r) # normalizing the result
        discounted_r /= np.std(discounted_r) # divide by standard deviation
        discounted_r = np.nan_to_num(discounted_r)
        return discounted_r
    
    def replay(self, states, actions, rewards):
        # reshape memory to appropriate shape for training
        states = np.vstack(states)
        actions = np.vstack(actions)
        
        if self.lstm_size == 1:
            actions = np.reshape(actions, (actions.shape[0], 1, actions.shape[1]))

        # Compute discounted rewards
        reward = self.discount_rewards(rewards)
        # Get Critic network predictions
        
        expected_reward = self.critic.predict(states)[:, 0]
        reward = np.reshape(reward, expected_reward.shape)
        
        # Compute advantages
        advantages = reward - expected_reward
        
        # training Actor and Critic networks
        self.actor.fit(states, actions, sample_weight=advantages, epochs=1, verbose=0)
        self.critic.fit(states, reward, epochs=1, verbose=0)



class Fugent_A3CAgent(A3CAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def onehot_actionMap(self, state):
        state = np.reshape(state, (1, self.history_length, self.stateSize+self.imgSize))
        prediction = self.actor.predict(state)[0]
        prediction = np.reshape(prediction, self.env.gridSize)
        return prediction    
        
    def critic_act(self, state):
        state = np.reshape(state, (1, self.history_length, self.stateSize+self.imgSize))
        prediction = float(self.critic.predict(state)[0][0])
        return prediction