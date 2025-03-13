import numpy as np 

class QLearningAgent: 
    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space
        self.q_table = np.zeros((observation_space.shape[0], action_space.n))
    
    def choose_actions(self, state): 
        return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, next_state): 
        self.q_table[state, action] = reward + 0.9 * np.max(self.q_table[next_state])
