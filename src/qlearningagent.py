import numpy as np 
import hashlib

class QLearningAgent: 
    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space
        self.q_table = np.zeros((10000, action_space.n))
    def _state_to_index(self, state): 
        return int(hashlib.md5(state).hexdigest(), 16) % self.q_table.shape[0]

    def choose_actions(self, state): 
        state_index = self._state_to_index(state)
        return np.argmax(self.q_table[state_index])
    

    def learn(self, state, action, reward, next_state):
        state_index = self._state_to_index(state)
        next_state_index = self._state_to_index(next_state)
        self.q_table[state_index, action] = reward + 0.9 * np.max(self.q_table[next_state_index])