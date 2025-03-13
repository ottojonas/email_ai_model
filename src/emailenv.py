import gym 
import numpy as np 

class EmailEnv(gym.env): 
    def __init__(self, X, y): 
        super(EmailEnv, self).__init__()
        self.X = X 
        self.y = y
        self.action_space = gym.spaces.Discrete(2) # ? two actions - to reply or not to reply, that is the question 
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(X.shape[1],), dtype=np.float32)
        self.current_index=0
    
    def reset(self):
        self.current_index = 0 
        return self.X[self.current_index].toarray().flatten()
    
    def step(self, action): 
        reward = 0 
        done = False 
        if action == self.y[self.current_index]:
            reward = 1 
        self.current_index += 1 
        if self.current_index >= len(self.X): 
            done = True 
        return self.X[self.current_index].toarray().flatten(), reward, done, {} 
