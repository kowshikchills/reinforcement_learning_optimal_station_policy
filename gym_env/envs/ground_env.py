import numpy as np
import random
import matplotlib.pyplot as plt
import gym
from gym import error, spaces, utils


def cartesian(*arrays):
    mesh = np.meshgrid(*arrays)  
    dim = len(mesh) 
    elements = mesh[0].size  
    flat = np.concatenate(mesh).ravel() 
    reshape = np.reshape(flat, (dim, elements)).T 
    return reshape

class PlayGround:
    def __init__(self,l,start_point='center', generation_policy= 'inverse_radial'):
        self.l = l
        self.start_point = start_point
        self.setup_PG_grid()
        self.init_station()
        self.generated = False
        self.traj_tracker = []
        self.degree_action_map = {0: 0, 2: 180, 1: 90, 3: 270}
        self.positional_grid = cartesian(np.arange(0,100),np.arange(0,100))
        if generation_policy == 'uniform':
            self.weights_ = np.ones(len(self.positional_grid))
        elif generation_policy == 'radial':
            self.weights_  = ((50-self.positional_grid.T[0])**2) + ((50-self.positional_grid.T[1])**2)
        elif generation_policy == 'inverse_radial':
            self.weights_ = 2*50**2 -1*((50-self.positional_grid.T[0])**2) + -1*((50-self.positional_grid.T[1])**2)

    def setup_PG_grid(self):
        self.grid = np.zeros((self.l, self.l))
        self.mask = np.ones((self.l,self.l))

    def init_station(self):
        if self.start_point == 'center':
            self.pos = [int(self.l/2),int(self.l/2)]
        elif self.start_point == 'edge':
            self.pos = [0,0]
        elif self.start_point == 'random':
            self.pos = [random.randrange(0, self.l),random.randrange(0, self.l)]
        self.pos =  np.array(self.pos)
        self.grid[self.pos[0]][self.pos[1]] = 1

    def apply_action(self,a):
        '''
        0 -> straight 
        1 -> right
        2 -> down
        3 -> left 
        4 -> no move
        '''
        self.grid[self.pos[0]][self.pos[1]] = 0
        self.new_pos = self.pos 
        self.traj_tracker.append(self.pos)

        if a ==0:
            self.new_pos[1] = self.new_pos[1] + 1
        elif a ==2:
            self.new_pos[1] = self.new_pos[1] - 1
        elif a ==1:
            self.new_pos[0] = self.new_pos[0] + 1
        elif a ==3:
            self.new_pos[0] = self.new_pos[0] - 1
        elif a == 4:
            self.new_pos = self.pos 
        '''
        
        Handle edge conditions 

        '''
        self.new_pos[0] = np.clip(self.new_pos[0],0,self.l - 1)
        self.new_pos[1] = np.clip(self.new_pos[1],0,self.l - 1)
        self.grid[self.new_pos[0]][self.new_pos[1]] = 1
        self.pos = self.new_pos

    def random_pickup_point_generator(self):
        if not self.generated:
            self.ran_pos = list(random.choices(self.positional_grid, weights= self.weights_ )[0])
            #self.ran_pos = [random.randrange(0, self.l),random.randrange(0, self.l)]
            self.grid[self.ran_pos[0]][self.ran_pos[1]] = -1
            self.generated = True


    def reset_playround(self):
        self.setup_PG_grid()
        self.init_station()
        self.generated = False
        self.traj_tracker = []



class playground_env(gym.Env):
    def __init__(self,l,start_point='center',generation_policy= 'inverse_radial', reward_params = [0.1,0.01,0.01,1,50,500,1e7]):
        self.PG = PlayGround(l,start_point,generation_policy)
        self.time_step_counter = 0
        self.time_step_counter_pickup = 0
        self.time_step_counter_station = 0
        self.reward_params = reward_params
        self.random_generator_frequency = self.reward_params[4] 
        
        self.action_space = spaces.Discrete(5)
        self.state_type = 'vec'
        if self.state_type == 'image':
            self.observation_space = spaces.Box(low=0, high=255, shape=(l,l,1), dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(low=-1, high=1, shape=(24,), dtype=np.float64)

    def generate_random_pickup(self):
        if self.time_step_counter_station == self.random_generator_frequency:
            self.PG.random_pickup_point_generator()
            

    def reward(self):
        if self.PG.generated:
            r_cd = -1*((self.PG.pos[0] - self.PG.ran_pos[0])**2 + (self.PG.pos[1] - self.PG.ran_pos[1])**2)**0.5
            r_cd_scale = self.reward_params[0]*r_cd/(self.PG.l*(2)**0.5)
            r = r_cd_scale - self.reward_params[1]
        else:
            r = - self.reward_params[2]
        return(r)

    def define_done(self):
        if self.time_step_counter_pickup > self.reward_params[5] or self.time_step_counter > self.reward_params[6]:
            return(True)
        else:
            return(False)

    def constuct_obs(self):
        if self.state_type == 'image':
            obs = self.PG.grid
            obs = obs.reshape(obs.shape[0],obs.shape[1],1)
            obs = obs.astype('uint8')
        else:
            obs_dict = {}
            obs_dict['x'] = self.PG.pos[0]
            obs_dict['y'] = self.PG.pos[1]
            obs_dict['edgesx1'] = 100 - self.PG.pos[0]
            obs_dict['edgesx2'] = self.PG.pos[0] - 0
            obs_dict['edgesy1'] = self.PG.pos[1] - 0
            obs_dict['edgesy2'] = 100 - self.PG.pos[1]
            obs_dict['centrex1'] = self.PG.pos[0] - 50
            obs_dict['centrex2'] = 50 - self.PG.pos[0]
            obs_dict['centrey1'] = self.PG.pos[1] - 50
            obs_dict['centrey2'] = 50 - self.PG.pos[1]
 
            if self.PG.generated:
                obs_dict['goal'] = 1
                obs_dict['rx'] = self.PG.ran_pos[0]
                obs_dict['ry'] = self.PG.ran_pos[1]
                obs_dict['redgesx1'] = 100 - self.PG.ran_pos[0]
                obs_dict['redgesx2'] = self.PG.ran_pos[0] - 0
                obs_dict['redgesy1'] = self.PG.ran_pos[1] - 0
                obs_dict['redgesy2'] = 100 - self.PG.ran_pos[1]
                obs_dict['rcentrex1'] = self.PG.ran_pos[0] - 50
                obs_dict['rcentrex2'] = 50 - self.PG.ran_pos[0]
                obs_dict['rcentrey1'] = self.PG.ran_pos[1] - 50
                obs_dict['rcentrey2'] = 50 - self.PG.ran_pos[1]
                obs_dict['deltax'] = self.PG.pos[0] - self.PG.ran_pos[0]
                obs_dict['deltay'] = self.PG.pos[1] - self.PG.ran_pos[1]
                obs_dict['deltaxy'] = ((self.PG.pos[0] - self.PG.ran_pos[0])**2 + (self.PG.pos[1] - self.PG.ran_pos[1])**2)**0.5

            else:
                obs_dict['goal'] = 0
                obs_dict['rx'] = 0
                obs_dict['ry'] = 0
                obs_dict['redgesx1'] = 0
                obs_dict['redgesx2'] = 0
                obs_dict['redgesy1'] = 0
                obs_dict['redgesy2'] = 0
                obs_dict['rcentrex1'] = 0
                obs_dict['rcentrex2'] = 0
                obs_dict['rcentrey1'] = 0
                obs_dict['rcentrey2'] = 0
                obs_dict['deltax'] = 0
                obs_dict['deltay'] = 0
                obs_dict['deltaxy'] = 0

        obs = np.array(list(obs_dict.values()))/100    
        return(obs)

    def step(self,a):
        r_goal = 0
        self.PG.apply_action(a)
        self.time_step_counter = self.time_step_counter + 1 
        if not self.PG.generated:
            self.time_step_counter_station = self.time_step_counter_station + 1 
            self.generate_random_pickup()
        else:
            self.time_step_counter_pickup = self.time_step_counter_pickup + 1
            dis = np.abs(self.PG.pos - self.PG.ran_pos)
            if dis[0]< 3 and dis[1]< 3:
                self.time_step_counter_station = 0
                self.PG.generated = False
                self.PG.grid[self.PG.ran_pos[0]][self.PG.ran_pos[1]] = 0
                r_goal =  self.reward_params[3]
                self.time_step_counter_pickup = 0
        obs = self.constuct_obs()
        r = self.reward()
        done = self.define_done()
        reward = r + r_goal 
        return(obs,reward,done,{}) 

    def reset(self):
        self.PG.reset_playround()
        self.time_step_counter = 0
        self.time_step_counter_pickup = 0
        self.time_step_counter_station = 0
        obs = self.constuct_obs()
        return(obs)
