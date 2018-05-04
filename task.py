import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 400
        self.action_high = 700
        self.action_size = 1

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        #dist = 1. - .3 * abs(np.linalg.norm(self.sim.pose[:3] - self.target_pos))
        #reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        #reward = 1.0 * (np.linalg.norm(self.target_pos - self.sim.pose[:3])**2)
        
        '''
        reward = -min(abs(self.target_pos[2] - self.sim.pose[2]), 20.0)  # reward = zero for matching target z, -ve as you go farther, upto -20
        if self.sim.pose[2] >= self.target_pos[2]:  # agent has crossed the target height
            reward += 10.0  # bonus reward
        elif self.sim.time > self.sim.runtime:  # agent has run out of time
            reward -= 10.0  # extra penalty
        elif self.sim.pose[2] < self.sim.init_pose[2]:
            reward -= 10.0 * self.sim.time # super penalty for drifting down from initial position

        v = self.sim.v[2] #check velocity
        if self.sim.pose[2] < self.target_pos[2]: #is below target
            if v > 0:
                reward += v
            else:
                reward -= v    
        else: #is above target
            if v < 0:
                reward += abs(v)
            else:
                reward -= v
        
        if self.sim.pose[2] == 0:
            reward -= 1000
        '''
        reward = self.sim.v[2]
        if self.sim.pose[2] == 0:
            reward -= 1000
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            rotor_speeds = rotor_speeds * 4 #same speed for all rotors
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state