import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import operator
import numpy as np

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.qtable = np.zeros((96,4))
        self.states=[]
        self.actions=["right","forward","left",None]
        self.previousstate=None
        self.previousreward=0
        self.previousaction = None
        self.alpha = .5
        self.gamma = .2
        self.epsilon = .1
    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = (inputs['light'],inputs['oncoming'],inputs['left'],self.next_waypoint)
                
        # TODO: Select action according to your policy
        if self.state in self.states:
            if random.random()<self.epsilon: action=random.choice(self.actions) #epsilon
            else:action = self.actions[np.argmax(self.qtable,1)[self.states.index(self.state)]]
        else: 
            action=random.choice(self.actions)
            self.states.append(self.state)
        # Execute action and get reward
        reward = self.env.act(self, action)
        
        # TODO: Learn policy based on state, action, reward
        if self.previousaction!=None:
            stin = self.states.index(self.previousstate)
            actin = self.actions.index(self.previousaction)
            self.qtable[stin,actin]=(1-self.alpha)*self.qtable[stin,actin] + self.alpha*(self.previousreward+self.gamma * np.argmax(self.qtable,1)[self.states.index(self.state)])
        self.previousaction=action
        self.previousreward = reward
        self.previousstate=self.state
         
         
        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
       # print len(self.states)

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=10)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    print a.qtable
if __name__ == '__main__':
    run()
