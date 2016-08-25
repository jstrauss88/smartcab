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
        self.qtable = {}
        self.actions=["right","forward","left",None]
        self.previousstate=None
        self.previousreward=0
        self.previousaction = None
        self.alpha = 0.9
        self.gamma = 0
        self.epsilon = 0
        self.success=0
        self.totalpenalties=0
        self.penalties = 0
    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        if self.penalties>0:self.totalpenalties+=1        
        self.previousstate==None
        self.previousreward=0
        self.previousaction = None
        self.penalties=0
    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = (inputs['light'],inputs['oncoming'],inputs['left'],self.next_waypoint)
                
        # TODO: Select action according to your policy
        if self.state in self.qtable.keys():
            if random.random()<self.epsilon: action=random.choice(self.actions) #epsilon
            else:action = self.actions[np.argmax(self.qtable[self.state])]
        else: 
            action=random.choice(self.actions)
            self.qtable[self.state]=[0,0,0,0]
        # Execute action and get reward
        reward = self.env.act(self, action)
        if reward==12:
            self.success+=1 #count successful trials
            
        elif reward<0:self.penalties+=1
        # TODO: Learn policy based on state, action, reward
        if self.previousaction!=None:
            self.qtable[self.previousstate][self.previousaction]=(1-self.alpha)*self.qtable[self.previousstate][self.previousaction] + self.alpha*(self.previousreward+self.gamma * max(self.qtable[self.state]))
        self.previousaction=self.actions.index(action)
        self.previousreward = reward
        self.previousstate=self.state
        
         
         
      #  print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
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
    trials = 100
    sim.run(n_trials=trials)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    if a.penalties>0:a.totalpenalties+=1
    print "%s successful trials out of %s total trials with %s penalties" % (a.success,trials,a.totalpenalties)

if __name__ == '__main__':
    run()
