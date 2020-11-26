# crawler.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu) and Ioannis Karamouzas (ioannis@g.clemson.edu)

"""
In this file, you should test your Q-learning implementation on the robot crawler environment 
that we saw in class. It is suggested to test your code in the grid world environments before this one.

The package `matplotlib` is needed for the program to run.


The Crawler environment has discrete state and action spaces
and provides both of model-based and model-free access.

It has the following properties:
    env.observation_space.n     # the number of states
    env.action_space.n          # the number of actions
  

Once a terminal state is reached the environment should be (re)initialized by
    s = env.reset()
where s is the initial state.
An experience (sample) can be collected from s by taking an action a as follows:
    s_, r, terminal, info = env.step(a)
where s_ is the resulted state by taking the action a,
      r is the reward achieved by taking the action a,
      terminal is a boolean flag to indicate if s_ is a terminal state, and
      info is just used to keep compatible with openAI gym library.


A Logger instance is provided for each function, through which you can
visualize the process of the algorithm.
You can visualize the value, v, and policy, pi, for the i-th iteration by
    logger.log(i, v, pi)
"""


# use random library if needed
import random
import numpy as np

def q_learning(env, logger):
    """
    Implement Q-learning to return a deterministic policy for all states.

    Parameters
    ----------
    env: CrawlerEnv
        the environment
    logger: app.grid_world.App.Logger
        a logger instance to perform test and record the iteration process.
    
    Returns
    -------
    pi: list or dict
        pi[s] should give a valid action,
        i.e. an integer in [0, env.action_space.n),
        as the optimal policy found by the algorithm for the state s.
    """

    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n
    gamma = 0.95   

    #########################
    # Adjust superparameters as you see fit
    #
    # parameter for the epsilon-greedy method to trade off exploration and exploitation
    eps = 1
    # learning rate for updating q values based on sample estimates
    alpha = 0.1
    v = [0] * NUM_STATES
    pi = [0] * NUM_STATES
    # maximum number of training iterations
    max_iterations = 6500
    #########################

### Please finish the code below ##############################################
###############################################################################
    Q = np.zeros((NUM_STATES, NUM_ACTIONS))
    s = env.reset()
    eps_initial = eps
    for i in range(0, max_iterations):
        eps_final = 0.2
        eps = max(eps_final, eps*0.9999)
        if np.random.rand() < eps:
            action = np.random.randint(0, NUM_ACTIONS)
        else:
            action = np.argmax(Q[s, :])
        s_, r , terminal, info = env.step(action)
        if terminal:
            target = r
            Q[s, action] = ((1-alpha)*Q[s, action]) + (alpha*target)
            pi[s] = np.argmax(Q[s, :])
            s = env.reset()
        else:
            action_ = np.argmax(Q[s_, :])
            target = r + gamma*Q[s_, action_]
            Q[s, action] = ((1-alpha)*Q[s,action]) + (alpha * target)
            pi[s] = np.argmax(Q[s, :])
            v[s] = max(Q[s])
            s, action = s_, action_
        logger.log(i, v, pi)
###############################################################################
    return pi


if __name__ == "__main__":
    from app.crawler import App
    import tkinter as tk
    
    algs = {
        "Q Learning": q_learning,
    }

    root = tk.Tk()
    App(algs, root)
    root.mainloop()