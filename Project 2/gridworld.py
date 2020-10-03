# Rahil Modi C14109603
# grid_world.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu) and Ioannis Karamouzas (ioannis@g.clemson.edu)

"""
In this assignment, you will implement three classic algorithm for 
solving Markov Decision Processes either offline or online. 
These algorithms include: value_iteration, policy_iteration and q_learning.
You will test your implementation on three grid world environments. 
You will also have the opportunity to use Q-learning to control a simulated robot 
in crawler.py

The package `matplotlib` is needed for the program to run.

The Grid World environment has discrete state and action spaces
and allows for both model-based and model-free access.

It has the following properties:
    env.observation_space.n     # the number of states
    env.action_space.n          # the number of actions
    env.trans_model             # the transition/dynamics model

In value_iteration and policy_iteration, you can access the transition model 
at a given state s and action by calling
    t = env.trans_model[s][a]
where s is an integer in the range [0, env.observation_space.n),
      a is an integer in the range [0, env.action_space.n), and
      t is a list of four-element tuples in the form of
        (p, s_, r, terminal)
where s_ is a new state reachable from the state s by taking the action a,
      p is the probability to reach s_ from s by a, i.e. p(s_|s, a),
      r is the reward of reaching s_ from s by a, and
      terminal is a boolean flag to indicate if s_ is a terminal state.

In q_learning, once a terminal state is reached the environment should be (re)initialized by
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
You can also only update the visualization of the v values by
    logger.log(i, v)
"""


# use random library if needed
import random
import numpy as np


def value_iteration(env, gamma, max_iterations, logger):
    """
    Implement value iteration to return a deterministic policy for all states.
    See lines 20-40 for details.  

    Parameters
    ----------
    env: GridWorld
        the environment
    gamma: float
        the reward discount factor
    max_iterations: integer
        the maximum number of value iterations that should be performed;
        the algorithm should terminate when max_iterations is exceeded.
        Hint: The value iteration may converge before reaching max_iterations.  
        In this case, you may want to exit the algorithm earlier. A way to check 
        if value iteration has already converged is to check whether 
        the max over (or sum of) L1 or L2 norms between the values before and
        after an iteration is small enough. For the Grid World environment, 1e-4
        is an acceptable tolerance.
    logger: app.grid_world.App.Logger
        a logger instance to perform test and record the iteration process
    
    Returns
    -------
    pi: list or dict
        pi[s] should give a valid action,
        i.e. an integer in [0, env.action_space.n),
        as the optimal policy found by the algorithm for the state s.
    """
    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n
    TRANSITION_MODEL = env.trans_model

    v = [0] * NUM_STATES
    pi = [0] * NUM_STATES
    # Visualize the value and policy 
    logger.log(0, v, pi)
    # At each iteration, you may need to keep track of pi to perform logging
   
### Please finish the code below ##############################################
###############################################################################
    for k in range(0, max_iterations):
        threshold = 0
        for current_state in range(NUM_STATES):
            expected_action = [0] * NUM_STATES
            for action in range(NUM_ACTIONS):
                for probability, next_state, reward, terminal in TRANSITION_MODEL[current_state][action]:
                    if terminal:
                        expected_action[action] += probability*reward
                    else:
                        expected_action[action] += probability * (reward + gamma * v[next_state])
            best_action = max(expected_action)
            threshold = max(threshold, abs(best_action - v[current_state]))
            v[current_state] = best_action
        if threshold < 0.000001:
            break
        for i in range(NUM_STATES):
            expected_action = [0] * NUM_STATES
            for action in range(NUM_ACTIONS):
                for probability, next_state, reward, terminal in TRANSITION_MODEL[i][action]:
                    if terminal:
                        expected_action[action] += probability*reward
                    else:
                        expected_action[action] += probability * (reward + gamma * v[next_state])
            optimal = max(range(len(expected_action)), key=lambda x: expected_action[x])
            pi[i] = optimal
        logger.log(k, v, pi)
###############################################################################
    return pi


def policy_iteration(env, gamma, max_iterations, logger):
    """
    Implement policy iteration to return a deterministic policy for all states.
    See lines 20-40 for details.  

    Parameters
    ----------
    env: GridWorld
        the environment
    gamma: float
        the reward discount factor
    max_iterations: integer
        the maximum number of policy iterations that should be performed;
        the algorithm should terminate when max_iterations is exceeded.
        Hint 1: Policy iteration may converge before reaching max_iterations. 
        In this case, you should exit the algorithm. A simple way to check 
        if the algorithm has already converged is by simply checking whether
        the policy at each state hasn't changed from the previous iteration.
        Hint 2: The value iteration during policy evaluation usually converges 
        very fast and policy evaluation should end upon convergence. A way to check 
        if policy evaluation has converged is to check whether the max over (or sum of) 
        L1 or L2 norm between the values before and after an iteration is small enough. 
        For the Grid World environment, 1e-4 is an acceptable tolerance.
    logger: app.grid_world.App.Logger
        a logger instance to record and visualize the iteration process.
        During policy evaluation, the V-values will be updated without changing the current policy; 
        here you can update the visualization of value by simply calling logger.log(i, v).
    
    Returns
    -------
    pi: list or dict
        pi[s] should give a valid action,
        i.e. an integer in [0, env.action_space.n),
        as the optimal policy found by the algorithm for the state s.
    """
    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n
    TRANSITION_MODEL = env.trans_model
    
    v = [0.0] * NUM_STATES
    pi = [random.randint(0, NUM_ACTIONS-1)] * NUM_STATES
    # Visualize the initial value and policy
    logger.log(0, v, pi)

### Please finish the code below ##############################################
###############################################################################

    for i in range(0, max_iterations):
        for k in range(0, max_iterations):
            threshold = 0
            v_old = v.copy()
            for current_state in range(NUM_STATES):
                val = 0
                for probability, next_state, reward, terminal in TRANSITION_MODEL[current_state][pi[current_state]]:
                    if terminal:
                        val += probability * reward
                    else:
                        val += probability * (reward + gamma * v_old[next_state])
                threshold = max(threshold, abs(v_old[current_state] - val))
                v[current_state] = val
            if threshold < 0.000001:
                break
        policy_stable = True
        for current_state in range(NUM_STATES):
            old_action = pi[current_state]
            expected_action = [0]*NUM_ACTIONS
            for action in range(NUM_ACTIONS):
                for probability, next_state, reward, terminal in TRANSITION_MODEL[current_state][action]:
                    if terminal:
                        expected_action[action] += probability * reward
                    else:
                        expected_action[action] += probability * (reward + gamma*v[next_state])
                        
            best_action = max(range(len(expected_action)), key=lambda x: expected_action[x])

            if old_action != best_action:
                policy_stable = False

            pi[current_state] = best_action

        logger.log(i, v, pi)

        if policy_stable:
            return pi        
###############################################################################


def q_learning(env, gamma, max_iterations, logger):
    """
    Implement Q-learning to return a deterministic policy for all states.
    Please note that in RL you do not have access to the transition model 
    and the reward function, i.e. you cannot call env.trans_model as above. 
    Instead you need to learn policies by collecting samples using env.step
    See lines 40-50 for more details. 

    Parameters
    ----------
    env: GridWorld
        the environment
    gamma: float
        the discount factor
    max_iterations: integer
        the maximum number of iterations (training episodes) that should be performed;
        the algorithm should terminate when max_iterations is exceeded.
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
    
    v = [0] * NUM_STATES
    pi = [0] * NUM_STATES
    # Visualize the initial value and policy
    logger.log(0, v, pi)

    #########################
    # Adjust superparameters as you see fit
    #
    # parameter for the epsilon-greedy method to trade off exploration and exploitation
    eps = 1
    # learning rate for updating q values based on sample estimates
    alpha = 0.1
    ##################
    

### Please finish the code below ##############################################
###############################################################################
    Q = np.zeros((NUM_STATES, NUM_ACTIONS))
    s = env.reset()
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
            logger.log(i, v, pi)
            s = env.reset()
        else:
            action_ = np.argmax(Q[s_, :])
            target = r + gamma*Q[s_, action_]
            Q[s, action] = ((1-alpha)*Q[s,action]) + (alpha * target)
            pi[s] = np.argmax(Q[s, :])
            v [s] = max(Q[s])
            s, action = s_, action_
###############################################################################
    return pi


if __name__ == "__main__":
    from app.grid_world import App
    import tkinter as tk

    algs = {
        "Value Iteration": value_iteration,
        "Policy Iteration": policy_iteration,
        "Q Learning": q_learning
   }
    worlds = {
        # o for obstacle
        # s for start cell
        "world1": App.DEFAULT_WORLD,
        "world2": lambda : [
            [10, "s", "s", "s", 1],
            [-10, -10, -10, -10, -10],
        ],
        "world3": lambda : [
            ["_", "_", "_", "_", "_"],
            ["_", "o", "_", "_", "_"],
            ["_", "o",   1, "_",  10],
            ["s", "_", "_", "_", "_"],
            [-10, -10, -10, -10, -10]
        ]
    }

    root = tk.Tk()
    App(algs, worlds, root)
    tk.mainloop()