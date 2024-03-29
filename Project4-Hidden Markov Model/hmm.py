# hmm.py
# Submitted by Rahil Modi - C14109603
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu) and Ioannis Karamouzas (ioannis@g.clemson.edu)
#

"""
In this assignment, your task is to implement a localization system to track the 
location of a tiger-agent based on noisy sensor readings. To do so, you will 
implement two HMM inference algorithms: the exact algorithm (ExactInference), which 
computes a full probability distribution of the agent's discretized location, and 
particle filtering (ParticleFilter), which assumes a particle-based representation 
of this same distribution.


To simplify things, we assume a discretized 2D environment equipped with a number
of landmarks that facilitate the tracking of the agent. Based on this:

(1) At each time step t, let X_t={x_t for each cell x} denote the discretized state  
space representing the possible locations that the agent can be. We assume there is  
a local conditional distribution p(x_t | x_{t−1}) which governs the agent's movement. 

(2) The agent is equipped with a distance-based sensor that can detect its distance  
to the landmarks installed in the environment. However, the sensor readings are noisy. 
For each landmark i, the sensor provides us with a distance e^i_t, which is a Gaussian
random variable with mean equal to the true distance between the landmark and a given 
cell, having variance sigma^2, i.e.
		e^i_t ~ N (||(x-l_x, y-l_y)||, sigma^2), 
where (x, y) is a coordinate of a cell and (l_x, l_y) is the location coordinate 
of the landmark. Assuming each observation is independent, the sensor model follows:
	   p(e_t|x_t) = Π_i PDF(e^i_t), 
where pdf denotes the probability density function of e^i.  
Note that the PDF does not return a probability (densities can exceed 1), but for this 
assignment, we can get away with treating it like a probability. 


Both the ExactInference and ParticleFilter classes derive from the Inference abstract 
class, which provides the following public properties and methods:
 Properties
   n_rows : int                        the number of rows of the discrtized map;
   n_cols : int                        the number of columns of the discretized map;
   belief : list[list[float]]          an n_rows x n_cols belief table; each table
                                       entry denotes the probability of the tracked
                                       agent being in the corresponding cell;
   
 Methods:
   get_coordinate(r:int, c:int) -> (float, float)
               returns the coordinate (x, y) of the given cell (r, c).
   transition_model(r:int, c:int) -> list[(float, (int, int))]
               returns the dynamics model given a state; the model is represented
               as a list of (p, (r_, c_)) where (r_, c_) is a target cell the agent
               may reach from the given state (r, c) and the related probability p.
   normalize(belief: list[list[float]]) -> list[list[float]]
               returns the normalized belief table.


In addition, we provide below the following function: 			   
     normal_pdf(mean:float, sigma: float, value:float) -> float
				returns the probability density function of a Gaussian with given mean 
				and standard deviation, evaluated at value
			   

Given the above, your task is to implement a tracker that (approximately) computes the 
posterior distribution P(X_t | E_1=e_1,…,E_t=e_t) (your beliefs of where the agent is), 
and update it for each t. To do so, for the ExactInference and ParticleFilter algorithms, 
you need to complete two functions: 

 - The observe() function updates the belief table to represent p(x_t|e_{1:t}),  
   where e_{1:t} is the collection of observed evidence from the start till timestep t.
 - The timeUpdate() function updates the belief table to represent p(x_{t+1}|e_{1:t})
   and tracks the agent's state at the next timestep.

"""


from app.hmm_app import App
from app.inference import Inference


# It is allowed to use functions from math and random libraries if needed.
import math, random
import numpy as np

def normal_pdf(mean, sigma, value):
    """Calculates the probability density function of 1D Gaussian with given mean 
    and standard deviation, evaluated at value. 
    """
    return math.exp(- ((value - mean)/sigma)**2 * 0.5) /((2*math.pi)**0.5 * sigma) 


class ExactInference(Inference):
    """Exact inference algorithm to calculate a belief distribution over the probability 
	of the agent being in each cell.
	"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize the belief table, 
        # assuming that the agent is located at each cell with equal probability.
        p = 1. / (self.n_cols*self.n_rows)
        self.belief = [[p]*self.n_cols for _ in range(self.n_rows)]


    def observe(self,
            observed_distances,
            landmarks
        ):
        """ Update the self.belief table to represent p(x_t|e_{1:t}) based on observed 
		distances e_t and the landmark locations.
        Hints:
          - p(x_t|e_{1:t}) = p(x_t|e_t, e_{1:t-1}) \propto p(e_t|x_t)p(x_t|e_{1:t-1}), 
			where p(e_t|x_t) = \prod_{i} PDF(e^i_{t}). 			
			Here, PDF denotes the probability density function of a Gaussian, 
			e_t = {e^i_t} is the collection of distances between the tracked agent and 
			each landmark i. Each observed distance e^i_{t} follows a Gausssian 
			distribution: e^i_t ~ N(||Coord(x_t)-l_i||, sigma^2), where sigma = App.SENSOR_NOISE, 
			Coord(x_t) is the (x,y) coordinate of the given state cell x_t, l_i is the coordinate 
			of the i-th landmark, and || __ || denotes the Euclidean distance.  	
		 - Use get_coordinate to convert row and column indices of cells into locations
		 - normal_pdf: computes the probability density function for a Gaussian
		 - Don't forget to normalize the belief after you update its probabilities by calling 
		   self.belief = self.normalize(self.belief)!

        Parameters
        ----------
        observed_distances: a list of distances between each landmark and the tracked agent.  
        landmarks: a list of the (x, y) coordinate  of each landmark.
        """
        # Please finish the code below
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                x,y = self.get_coordinate(row,col)
                weight = 1
                for i in range(4):
                    posterior = self.belief[row][col]
                    pos_x = landmarks[i][0]
                    pos_y = landmarks[i][1]
                    distance = math.sqrt((pos_x - x)**2 + (pos_y - y)**2)
                    prob_df = normal_pdf(distance,App.SENSOR_NOISE,observed_distances[i])
                    weight *= prob_df
                self.belief[row][col] = posterior * weight
        self.belief = self.normalize(self.belief)

    def timeUpdate(self):
        """ Update the self.belief table to represent p(x_{t+1}|e_{1:t}) based on the passing 
		of one time step.
        Hints:
          - p(x_{t+1}|e_{1:t}) = \sum_{x_t} p(x_{t+1}|x_t)p(x_t|e_{1:t}),
            where p(x_{t+1}|x_t) is the transition probability which can be obtained by 
		    the function self.transition_model(r, c) fo the given x_t = (r, c).
		  - Be careful to use the current self.belief distribution to compute updated beliefs
		    rather than incrementally update the self.belief table.
		  - Don't forget to normalize the belief after you update its probabilities by calling 
		    self.belief = self.normalize(self.belief)!
        """
        # Please finish the code below 
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                prob_posterior = self.belief[row][col]
                actions = self.transition_model(row, col)
                for a in actions:
                    prob_transition = a[0]
                    new_row,new_col = a[1]
                    self.belief[new_row][new_col] += prob_posterior * prob_transition
        self.belief = self.normalize(self.belief)
        
class ParticleFilter(Inference):
    """ Use a set of particles to calculate an approximate belief distribution over the probability 
	of the agent being in each cell.
    """
    # total number of particles    
    NUM_PARTICLES = 200

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize the distribution of particles randomly.
        # Each element in self.particles represents a cell (r, c)
        self.particles = [None]*self.NUM_PARTICLES
        for i in range(self.NUM_PARTICLES):
            r, c = random.randint(0, self.n_rows-1), random.randint(0, self.n_cols-1)
            self.particles[i]=(r,c)
        self.updateBelief()
     
    # Convert the particle distributions to belief table
    def updateBelief(self):
        belief = [[0]*self.n_cols for _ in range(self.n_rows)]
        for p in self.particles:
            belief[p[0]][p[1]] += 1
        self.belief = self.normalize(belief)  


    def observe(self,
            observed_distances,
            landmarks
        ):
        """ Update the belief table to represent p(x_t|e_{1:t}) based on observed distances e_t
		and the landmark locations. 
		
		The whole process involves three steps:
        1. Compute the weight w_i of each partice i in self.particles as w_i = p(e_t|x_i), 
		where x_i denotes the grid cell of particle i. The emission probability associated 
		with the observed distance is computed as in the ExactInference algorithm. 
		2. Resample NUM_PARTICLES new particles by drawing with replacement from self.particles
		according to the weights computed in the previous step. 
        3. For visualization purposes, update the belief table based on the new particle distribution
		by calling self.updateBelief().

        Hints:
		  - You can use random.choices to draw samples with replacement.  
          - As multiple particles can reside in the same cell, they will share the same weight 
            w = p(e_t|x), where x = (r, c) denotes the grid cell. Hence, you may want to re-use 
            the weight as needed rather than re-compute it. 
		 
        Parameters
        ----------
        observed_distances: a list of distances between each landmark and the tracked agent.  
        landmarks: a list of the coordinate (x, y) of each landmark.
        """
        # Please finish the code below
        weight=[]
        for particle in self.particles:
            row,col = particle
            x,y = self.get_coordinate(row,col)
            product = 1
            for i in range(4):
                pos_x = landmarks[i][0]
                pos_y = landmarks[i][1]
                distance = math.sqrt((pos_x - x)**2 + (pos_y - y)**2)
                product *= normal_pdf(distance, App.SENSOR_NOISE, observed_distances[i]) 
            weight.append(product)
        prev_particles = self.particles[:]
        for i in range(self.NUM_PARTICLES):
            new_particle = (random.choices(prev_particles, weights=weight, k=1))[0]
            self.particles[i] = new_particle
        self.updateBelief()

    def timeUpdate(self):
        """ Update the self.belief table to represent p(x_{t+1}|e_{1:t}) based on the passing
		of one time step.
		
		The whole process involves two steps:
		1. Given the particle distribution self.particles at current time t, we want
		to propose the particle distribution at time t+1. To do so, we can sample  
		each particle using the transition model to see where it would end up. 
		2. For visualization purposes, update the belief table based on the new particle distribution
		by calling self.updateBelief().

		
        Hint:
          - A particle at (r, c) will reach a new cell or stay at its current cell with
            probability that can be obtained by the function self.transition_model(r,c).
        """
        # Please finish the code below 
        for j in range(len(self.particles)): 
            actions = self.transition_model(self.particles[j][0],self.particles[j][1])
            weight = [0] * len(actions)
            for i in range(len(actions)):
                weight[i] = (actions[i][0])
            new_cell = random.choices (actions, weights=weight, k=1) 
            self.particles[j] = new_cell[0][1]
        self.updateBelief()

if __name__ == "__main__":
    import tkinter as tk

    algs = {
        "Exact Inference": ExactInference, 
        "Particle Filter": ParticleFilter
    }
    root = tk.Tk()
    App(5, algs, root)
    tk.mainloop()
