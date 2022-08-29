import torch
import torch.nn
import gym
import argparse
import numpy as np
import math
import nevergrad as ng
import cma

from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from sb3_contrib import TRPO

class SimOpt(object):
    """
    Available methods.
    
    distribution_optimization(). It finds an optimal distribution of parameters of interest.
    
    get_optimum(). Getter method, to be called after 'distribution_optimization()'. It returns the optimal value of the objective function.
    
    get_optimal_values(). Getter method, to be called after 'distribution_optimization()'. It returns the optimal distribution for each parameter of interest in the form list(Tuple(<mean>, <variance>)). 
    """
    
    def __init__(self, source, target):
        """
        Parameters.
        
        source: string,
            source environment name to be passed to gym simulated environment object.
            
        target: string,
            target environment name to be passed to gym simulated environment object.
        """
        
        self.source = source
        self.target = target
        return

    def distribution_optimization(self, training_algorithm, initPhi, normalize, logspace, budget, n_iterations, T_first, phi_initial_values, phi_bounds, l_norm_space, importance_weights, norms_weights):
        """
        Parameters.
        
        training_algorithm: string,
            policy gradient algorithm used to train the agent from stable_baselines3 or sb3_contrib libraries. Possible choices: ['PPO', 'TRPO'].
            
        initPhi: string,
            intitialization method for the distribution parameter phi. Possible choices: ['fixed', 'random'].
            
        normalize: boolean,
            normalize phi in the space [0, n].
            
        logspace: boolean,
            normalize variances belonging to phi in a logarithmic space. It makes senses only if 'normalize' is set to True.
            
        budget: int,
            number evaluations of the objective function performed by the optimizer (i.e. number of samples from the distribution).
            
        n_iterations: int,
            number of iterations of the optimization algorithm.
            
        T_first: string,
            T-first value in discrepancy function as in our paper. Possible choice: ['max', 'min', 'fixed:<number>'].
            
        phi_initial_values: list(int),
            initial values for the distribution parameter phi in the first iteration of the algorithm.
            
        phi_bounds: list(int),
            bounds values for the distribution parameter phi.
            
        l_norm_space: int,
            length of the normalized space if 'normalize' is set to True.
            
        importance_weights: list(int),
            importance weights for each dimension in the discrepancy function as in our paper. Each element in the list corresponds to a single dimension of the observation space. 
        
        norms_weights: list(int),
            weights for each norm in the discrepancy function as in our paper. The first element of the list corresponds to the 1-norm of the observation vector, the second element corresponds to the 2-norm.
        """

        self.dimensions_ImportanceWeights = importance_weights
        self.norms_weights = norms_weights

        env_source = gym.make(self.source)
        self.n_paramsToBeRandomized = len(env_source.get_parametersToBeRandomized())
        env_source.close()

        self.__set_phiBounds(phi_bounds)
        phi0 = self.__set_initialPhi(initPhi, phi_initial_values)
        self.__set_searchSpace_bounds()


        phi = phi0
        for i in range(n_iterations):
            env_source = gym.make(self.source)
            env_source.reset()
            model = self.__get_model(training_algorithm, env_source)
            for i in range(100):
                env_source.set_random_parameters(phi)
                model.learn(total_timesteps=10000)
            #model.save("model.mdl")
            env_source.close()

            #Collect 1 rollout in real word
            env_target = gym.make(self.target)
            traj_obs = []
            obs = env_target.reset()
            self.starting_state = obs
            traj_obs.append(obs)
            cum_reward = 0
            done = False
            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = env_target.step(action)
                traj_obs.append(obs)
                cum_reward += reward
            #print(cum_reward)
            tau_real = np.array(traj_obs)
            env_target.close()

            self.__set_length_normalized_space(l_norm_space)
            searchSpace = []
            for i in range(self.n_paramsToBeRandomized):
                if normalize:
                    mean = ng.p.Scalar(init=self.__normalize_value(phi[i][0], self.bounds[i][0][0], self.bounds[i][0][1], 0, self.length_normalized_space)).set_bounds(lower=0, upper=self.length_normalized_space)
                    if logspace:
                        standard_deviation = ng.p.Scalar(init=self.__normalize_value_log(phi[i][1], self.bounds[i][1][0], self.bounds[i][1][1], 0, self.length_normalized_space)).set_bounds(lower=0, upper=self.length_normalized_space)
                    else:
                        standard_deviation = ng.p.Scalar(init=self.__normalize_value(phi[i][1], self.bounds[i][1][0], self.bounds[i][1][1], 0, self.length_normalized_space)).set_bounds(lower=0, upper=self.length_normalized_space)
                else:
                    mean = ng.p.Scalar(init=phi[i][0]).set_bounds(lower=self.bounds[i][0][0], upper=self.bounds[i][0][1])
                    standard_deviation = ng.p.Scalar(init=phi[i][1]).set_bounds(lower=self.bounds[i][1][0], upper=self.bounds[i][1][1])
                searchSpace.append(mean)
                searchSpace.append(standard_deviation)

            params = ng.p.Tuple(*searchSpace)
            instrumentation = ng.p.Instrumentation(params=params, normalize=normalize, logspace=logspace, model=model, tau_real=tau_real, T_first=T_first)
            cmaES_optimizer = ng.optimizers.CMA(parametrization=instrumentation, budget=budget)
            recommendation = cmaES_optimizer.minimize(self.__objective_function)


            rec = []
            if normalize:
                for i in range(int(len(recommendation.value[1]['params'])/2)):
                    mean = recommendation.value[1]['params'][i*2]
                    standard_deviation = recommendation.value[1]['params'][i*2+1]
                    mean = self.__denormalize_value(mean, self.bounds[i][0][0], self.bounds[i][0][1], 0, self.length_normalized_space)
                    if logspace:
                        standard_deviation = self.__denormalize_value_log(standard_deviation, self.bounds[i][1][0], self.bounds[i][1][1], 0, self.length_normalized_space)
                    else:
                        standard_deviation = self.__denormalize_value(standard_deviation, self.bounds[i][1][0], self.bounds[i][1][1], 0, self.length_normalized_space)
                    rec.append(mean)
                    rec.append(standard_deviation)
            else:
                rec = recommendation.value[1]['params']
            optimum =  self.__objective_function(**recommendation.kwargs)
            #print(rec, optimum)

            phi_optim = rec
            phi = []
            for i in range(int(len(phi_optim)/2)):
                mean = phi_optim[i*2]
                standard_deviation = phi_optim[i*2+1]
                phi.append((mean, standard_deviation))

        self.optimum = optimum
        self.optimal_values = phi


    def get_optimum(self):
        return self.optimum

    def get_optimal_values(self):
        return self.optimal_values.copy()

    def __set_initialPhi(self, initPhi, initial_values):
        phi = []
        if initPhi=='fixed':
            mean = initial_values[0]
            standard_deviation = initial_values[1]
            phi.append((mean, standard_deviation))
            mean = initial_values[2]
            standard_deviation = initial_values[3]
            phi.append((mean, standard_deviation))
            mean = initial_values[4]
            standard_deviation = initial_values[5]
            phi.append((mean, standard_deviation))
            mean = initial_values[6]
            standard_deviation = initial_values[7]
            phi.append((mean, standard_deviation))
        if initPhi=='random':
            for i in range(self.n_paramsToBeRandomized):
                mean = np.random.rand()*(self.bounds[i][0][1]-self.bounds[i][0][0])
                while (mean<=self.bounds[i][0][0]+(self.bounds[i][0][1]-self.bounds[i][0][0])/3) | (mean>=self.bounds[i][0][1]-(self.bounds[i][0][1]-self.bounds[i][0][0])/3): #if values are to close to bounds, resample
                    mean = np.random.rand()*(self.bounds[i][0][1]-self.bounds[i][0][0])
                standard_deviation = np.random.rand()*(self.bounds[i][1][1]-self.bounds[i][1][0])
                while (standard_deviation<=self.bounds[i][1][0]+(self.bounds[i][1][1]-self.bounds[i][1][0])/3) | (standard_deviation>=self.bounds[i][1][1]-(self.bounds[i][1][1]-self.bounds[i][1][0])/3): #if values are to close to bounds, resample
                    standard_deviation = np.random.rand()*(self.bounds[i][1][1]-self.bounds[i][1][0])
                phi.append((mean, standard_deviation))
        if len(phi)==0:
            raise NotImplementedError('Initialization for phi not found.')
        return phi

    def __set_phiBounds(self, phi_bounds):
        self.bounds = []
        for i in range(self.n_paramsToBeRandomized):
            lower = phi_bounds[i*4]
            upper = phi_bounds[i*4+1]
            mean = (lower, upper)
            lower = phi_bounds[i*4+2]
            upper = phi_bounds[i*4+3]
            standard_deviation = (lower, upper)
            self.bounds.append((mean, standard_deviation))

    def __set_searchSpace_bounds(self):
        self.searchSpace_bounds = []
        for i in range(self.n_paramsToBeRandomized):
            lower = 0.001
            upper = 12
            self.searchSpace_bounds.append((lower, upper))

    def __set_length_normalized_space(self, l_norm_space):
        self.length_normalized_space = l_norm_space

    def __get_model(self, algorithm, env):
        model = None
        if algorithm=='PPO':
            model = PPO(MlpPolicy, env)
            #model = PPO(MlpPolicy, env, verbose = 1) to print info during training
        if algorithm=='TRPO':
            model = TRPO(MlpPolicy, env)
            #model = TRPO(MlpPolicy, env, verbose = 1) to print info during training
        if model is None:
            raise NotImplementedError('Training algorithm not found')
        return model

    def __compute_discrepancy(self, tau_real, tau_sim, T_first):
        error_msg = ''
        obj = None

        obs_dim = tau_sim.shape[1]
        horizon_diff = tau_sim.shape[0]-tau_real.shape[0]
        if T_first=='max':
            if horizon_diff!=0:
                queue = np.zeros((abs(horizon_diff), obs_dim))
            if horizon_diff>0:
                tau_real = np.concatenate((tau_real, queue))
            if horizon_diff<0:
                tau_sim = np.concatenate((tau_sim, queue))
            obj = 0
        if T_first=='min':
            if horizon_diff>0:
                tau_sim = tau_sim[:tau_real.shape[0]]
            if horizon_diff<0:
                tau_real = tau_real[:tau_sim.shape[0]]
            obj = 0
        if T_first.split(":")[0]=='fixed':
            try:
                T_first_value = T_first.split(":")[1]
                try:
                    T_first_value = int(T_first.split(":")[1])
                    if (T_first_value>tau_real.shape[0]) | (T_first_value>tau_sim.shape[0]):
                        error_msg = ' (T_first value not compatible)'
                    else:
                        tau_sim = tau_sim[:T_first_value]
                        tau_real = tau_real[:T_first_value]
                        obj = 0
                except ValueError:
                    error_msg = ' (Incorrect sintax)'
            except IndexError:
                error_msg = ' (Incorrect sintax)'
        if obj is None:
            raise NotImplementedError('T_first value not found' + error_msg)
        diff = tau_sim - tau_real
        dimensions_ImportanceWeights = np.array(self.dimensions_ImportanceWeights)
        diff = dimensions_ImportanceWeights*diff
        l1Norm = np.linalg.norm(diff, ord=1, axis=1)
        l2Norm = np.linalg.norm(diff, ord=2, axis=1)
        l1_weight = self.norms_weights[0]
        l2_weight = self.norms_weights[1]
        obj = l1_weight*np.sum(l1Norm) + l2_weight*np.sum(l2Norm)
        #print(obj)
        return obj

    def __objective_function(self, params, normalize, logspace, model, tau_real, T_first):
        if normalize:
            den_params = []
            for i in range(int(len(params)/2)):
                mean = params[i*2]
                standard_deviation = params[i*2+1]
                mean = self.__denormalize_value(mean, self.bounds[i][0][0], self.bounds[i][0][1], 0, self.length_normalized_space)
                if logspace:
                    standard_deviation = self.__denormalize_value_log(standard_deviation, self.bounds[i][1][0], self.bounds[i][1][1], 0, self.length_normalized_space)
                else:
                    standard_deviation = self.__denormalize_value(standard_deviation, self.bounds[i][1][0], self.bounds[i][1][1], 0, self.length_normalized_space)
                den_params.append(mean)
                den_params.append(standard_deviation)
            params = den_params

        samples = []
        for i in range(int(len(params)/2)):
            mean = params[i*2]
            standard_deviation = params[i*2+1]
            sample = np.random.normal(mean,standard_deviation,1).astype(float)
            while (sample<self.searchSpace_bounds[i][0]) | (sample>self.searchSpace_bounds[i][1]):
                sample = np.random.normal(mean,standard_deviation,1).astype(float) #resampling
            samples.append(sample)

        #Collect 1 rollout in simulation
        env = gym.make(self.source)
        env.set_random_parametersBySamples(samples[0], samples[1], samples[2], samples[3])
        traj_obs = []
        env.reset()
        env.set_mujoco_state(self.starting_state)
        obs = env.get_mujoco_current_state() #'.get_mujoco_current_state()' not necessary, only to check. otherwise: obs = self.starting_state
        traj_obs.append(obs)
        cum_reward = 0
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            traj_obs.append(obs)
            cum_reward += reward
        #print(cum_reward)
        tau_sim = np.array(traj_obs)
        env.close()

        disc = self.__compute_discrepancy(tau_real, tau_sim, T_first)
        return disc

    def __normalize_value(self, value, original_low, original_upper, normal_low, normal_upper):
        original_length = original_upper-original_low
        normal_length = normal_upper-normal_low
        factor = normal_length/original_length
        original_offset = value-original_low
        normalized_value = (original_offset)*factor+normal_low
        return normalized_value

    def __denormalize_value(self, normalized_value, original_low, original_upper, normal_low, normal_upper):
        original_length = original_upper-original_low
        normal_length = normal_upper-normal_low
        factor = original_length/normal_length
        normal_offset = normalized_value-normal_low
        value = (normal_offset)*factor+original_low
        return value

    def __normalize_value_log(self, value, original_low, original_upper, normal_low, normal_upper):
        original_length = math.log(original_upper)-math.log(original_low)
        normal_length = normal_upper-normal_low
        factor = normal_length/original_length
        original_offset = math.log(value)-math.log(original_low)
        normalized_value = (original_offset)*factor+normal_low
        return normalized_value

    def __denormalize_value_log(self, normalized_value, original_low, original_upper, normal_low, normal_upper):
        # it's the inverse function of '__normalize_value_log', after applying properties of logarithms
        original_length = math.log(original_upper)-math.log(original_low)
        normal_length = normal_upper-normal_low
        factor = original_length/normal_length
        normal_offset = normalized_value-normal_low
        value = (normal_offset)*factor+math.log(original_low)
        return math.exp(value)
