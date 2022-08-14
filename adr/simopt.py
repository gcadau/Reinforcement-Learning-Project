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

    def __init__(self):

        return

    def distribution_optimization(self, training_algorithm, initPhi, normalize, logspace, budget, n_iterations, T_first):

        env_source = gym.make('CustomHopper-source-v0')
        self.n_paramsToBeRandomized = len(env_source.get_parametersToBeRandomized())
        env_source.close()
        phi0 = self.__set_initialPhi(initPhi)

        self.__set_phiBounds()
        self.__set_searchSpace_bounds()


        phi = phi0
        for i in range(n_iterations):
            env_source = gym.make('CustomHopper-source-v0')
            env_source.reset()
            model = self.__get_model(training_algorithm, env_source)
            for i in range(100): #####
                env_source.set_random_parameters(phi)
                print(env_source.get_parameters(), 'for:' , i)
                model.learn(total_timesteps=10000)
            model.save("model_ppo.mdl")
            env_source.close()

            #Collect 1 rollout in real word
            env_target = gym.make('CustomHopper-target-v0')
            traj_obs = []
            obs = env_target.reset()
            traj_obs.append(obs)
            train_reward = 0
            done = False
            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = env_target.step(action)
                traj_obs.append(obs)
                train_reward += reward
            #print(train_reward)
            tau_real = np.array(traj_obs)
            env_target.close()

            searchSpace = []
            for i in range(self.n_paramsToBeRandomized):
                if normalize:
                    pass # TODO:
                    if logspace:
                        pass # TODO:
                    else:
                        pass # TODO:
                else:
                    mean = ng.p.Scalar(init=phi[i][0]).set_bounds(lower=self.bounds[i][0][0], upper=self.bounds[i][0][1])
                    standard_deviation = ng.p.Scalar(init=phi[i][1]).set_bounds(lower=self.bounds[i][1][0], upper=self.bounds[i][1][1])
                searchSpace.append(mean)
                searchSpace.append(standard_deviation)

            params = ng.p.Tuple(*searchSpace)
            instrumentation = ng.p.Instrumentation(params=params, normalize=normalize, model=model, tau_real=tau_real, T_first=T_first)
            cmaES_optimizer = ng.optimizers.CMA(parametrization=instrumentation, budget=budget)
            recommendation = cmaES_optimizer.minimize(self.__objective_function)

            #get __objective_function value, recommended ## TODO: if normalize
            #print(recommendation)
            print(recommendation.value[1]['params'], self.__objective_function(**recommendation.kwargs))

            phi_optim = recommendation.value[1]['params']
            phi = []
            for i in range(int(len(phi_optim)/2)):
                mean = phi_optim[i*2]
                standard_deviation = phi_optim[i*2+1]
                phi.append((mean, standard_deviation))



    def __set_initialPhi(self, initPhi):
        phi = []
        if initPhi=='fixed':
            mean = 4.5
            standard_deviation = 1
            phi.append((mean, standard_deviation))
            mean = 2.8
            standard_deviation = 1
            phi.append((mean, standard_deviation))
            mean = 4.5
            standard_deviation = 1
            phi.append((mean, standard_deviation))
        if initPhi=='random':
            for i in range(self.n_paramsToBeRandomized):
                #mean =
                #standard_deviation =
                phi.append((mean, standard_deviation))
        if len(phi)==0:
            raise NotImplementedError('Initialization for phi not found.')
        return phi

    def __set_phiBounds(self):
        self.bounds = []
        for i in range(self.n_paramsToBeRandomized):
            lower = 0.7
            upper = 8.5
            mean = (lower, upper)
            lower = 0.00001
            upper = 2
            standard_deviation = (lower, upper)
            self.bounds.append((mean, standard_deviation))

    def __set_searchSpace_bounds(self):
        self.searchSpace_bounds = []
        for i in range(self.n_paramsToBeRandomized):
            lower = 0.001
            upper = 12
            self.searchSpace_bounds.append((lower, upper))

    def __get_model(self, algorithm, env):
        model = None
        if algorithm=='PPO':
            model = PPO(MlpPolicy, env, verbose = 1)
        if algorithm=='TRPO':
            model = TRPO(MlpPolicy, env, verbose = 1)
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
                        error_msg = ' (T_first_value not compatible)'
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
        dimensions_ImportanceWeights = np.ones((obs_dim,))
        diff = dimensions_ImportanceWeights*diff
        l1Norm = np.linalg.norm(diff, ord=1, axis=1)
        l2Norm = np.linalg.norm(diff, ord=2, axis=1)
        #print(l1Norm)
        #print(l2Norm)
        l1_weight = 1
        l2_weight = 1
        obj = l1_weight*np.sum(l1Norm) + l2_weight*np.sum(l2Norm)
        #print(obj)
        return obj

    def __objective_function(self, params, normalize, model, tau_real, T_first):
        if normalize:
            pass# TODO: denormalizeBounds

        samples = []
        for i in range(int(len(params)/2)):
            mean = params[i*2]
            standard_deviation = params[i*2+1]
            sample = np.random.normal(mean,standard_deviation,1).astype(float)
            while (sample<self.searchSpace_bounds[i][0]) | (sample>self.searchSpace_bounds[i][1]):
                sample = np.random.normal(mean,standard_deviation,1).astype(float) #resampling
            samples.append(sample)

        #Collect 1 rollout in simulation
        env = gym.make('CustomHopper-source-v0')
        env.set_random_parametersBySamples(samples[0], samples[1], samples[2])
        print(env.get_parameters())
        traj_obs = []
        obs = env.reset()
        traj_obs.append(obs)
        train_reward = 0
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            traj_obs.append(obs)
            #env.render()
            train_reward += reward
        #print(train_reward)
        tau_sim = np.array(traj_obs)
        env.close()

        disc = self.__compute_discrepancy(tau_real, tau_sim, T_first)
        return disc
