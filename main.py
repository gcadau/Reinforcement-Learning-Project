import argparse

import torch
import torch.nn
import gym
from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from sb3_contrib import TRPO

from adr.simopt import SimOpt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--training_algorithm', default='PPO', type=str, help='training algorithm [PPO, TRPO]')
    parser.add_argument('--initialPhi', default='fixed', type=str, help='initial values for phi [fixed, random]')
    parser.add_argument('--normalize', default=False, action='store_true', help='normalize dynamics parameters search space to [0,n] (n depends on the implementation)')
    parser.add_argument('--logspace', default=False, action='store_true', help="use a log space for standard deviations (makes senses only if 'normalize' is set to True)")
    parser.add_argument('--budget', default=1000, type=int, help='Number of evaluations in the optimization problem (i.e.: number of samples from the distribution)')
    parser.add_argument('--n_iterations', default=1, type=int, help='Number of iterations in SimOpt algorithm')
    parser.add_argument('--T_first', default='max', type=str, help='T-first value in discrepancy function [max, min, fixed:<number>]')
    parser.add_argument('--algorithm_parameters_filePath', default=None, type=str, help='Path of the file with the values ​​of the algorithm parameters')
    parser.add_argument('--episodes', default=50, type=int, help='Number of test episodes')
    parser.add_argument('--render', default=False, action='store_true', help='Render the simulator')

    return parser.parse_args()

def get_model(algorithm, env):
    model = None
    if algorithm=='PPO':
        model = PPO(MlpPolicy, env, verbose = 1)
    if algorithm=='TRPO':
        model = TRPO(MlpPolicy, env, verbose = 1)
    if model is None:
        raise NotImplementedError('Training algorithm not found')
    return model

def main():

    args = parse_args()
    filePath = args.algorithm_parameters_filePath
    if filePath is None: #default values
        phi_initial_values = [4.5, 1, 2.8, 1, 4.5, 1]
        phi_bounds = [0.7, 8.5, 0.00001, 2, 0.7, 8.5, 0.00001, 2, 0.7, 8.5, 0.00001, 2]
        length_normalized_space = 4
        importance_weights = 11*[float(1)]
        norms_weights = 2*[float(1)]
    else:
        with open(filePath) as f:
            params = {l.split(":")[0]: l.split(":")[1] for l in f.read().splitlines()}
            try:
                phi_initial_values = [float(i) for i in params['phi initial values'].split(" ")]
            except KeyError:
                phi_initial_values = [4.5, 1, 2.8, 1, 4.5, 1] #default values
            try:
                phi_bounds = [float(i) for i in params['phi bounds'].split(" ")]
            except KeyError:
                phi_bounds = [0.7, 8.5, 0.00001, 2, 0.7, 8.5, 0.00001, 2, 0.7, 8.5, 0.00001, 2] #default values
            try:
                length_normalized_space = int(params['length normalized space'])
            except KeyError:
                length_normalized_space = 4 #default values
            try:
                importance_weights = [float(i) for i in params['importance weights'].split(" ")]
            except KeyError:
                importance_weights = 11*[float(1)] #default values
            try:
                norms_weights = [int(i) for i in params['norms weights'].split(" ")]
            except KeyError:
                norms_weights = 2*[float(1)] #default values


    source = 'CustomHopper-source-v0'
    target = 'CustomHopper-target-v0'

    simopt = SimOpt(source=source, target=target)
    simopt.distribution_optimization(training_algorithm=args.training_algorithm, initPhi=args.initialPhi, normalize=args.normalize, logspace=args.logspace, budget=args.budget, n_iterations=args.n_iterations, T_first=args.T_first, phi_initial_values=phi_initial_values, phi_bounds=phi_bounds, l_norm_space=length_normalized_space, importance_weights=importance_weights, norms_weights=norms_weights)

    env_source = gym.make(source)
    print("Initial dynamics parameters")
    print(env_source.get_parameters())
    print("\n")

    print("Optimal dynamics parameters (each element (mean, standard_deviation) refers to a gaussian distribution)")
    phi_optimal_values = simopt.get_optimal_values()
    print(phi_optimal_values)
    print("\n")
    print("Optimal value of the discrepancy function")
    optimum = simopt.get_optimum()
    print(optimum)
    print("\n")

    env_source.reset()
    model = get_model(args.training_algorithm, env_source)
    for i in range(100):
        env_source.set_random_parameters(phi_optimal_values)
        model.learn(total_timesteps=10000)
    #model.save("model.mdl")
    env_source.close()

    env_target = gym.make(target)
    for episode in range(args.episodes):
        obs = env_target.reset()
        cum_reward = 0
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env_target.step(action)
            if args.render:
                env_target.render()
            cum_reward += reward
            print(f"Episode: {episode} | Return: {cum_reward}")
    env_target.close()




if __name__ == '__main__':
    main()
