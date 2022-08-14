import argparse
from adr.simopt import SimOpt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=100000, type=int, help='Number of training episodes') #da integrare
    parser.add_argument('--print-every', default=1, type=int, help='Print info every <> episodes') #changed #da integrare
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]') #da togliere?
    parser.add_argument('--training_algorithm', default='PPO', type=str, help='training algorithm [PPO, TRPO]')
    parser.add_argument('--initialPhi', default='fixed', type=str, help='initial values for phi [fixed, random]')
    parser.add_argument('--normalize', default=False, action='store_true', help='normalize parametes search space to [0,4]')
    parser.add_argument('--logspace', default=False, action='store_true', help="use a log space for standard deviations (makes senses only if 'normalize' is set to True)")
    parser.add_argument('--budget', default=1000, type=int, help='Number of evaluations in the optimization problem (i.e.: number of samples from the distribution)')
    parser.add_argument('--n_iterations', default=1, type=int, help='Number of iterations in SimOpt algorithm')
    parser.add_argument('--T_first', default='max', type=str, help='T-firtst value in discrepancy function [max, min, fixed:<number>]')

    return parser.parse_args()

def main():

    args = parse_args()



    #print('Action space:', env.action_space)
    #print('State space:', env.observation_space)
    #print('Dynamics parameters:', env.get_parameters())

    simopt = SimOpt()

    simopt.distribution_optimization(training_algorithm=args.training_algorithm, initPhi=args.initialPhi, normalize=args.normalize, logspace=args.logspace, budget=args.budget, n_iterations=args.n_iterations, T_first=args.T_first)

    #get opt values, recommend paramters (2 modi)




if __name__ == '__main__':
    main()
