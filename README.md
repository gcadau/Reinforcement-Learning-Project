## SimOpt algorithm.
    main.py file contains the code needed to launch several iterations of the SimOpt algorithm, to get the optimal distribution found and to use it to train an agent in the source target and test its performances on the target environment.
    
    To launch it:
    python3 main.py --device <device> --training_algorithm <training_algorithm> --initialPhi <initialPhi> --normalize --logspace --budget <budget> --n_iterations <n_iterations> --T_first <T_first> ----algorithm_parameters_filePath <filePath> --episodes <episodes> --render
    Possible arguments.
    <device>: network device [cpu, cuda], default='cpu'
    <training_algorithm>: training algorithm [PPO, TRPO], default='PPO'
    <initialPhi>: initial values for phi [fixed, random], default='fixed'
    --normalize: normalize dynamics parameters search space to [0,n] (n depends on the implementation), default=False
    --logspace: use a log space for variances (makes senses only if 'normalize' is set to True), default=False
    <budget>: Number of evaluations in the optimization problem (i.e.: number of samples from the distribution), default=1000
    <n_iterations>: number of iterations in SimOpt algorithm, default=1
    <T_first>: T-first value in discrepancy function [max, min, fixed:<number>], default='max'
    <filePath>: path of the file with the values of the algorithm parameters, default=None
    <episodes>: number of test episodes, default=50
    --render': render the simulator, default=False
        If the path of the file is not passed as an argument, the defaults parameters values are considered.
    Format of the file.
    <parameter name>:<parameters values separated by a single space>\n
    ...
    <parameter name>:<parameters values separated by a single space>
       -- If <parameter name> is missing for a specific parameter, the default values are considered for that parameter. 
       -- <parameter name> possible choices: <phi_initial_values>, <phi_bounds>, <length_normalized_space>, <importance_weights>, <norms_weights>.

    default values:
      phi initial values:4.5 1 2.8 1 4.5 1
      phi bounds:0.7 8.5 0.00001 2 0.7 8.5 0.00001 2 0.7 8.5 0.00001 2
      length normalized space:4
      importance weights:1 1 1 1 1 1 1 1 1 1 1
      norms weights:1 1
