import numpy as np

class Rewards:
    
    @staticmethod
    def waiting_count(eng,intersection_state,roads,summary):
        waiting = eng.get_lane_waiting_vehicle_count()
        sum = 0
        for lane in roads:
            sum += waiting[lane]
        return sum
    
    @staticmethod
    def avg_travel_time(eng,intersection_state,roads,summary):
        return eng.get_average_travel_time()
    
    @staticmethod
    def delay_from_opt(eng,intersection_state,roads,summary):
        v_num = 0
        d = np.zeros(len(roads))
        count_dict = eng.get_lane_vehicle_count()
        speed_dict = eng.get_vehicle_speed()
        lane_v = eng.get_lane_vehicles()
        for idx,lane in enumerate(roads):
            if lane in count_dict and lane in lane_v:
                for vehicle_id in lane_v[lane]:
                    d[idx]+= max(0,1 - speed_dict[vehicle_id] / summary['maxSpeed'])
                v_num += count_dict[lane]
        return np.sum(d) / v_num
    
    @staticmethod
    def exp_delay_from_opt(eng,intersection_state,roads,summary):
        C = 1.45
        v_num = 0
        val = np.zeros(len(roads))
        count_dict = eng.get_lane_vehicle_count()
        speed_dict = eng.get_vehicle_speed()
        lane_v = eng.get_lane_vehicles()
        dist_v = eng.get_vehicle_distance()
        for idx,lane in enumerate(roads):
            if lane in count_dict and lane in lane_v:
                for vehicle_id in lane_v[lane]:
                    leader = eng.get_leader(vehicle_id) 
                    w = dist_v[vehicle_id]
                    w -= dist_v[vehicle_id] if leader != "" else 0
                    d = max(0,1 -speed_dict[vehicle_id] / summary['maxSpeed'])
                    val[idx] += C ** (w*d)
                v_num += count_dict[lane]
        return (np.sum(val)-1)/ v_num
    
    @staticmethod
    def get(name):
        if name ==  'waiting_count':
            return Rewards.waiting_count
        elif name == 'avg_travel_time':
            return Rewards.avg_travel_time
        elif name == 'delay_from_opt':
            return Rewards.delay_from_opt
        elif name == 'exp_delay_from_opt':
            return Rewards.exp_delay_from_opt

class AlgorithemsConfig:
    
    # Adds the following updates to the (base) `Trainer` config in
    # rllib/agents/trainer.py (`COMMON_CONFIG` dict).
    PPO = {
        # Should use a critic as a baseline (otherwise don't use value baseline;
        # required for using GAE).
        "use_critic": True,
        # If true, use the Generalized Advantage Estimator (GAE)
        # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
        "use_gae": True,
        # The GAE (lambda) parameter.
        "lambda": 1.0,
        # Initial coefficient for KL divergence.
        "kl_coeff": 0.2,
        # Size of batches collected from each worker.
        "rollout_fragment_length": 200,
        # Number of timesteps collected for each SGD round. This defines the size
        # of each SGD epoch.
        "train_batch_size": 4000,
        # Total SGD batch size across all devices for SGD. This defines the
        # minibatch size within each epoch.
        "sgd_minibatch_size": 128,
        # Whether to shuffle sequences in the batch when training (recommended).
        "shuffle_sequences": True,
        # Number of SGD iterations in each outer loop (i.e., number of epochs to
        # execute per train batch).
        "num_sgd_iter": 30,
        # Stepsize of SGD.
        "lr": 5e-5,
        # Learning rate schedule.
        "lr_schedule": None,
        # Coefficient of the value function loss. IMPORTANT: you must tune this if
        # you set vf_share_layers=True inside your model's config.
        "vf_loss_coeff": 1.0,
        "model": {
            # Share layers for value function. If you set this to True, it's
            # important to tune vf_loss_coeff.
            "vf_share_layers": False,
        },
        # Coefficient of the entropy regularizer.
        "entropy_coeff": 0.0,
        # Decay schedule for the entropy regularizer.
        "entropy_coeff_schedule": None,
        # PPO clip parameter.
        "clip_param": 0.3,
        # Clip param for the value function. Note that this is sensitive to the
        # scale of the rewards. If your expected V is large, increase this.
        "vf_clip_param": 10.0,
        # If specified, clip the global norm of gradients by this amount.
        "grad_clip": None,
        # Target value for KL divergence.
        "kl_target": 0.01,
        # Whether to rollout "complete_episodes" or "truncate_episodes".
        "batch_mode": "truncate_episodes",
    }

    # Adds the following updates to the (base) `Trainer` config in
    # rllib/agents/trainer.py (`COMMON_CONFIG` dict).
    SAC = {
        # === Model ===
        # Use two Q-networks (instead of one) for action-value estimation.
        # Note: Each Q-network will have its own target network.
        "twin_q": True,
        # Model options for the Q network(s). These will override MODEL_DEFAULTS.
        # The `Q_model` dict is treated just as the top-level `model` dict in
        # setting up the Q-network(s) (2 if twin_q=True).
        # That means, you can do for different observation spaces:
        # obs=Box(1D) -> Tuple(Box(1D) + Action) -> concat -> post_fcnet
        # obs=Box(3D) -> Tuple(Box(3D) + Action) -> vision-net -> concat w/ action
        #   -> post_fcnet
        # obs=Tuple(Box(1D), Box(3D)) -> Tuple(Box(1D), Box(3D), Action)
        #   -> vision-net -> concat w/ Box(1D) and action -> post_fcnet
        # You can also have SAC use your custom_model as Q-model(s), by simply
        # specifying the `custom_model` sub-key in below dict (just like you would
        # do in the top-level `model` dict.
        "Q_model": {
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "relu",
            "post_fcnet_hiddens": [],
            "post_fcnet_activation": None,
            "custom_model": None,  # Use this to define custom Q-model(s).
            "custom_model_config": {},
        },
        # Model options for the policy function (see `Q_model` above for details).
        # The difference to `Q_model` above is that no action concat'ing is
        # performed before the post_fcnet stack.
        "policy_model": {
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "relu",
            "post_fcnet_hiddens": [],
            "post_fcnet_activation": None,
            "custom_model": None,  # Use this to define a custom policy model.
            "custom_model_config": {},
        },
        # Actions are already normalized, no need to clip them further.
        "clip_actions": False,

        # === Learning ===
        # Update the target by \tau * policy + (1-\tau) * target_policy.
        "tau": 5e-3,
        # Initial value to use for the entropy weight alpha.
        "initial_alpha": 1.0,
        # Target entropy lower bound. If "auto", will be set to -|A| (e.g. -2.0 for
        # Discrete(2), -3.0 for Box(shape=(3,))).
        # This is the inverse of reward scale, and will be optimized automatically.
        "target_entropy": "auto",
        # N-step target updates. If >1, sars' tuples in trajectories will be
        # postprocessed to become sa[discounted sum of R][s t+n] tuples.
        "n_step": 1,
        # Number of env steps to optimize for before returning.
        "timesteps_per_iteration": 100,

        # === Replay buffer ===
        # Size of the replay buffer (in time steps).
        "replay_buffer_config": {
            "type": "MultiAgentReplayBuffer",
            "capacity": int(1e6),
        },
        # Set this to True, if you want the contents of your buffer(s) to be
        # stored in any saved checkpoints as well.
        # Warnings will be created if:
        # - This is True AND restoring from a checkpoint that contains no buffer
        #   data.
        # - This is False AND restoring from a checkpoint that does contain
        #   buffer data.
        "store_buffer_in_checkpoints": False,
        # If True prioritized replay buffer will be used.
        "prioritized_replay": False,
        "prioritized_replay_alpha": 0.6,
        "prioritized_replay_beta": 0.4,
        "prioritized_replay_eps": 1e-6,
        "prioritized_replay_beta_annealing_timesteps": 20000,
        "final_prioritized_replay_beta": 0.4,
        # Whether to LZ4 compress observations
        "compress_observations": False,

        # The intensity with which to update the model (vs collecting samples from
        # the env). If None, uses the "natural" value of:
        # `train_batch_size` / (`rollout_fragment_length` x `num_workers` x
        # `num_envs_per_worker`).
        # If provided, will make sure that the ratio between ts inserted into and
        # sampled from the buffer matches the given value.
        # Example:
        #   training_intensity=1000.0
        #   train_batch_size=250 rollout_fragment_length=1
        #   num_workers=1 (or 0) num_envs_per_worker=1
        #   -> natural value = 250 / 1 = 250.0
        #   -> will make sure that replay+train op will be executed 4x as
        #      often as rollout+insert op (4 * 250 = 1000).
        # See: rllib/agents/dqn/dqn.py::calculate_rr_weights for further details.
        "training_intensity": None,

        # === Optimization ===
        "optimization": {
            "actor_learning_rate": 3e-4,
            "critic_learning_rate": 3e-4,
            "entropy_learning_rate": 3e-4,
        },
        # If not None, clip gradients during optimization at this value.
        "grad_clip": None,
        # How many steps of the model to sample before learning starts.
        "learning_starts": 1500,
        # Update the replay buffer with this many samples at once. Note that this
        # setting applies per-worker if num_workers > 1.
        "rollout_fragment_length": 1,
        # Size of a batched sampled from replay buffer for training.
        "train_batch_size": 256,
        # Update the target network every `target_network_update_freq` steps.
        "target_network_update_freq": 0,
        # Whether the loss should be calculated deterministically (w/o the
        # stochastic action sampling step). True only useful for cont. actions and
        # for debugging!
        "_deterministic_loss": False,
        # Use a Beta-distribution instead of a SquashedGaussian for bounded,
        # continuous action spaces (not recommended, for debugging only).
        "_use_beta_distribution": False,
    }
  
    
    A3C = {
        # Should use a critic as a baseline (otherwise don't use value baseline;
        # required for using GAE).
        "use_critic": True,
        # If true, use the Generalized Advantage Estimator (GAE)
        # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
        "use_gae": True,
        # Size of rollout batch
        "rollout_fragment_length": 10,
        # GAE(gamma) parameter
        "lambda": 1.0,
        # Max global norm for each gradient calculated by worker
        "grad_clip": 40.0,
        # Learning rate
        "lr": 0.0001,
        # Learning rate schedule
        "lr_schedule": None,
        # Value Function Loss coefficient
        "vf_loss_coeff": 0.5,
        # Entropy coefficient
        "entropy_coeff": 0.01,
        # Entropy coefficient schedule
        "entropy_coeff_schedule": None,
        # Min time (in seconds) per reporting.
        # This causes not every call to `training_iteration` to be reported,
        # but to wait until n seconds have passed and then to summarize the
        # thus far collected results.
        "min_time_s_per_reporting": 5,
        # Workers sample async. Note that this increases the effective
        # rollout_fragment_length by up to 5x due to async buffering of batches.
        "sample_async": True,
        # Use the Trainer's `training_iteration` function instead of `execution_plan`.
        # Fixes a severe performance problem with A3C. Setting this to True leads to a
        # speedup of up to 3x for a large number of workers and heavier
        # gradient computations (e.g. ray/rllib/tuned_examples/a3c/pong-a3c.yaml)).
        "_disable_execution_plan_api": True,
    }
        
    
class ModelConfig:
    MODEL_DEFAULTS = {
        # Experimental flag.
        # If True, try to use a native (tf.keras.Model or torch.Module) default
        # model instead of our built-in ModelV2 defaults.
        # If False (default), use "classic" ModelV2 default models.
        # Note that this currently only works for:
        # 1) framework != torch AND
        # 2) fully connected and CNN default networks as well as
        # auto-wrapped LSTM- and attention nets.
        "_use_default_native_models": False,
        # Experimental flag.
        # If True, user specified no preprocessor to be created
        # (via config._disable_preprocessor_api=True). If True, observations
        # will arrive in model as they are returned by the env.
        "_disable_preprocessor_api": False,
        # Experimental flag.
        # If True, RLlib will no longer flatten the policy-computed actions into
        # a single tensor (for storage in SampleCollectors/output files/etc..),
        # but leave (possibly nested) actions as-is. Disabling flattening affects:
        # - SampleCollectors: Have to store possibly nested action structs.
        # - Models that have the previous action(s) as part of their input.
        # - Algorithms reading from offline files (incl. action information).
        "_disable_action_flattening": False,

        # === Built-in options ===
        # FullyConnectedNetwork (tf and torch): rllib.models.tf|torch.fcnet.py
        # These are used if no custom model is specified and the input space is 1D.
        # Number of hidden layers to be used.
        "fcnet_hiddens": [256, 256],
        # Activation function descriptor.
        # Supported values are: "tanh", "relu", "swish" (or "silu"),
        # "linear" (or None).
        "fcnet_activation": "tanh",

        # VisionNetwork (tf and torch): rllib.models.tf|torch.visionnet.py
        # These are used if no custom model is specified and the input space is 2D.
        # Filter config: List of [out_channels, kernel, stride] for each filter.
        # Example:
        # Use None for making RLlib try to find a default filter setup given the
        # observation space.
        "conv_filters": None,
        # Activation function descriptor.
        # Supported values are: "tanh", "relu", "swish" (or "silu"),
        # "linear" (or None).
        "conv_activation": "relu",

        # Some default models support a final FC stack of n Dense layers with given
        # activation:
        # - Complex observation spaces: Image components are fed through
        #   VisionNets, flat Boxes are left as-is, Discrete are one-hot'd, then
        #   everything is concated and pushed through this final FC stack.
        # - VisionNets (CNNs), e.g. after the CNN stack, there may be
        #   additional Dense layers.
        # - FullyConnectedNetworks will have this additional FCStack as well
        # (that's why it's empty by default).
        "post_fcnet_hiddens": [],
        "post_fcnet_activation": "relu",

        # For DiagGaussian action distributions, make the second half of the model
        # outputs floating bias variables instead of state-dependent. This only
        # has an effect is using the default fully connected net.
        "free_log_std": False,
        # Whether to skip the final linear layer used to resize the hidden layer
        # outputs to size `num_outputs`. If True, then the last hidden layer
        # should already match num_outputs.
        "no_final_linear": False,
        # Whether layers should be shared for the value function.
        "vf_share_layers": True,

        # == LSTM ==
        # Whether to wrap the model with an LSTM.
        "use_lstm": False,
        # Max seq len for training the LSTM, defaults to 20.
        "max_seq_len": 20,
        # Size of the LSTM cell.
        "lstm_cell_size": 256,
        # Whether to feed a_{t-1} to LSTM (one-hot encoded if discrete).
        "lstm_use_prev_action": False,
        # Whether to feed r_{t-1} to LSTM.
        "lstm_use_prev_reward": False,
        # Whether the LSTM is time-major (TxBx..) or batch-major (BxTx..).
        "_time_major": False,

        # == Attention Nets (experimental: torch-version is untested) ==
        # Whether to use a GTrXL ("Gru transformer XL"; attention net) as the
        # wrapper Model around the default Model.
        "use_attention": False,
        # The number of transformer units within GTrXL.
        # A transformer unit in GTrXL consists of a) MultiHeadAttention module and
        # b) a position-wise MLP.
        "attention_num_transformer_units": 1,
        # The input and output size of each transformer unit.
        "attention_dim": 64,
        # The number of attention heads within the MultiHeadAttention units.
        "attention_num_heads": 1,
        # The dim of a single head (within the MultiHeadAttention units).
        "attention_head_dim": 32,
        # The memory sizes for inference and training.
        "attention_memory_inference": 50,
        "attention_memory_training": 50,
        # The output dim of the position-wise MLP.
        "attention_position_wise_mlp_dim": 32,
        # The initial bias values for the 2 GRU gates within a transformer unit.
        "attention_init_gru_gate_bias": 2.0,
        # Whether to feed a_{t-n:t-1} to GTrXL (one-hot encoded if discrete).
        "attention_use_n_prev_actions": 0,
        # Whether to feed r_{t-n:t-1} to GTrXL.
        "attention_use_n_prev_rewards": 0,

        # == Atari ==
        # Set to True to enable 4x stacking behavior.
        "framestack": True,
        # Final resized frame dimension
        "dim": 84,
        # (deprecated) Converts ATARI frame to 1 Channel Grayscale image
        "grayscale": False,
        # (deprecated) Changes frame to range from [-1, 1] if true
        "zero_mean": True,

        # === Options for custom models ===
        # Name of a custom model to use
        "custom_model": None,
        # Extra options to pass to the custom classes. These will be available to
        # the Model's constructor in the model_config field. Also, they will be
        # attempted to be passed as **kwargs to ModelV2 models. For an example,
        # see rllib/models/[tf|torch]/attention_net.py.
        "custom_model_config": {},
        # Name of a custom action distribution to use.
        "custom_action_dist": None,
        # Custom preprocessors are deprecated. Please use a wrapper class around
        # your environment instead to preprocess observations.
        "custom_preprocessor": None,
    }
    