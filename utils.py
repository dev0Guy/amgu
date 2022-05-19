import numpy as np

class Rewards:
    @staticmethod
    def waiting_count(eng,intersection_state,roads,summary,in_lanes,out_lanes):
        waiting = eng.get_lane_waiting_vehicle_count()  
        running = eng.get_lane_vehicle_count()
        sum = 0
        for lane in roads:
            if lane in in_lanes:
                sum +=  waiting[lane]
            else: 
                sum -= waiting[lane] + running[lane]
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
                    d[idx]+= max(0,1 -speed_dict[vehicle_id] / summary['maxSpeed'])
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
        "grad_clip": 10000.0,
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
        "_disable_execution_plan_api": False,
    }
    
class ModelConfig:
    
    FCN = {
        "_use_default_native_models": False,
        "_disable_preprocessor_api": False,
        "_disable_action_flattening": False,
        "fcnet_hiddens": [255,255,255],
        "fcnet_activation": "relu",
    }
    
    CNN = {
        "_use_default_native_models": False,
        "_disable_preprocessor_api": True,
        "_disable_action_flattening": False,
        "conv_filters": [[8,[3,72],1]],
        "conv_activation": "relu",
    }
    
    LSTM = {
        "use_lstm": True,
        "max_seq_len": 20,
        "lstm_cell_size": 256,
        "lstm_use_prev_action": True,
        "lstm_use_prev_reward": True,
        "_time_major": False,
    }
    
    
    
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
        