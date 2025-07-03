#!/usr/bin/env python3
"""
Train a DQN agent to play Atari Breakout using keras-rl2 and gymnasium
"""

import numpy as np
import gymnasium as gym
from gymnasium.wrappers import StepAPICompatibility
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input, Permute, Reshape
from tensorflow.keras.optimizers.legacy import Adam

# Fix keras-rl2 compatibility issue
import tensorflow.keras
tensorflow.keras.__version__ = '2.15.0'

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory


class GymnasiumWrapper(gym.Wrapper):
    """
    Wrapper to make Gymnasium compatible with keras-rl2
    """
    def __init__(self, env):
        super().__init__(env)
        
    def reset(self, **kwargs):
        """Reset environment and return only observation (keras-rl expects single return)"""
        obs, info = self.env.reset(**kwargs)
        return obs
    
    def step(self, action):
        """Step environment and return old API format (obs, reward, done, info)"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info


def create_model(input_shape, nb_actions):
    """
    Create a CNN model for Atari games
    """
    from tensorflow.keras.layers import Input, Permute, Reshape
    from tensorflow.keras.models import Model

    # Create input layer - keras-rl2 will pass (batch, window_length, *input_shape)
    # So we expect (batch, 1, 4, 84, 84) and need to reshape to (batch, 84, 84, 4)
    input_layer = Input(shape=(1,) + input_shape)

    # Remove the window dimension and permute to channels-last format
    x = Reshape(input_shape)(input_layer)  # (batch, 4, 84, 84)
    x = Permute((2, 3, 1))(x)  # (batch, 84, 84, 4)

    # Convolutional layers for image processing
    x = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(x)
    x = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(x)

    # Flatten and dense layers
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    output = Dense(nb_actions, activation='linear')(x)

    model = Model(inputs=input_layer, outputs=output)
    return model


def preprocess_observation(obs):
    """
    Preprocess observation for training
    """
    # Convert to grayscale and resize
    obs = np.dot(obs[...,:3], [0.299, 0.587, 0.114])
    obs = obs.astype(np.uint8)
    return obs


def main():
    """
    Main training function
    """
    # Create environment with frameskip disabled
    env = gym.make('ALE/Breakout-v5', render_mode='rgb_array', frameskip=1)
    
    # Apply preprocessing wrappers
    env = gym.wrappers.AtariPreprocessing(env, 
                                         noop_max=30,
                                         frame_skip=4,
                                         screen_size=84,
                                         terminal_on_life_loss=True,
                                         grayscale_obs=True,
                                         grayscale_newaxis=False,
                                         scale_obs=True)
    
    # Stack frames
    env = gym.wrappers.FrameStack(env, 4)
    
    # Apply compatibility wrapper
    env = GymnasiumWrapper(env)
    
    # Get environment info
    nb_actions = env.action_space.n
    input_shape = env.observation_space.shape
    
    print(f"Environment: {env}")
    print(f"Number of actions: {nb_actions}")
    print(f"Input shape: {input_shape}")
    
    # Create the model
    model = create_model(input_shape, nb_actions)
    print(model.summary())
    
    # Configure memory
    memory = SequentialMemory(limit=1000000, window_length=1)
    
    # Configure policy
    policy = EpsGreedyQPolicy(eps=1.0)
    
    # Create DQN agent
    dqn = DQNAgent(model=model,
                   nb_actions=nb_actions,
                   memory=memory,
                   nb_steps_warmup=50000,
                   target_model_update=10000,
                   policy=policy,
                   gamma=0.99,
                   batch_size=32,
                   train_interval=4,
                   delta_clip=1.0)
    
    # Compile agent
    dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])
    
    # Train the agent
    print("Starting training...")
    history = dqn.fit(env, 
                     nb_steps=2000000,
                     visualize=False,
                     verbose=1,
                     nb_max_episode_steps=10000,
                     log_interval=10000)
    
    # Save the policy network
    dqn.save_weights('policy.h5', overwrite=True)
    print("Training completed! Policy saved as policy.h5")
    
    # Close environment
    env.close()


if __name__ == '__main__':
    main()
