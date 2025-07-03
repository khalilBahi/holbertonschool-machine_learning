#!/usr/bin/env python3
"""
Play Atari Breakout using a trained DQN agent
"""

import numpy as np
import gymnasium as gym
from gymnasium.wrappers import StepAPICompatibility
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input, Permute, Reshape
from tensorflow.keras.optimizers.legacy import Adam
import time

# Fix keras-rl2 compatibility issue
import tensorflow.keras
tensorflow.keras.__version__ = '2.15.0'

from rl.agents.dqn import DQNAgent
from rl.policy import GreedyQPolicy
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
    Create a CNN model for Atari games (same architecture as training)
    """
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


def main():
    """
    Main playing function
    """
    # Create environment with human rendering and frameskip disabled
    env = gym.make('ALE/Breakout-v5', render_mode='human', frameskip=1)
    
    # Apply preprocessing wrappers (same as training)
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
    
    # Create the model (same architecture as training)
    model = create_model(input_shape, nb_actions)
    
    # Configure memory (minimal for playing)
    memory = SequentialMemory(limit=1000, window_length=1)
    
    # Configure greedy policy (no exploration)
    policy = GreedyQPolicy()
    
    # Create DQN agent
    dqn = DQNAgent(model=model,
                   nb_actions=nb_actions,
                   memory=memory,
                   nb_steps_warmup=0,
                   target_model_update=1,
                   policy=policy,
                   gamma=0.99,
                   batch_size=32,
                   train_interval=1,
                   delta_clip=1.0)
    
    # Compile agent
    dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])
    
    # Load the trained weights
    try:
        dqn.load_weights('policy.h5')
        print("Successfully loaded trained policy from policy.h5")
    except Exception as e:
        print(f"Error loading policy.h5: {e}")
        print("Make sure you have trained the agent first by running train.py")
        return
    
    # Play episodes
    print("Starting to play...")
    print("Press Ctrl+C to stop")
    
    try:
        # Test the agent for multiple episodes
        for episode in range(10):
            print(f"\nEpisode {episode + 1}")
            
            obs = env.reset()
            total_reward = 0
            steps = 0
            done = False
            
            while not done:
                # Agent selects action
                action = dqn.forward(obs)
                
                # Take action in environment
                obs, reward, done, info = env.step(action)
                total_reward += reward
                steps += 1
                
                # Add small delay to make it watchable
                time.sleep(0.02)
                
                # Break if episode is too long
                if steps > 10000:
                    break
            
            print(f"Episode {episode + 1} finished: {steps} steps, total reward: {total_reward}")
            
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    finally:
        # Close environment
        env.close()
        print("Environment closed")


if __name__ == '__main__':
    main()
