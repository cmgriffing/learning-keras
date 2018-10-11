import gym
import numpy

env = gym.make('Skiing-v0')
numpy.random.seed(123)
env.seed(123)

for i_episode in range(300):
    observation = env.reset()
    for t in range(5000):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

        if done:
            break
