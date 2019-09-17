import gym
import numpy as np
import time, pickle, os

env = gym.make('FrozenLake-v0')

epsilon = 0
total_episodes = 1000
max_steps = 20

lr_rate = 0.81
gamma = 0.96

Q = np.zeros((env.observation_space.n, env.action_space.n))
    
def phil_reward(state):
	reward = [0,0.1,0.1,0.1,0.1,-0.1,0.1,-0.1,0.1,0.1,0.1,-0.1,-0.1,0.1,0.1,100]
	return reward[state]

def choose_action(state):
    action=0
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])
    return action

def learn(state, state2, reward, action):
    predict = Q[state, action]
    target = reward + gamma * np.max(Q[state2, :])
    Q[state, action] = Q[state, action] + lr_rate * (target - predict)

# Start
for episode in range(total_episodes):
    print('ep: ',episode)
    state = env.reset()
    t = 0
    
    while t < max_steps:
        env.render()

        action = choose_action(state)  
        print('action chosen: ', action)
        state2, reward, done, info = env.step(action)  

        reward = phil_reward(state2)

        if (state == state2):
        	reward = -0.1

        print('reward: ', reward)
        print('done?: ', done)
        learn(state, state2, reward, action)
        print('old state: ',state)
        print('new state: ',state2)
        print('info: ', info)
        # print('Q: ',Q)
        state = state2

        t += 1
       
        if done:
            break

        time.sleep(0.1)

    if episode % 100 == 0:
      	with open("savedQ/frozenLake_qTable_"+str(episode)+".pkl", 'wb') as f:
    	    pickle.dump(Q, f)

print(Q)

