from solution.DDPG_Agent import Agent
from task import Task
import numpy as np
import matplotlib.pyplot as plt

graph_episodes = []
graph_rewards = []

# time limit of the episode
runtime = 4.

# POSSIBLE TASKS
#            init_pose     target_pos
# takeoff :  ( 0,0,0 )     ( 0,0,10 )
# landing :  ( 0,0,10 )    ( 0,0,0 )
# hover :    ( 0,0,10 )    ( 10,0,10 )

init_pose = np.array(
    [0., 0., 10.,
     0., 0., 0.])

target_pos = np.array([0., 0., 0.])


# initial velocities
init_velocities = np.array([0., 0., 0.])

# initial angle velocities
init_angle_velocities = np.array([0., 0., 0.])

# Setup
task = Task(init_pose=init_pose, init_velocities=init_velocities,
            init_angle_velocities=init_angle_velocities, runtime=runtime,
            state_size=init_pose.shape[0], target_pos=target_pos)

agent = Agent(task)


num_episodes = 500
for i_episode in range(1, num_episodes+1):
    state = agent.reset_episode()

    while True:
        action = agent.act(state)
        next_state, reward, done = task.step(action)
        agent.step(action=action, reward=reward, next_state=next_state, done=done)
        state = next_state
        if done:
            break

    if i_episode % 100 == 0:
        graph_episodes.append(i_episode / 100)
        graph_rewards.append(reward)
        print('episode', i_episode, 'reward', reward)


plt.figure(1, figsize=(15, 6))

'''
plt.subplot(131)
plt.plot(results['time'], results['x'], label='x_pos')
plt.plot(results['time'], results['y'], label='y_pos')
plt.plot(results['time'], results['z'], label='z_pos')
plt.legend()
plt.title('time_x_position')

plt.subplot(132)
plt.plot(results['time'], results['x_velocity'], label='x_hat')
plt.plot(results['time'], results['y_velocity'], label='y_hat')
plt.plot(results['time'], results['z_velocity'], label='z_hat')
plt.legend()
plt.title('time_x_velocity')
'''

plt.subplot(121)
plt.plot(graph_episodes, graph_rewards, label='reward')
plt.legend()
plt.title('time_x_rewards')

plt.savefig('graphs.png')
