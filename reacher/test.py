import gym
import time
import numpy as np

env = gym.make("Reacher-v2")
o=env.reset()

print(env.observation_space.high)
print(env.observation_space.low)

init_qpos = env.env.init_qpos
init_qvel = env.env.init_qvel

init_qpos[2] = .27
init_qpos[3] = .27

print(init_qpos)
print(init_qvel)

# time.sleep(50)

init_qpos[0] = 0 # need to look more closely at these ranges/limits
init_qpos[1] = 1

env.env.set_state(init_qpos, init_qvel)

for i in range(1000):
    a = env.action_space.sample()
    state, r, d, _ = env.step(a)

    env.render()
    time.sleep(15)
    print("-------------")

    # svec = env.env.state_vector()
    # deg = np.degrees(svec[:2]) % 360
    # print(np.degrees(svec[:2]))
    # print(svec[:2])
    # print(np.radians(deg))
    # print(np.isclose(np.cos(svec[:2]), np.cos(np.radians(deg))))
    # print(np.isclose(np.sin(svec[:2]), np.sin(np.radians(deg))))
    # print(np.cos(svec[:2]), np.cos(np.radians(deg)))

    # need to convert svec[0:2] to the range -pi, pi
    # np.unwrap([svec])

# let's say I want to construct a specific form for the state to learn on
# x/y pos of fingertip + state_vector[0:2] + state_vector[4:6]
# then how to discretize?
# what is the full range of the grid? --> use for first THREE (what is the third item?)
    # also use for discretizing the xy plane
# then what is the range for the angular pos/vel? => use for last 4



# need to normalize angles -- between 0,2pi (or -pi, pi)
# do I need to normalize angular velocity?