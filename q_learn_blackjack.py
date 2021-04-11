import gym
import numpy as np

# import pickle  # -- to save/load trained policy

def state_to_ind(state):
    # State - The observation of a 3-tuple of: 
    #    - the players current sum,
    #    - the dealer's one showing card (1-10 where 1 is ace),
    #    - and whether or not the player holds a usable ace (0 or 1).   
    # state: [(players_sum),(shown_card),(usable_ace)]

    if state[0]<=11 or state[0]>=21:
        return -1

    # linear indeces in 3d array shape
    lin_inds = np.arange(10*10*2).reshape([10,10,2])
    x = state[0]-12
    y = state[1]-1
    z = 1 if state[2]==False else 0 
    return lin_inds[x,y,z]

def sample_action(policy, state):
    
    current_sum = state[0]

    nS = 10*10*2
    nA = 2

    # if our sum is <= 11, always hit
    # since we cannot exceed 21
    if current_sum <= 11:
        return 1

    # else pick at random
    all_actions = np.arange(nA)
    return np.random.choice(all_actions, p=policy[state_to_ind(state)])

def epsilon_greedy_policy_improve(Q_value, nS, nA, epsilon):
    """Given the Q_value function and epsilon generate a new epsilon-greedy policy.
    IF TWO ACTIONS HAVE THE SAME MAXIMUM Q VALUE, THEY MUST BOTH BE EXECUTED EQUALLY LIKELY.
    THIS IS IMPORTANT FOR EXPLORATION.

    Parameters
    ----------
    Q_value: np.ndarray[nS, nA]
        Defined similar to the input of `mc_policy_evaluation`.
    nS: int
        number of states
    nA: int
        number of actions
    epsilon: float
        current value of epsilon

    Returns
    -------
    new_policy: np.ndarray[nS, nA]
        The new epsilon-greedy policy according. The shape of the new policy is
        as described in `sample_action`.
    """

    new_policy = epsilon * np.ones((nS, nA)) / nA

    for s in range(nS):
        max_a = np.argwhere(Q_value[s] == np.amax(Q_value[s])).flatten()
        for a in range(nA):
            if a in max_a:
                new_policy[s,a] = epsilon/nA + (1 - epsilon)/len(max_a)
            else:
                new_policy[s,a] = epsilon/nA

    return new_policy

def updateQ_policy(s_t1, policy, Q_value, epsilon):
    a_t1 = sample_action(policy, s_t1)
    s_t2, r_t1, done, _ = env.step(a_t1)

    if s_t2<=11:
        return Q_value, policy
    
    s_t1_ind = state_to_ind(s_t1)
    s_t2_ind = state_to_ind(s_t2)
    Q_value[s_t1_ind, a_t1] += alpha*(r_t1 + gamma*np.max(Q_value[s_t2_ind]) - Q_value[s_t1_ind, a_t1])
    i += 1
    epsilon = epsilon/i
    policy = epsilon_greedy_policy_improve(Q_value, nS, nA, epsilon)

    return Q_value, policy 

def qlearning(env, iterations=1000, gamma=0.9, alpha=0.1):
    """This function implements the Q-Learning policy iteration for finding
    the optimal policy.

    Parameters
    ----------
    env: given enviroment, here frozenlake
    iterations: int
        the number of iterations to try
    gamma: float
        discount factor
    alpha: float
        The learning rate during Q-value updates

    Returns:
    ----------
    Q_value: np.ndarray[nS, nA]
        The Q_value at the end of iterations
    det_policy: np.ndarray[nS]
        The greedy (i.e., deterministic policy)
    """
    # states: [(player_sum, shown_card, usable_ace)]
    # [(12-21),(1-10),(True,False)]
    nS = 10*10*2  # number of states
    nA = 2  # number of actions: hit or stand
    Q_value = np.zeros((nS, nA))
    policy = np.ones((nS,nA))/nA
    epsilon = 1
    s_t1 = env.reset()  # reset the environment and place the agent in the start square
    s_t1_ind = state_to_ind(s_t1)

    i = 0
    while True:
        a_t1 = sample_action(policy, s_t1)
        s_t2, r_t1, done, _ = env.step(a_t1)
        if s_t2[0]<=11:
            continue
        print(s_t1,s_t2)
        s_t2_ind = state_to_ind(s_t2)
        Q_value[s_t1_ind, a_t1] += alpha*(r_t1 + gamma*np.max(Q_value[s_t2_ind]) - Q_value[s_t1_ind, a_t1])
        
        i += 1
        epsilon = epsilon/i
        policy = epsilon_greedy_policy_improve(Q_value, nS, nA, epsilon)
        
        s_t1 = s_t2
        s_t1_ind = s_t2_ind

        if done: # if episode ends update Q and reset our agent
            a_t1 = sample_action(policy, s_t1)
            s_t2, r_t1, done, _ = env.step(a_t1)
            Q_value[s_t1_ind, a_t1] += alpha*(r_t1 + gamma*np.max(Q_value[s_t2_ind]) - Q_value[s_t1_ind, a_t1])
            s_t1 = env.reset()
            s_t1_ind = state_to_ind(s_t1)
     
        if i >= iterations:
            break
    
    ############################
    det_policy = np.argmax(Q_value, axis=1)
    return Q_value, det_policy

def test_performance(env, policy, nb_episodes=500, max_steps=500):
    """
      This function evaluate the success rate of the policy in reaching
      the goal.

      Parameters
      ----------
      env: gym.core.Environment
        Environment to play on. Must have nS, nA, and P as
        attributes.
      Policy: np.array of shape [env.nS]
        The action to take at a given state
      nb_episodes: int
        number of episodes to evaluate over
      max_steps: int
        maximum number of steps in each episode
    """
    win = 0
    draw = 0
    loss = 0
    res_reward=0
    for i in range(nb_episodes):
        state = env.reset()
        done = False
        for j in range(max_steps):
            action = 1 if state[0]<=11 else policy[state_to_ind(state)]
            state, reward, done, _ = env.step(action)
            res_reward+=reward
            if done:
                if reward>0:
                    win+=1
                elif reward==0:
                    draw+=1
                else:
                    loss+=1
                break
    print("\nSuccess rate over {} episodes:\nwins = {:.2f}%\ndraws = {:.2f}%\nlosses = {:.2f}%\n\n".format(nb_episodes,win/nb_episodes*100,draw/nb_episodes*100,loss/nb_episodes*100))


if __name__ == "__main__":
    env = gym.make("Blackjack-v0")
    nS = 10*10*2  # number of states for policy improvement
    nA = 2  # number of actions: hit or stand
    Q_value = np.zeros((nS, nA))
    policy = np.ones((nS,nA))/nA
    state = env._get_obs()
    Q_ql, policy_ql = qlearning(env, iterations=10000, gamma=0.9, alpha=0.1)
    test_performance(env, policy_ql, nb_episodes=500, max_steps=500)