import gym
import numpy as np
from blackjack_env import *

import pickle  # -- to save/load trained policy

def state_to_ind(state):
    # State - The observation of a 3-tuple of: 
    #    - the players current sum,
    #    - the dealer's one showing card (1-10 where 1 is ace),
    #    - and whether or not the player holds a usable ace (0 or 1).   
    # state: [(players_sum),(shown_card),(usable_ace)]
    if state[0]<=11 or state[0]>21:
        return -1

    # linear indeces in 3d array shape
    lin_inds = np.arange(10*10*2*3).reshape([10,10,2,3])
    x = state[0]-12
    y = state[1]-1
    z = 1 if state[2]==False else 0
    # 0 - card count == 0
    # 1 - card count is positive
    # 2 - card count is negative
    w = 0
    if state[3] > 0:
        w = 1
    elif state[3] < 0:
        w = 2
    return lin_inds[x,y,z,w]

def sample_action(policy, state):
    
    current_sum = state[0]

    # if our sum is <= 11, always hit
    # since we cannot exceed 21
    if current_sum <= 11:
        return 1

    # else pick at random
    all_actions = np.arange(env.nA)
    return np.random.choice(all_actions, p=policy[state_to_ind(state)])

def take_one_step(env, policy, state):
    """
    This function takes one step in the environment according to the stochastic policy.

    Parameters
    ----------
        env: given enviroment, here frozenlake
        policy: np.ndarray[env.nS, env.nA]
            See the description in `sample_action`.
        state: int
            The current state where the agent is in the environment

    Returns
    -------
        action: int
            the action that was chosen from the stochastic policy.
        reward: float
            the reward that was obtained during this step
        new_state: int
            the new state that the agent transitioned to
        done: boolean
            If done is `True` this indicates that we have entered a terminating state
            (i.e, `new_state` is a terminating state).

    """
    action = sample_action(policy, state)
    new_state, reward, done, _ = env.step(action)
    return action, reward, new_state, done


def generate_episode(env, policy, max_steps=500):
    """
    Since Monte Carlo methods are based on learning from episodes write a function `random_episode`
    that generates an episode given the frozenlake environment and a policy.

    Parameters
    ----------
        env: given enviroment, here frozenlake
        policy: np.ndarray[env.nS, env.nA]
            See the description in `sample_action`.
        max_steps: int
            The maximum number of steps that the episode could take. If a terminating state
            is not reached within this time, terminate the episode.

    Returns
    -------
        episode: list of [(state, action, reward)] triplet.
            For example, [(0,1,0),(4,2,0)] indicates that in the first time
            we were in state 0 took action 1 and observed reward 0
            (it also means we transitioned to state 4). Similarly, in the
            second time step we are in state 4 took action 2 and observed reward 0.

    """
    episode = []
    curr_state = env.reset()  # reset the environment and place the agent in the start square
    ############################
    # YOUR IMPLEMENTATION HERE #

    steps = 0
    while True:
        action, reward, new_state, done = take_one_step(env, policy, curr_state)
        episode.append((curr_state, action, reward))
        curr_state = new_state
        steps+=1
        if done or steps >= max_steps:
            break
        
    ############################
    return episode

def generate_returns(episode, gamma=0.9):
    """
    Given an episode, generate the total return from each step in the episode based on the
    discount factor gamma. For example, let the episode be:
    [(0,1,1),(4,2,-1),(6,3,0),(8,0,2)]
    and gamma=0.9. Then the total return in the first time step is:
    1 + 0.9 * -1 + 0.9^2 * 0 + 0.9^3 * 2
    In the second time step it is:
    -1 + 0.9 * 0 + 0.9^2 * 2
    In the third time step it is:
    0 + 0.9 * 2
    And finally, in the last time step it is:
    2

    Parameters
    ----------
        episode: list
            The episode is assumed to be in the same format as the output of the `generate_episode`
            described above.
        gamma: float
            This is the discount factor, which is a number between 0 and 1.

    Returns
    -------
        epi_returns: np.ndarray[len(episode)]
            The array containing the total returns for each step of the episode.

    """
    len_episode = len(episode)
    epi_returns = np.zeros(len_episode)
    ############################
    # YOUR IMPLEMENTATION HERE #
    # HINT: Representing immediate reward as a vector and
    # using a vector of powers of gamma along with `np.dot` will
    # make this much easier to implement in a few lines of code.
    # You don't need to use this approach however and use whatever works for you. #

    immediate_return = [x[2] for x in episode]
    gammas = [gamma**i for i in range(len_episode)]
    for i in range(len_episode):
        epi_returns[i] = np.dot(immediate_return[i:], gammas[:len_episode-i])

    ############################
    return epi_returns

def epsilon_greedy_policy_improve(Q_value, nS, nA, epsilon):
    """Given the Q_value function and epsilon generate a new epsilon-greedy policy.
    IF TWO ACTIONS HAVE THE SAME MAXIMUM Q VALUE, THEY MUST BOTH BE EXECUTED EQUALLY LIKELY.
    THIS IS IMPORTANT FOR EXPLORATION.

    Parameters
    ----------
    Q_value: np.ndarray[env.nS, env.nA]
        Defined similar to the input of `mc_policy_evaluation`.
    env.nS: int
        number of states
    env.nA: int
        number of actions
    epsilon: float
        current value of epsilon

    Returns
    -------
    new_policy: np.ndarray[env.nS, env.nA]
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


def mc_policy_evaluation(env, policy, Q_value, n_visits, gamma=0.9):
    """Update the current Q_values and n_visits by generating one random episode
    and using the given policy and the Monte Carlo first-visit approach.

    Parameters
    ----------
        env: given enviroment, here frozenlake
        policy: np.ndarray[env.nS, env.nA]
            See the description in `sample_action`.
        Q_value: np.ndarray[env.nS, env.nA]
            The current Q_values. This is a matrix (i.e., 2D array) of size
            numb_states (env.nS) x numb_actions (env.nA). For example, `Q_value[0, 1]` is the current
            estimate of the Q_value for state 0 and action 1.
        n_visits: np.ndarray[env.nS, env.nA]
            The current number of times a (state, action) pair has been visited.
            This is a matrix (i.e., 2D array) of size numb_states (env.nS) x numb_actions (env.nA).
            For example, `n_visits[0, 1]` is the current number of times action 1 has been performed in state 0.
        gamma: float
            This is the discount factor, which is a number between 0 and 1.
    Returns
    -------
    value_function: np.ndarray[env.nS]
        The value function of the given policy, where value_function[s] is
        the value of state s
    """
    episode = generate_episode(env, policy)
    returns = generate_returns(episode, gamma=gamma)
    visit_flag = np.zeros((env.nS, env.nA))
    ############################
    # YOUR IMPLEMENTATION HERE #
    for t in range(len(episode)):
        s, a, r = episode[t]
        s_ind = state_to_ind(s)
        if s_ind >= 0:
            if visit_flag[s_ind,a] == 0:
                n_visits[s_ind, a] += 1
                Q_value[s_ind,a] += 1/n_visits[s_ind,a]*(returns[t] - Q_value[s_ind,a])
                visit_flag[s_ind,a] += 1
    
    ############################
    return Q_value, n_visits

def mc_glie(env, iterations=1000, gamma=0.9, policy=None):
    """This function implements the first-visit Monte Carlo GLIE policy iteration for finding
    the optimal policy.

    Parameters
    ----------
    env: given enviroment, here frozenlake
    iterations: int
        the number of iterations to try
    gamma: float
        discount factor

    Returns:
    ----------
    Q_value: np.ndarray[env.nS, env.nA]
        The Q_value at the end of iterations
    det_policy: np.ndarray[env.nS]
        The greedy (i.e., deterministic policy)
    """
    Q_value = np.zeros((env.nS, env.nA))
    n_visits = np.zeros((env.nS, env.nA))
    if policy is None:
        policy = np.ones((env.nS,env.nA))/env.nA  # initially all actions are equally likely
    epsilon = 1
    ############################
    # YOUR IMPLEMENTATION HERE #
    i = 0
    while True:
        Q_value, n_visits = mc_policy_evaluation(env, policy, Q_value, n_visits)
        i += 1
        epsilon = epsilon/i
        policy = epsilon_greedy_policy_improve(Q_value, env.nS, env.nA, epsilon)

        if i >= iterations:
            break

    ############################
    det_policy = np.argmax(Q_value, axis=1)
    return Q_value, det_policy


def td_sarsa(env, iterations=1000, gamma=0.9, alpha=0.1):
    """This function implements the temporal-difference SARSA policy iteration for finding
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

    nS = env.nS  # number of states
    nA = env.nA  # number of actions
    Q_value = np.zeros((nS, nA))
    policy = np.ones((env.nS,env.nA))/env.nA
    epsilon = 1
    s_t1 = env.reset()  # reset the environment and place the agent in the start square
    a_t1 = sample_action(policy, s_t1)
    ############################
    # YOUR IMPLEMENTATION HERE #
    # HINT: Don't forget to decay epsilon according to GLIE

    for i in range(iterations):
    	
        d_t1 = False
        epsilon = 1 / (1 + 2)
        
        while (d_t1 == False):
            
            
            s_t2, r_t1, d_t1, _ = env.step(a_t1)
            a_t2 = sample_action(policy, s_t2)
            
            Q_value[s_t1, a_t1] += alpha * (r_t1 + (gamma * Q_value[s_t2, a_t2]) - Q_value[s_t1, a_t1])
            
            actions = Q_value[s_t1]
            max_index = np.argwhere(actions == np.amax(actions))
            max_list = max_index.flatten().tolist()
            
            policy[s_t1] = (epsilon / nA)
            
            for max_elem in max_list:
                
                policy[s_t1, max_elem] += (1-epsilon) / len(max_list)
            
            s_t1 = s_t2
            a_t1 = a_t2
    	
        s_t1 = env.reset()
        a_t1 = sample_action(policy, s_t1)
        
    ############################
    det_policy = np.argmax(Q_value, axis=1)
    return Q_value, det_policy


def qlearning(env, iterations=1000, gamma=0.9, alpha=0.1, policy=None, Q_value=None):
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
    Q_value: np.ndarray[env.nS, env.nA]
        The Q_value at the end of iterations
    det_policy: np.ndarray[env.nS]
        The greedy (i.e., deterministic policy)
    """
    if Q_value is None:
        Q_value = np.zeros((env.nS, env.nA))
    if policy is None:
        policy = np.ones((env.nS,env.nA))/env.nA
    epsilon = 1
    s_t1 = env.reset()  # reset the environment and place the agent in the start square
    s_t1_ind = state_to_ind(s_t1)

    i = 0
    while True:
        a_t1 = sample_action(policy, s_t1)
        s_t2, r_t1, done, _ = env.step(a_t1)
        if s_t2[0]<=11:
            continue
        s_t2_ind = state_to_ind(s_t2)
        Q_value[s_t1_ind, a_t1] += alpha*(r_t1 + gamma*np.max(Q_value[s_t2_ind]) - Q_value[s_t1_ind, a_t1])
        
        i += 1
        epsilon = epsilon/i
        policy = epsilon_greedy_policy_improve(Q_value, env.nS, env.nA, epsilon)
        
        s_t1 = s_t2
        s_t1_ind = s_t2_ind

        if done: # if episode ends update Q and reset our agent
            s_t2, r_t1, done, _ = env.step(a_t1)
            Q_value[s_t1_ind, a_t1] += alpha*(r_t1 + gamma*np.max(Q_value[s_t2_ind]) - Q_value[s_t1_ind, a_t1])
            s_t1 = env.reset()
            s_t1_ind = state_to_ind(s_t1)
     
        if i >= iterations:
            break
    
    ############################
    det_policy = np.argmax(Q_value, axis=1)
    return Q_value, det_policy

def test_performance(env, policy, nb_episodes=5000, max_steps=500):
    """
      This function evaluate the success rate of the policy in reaching
      the goal.

      Parameters
      ----------
      env: gym.core.Environment
        Environment to play on. Must have env.nS, env.nA, and P as
        attributes.
      Policy: np.array of shape [env.env.nS]
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
            if done:
                res_reward+=reward
                if reward>0:
                    win+=1
                elif reward==0:
                    draw+=1
                else:
                    loss+=1
                break
    print(("\nSuccess Rate Over {} Episodes:\n\n"
           "Wins = {:.2f}%\nDraws = {:.2f}%\nLosses = {:.2f}%\n\n"
           "Average Reward={:.2f}")
    .format(nb_episodes,win/nb_episodes*100,draw/nb_episodes*100,loss/nb_episodes*100,res_reward/nb_episodes))

def call_mc(env, policy):
    
    Q_mc, policy_mc = mc_glie(env, iterations=1000, gamma=0.9, policy=policy)
    print("Monte Carlo\n")
    print(policy_mc)
    test_performance(env, policy_mc)
    
def call_q(env, policy):
    
    Q_ql, policy_ql = qlearning(env, iterations=1000, gamma=0.9, alpha=0.05)
    print("Q Learning\n")
    print(policy_ql)
    test_performance(env, policy_ql)
    
def call_td(env, policy):
    
    Q_td, policy_td = qlearning(env, iterations=1000, gamma=0.9, alpha=0.05)
    print("Sarsa\n")
    print(policy_td)
    test_performance(env, policy_td)
        
if __name__ == "__main__":
    env = Blackjack()
    # env = gym.make("Blackjack-v0")
    nS = env.nS  # number of states for policy improvement
    nA = env.nA  # number of actions: hit or stand
    Q_value = np.zeros((env.nS, env.nA))
    policy = np.ones((env.nS,env.nA))/env.nA

    ## Monte-Carlo
    # with open("mc_policy.pkl", "rb") as input_file:
    #     policy = pickle.load(input_file) # load saved policy

    call_mc(env, policy)

    # with open("mc_det_policy.pkl", "wb") as output_file:
    #     pickle.dump(policy_mc, output_file)

    call_q(env, policy)
    
    call_td(env, policy)
    
    