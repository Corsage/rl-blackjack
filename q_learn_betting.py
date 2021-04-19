import gym
import numpy as np
from betting_env import BlackjackBetting

import pickle  # -- to save/load trained policy

def sample_action(policy, state):
    all_actions = np.arange(env.nA)
    return np.random.choice(all_actions, p=policy[state_to_ind(state)])

def state_to_ind(state):
    return (state+20)

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
    # s_t1 = env.reset()  # reset the environment
    s_t1 = 0
    s_t1_ind = state_to_ind(s_t1)

    i = 0
    while True:
        a_t1 = sample_action(policy, s_t1)
        s_t2, r_t1, done, _ = env.step(a_t1)
        s_t2_ind = state_to_ind(s_t2)
        Q_value[s_t1_ind, a_t1] += alpha*(r_t1 + gamma*np.max(Q_value[s_t2_ind]) - Q_value[s_t1_ind, a_t1])
        
        i += 1
        epsilon = epsilon/i
        policy = epsilon_greedy_policy_improve(Q_value, env.nS, env.nA, epsilon)
        
        s_t1 = s_t2
        s_t1_ind = s_t2_ind

        if done: # if episode ends update Q and reset our agent
            s_t1 = env.reset()
            s_t1_ind = state_to_ind(s_t1)
     
        if i >= iterations:
            # save policy to continue training from this point
            with open("qlearn_bet_policy.pkl", "wb") as output_file:
                pickle.dump(policy, output_file)
            with open("qlearn_bet_qvalue.pkl", "wb") as output_file:
                pickle.dump(Q_value, output_file)
            break
    
    det_policy = np.argmax(Q_value, axis=1)
    return Q_value, det_policy

def test_performance(env, policy, nb_episodes=1000, max_steps=10):
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
    res_reward=0
    balance=0
    wins = 0
    for i in range(nb_episodes):
        state = env.reset()
        done = False
        local_win = 0
        local_reward = 0
        for j in range(max_steps):
            action = policy[state]
            state, reward, done, _ = env.step(action)
            if reward > 0:
                local_win += 1
            local_reward += reward
            if done:
                break
        balance += env.player.balance
        res_reward += local_reward/(j+1)
        wins += local_win/(j+1)
    print(("\nSuccess Rate Over {} Episodes:\n\n"
           "Average balance={:.2f}\n"
           "Average Reward={:.2f}\n"
           "Average wins={:.2f}%\n")
    .format(nb_episodes,balance/nb_episodes,res_reward/nb_episodes, wins/nb_episodes*100))

def call_q(env):
    
    Q_ql, policy_ql = qlearning(env, iterations=1000, gamma=0.9, alpha=0.01)
    print("Q Learning\n")
    print(policy_ql)
    test_performance(env, policy_ql)
    
        
if __name__ == "__main__":
    env = BlackjackBetting()

    print("\nBetting lowest:\n")
    bet_lowest_policy = np.repeat(np.array([0]), repeats=41, axis=0)
    print(bet_lowest_policy) 
    test_performance(env, bet_lowest_policy)

    print("\nOptimal policy:\n")
    call_q(env)
    