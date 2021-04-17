import numpy as np
import pickle 

from blackjack_env import Blackjack
from blackjack_env_cardcount import Blackjack as Blackjack_count
from train_blackjack import test_performance
from train_cardcount import test_performance as test_card_count

if __name__ == "__main__":
    env1 = Blackjack()
    env2 = Blackjack_count()
    
    with open("qlearn_qvalue.pkl", "rb") as input_file:
        Q_value = pickle.load(input_file)
    
    with open("qlearn_qvalue_cardcount.pkl", "rb") as input_file:
        Q_value_count = pickle.load(input_file)
    
    det_policy = np.argmax(Q_value, axis=1)
    test_performance(env1, det_policy, nb_episodes=1000, max_steps=500)

    det_policy_count = np.argmax(Q_value_count, axis=1)
    test_card_count(env2, det_policy_count, nb_episodes=1000, max_steps=500)