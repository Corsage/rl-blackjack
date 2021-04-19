# rl-blackjack
A reinforcement learning agent that learns to card count in Blackjack.

There are two files to run:
q_learn_blackjack.py is where our agent learns the game and how to count cards.
q_learn_betting.py is where out agent learns how much to bet depending on the
weight of the deck. It simulates the game using our policy from q_learn_blackjack.
Uncomment whichever call function in main you would like to use and run.
You can alter the iteration size in their respective call functions.
You can also alter the number of episodes it runs on to get its average reward in
the test_performance function.
Using np.loadtxt(filename), you can load in a policy to test its performance.
We left one commented out in q_learn_blackjack.py, this is the policy that
worked best for us.

Our environment is blackjack_env. There is nothing for you to change here but
you can if you'd like. There are comments to help you understand why everything
is  where it is.

All the policies we gathered are in folders:
history and history_t are the policies gathered for the card counting agent when
we used the range -20 to 20 as our weight.
weight and weight_t are the policies gathered for the card counting agentwhen
we used the range -3 to 3 as our weight.
bet and bet_t are the policies gathered for the betting agent.