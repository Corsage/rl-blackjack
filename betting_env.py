import random
import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pickle
from blackjack_env import *
from q_learn_blackjack import state_to_ind

class BlackjackBetting:
    """ A Blackjack class that emulates the game. """

    SUITS = ('H', 'D', 'C', 'S')
    RANKINGS = ('A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K')
    VALUES = { 'A': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, 'J': 10, 'Q': 10, 'K': 10 }

    BETS = ( 10, 20, 50 ) # for simplicity - discrete number of bet options 

    def __init__(self, decks = 1, rounds = 1):
        self.nA = 3
        self.nS = 41 # card-count -20..+20

        self.deck = []
        self.history = 0
        self.history_dealer = 0

        self.current_bet = 0

        self.end_episode = False
  
        self.player = Player(PlayerType.PERSON)
        self.dealer = Player(PlayerType.DEALER)

        # create the deck of cards
        self.deck = []
        for suit in self.SUITS:
            for rank in self.RANKINGS:
                self.deck.append(Card(suit, rank))
        self.shuffle()

        with open("qlearn_policy.pkl", "rb") as input_file:
            self.hit_policy = pickle.load(input_file) # load saved policy
    
    def add_weight(self, card, player = ''):
        value = card.get_value()
        # if player == 'dealer':
        #     if (value > 2 and value < 7):
        #         self.history_dealer += 1
        #     elif (value == 1 or value == 10):
        #         self.history_dealer -= 1
        # else:
        if (value >= 2 and value < 7):
            self.history += 1
        elif (value == 1 or value == 10):
            self.history -= 1

    def shuffle(self):
        random.shuffle(self.deck)

    def get_history(self):
        return self.history

    def deal(self):
        # Every time we deal a card, it goes into our history.
        card = self.deck.pop()
        self.add_weight(card)
        return card
    
    def dealer_deal(self):
        # Same as deal except these cards would be facedown to the player.
        card = self.deck.pop()
        self.add_weight(card, 'dealer')
        return card
    
    def reward_scaling(self):
        
        if self.history > 0:
            return (self.history / (self.history + 4))
        
        elif self.history < 0:
            return (-self.history / (self.history - 4))
        
        return 0

    def distribute_winnings(self):
        # If both the player and the dealer have a tie—including
        # with a blackjack—the bet is a tie or “push”.
        # If both the dealer and player bust, the player loses.
        
        scaling = self.reward_scaling()
        
        if not(self.player.did_bust()):
            
            if self.dealer.did_bust():
                # reward = 1 + scaling
                reward = 1
                self.player.add_to_balance(self.current_bet)
            
            elif self.player.sum_hand() == self.dealer.sum_hand():
                # reward = 0 - (scaling / 2)
                reward = 0
                # no change in balance, player gets their money back

            elif self.player.sum_hand() == 21:
                # if Blackjack - 1.5*bet_amount
                self.current_bet = 1.5*self.current_bet
                # reward = 1 - scaling
                reward = 1
                self.player.add_to_balance(self.current_bet)

            elif self.player.sum_hand() > self.dealer.sum_hand():
                # reward = 1 - scaling
                reward = 1
                self.player.add_to_balance(self.current_bet)

            else:
                # reward = -1 + scaling
                reward = -1
                self.player.add_to_balance(-self.current_bet)

        else:
            # reward = -1 - scaling
            reward = -1
            self.player.add_to_balance(-self.current_bet)
        self.history += self.history_dealer
        reward *= self.current_bet/10 # greater reward if bet is bigger
        return reward

    def get_obs(self):
        return (self.history)
    
    def sample_action(self):
        state = (self.player.sum_hand(), self.dealer.get_cards()[0].get_value(), self.player.has_ace())
        current_sum = state[0]

        # if our sum is <= 11, always hit
        # since we cannot exceed 21
        if current_sum <= 11:
            return 1

        # else pick at random
        all_actions = np.arange(2)
        return np.random.choice(all_actions, p=self.hit_policy[state_to_ind(state)])

    def sample_round(self):
        self.reset()
        a = self.sample_action()
        done = False
        while a==1 and not done:
            self.player.hit(self.deal())
            done = self.player.did_bust()
            a = self.sample_action()

        done = True
        self.player.stand()
        while self.dealer.sum_hand() < 17:
            self.dealer.hit(self.dealer_deal())
        reward = self.distribute_winnings()
        return reward

    def step(self, bet):
        self.current_bet = self.BETS[bet]
        reward = self.sample_round()
        done = self.end_episode
        self.end_episode = False
        return self.get_obs(), reward, done, {}

    def reset(self):
        self.player.reset_hand()
        self.dealer.reset_hand()
        # Check if the deck is 1/3 full.
        if len(self.deck) < 17:
            self.deck = []
            self.end_episode = True
            self.player.balance = INITIAL_BALANCE
            
            for suit in self.SUITS:
                for rank in self.RANKINGS:
                    self.deck.append(Card(suit, rank))

            self.shuffle()
            self.history = 0
            self.history_dealer = 0


        # deal cards to player and dealer
        self.player.hit(self.deal())
        self.dealer.hit(self.dealer_deal())
        
        self.player.hit(self.deal())
        self.dealer.hit(self.deal())
        return self.get_obs()


if __name__ == "__main__":
    env = BlackjackBetting()
    for i in range(2):
        print('\n\n', env.player.balance)
        print(env.step(2))
        print(env.player.balance)
