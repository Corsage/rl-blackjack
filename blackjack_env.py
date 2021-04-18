import random
import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding

INITIAL_BALANCE = 1000
NUM_DECKS = 6
import random
from enum import Enum

# Set to True to see debug messages.
DEBUGGING = True

class Card:
    """ A basic card class for Blackjack. """

    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank

    def __str__(self):
        return self.suit + self.rank
    
    def get_suit(self):
        return self.suit
    
    def get_rank(self):
        return self.rank

    def get_value(self):
        return Blackjack.VALUES[self.rank]

class PlayerType(Enum):
    """ A self.player can either be a dealer or a person. """
    DEALER = "DEALER"
    PERSON = "PERSON"

class Player:
    """ A basic self.player class for Blackjack. """

    def __init__(self, playerType):
        self.cards = []
        self.playerType = playerType
        self.score = 0

        # Distinction between busting and stopping.
        self.busted = False
        self.stop = False

        # For dealers.
        self.hidden_card = self.playerType == PlayerType.DEALER

    def __str__(self):
        info = self.playerType.name + " score: " + str(self.score) + "\n"
        hand = ""
        for card in self.cards:
            hand += str(card) + " "

        hand += "SUM: " + str(self.sum_hand())

        return info + hand

    def add_to_score(self, num):
        self.score += num

    def get_cards(self):
        return self.cards

    def did_bust(self):
        return self.busted

    def has_ace(self):
        for card in self.cards:
            value = card.get_value()
            if value == 1:
                return True

    def sum_hand(self):
        hand = 0
        has_ace = False

        for card in self.cards:
            value = card.get_value()

            if value == 1:
                has_ace = True

            hand += value

        # Check if ace as 11 helps.
        if has_ace and (hand + 10) <= 21:
            return hand + 10
        
        return hand
    
    def ask(self):
        if self.playerType == PlayerType.PERSON:
            temp = str(input("Hit? (Y/N): "))

            if temp == 'Y' or temp == 'y':
                return True

            self.stand()
            return False

        elif self.playerType == PlayerType.DEALER:
            # Dealer must hit in certain conditions.
            # If the total is 17 or more, it must stand.
            # If the total is 16 or under, they must take a card.
            # If the dealer has an ace, and counting it as 11
            # would bring the total to 17 or more (but not over 21),
            # the dealer must count the ace as 11 and stand.

            if self.sum_hand() >= 17:
                self.stand()
                return False
            else:
                return True

    def hit(self, card):
        self.cards.append(card)

        # Check if busted.
        if self.sum_hand() > 21:
            self.busted = True

    def stand(self):
        self.stop = True

class Blackjack:
    """ A Blackjack class that emulates the game. """

    SUITS = ('H', 'D', 'C', 'S')
    RANKINGS = ('A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K')
    VALUES = { 'A': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, 'J': 10, 'Q': 10, 'K': 10 }

    def __init__(self, decks = 1, rounds = 1):
        self.nA = 2
        self.nS = 10*10*2
        self.rounds = rounds
        self.decks = decks

        self.reset()
    
    def add_weight(self, card, player = ''):
        value = card.get_value()
        if player == 'dealer':
            if (value > 2 and value < 7):
                self.history_dealer += 1
            elif (value == 1 or value == 10):
                self.history_dealer -= 1
        else:
            if (value > 2 and value < 7):
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
        # for player in self.players:
        
        scaling = self.reward_scaling()
        
        if not(self.player.did_bust()):
            
            if self.dealer.did_bust():
                reward = 1 + scaling
                self.player.add_to_score(1)
                self.dealer.add_to_score(-1)
            
            elif self.player.sum_hand() > self.dealer.sum_hand():
                reward = 1 - scaling
                self.player.add_to_score(1)
                self.dealer.add_to_score(-1)
            
            elif self.player.sum_hand() == self.dealer.sum_hand():
                reward = 0 - (scaling / 2)
                self.player.add_to_score(0)
                self.dealer.add_to_score(0)

            else:
                reward = -1 + scaling
                self.player.add_to_score(-1)
                self.dealer.add_to_score(1)

        else:
            reward = -1 - scaling
            self.player.add_to_score(-1)
            self.dealer.add_to_score(1)
        
        self.history += self.history_dealer
        
        return reward

    def get_obs(self):
        return (self.player.sum_hand(), self.dealer.get_cards()[0].get_value(), self.player.has_ace())
    
    def step(self, a):
        reward = 0
        if a==1: # if action == 'hit'
            self.player.hit(self.deal())
            done = self.player.did_bust()
        else:   # action == 'stand'
            done = True
            self.player.stand()
            while self.dealer.sum_hand() < 17:
                self.dealer.hit(self.dealer_deal())

        if done:
            reward = self.distribute_winnings()
        return self.get_obs(), reward, done, {}

    def reset(self):
        
        self.deck = []
        self.history = 0
        self.history_dealer = 0
        self.player = Player(PlayerType.PERSON)
        self.dealer = Player(PlayerType.DEALER)
        self.reward = 0

        # initialize the deck
        for suit in self.SUITS:
            for rank in self.RANKINGS:
                self.deck.append(Card(suit, rank))
        self.shuffle()

        # deal cards to player and dealer
        self.player.hit(self.deal())
        self.dealer.hit(self.dealer_deal())
        
        self.player.hit(self.deal())
        self.dealer.hit(self.deal())

        return self.get_obs()

