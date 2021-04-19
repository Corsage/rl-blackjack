import random
import numpy as np

INITIAL_BALANCE = 1000
NUM_DECKS = 6
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
        self.nS = 10*10*2*41
        self.rounds = rounds
        self.decks = decks

        self.deck = []

        self.reset()
    
    def add_weight(self, card, player = ''):
        # Basic card counting technique:
            # If the card is between 2 and 6, add 1
            # If the card is an ace or a 10/face, sutract 1
        # The value at any given time is known as the weight of the deck.
        # When the deck is refilled, the weight of the deck will reset,
        # which can be seen in the reset() function.
        
        value = card.get_value()
        if player == 'dealer':
            if (value > 1 and value < 7):
                self.history_dealer += 1
            elif (value == 1 or value == 10):
                self.history_dealer -= 1
        else:
            if (value > 1 and value < 7):
                self.history += 1
            elif (value == 1 or value == 10):
                self.history -= 1

    def shuffle(self):
        random.shuffle(self.deck)

    def get_history(self):
        return self.history
    
    def get_weight(self):
        return self.weight

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
        # We need add a reward scaling based on the weight of the deck.
        # The weight of the deck is kept in history, so we take a function
        # of that and add (or subtract) that value in distribute_winnings.
        # For example, we want to add this scaling to punish it more harshly
        # if it busted on a heavy deck (history > 0), or less harshly if it 
        # busted on a light deck (history < 0).
        
        history = self.get_history()
        
        if history > 0:
            return (history / (history + 2))
        
        elif history < 0:
            return (-history / (history - 2))
        
        return 0

    def distribute_winnings(self):
        # If both the player and the dealer have a tie—including
        # with a blackjack—the bet is a tie or “push”.
        # If both the dealer and player bust, the player loses.
        # for player in self.players:
        
        scaling = self.reward_scaling()
        
        if not(self.player.did_bust()):
            
            if self.dealer.did_bust():
                reward = 1 + scaling/1.5
                #reward = 1
                self.player.add_to_score(1)
                self.dealer.add_to_score(-1)
            
            elif self.player.sum_hand() > self.dealer.sum_hand():
                reward = 1 - scaling/2
                #reward = 1
                self.player.add_to_score(1)
                self.dealer.add_to_score(-1)
            
            elif self.player.sum_hand() == self.dealer.sum_hand():
                reward = 0 - scaling/3
                #reward = 0
                self.player.add_to_score(0)
                self.dealer.add_to_score(0)

            else:
                reward = -1 + scaling
                #reward = -1
                self.player.add_to_score(-1)
                self.dealer.add_to_score(1)

        else:
            reward = -1 - scaling
            #reward = -1
            self.player.add_to_score(-1)
            self.dealer.add_to_score(1)
        
        self.history += self.history_dealer
        self.history_dealer = 0
        
        return reward

    def state_to_ind(self, state):
        # State - The observation of a 4-tuple of: 
        #    - the players current sum,
        #    - the dealer's one showing card (1-10 where 1 is ace),
        #    - and whether or not the player holds a usable ace (0 or 1).  
        #    - weight of the deck
        # state: [(players_sum),(shown_card),(usable_ace),(weight)]
    
        if state[0]<=11 or state[0]>21:
            return -1
    
        # linear indeces in 4d array shape
        lin_inds = np.arange(10*10*2*41).reshape([10,10,2,41])
        x = state[0]-12
        y = state[1]-1
        z = 1 if state[2]==False else 0
        w = state[3] + 20
        return lin_inds[x,y,z,w]

    def get_obs(self):
        return (self.player.sum_hand(), self.dealer.get_cards()[0].get_value(), self.player.has_ace(), self.get_history())
    
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

    def update_weight(self):
        # To minimize our states, in case 41 is too many, this will
        # partition the different values of history into ranges whose
        # values are kept in weight.
        
        if self.get_history() == 0:
            self.weight = 0
        elif self.get_history() == 1 or self.get_history() == 2:
            self.weight = 1
        elif self.get_history() == -1 or self.get_history() == -2:
            self.weight = -1
        elif self.get_history() == 3 or self.get_history() == 4:
            self.weight = 2
        elif self.get_history() == -3 or self.get_history() == -4:
            self.weight = -2
        elif self.get_history() > 4:
            self.weight = 3
        elif self.get_history() < -4:
            self.weight = -3
        else:
            self.weight = self.get_history()

    def reset(self):
        self.player = Player(PlayerType.PERSON)
        self.dealer = Player(PlayerType.DEALER)
        self.reward = 0

        # Check if the deck is 1/3 full.
        if len(self.deck) < 17:
            self.deck = []
            
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
        
        self.update_weight()

        return self.get_obs()

class Bet:
    """ A Betting class that emulates the game and awards points based on bets. """

    def __init__(self):
        self.nA = 3
        self.nS = 7
        self.blackjack = Blackjack()
        
        # Load in our best policy to simulate the game
        self.policy = np.loadtxt('history_t/td_1p5_2_3_1_1_1mil.txt')

        self.reset()
        
    def get_obs(self):
        
        return (self.blackjack.get_weight())
    
    def state_to_ind(self, state):
        
        return (state + 3)
    
    def step(self, a, max_steps = 500):
        # Simulate one hand of blackjack and give the reward multiplied by the
        # bet chosen to the agent. Bet is raised by 1 because while the acions'
        # indices are 0, 1, and 2, the agent is betting 1, 2, or 3.
        
        state = self.blackjack.reset()
        done = False
        
        for j in range(max_steps):
            
            action = 1 if state[0]<=11 else self.policy[self.blackjack.state_to_ind(state)]
            state, reward, done, _ = self.blackjack.step(action)
            
            if done:
                
                reward *= (a + 1)
                break
            
        return self.get_obs(), reward, done, {}

    def reset(self):
        
        return self.get_obs()