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
    """ A player can either be a dealer or a person. """
    DEALER = "DEALER"
    PERSON = "PERSON"

class Player:
    """ A basic player class for Blackjack. """

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

    def __init__(self, decks = 1, players = 1, rounds = 1):
        self.deck = []
        self.history = []
        self.players = []
        self.rounds = rounds
        self.decks = decks
        self.dealer = Player(PlayerType.DEALER)

        # Create x players.
        for i in range(players):
            self.players.append(Player(PlayerType.PERSON))

    def shuffle(self):
        random.shuffle(self.deck)

    def get_history(self):
        return self.history

    def deal(self):
        # Every time we deal a card, it goes into our history.
        card = self.deck.pop()
        self.history.append(card)
        return card

    def distribute_winnings(self):
        # If both the player and the dealer have a tie—including
        # with a blackjack—the bet is a tie or “push”.
        # If both the dealer and player bust, the player loses.
        for player in self.players:
            if not(player.did_bust()):
                if self.dealer.did_bust() or (player.sum_hand() > self.dealer.sum_hand()) :
                    player.add_to_score(1)
                    self.dealer.add_to_score(-1)

                elif player.sum_hand() == self.dealer.sum_hand():
                    player.add_to_score(0)
                    self.dealer.add_to_score(0)

                else:
                    player.add_to_score(-1)
                    self.dealer.add_to_score(1)
            else:
                player.add_to_score(-1)
                self.dealer.add_to_score(1)


    def start(self):
        # Start the game.
        # Note: We do NOT shuffle per game, only once per episode.

        # Create x deck of cards into one.
        for i in range(self.decks):
            for suit in self.SUITS:
                for rank in self.RANKINGS:
                    self.deck.append(Card(suit, rank))

        self.shuffle()

        # Begin games.
        for i in range(self.rounds):
            if DEBUGGING:
                print("BEGINNING GAME")
                print("History:", *self.history)

            # Each player is dealt two cards first.
            for player in self.players:
                player.hit(self.deal())
                player.hit(self.deal())

            if DEBUGGING:
                print("Players are dealt cards.")

            # Next, the dealer is dealt two cards.
            self.dealer.hit(self.deal())
            self.dealer.hit(self.deal())

            if DEBUGGING:
                print("Dealer is dealt cards.")

            # Keep asking each player if they want to hit.
            asking = True

            while asking:
                asking = False
                if DEBUGGING:
                    print("Asking players.")

                for player in self.players:
                    if player.ask():
                        asking = True
                        player.hit(self.deal())

            # Once all players are done, dealer reveals their hidden card.
            # Dealer then is asked to hit.

            if DEBUGGING:
                print("Asking dealer.")

            while self.dealer.ask():
                self.dealer.hit(self.deal())

            # Once completed, we distribute winnings.
            self.distribute_winnings()

            if DEBUGGING:
                for player in self.players:
                    print(player)

                print(self.dealer)

if __name__ == "__main__":
    game = Blackjack(rounds=5)
    game.start()


