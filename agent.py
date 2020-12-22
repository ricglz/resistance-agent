#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Logic for the Agent bot"""
from functools import reduce

from numpy import array, float32, int32
from pandas import read_csv
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.models import load_model

from player import Bot

path_to_folder = 'bots/agent/resources'

def create_classifier():
    filepath = f'{path_to_folder}/vote_info_file.csv'
    names = ['Mission #', 'team_sus_avg', 'game_sus_avg', 'Is spy', 'Success']
    df = read_csv(filepath, names=names)
    inputs = df.values[:,:-1].astype(float32)
    outputs = df.values[:,-1].astype(int32)

    train_set_size = int(len(df) * 0.7)
    training_set = inputs[:train_set_size], outputs[:train_set_size]

    inputs_train, labels_train = training_set

    classifer = DecisionTreeClassifier(max_depth=10, min_samples_leaf=1)
    classifer.fit(inputs_train, labels_train)

    return classifer

def get_scaler(orig_filename):
    inputs = read_csv(orig_filename, header=None).values[:,:-1]
    scaler = StandardScaler()
    scaler.fit(inputs)
    return scaler

model = load_model(f'{path_to_folder}/spy_classifier')
model_scaler = get_scaler(f'{path_to_folder}/orig_LoggerBot.log')

classifier = create_classifier()
classifier_scaler = get_scaler(f'{path_to_folder}/orig_vote_info_file.csv')

def dict_counter(players):
    """Returns dictionary containing counter of a player"""
    return {player: 0 for player in players}

def dict_array_counter(players):
    """Returns dictionary containing counter of a player"""
    return {player: [0] * 7 for player in players}

def join_elements(arr):
    return ','.join(map(str, arr))

class Agent(Bot):
    """Bot that's a very good spy"""
    # Attributes
    failed_missions_been_on = {}
    missions_been_on = {}
    suspect_missions_voted_up = {}
    suspect_missions_voted_down = {}
    perfect_team_counts = []
    spies = []
    write_information = False

    # ------------ Helpers -------------------
    def mission_total_suspect_count(self, team):
        """
        Return the total count of failed missions of all the players
        in a team
        """
        count_suspect = lambda acc, player: acc + self.failed_missions_been_on[player]
        return reduce(count_suspect, team, 0)

    def mission_suspect_average(self, team):
        """
        Return the average count of failed missions of all the players
        in a team
        """
        return self.mission_total_suspect_count(team) / len(team)

    # ------- Neural Network functions ------
    def __get_input_vector__(self, player):
        input_vector = [self.missions_been_on[player],self.failed_missions_been_on[player]]
        input_vector += self.suspect_missions_voted_up[player] + \
                self.suspect_missions_voted_down[player]
        return model_scaler.transform(array(input_vector).reshape(1, -1))[0]

    def calculate_spy_probs(self, players):
        input_vectors = array(list(map(self.__get_input_vector__, players)))
        outputs = model(input_vectors)
        probabilities = { player: outputs[index, 1] for index, player in enumerate(players) }
        return probabilities # This returns a dictionary of {player: spyProbability}

    def get_players_probs(self, players, sort=False, probs=False):
        spy_probs = self.calculate_spy_probs(players).items()
        if sort:
            spy_probs = sorted(spy_probs, key=lambda item: item[1])
        return [v for k, v in spy_probs] if probs else [k for k, v in spy_probs]

    def player_probs_avg(self, players):
        player_probs = self.get_players_probs(players, probs=True)
        get_numpy = lambda tensor: tensor.numpy()
        player_probs = list(map(get_numpy, player_probs))

        return sum(player_probs) / len(player_probs)

    # ----------- Listeners ----------------
    def onGameRevealed(self, players, spies):
        self.failed_missions_been_on = dict_counter(players)
        self.missions_been_on = dict_counter(players)
        self.suspect_missions_voted_up = dict_array_counter(players)
        self.suspect_missions_voted_down = dict_array_counter(players)
        self.perfect_team_counts = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
        self.spies = spies

    def onVoteComplete(self, votes):
        current_suspect_count = self.mission_total_suspect_count(self.game.team)
        for index, vote in enumerate(votes):
            player = self.game.players[index]
            if vote:
                self.suspect_missions_voted_up[player][current_suspect_count] += 1
            else:
                self.suspect_missions_voted_down[player][current_suspect_count] += 1

    def onMissionComplete(self, sabotaged):
        was_sabotaged = sabotaged > 0

        # Count missions
        for player in self.game.team:
            self.missions_been_on[player] += 1
            if was_sabotaged:
                self.failed_missions_been_on[player] += 1

        # Count when voting for the perfect team
        team_sus_avg = self.player_probs_avg(self.game.team)
        game_sus_avg = self.player_probs_avg(self.game.players)
        successful_mission = was_sabotaged if self.spy else not was_sabotaged

        self.perfect_team_counts[self.game.turn - 1][0] = team_sus_avg
        self.perfect_team_counts[self.game.turn - 1][1] = game_sus_avg
        self.perfect_team_counts[self.game.turn - 1][2] = int(self.spy)
        self.perfect_team_counts[self.game.turn - 1][3] = int(successful_mission)

    # --------- Logging information --------
    def __print_player_information__(self, spies):
        for player in self.game.players:
            failed_missions_been_on = self.failed_missions_been_on[player]
            missions_been_on = self.missions_been_on[player]
            voted_up_total = self.suspect_missions_voted_up[player]
            voted_down_total = self.suspect_missions_voted_down[player]
            is_spy = int(player in spies)
            self.log.debug(
                '%s,%s,%s,%s,%s',
                failed_missions_been_on,
                missions_been_on,
                join_elements(voted_up_total),
                join_elements(voted_down_total),
                is_spy)

    def __print_vote_information__(self):
        vote_information = list(map(join_elements, self.perfect_team_counts))
        vote_info_file = open('vote_info_file.csv', 'a')
        for index in range(self.game.turn - 1):
            vote_info_file.write(f'{index + 1},{vote_information[index]}\n')
        vote_info_file.close()

    def onGameComplete(self, _win, spies):
        """What to do when the game is completed"""
        if self.write_information:
            self.__print_player_information__(spies)
            self.__print_vote_information__()

    # ----- Decision tree functions -------
    def __get_classifier_input__(self):
        team_sus_avg = self.player_probs_avg(self.game.team)
        game_sus_avg = self.player_probs_avg(self.game.players)
        turn = self.game.turn

        non_normalized_inputs = [turn, team_sus_avg, game_sus_avg, self.spy]
        return classifier_scaler.transform([non_normalized_inputs])

    # ---------- Actions -----------------
    def select(self, _players, count):
        """
        Just select those players who are not a spy and that has less
        probability of being a spy
        """
        spies = self.spies
        others_filtered_by_spy = [player for player in self.others() if player not in spies]

        players_sorted = self.get_players_probs(others_filtered_by_spy, sort=True)

        return [self] + players_sorted[:count - 1]

    def vote(self, team):
        """
        Voting is based if the player is a spy or not

        If it's not a spy then it will vote in favor if any of
        the following is true
            - Team is of two people
            - I'm on the team

        Otherwise, if the player's a spy, it will base his decision
        on the classifier
        """
        if not self.spy:
            return len(team) == 2 or self in team
        classifier_inputs = self.__get_classifier_input__()
        output = classifier.predict(classifier_inputs)[0]
        return bool(output)

    def sabotage(self):
        """
        Sabotage if it's the last mission,
        OR if ALL of the next happens:
            - The team is of at least 3 players
            - The team does not contains other spy
        """
        team = self.game.team
        spies_in_team = [player for player in team if player in self.spies]
        return (self.game.turn == 5) or \
            (len(team) == 3 and len(spies_in_team) == 1)
