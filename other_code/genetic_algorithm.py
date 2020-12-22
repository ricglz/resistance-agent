"""Genetic algorithm for finding the optimal weights"""
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import argmax
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

NAMES = [
    'MissionsBeenOn', 'FailedMissionsBeenOn', 'VotedUp0', 'VotedUp1', 'VotedUp2',
    'VotedUp3', 'VotedUp4', 'VotedUp5', 'VotedUp6', 'VotedDown0', 'VotedDown1',
    'VotedDown2', 'VotedDown3', 'VotedDown4', 'VotedDown5', 'VotedDown6', 'Spy',
]
INPUT_LEN = len(NAMES) - 1
OUTPUT_LEN = 2

train_x = []
train_y = []
val_x = []
val_y = []

def create_sets():
    """Creating training and validation sets"""
    global train_x, train_y, val_x, val_y

    print('Creating sets')

    dataframe = pd.read_csv('LoggerBot.log', names=NAMES).sample(frac=1)
    inputs = dataframe.values[:,:-1].astype(np.float32)
    outputs = dataframe.values[:,-1].astype(np.int32)

    train_set_size = int(len(dataframe) * 0.7)
    train_x, train_y = inputs[:train_set_size], outputs[:train_set_size]
    val_x, val_y = inputs[train_set_size:], outputs[train_set_size:]

def get_weights(layers):
    """Get weights of a keras layer"""
    get_layer_weights = lambda layer: layer.get_weights()[0]
    return list(map(get_layer_weights, layers))

class NeuralNetwork():
    accuracy = 0
    model = None
    layers = []

    # Constructor
    def __init__(self, child_weights=None):
        first_size, second_size = 10, 10
        model = Sequential(name='my_custom_neural_network')

        # If no weights provided randomly generate them
        if child_weights is None:
            # Layers are added to the model
            model.add(Dense(first_size, activation='tanh', input_shape=(INPUT_LEN,)))
            model.add(Dense(second_size, activation='tanh'))
            model.add(Dense(second_size, activation='tanh'))
            model.add(Dense(OUTPUT_LEN, activation='softmax'))
        # If weights are provided set them within the layers
        else:
            # Set weights within the layers
            model.add(
                Dense(
                    first_size,
                    input_shape=(INPUT_LEN,),
                    activation='tanh',
                    weights=[child_weights[0], np.zeros(first_size)],
                    name='inputs')
                )
            model.add(
                Dense(
                 second_size,
                 activation='tanh',
                 weights=[child_weights[1], np.zeros(second_size)],
                 name='dense_1')
            )
            model.add(
                Dense(
                 second_size,
                 activation='tanh',
                 weights=[child_weights[2], np.zeros(second_size)],
                 name='dense_2')
            )
            model.add(
                Dense(
                 OUTPUT_LEN,
                 weights=[child_weights[3], np.zeros(OUTPUT_LEN)],
                 name='predictions',
                 activation='softmax')
            )
        self.model = model
        self.layers = self.model.layers

    def __gt__(self, other):
        return self.accuracy > other.accuracy

    # Function for forward propagating a row vector of a matrix
    def forward_propagation(self):
        """
        Forward propagate to know which is the current accuracy of
        the model based on the training data
        """
        pred_y = argmax(self.model.predict(train_x), axis=1)

        accuracy_func = Accuracy()
        accuracy_func.update_state(pred_y, train_y)
        self.accuracy = accuracy_func.result().numpy()

    # Standard Backpropagation
    def compile_train(self, epochs):
        """Compile and fit the model through different epochs"""
        self.model.compile(
            optimizer=Adam(0.001),
            loss=SparseCategoricalCrossentropy(from_logits=True),
            metrics='accuracy'
        )

        self.model.fit(
            train_x,
            train_y,
            batch_size=50,
            epochs=epochs,
            validation_data=(val_x, val_y),
            verbose=1
        )

    def save_accuracy_chart(self):
        """Save figure of the accuracy as an image"""
        history = self.model.history.history
        fig = plt.figure()
        plt.plot(history['accuracy'], label='Training Accuracy')
        plt.plot(history['val_accuracy'],label='Validation Set Accuracy')
        plt.legend()
        fig.savefig('model_accuracy.png')

# Chance to mutate weights
def mutation(child_weights):
    """Mutation function
    This function has the possibility of mutating each of the weights
    """
    for index, _ in enumerate(child_weights):
        # Add a chance for random mutation
        has_mutation = random.uniform(0, 1)
        if has_mutation <= .1:
            child_weights[index] *= random.randint(0, 5)

def dynamic_crossover(nn1, nn2):
    """Performs a crossover for mating the 2 neural networks"""
    # Lists for respective weights
    nn1_weights = get_weights(nn1.layers)
    nn2_weights = get_weights(nn2.layers)
    child_weights = []

    # Iterate through all weights from all layers for crossover
    for index, _ in enumerate(nn1_weights):
        # Get single point to split the matrix in parents based on # of cols
        coulmns = np.shape(nn1_weights[index])[1]-1
        split = random.randint(0, coulmns)
        # Iterate through after a single point and set the remaing cols to nn_2
        for j in range(split, coulmns):
            nn1_weights[index][:, j] = nn2_weights[index][:, j]

        # After crossover add weights to child
        child_weights.append(nn1_weights[index])

    # Add a chance for mutation
    mutation(child_weights)

    # Create and return child object
    return NeuralNetwork(child_weights)

def propagate_networks(networks, pool):
    """Forward propagate the neural networks to compute a accuracy score"""
    for network in networks:
        network.forward_propagation()
        pool.append(network)

def tournament_selection(pool):
    """Tournament selection of the pool"""
    return max(random.sample(pool, len(pool) // 5))

def genetic_algorithm():
    """Genetic algorithm used"""
    generation, pop_size, max_gen = 1, 2, 10

    # Create a List of all active GeneticNeuralNetworks
    networks = [NeuralNetwork() for _ in range(pop_size)]
    pool = []

    # Cache max accuracy
    max_accuracy = 0
    max_pool_size = pop_size * 5
    # max accuracy Weights
    optimal_weights = []

    top_amount = 5
    top_partners = pop_size // top_amount

    # Evolution Loop
    while max_accuracy < 0.9 and generation <= max_gen:
        # Log the current generation
        print(f'Generation: {generation}')

        propagate_networks(networks, pool)

        # Clear for propagation of next children
        networks.clear()

        # Sort based on accuracy
        pool.sort(reverse=True)

        # To avoid a very big pool I will slice the list with the best of them
        pool = pool[:max_pool_size]

        # Find Max accuracy and Log Associated Weights
        best_individual = pool[0]
        if best_individual.accuracy > max_accuracy:
            max_accuracy = best_individual.accuracy
            print(f'Max accuracy: {max_accuracy}')
            # Iterate through layers, get weights, and append to optimal
            optimal_weights = get_weights(best_individual.layers)

        # Crossover, top 5 randomly select 2 partners for child
        for best in pool[:top_amount]:
            for _ in range(top_partners):
                # Create a child and add to networks
                best_of_tournament = tournament_selection(pool)

                # Add to networks to calculate accuracy score next iteration
                networks.append(dynamic_crossover(best, best_of_tournament))

        generation += 1

    return optimal_weights

def obtain_best_model(optimal_weights):
    """Create best model based on the optimal weights"""
    gnn = NeuralNetwork(optimal_weights)
    gnn.compile_train(5)

    gnn.save_accuracy_chart()

    gnn.model.save('spy_classifier')

def main():
    """Main process of the algorithm"""
    create_sets()
    optimal_weights = genetic_algorithm()
    obtain_best_model(optimal_weights)

if __name__ == "__main__":
    main()
