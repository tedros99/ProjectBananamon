import time
import argparse
import numpy as np
import os
import pdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input
from keras.optimizers import *

import pkmn_visuals

ROOT = os.getcwd()  # root folder of this code


parser = argparse.ArgumentParser(description="Train a perceptron to classify pokemon by their stats.")
parser.add_argument('-t', '--test',
                    help='path to testing file, defaults to ROOT/pokemon_data_test.csv',
                    default=os.path.join(ROOT, 'data\pokemon_new_data.csv'))
parser.add_argument('-s', '--stats',
                    help='path to training file, defaults to ROOT/PokemonData.csv',
                    default=os.path.join(ROOT, 'data\pokemon_base_stats.csv'))

PKMN_TYPES = ["Normal", "Fire", "Water", "Grass", "Flying", "Fighting", 
"Poison", "Electric", "Ground", "Rock", "Psychic", "Ice", 
"Bug", "Ghost", "Steel", "Dragon", "Dark", "Fairy"]

def main(args):
    testfile = os.path.expanduser(args.test)
    statsfile = os.path.expanduser(args.stats)

    print("loading files...")
    testcsv = np.loadtxt(testfile, dtype=str, delimiter=",")
    statscsv = np.loadtxt(statsfile, dtype=str, delimiter=",")

    print("creating label mask...")
    label_mask = np.array([[1 if type in pkmn[2:4] else 0 for type in PKMN_TYPES] for pkmn in statscsv[1:, :]])
    test_label_mask = np.array([[1 if type in pkmn[2:4] else 0 for type in PKMN_TYPES] for pkmn in testcsv[1:, :]])

    print("getting pkmn stats...")
    pkmn_stats = np.array(statscsv[1:, 4:10], dtype=int)
    test_pkmn_stats = np.array(testcsv[1:, 4:10], dtype=int)

    # lets try using the RATIO of stats!
    pkmn_stats_ratios = np.array([[pkmn_stats[row, col] / sum(pkmn_stats[row]) for col in range(len(pkmn_stats[row]))] for row in range(len(pkmn_stats))])
    # nts did not work with the same model: try other models!
    print("creating the MLP classifier...")
    model = Sequential()
    model.add(Input(shape=(len(pkmn_stats[0]),),name='input'))
    model.add(Dense(units=len(PKMN_TYPES) ** 2, activation="relu", name="type1_hidden"))
    model.add(Dense(units=len(PKMN_TYPES) ** 2, activation="relu", name="type2_hidden"))
    model.add(Dense(units=len(PKMN_TYPES) ** 2, activation="relu", name="extra_hidden_layer"))
    model.add(Dense(units=len(PKMN_TYPES) ** 2, activation="relu", name="extra_hidden_layer2"))
    model.add(Dense(units=len(PKMN_TYPES) ** 2, activation="relu", name="extra_hidden_layer3"))
    model.add(Dense(units=len(PKMN_TYPES), activation="softmax", name="output"))
    model.summary()

    model.compile(
        loss='binary_crossentropy', # current best: binary_crossentropy, mse
        optimizer=Adagrad(learning_rate=0.01), # Adagrad for base stats
        metrics=['accuracy'])

    print("training the MLP classifier...")
    history = model.fit(x=pkmn_stats, y=label_mask,
        batch_size=10,
        epochs=200,
        verbose=1)

    # test the network on training data
    print("testing the MLP classifier on the training data...")
    metrics = model.evaluate(x=pkmn_stats, y=label_mask, verbose=0)
    print(f"accuracy = {metrics[1]}")

    # test the network on training data
    print("testing the MLP classifier on the testing data...")
    metrics = model.evaluate(x=test_pkmn_stats, y=test_label_mask, verbose=0)
    print(f"accuracy = {metrics[1]}")

    labels = model.predict(pkmn_stats) # shape (721, 18)
    type_predictions = np.array([np.argsort(pkmn)[-2:][::-1] for pkmn in softmax(labels)])
    full_type_prediction_probabilities = np.array([softmax(labels[n]) for n in range(len(labels))])
    type_prediction_probabilities = np.array(
        [[softmax(labels[n])[type_predictions[n, 0]], 
        softmax(labels[n])[type_predictions[n, 1]]] 
        for n in range(len(labels))]
        )
    # type_predictions contains the 2 most predicted types for all pkmn
    # type_prediction_probabilities contains the probabilities corresponding to those 2 types for all pkmn



    # display the visuals
    pkmn_visuals.main(type_predictions, type_prediction_probabilities)




def softmax(x):
    return(np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum())


if __name__ == "__main__":
    main(parser.parse_args())



