import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pdb
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasRegressor

ROOT = os.path.dirname(os.path.abspath(__file__))

def main():
    # Load data
    print('Loading data...')
    basePokemonCSV = np.loadtxt(os.path.join(ROOT, 'PokemonData.csv'), dtype=str, delimiter=',')
    BMIPokemonCSV = np.loadtxt(os.path.join(ROOT, 'pokemon.csv'), dtype=str, delimiter=',')
    pokemonTestData = np.loadtxt(os.path.join(ROOT, 'pokemon_data_test.csv'), dtype=str, delimiter=',')

    x_train = np.array(basePokemonCSV[1:, 4:10], dtype=float)
    height_train = np.array(BMIPokemonCSV[1:, 2], dtype=float)
    weight_train = np.array(BMIPokemonCSV[1:, 3], dtype=float)

    x_test = np.array(pokemonTestData[1:, 4:10], dtype=float)
    height_test = np.array(pokemonTestData[1:, 10], dtype=float)
    weight_test = np.array(pokemonTestData[1:, 11], dtype=float)

    # BMI_train = np.array([(float(i[1]) / (float(i[0])**2), 4) for i in t_train])
    # BMI_test = np.array([(float(i[1]) / (float(i[0])**2), 4) for i in t_test])

    # pdb.set_trace()

    height = Sequential()
    height.add(Dense(12, input_dim=6, kernel_initializer='normal', activation='relu'))
    height.add(Dense(16, activation='relu'))
    height.add(Dense(32, activation='relu'))
    height.add(Dense(16, activation='relu'))
    height.add(Dense(1, activation='linear'))
    height.summary()

    height.compile(
        loss='mse',
        optimizer=Adam(),
        metrics=['mse']
    )

    input('Press <enter> to continue')


    callback = EarlyStopping(
        monitor='loss',
        min_delta=1e-3,
        patience=10,
        verbose=1
    )

    # Network training
    history_height = height.fit(x_train, height_train,
        epochs=250,
        batch_size=5,
        callbacks=[callback],
        verbose=1,
        validation_split = 0.3
    )

    weight = Sequential()
    weight.add(Dense(12, input_dim=6, kernel_initializer='normal', activation='relu'))
    weight.add(Dense(32, activation='relu'))
    weight.add(Dense(64, activation='relu'))
    weight.add(Dense(32, activation='relu'))
    weight.add(Dense(1, activation='linear'))
    weight.summary()

    input('Press <enter> to continue')

    weight.compile(
        loss='mse',
        optimizer=Adam(),
        metrics=['mse']
    )

    history_weight = weight.fit(x_train, weight_train,
        epochs=250,
        batch_size=5,
        callbacks=[callback],
        verbose=1,
        validation_split = 0.3
    )

    train_metrics = height.evaluate(x_train, height_train, verbose=0)
    print('TRAINING DATA')
    print(f'height loss = {train_metrics[0]:0.4f}')
    train_metrics = weight.evaluate(x_train, weight_train, verbose=0)
    print(f'weight loss = {train_metrics[0]:0.4f}')
    print()

    test_metrics = height.evaluate(x_test, height_test, verbose=0)
    print('TESTING DATA')
    print(f'height loss = {test_metrics[0]:0.4f}')
    test_metrics = weight.evaluate(x_test, weight_test, verbose=0)
    print(f'weight loss = {test_metrics[0]:0.4f}')

    train_h_labels = height.predict(x_train, verbose=0)
    train_w_labels = weight.predict(x_train, verbose=0)

    test_h_labels = height.predict(x_test, verbose=0)
    test_w_labels = weight.predict(x_test, verbose=0)

    x_train_sums = np.array([np.sum(x) for x in x_train])
    x_test_sums = np.array([np.sum(x) for x in x_test])

    plt.figure('stats to height')
    plt.axis([0,800,0,14])
    plt.plot(x_train_sums, height_train, 'b.', x_test_sums, height_test, 'r.')
    plt.title("TARGETS")
    plt.xlabel("BASE STAT TOTALS")
    plt.ylabel("HEIGHT")

    plt.figure('stats to height pred')
    plt.axis([0,800,0,14])
    plt.plot(x_train_sums, train_h_labels, 'b.', x_test_sums, test_h_labels, 'r.')
    plt.title("PREDICTIONS")
    plt.xlabel("BASE STAT TOTALS")
    plt.ylabel("HEIGHT")

    plt.show()

    input('Press <enter> to continue')

    plt.figure('stats to weight')
    plt.axis([0,800,0,800])
    plt.plot(x_train_sums, weight_train, 'b.', x_test_sums, weight_test, 'r.')
    plt.title("TARGETS")
    plt.xlabel("BASE STAT TOTALS")
    plt.ylabel("WEIGHT")

    plt.figure('stats to weight pred')
    plt.axis([0,800,0,800])
    plt.plot(x_train_sums, train_w_labels, 'b.', x_test_sums, test_w_labels, 'r.')
    plt.title("PREDICTIONS")
    plt.xlabel("BASE STAT TOTALS")
    plt.ylabel("WEIGHT")

    plt.show()

if __name__ == "__main__":
    main()
