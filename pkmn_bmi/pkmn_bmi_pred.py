import matplotlib.pyplot as plt
import numpy as np
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

    BMI_train = np.array(weight_train / (height_train**2))
    BMI_test = np.array(weight_test / (height_test**2))

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

    # Network training
    history_height = height.fit(x_train, height_train,
        epochs=250,
        batch_size=5,
        verbose=1,
        validation_split = 0.3
    )

    train_metrics = height.evaluate(x_train, height_train, verbose=0)
    print('TRAINING DATA')
    print(f'height loss = {train_metrics[0]:0.4f}')

    print()

    test_metrics = height.evaluate(x_test, height_test, verbose=0)
    print('TESTING DATA')
    print(f'height loss = {test_metrics[0]:0.4f}')

    weight = Sequential()
    weight.add(Dense(12, input_dim=6, kernel_initializer='normal', activation='relu'))
    weight.add(Dense(16, activation='relu'))
    weight.add(Dense(32, activation='relu'))
    weight.add(Dense(64, activation='relu'))
    weight.add(Dense(32, activation='relu'))
    weight.add(Dense(16, activation='relu'))
    weight.add(Dense(1, activation='linear'))
    weight.summary()

    input('Press <enter> to continue')

    weight.compile(
        loss='mse',
        optimizer=Adam(),
        metrics=['mse']
    )

    history_weight = weight.fit(x_train, weight_train,
        epochs=300,
        batch_size=5,
        verbose=1,
        validation_split = 0.3
    )

    train_metrics = weight.evaluate(x_train, weight_train, verbose=0)
    print('TRAINING DATA')
    print(f'weight loss = {train_metrics[0]:0.4f}')
    print()

    test_metrics = weight.evaluate(x_test, weight_test, verbose=0)
    print('TESTING DATA')
    print(f'weight loss = {test_metrics[0]:0.4f}')

    train_h_labels = height.predict(x_train, verbose=0)
    train_w_labels = weight.predict(x_train, verbose=0)

    test_h_labels = height.predict(x_test, verbose=0)
    test_w_labels = weight.predict(x_test, verbose=0)

    x_train_sums = np.array([np.sum(x) for x in x_train])
    x_test_sums = np.array([np.sum(x) for x in x_test])

    BMI_train_pred = np.array(train_w_labels / (train_h_labels**2))
    BMI_test_pred = np.array(test_w_labels / (test_h_labels**2))

    input('Press <enter> to continue')

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

    plt.show(block=False)

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

    while(True):
        input("<enter> to predict your own pokemon!")
        hp = input("HP: ")
        attack = input("Attack: ")
        sp_attack = input("Special Attack: ")
        defence = input("Defence: ")
        sp_defence = input("Special Defence: ")
        speed = input("Speed: ")
        
        your_stats = np.array([[hp, attack, sp_attack, defence, sp_defence, speed]], dtype=float)
        your_height = height.predict(your_stats, verbose=0)
        your_weight = weight.predict(your_stats, verbose=0)
        your_BMI = your_weight / (your_height**2)

        print(f'Height: {your_height[0]}')
        print(f'Weight: {your_weight[0]}')
        print(f'BMI: {your_BMI[0]}')
        print()
        yn = input("Would you like to make another pokemon? y/n ")
        
        if(yn == 'n'):
            break


if __name__ == "__main__":
    main()
