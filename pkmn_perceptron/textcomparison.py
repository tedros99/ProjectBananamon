import time
import argparse
import numpy as np
import os
import pdb

ROOT = os.path.dirname(os.path.abspath(__file__))  # root directory of this code


parser = argparse.ArgumentParser(description="Compare the names of pokemon between 2 csv's.")
parser.add_argument('-w', '--weight',
                    help='path to weight file, defaults to ROOT/pokemon.csv',
                    default=os.path.join(ROOT, 'pokemon.csv'))
parser.add_argument('-s', '--stats',
                    help='path to stats file, defaults to ROOT/PokemonData.csv',
                    default=os.path.join(ROOT, 'PokemonData.csv'))


def main(args):
    weightfile = os.path.expanduser(args.weight)
    statsfile = os.path.expanduser(args.stats)

    print("loading files...")
    weightcsv = np.loadtxt(weightfile, dtype=str, delimiter=",")
    statscsv = np.loadtxt(statsfile, dtype=str, delimiter=",")
    pdb.set_trace()

    print("comparing data")
    for index in range(len(weightcsv)):
        print("row" + str(index))
        print("Name equality:" + str((weightcsv[index, 1] == statscsv[index, 1])))

        if (index % 20 == 0):
            answer = input("continue for 20?")
            if (answer == "n"):
                break


if __name__ == "__main__":
    main(parser.parse_args())



