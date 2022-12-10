from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
import os

ROOT = os.getcwd()  # root folder of this code
# find data files
train_data_file = os.path.join(ROOT, 'data/pokemon_data_fixed.csv')
train_labels_file = os.path.join(ROOT, 'data/pokemon_data_labels.csv')
attributes_file = os.path.join(ROOT, 'data/attributes.txt')
test_data_file = os.path.join(ROOT, 'data/pokemon_data_test_fixed.csv')
test_labels_file = os.path.join(ROOT, 'data/pokemon_data_test_labels.csv')

# Load all the data
train_data = np.loadtxt(train_data_file, dtype=int, delimiter=",")
train_labels = np.loadtxt(train_labels_file, dtype=int, delimiter=",")
attributes = np.loadtxt(attributes_file, dtype=str)
test_data = np.loadtxt(test_data_file, dtype=float, delimiter=",")
test_labels = np.loadtxt(test_labels_file, dtype=float, delimiter=",")

# Init the base decision tree
dtree = DecisionTreeClassifier(random_state=5)

params = {
    'max_depth': range(1, 20),
    'criterion': ["gini", "entropy"],
    'min_samples_leaf': range(1, 20),
    'min_samples_split': range(2, 20)
}

# Train the tree using k-fold
print("Using k-fold cross validation to train the tree... (May take some time...)")
k_dtree = GridSearchCV(dtree, params, n_jobs=2)
k_dtree.fit(train_data, train_labels)
print("\nHighest model score:", k_dtree.best_score_, " | Best (hyper)parameters:", k_dtree.best_params_, "\n")
final_params = k_dtree.best_params_

# Use the parameters from the supposed best model
new_tree = DecisionTreeClassifier(
    max_depth=k_dtree.best_params_['max_depth'],
    criterion=k_dtree.best_params_['criterion'],
    min_samples_leaf=k_dtree.best_params_['min_samples_leaf'],
    min_samples_split=k_dtree.best_params_['min_samples_split'],
)

new_tree.fit(train_data, train_labels)

y_pred = new_tree.predict(test_data)
acc1 = np.mean(test_labels == y_pred)
y_pred2 = new_tree.predict(train_data)
acc2 = np.mean(train_labels == y_pred2)
print('Testing accuracy: {:.2f}'.format(acc1))
print('Training accuracy: {:.2f}\n'.format(acc2))

print("Confusion matrix:")
cm = confusion_matrix(test_labels, y_pred)
print(cm)

print("\nPlotting the tree...")
plot_tree(new_tree, feature_names=attributes, class_names=[str(x) for x in range(7)], fontsize=6, filled=True, rounded=True)
plt.show()

#Test the model with custom input!
user = input("Would you like to test the model with your own pokemon? (Y/N)\n")
if user == "Y" or "y":
    num = int(input("Please input how many you would like to test: "))
    for i in range(num):
        print("Please input your pokemon's data:")
        pkmn_hp = input("Base health:\n")
        pkmn_atk = input("Base attack:\n")
        pkmn_def = input("Base defense:\n")
        pkmn_spatk = input("Base special attack:\n")
        pkmn_spdef = input("Base special defense:\n")
        pkmn_spd = input("Base speed:\n")
        final = [[pkmn_hp, pkmn_atk, pkmn_def, pkmn_spatk, pkmn_spdef, pkmn_spd]]
        print("Your pokemon is predicted to be in Gen", new_tree.predict(final)[0])
