import os

import tkinter as tk
from PIL import Image, ImageTk

INDEX = 0
TYPE_PREDICTIONS = None
TYPE_PREDICTION_PROBABILITIES = None
type_1_label, type_2_label, type_1_probability_label, type_2_probability_label, index_text_box = None, None, None, None, None

PKMN_TYPES = ["Normal", "Fire", "Water", "Grass", "Flying", "Fighting", 
"Poison", "Electric", "Ground", "Rock", "Psychic", "Ice", 
"Bug", "Ghost", "Steel", "Dragon", "Dark", "Fairy"]

ROOT = os.path.dirname(os.path.abspath(__file__))  # root directory of this code
def main(type_predictions, type_prediction_probabilities):
    global TYPE_PREDICTIONS
    global TYPE_PREDICTION_PROBABILITIES

    global type_1_label, type_2_label, type_1_probability_label, type_2_probability_label, index_text_box


    TYPE_PREDICTIONS = type_predictions
    TYPE_PREDICTION_PROBABILITIES = type_prediction_probabilities


    root = tk.Tk()
    root.geometry("400x400")


    type_1_label = tk.Label(master=root, text = "Type 1 Label")
    type_1_label.place(x=50, y=275)

    type_2_label = tk.Label(master=root, text = "Type 2 Label")
    type_2_label.place(x=200, y=275)

    type_1_probability_label = tk.Label(master=root, text = "Type 1 probability")
    type_1_probability_label.place(x=50, y=325)

    type_2_probability_label = tk.Label(master=root, text = "Type 2 probability")
    type_2_probability_label.place(x=200, y=325)

    index_text_box = tk.Text(master=root, height=1, width = 12)
    index_text_box.place(x=275, y=150)
    

    button = tk.Button(master=root,
        command=button_command, 
        text = "Next pokemon?")
    button.place(x=275, y=50)

    button2 = tk.Button(master=root,
        command=set_index, 
        text = "Go!")
    button2.place(x=275, y=170)

    root.mainloop()

def get_pokemon_image(index):
    path = os.path.join(ROOT, f'pokemon/{index+1}.png')
    pil_image = Image.open(path)
    return ImageTk.PhotoImage(pil_image)
def place_pokemon_image(image):
    label = tk.Label(image=image)
    label.image = image

    label.place(x=0, y=0)


def button_command():
    global INDEX
    global TYPE_PREDICTIONS
    global TYPE_PREDICTION_PROBABILITIES

    global type_1_label, type_2_label, type_1_probability_label, type_2_probability_label

    # update pokemon image
    image = get_pokemon_image(INDEX)
    place_pokemon_image(image)

    # update type labels
    type_1_label.config(text= f"{PKMN_TYPES[TYPE_PREDICTIONS[INDEX][0]]}")
    type_2_label.config(text= f"{PKMN_TYPES[TYPE_PREDICTIONS[INDEX][1]]}")
    # update type percentages
    type_1_probability_label.config(text = f"{TYPE_PREDICTION_PROBABILITIES[INDEX][0] * 100:.2f}%")
    type_2_probability_label.config(text = f"{TYPE_PREDICTION_PROBABILITIES[INDEX][1] * 100:.2f}%")
    
    INDEX += 1
def set_index():
    global INDEX
    global index_text_box

    input = index_text_box.get('1.0', 'end-1c')
    input = int(input)

    INDEX = input - 1
    button_command()

if __name__ == "__main__":
    main([[3, 10]], [[0.11257, 0.05623]]) # just bulbasaur