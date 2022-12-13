# Project Bananamon
## Description:
  Our basic idea was to determine various factors of pokemon given their base stats. These factors include their type, BMI, and the generation they come from. There are two machine learning algorithms used in this project: neural networks and decision trees.

## Installation:
  - Python 3.7 or higher required. 
  - Using the Python pip installer, you must install various packages;
    - `pip install tensorflow`
    - `pip install numpy`
    - `pip install pdb`
    - `pip install Pillow`
    - `pip install tkinter`
    - `pip install matplotlib`
    - `pip install sklearn`
## Usage:
  - There are 3 separate code files which can be run independently depending on what you want to see:
    - `BMIPrediction.py` (bmi) located at `~/project_bananamon/pkmn_bmi/BMIPrediction.py`
    - `pkmn_mlp.py` (type) located at `~/project_bananamon/pkmn_perceptron/pkmn_mlp.py`
    - `dtree_kfold.py` (generation) located at `~/project_bananamon/pkmn_dtree/dtree_kfold.py`
    - In order to run the programs, you should navigate your IDE to the main (Outer) folder before attempting to run, as shown using git bash
    - ![Example navigation](https://cdn.discordapp.com/attachments/721541122094006292/1052035833143754804/image.png)
  - Note that the datasets required for each program are already included at `~/project_bananamon/data`
  - To run the program, the user can use any typical method, however using the git bash is recommended
  - Examples of the various programs in action:
  - ![pkmn_perceptron](https://cdn.discordapp.com/attachments/721541122094006292/1052038955211698227/image.png)
  - ![pkmn_dtree](https://cdn.discordapp.com/attachments/721541122094006292/1052039667672948808/image.png)
  - ![pkmn_bmi](https://cdn.discordapp.com/attachments/721541122094006292/1052040559952416848/image.png)
## Credits:
  - `pkmn_bmi` by Quade Leonard
  - `pkmn_perceptron` by Devan Burke
  - `pkmn_dtree` by Tedros Lafalaise
  - **Pokemon** by MICHAEL LOMUSCIO from [Kaggle](https://www.kaggle.com/datasets/mlomuscio/pokemon)
    - dataset containing the types and stats for all pokemon through generation 7
  - **The Complete Pokemon Dataset** by ROUNAK BANIK from [Kaggle](https://www.kaggle.com/datasets/rounakbanik/pokemon)
    - dataset containing the weights and heights for all pokemon through generation 7
  - **Pokemon Images Dataset** by KVPRATAMA from [Kaggle](https://www.kaggle.com/datasets/kvpratama/pokemon-images-dataset)
    - dataset containing images for all the pokemon through generation 7
  - **Pokemon Scarlet and Violet Data** by Serebii.net from [their website](https://www.serebii.net/scarletviolet/pokemon.shtml)
  - Special thanks to Tedros for compiling, sorting, and reformatting all the data
  
    
