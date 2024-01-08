### How to run:

* simple: run in terminal command: `python3 test_game.py`
* choose board: in file `test_game.py` uncomment *env* with **import_board** argument and run above command 
* infinite world: in above-mentioned file uncomment *env* with **size** argument and run command

### Implementation of env:

- [X] dynamic size of board
- [X] walls
- [X] import of board from txt file
- [X] start position: random
- [X] start position: selected
- [X] limited view
- [ ] dynamic appearing poisoned fruits

### Description of boards:

- **\'#\'**: means a wall
- any for letter: **\'W\'**, **\'S\'**, **\'A\'**, **\'D\'**, **\'R\'** means start position of snake. Letters 'wsad' also means the direction, in which 'R' sends for random direction 
- board must be a square

### SimpleObsEnv
This environment introduces a subclass of the primary SnakeEnv, designed to alter the observation space for a more limited view.
In this modified version, the snake's perception is restricted to detecting imminent dangers, such as its own body or walls, directly in front of it or to its left and right sides. 
Additionally, the snake no longer has direct knowledge of the fruit's location. Instead, it can sense the fruit's presence only when it is directly above, below, or to either side of it. 
The final aspect, concerning the snake's direction, remains unchanged.
    
This will simplify the Q-learning algorithm, making it easier to compute.
