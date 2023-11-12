### Current stage of development:
- Creating the environment for RL
- Testing environment on first reinforcement learning

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
- [ ] limited view
- [ ] dynamic appearing poisoned fruits

### Description of boards:

- **\'#\'**: means a wall
- any for letter: **\'W\'**, **\'S\'**, **\'A\'**, **\'D\'**, **\'R\'** means start position of snake. Letters 'wsad' also means the direction, in which 'R' sends for random direction 
- board must be a square