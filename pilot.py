from adn import Adn
from gameModule import (
    RIGHT,
    LEFT,
    DOWN,
    UP,
    SNAKE_CHAR,
    FOOD_CHAR,
    WALL_CHAR,
)


class Pilot:
    """ Represents the AI that plays RaceAI """

    def __init__(self, adn): 
        self.adn = adn
        self.fitness = None
        self.nbrMove = 0
        self.previous_moves = []
        self.MOVEMENT = (RIGHT, LEFT, UP, DOWN)

    def choose_next_move(self, state):
        """ Choose a new move based on its state.
        Return the movement choice of the snake (tuple) """
        
        vision = self.get_simplified_state(state)
        
        self.nbrMove += 1
        movesValues = self.adn.neural_network_forward(vision) 
        choice = 0

        movesValues = movesValues.tolist()

        # Chooses the best move (the move with the highest value)
        for i in range(1, len(movesValues[0])):
            if movesValues[0][i] > movesValues[0][choice]:
                choice = i

        
        self.previous_moves.append(self.MOVEMENT[choice])
        if len(self.previous_moves) >= 3:
            self.previous_moves.pop(0)
        return self.MOVEMENT[choice]

    def get_simplified_state(self, state):
        """
        returns a matrix of elements surrounding the snake and the previous two
        moves, this serves as the input for the neural network.
        """
        res = self.get_line_elem(RIGHT, state)
        res += self.get_line_elem((DOWN[0], RIGHT[1]), state)
        res += self.get_line_elem(DOWN, state)
        res += self.get_line_elem((DOWN[0], LEFT[1]), state)
        res += self.get_line_elem(LEFT, state)
        res += self.get_line_elem((UP[0], LEFT[1]), state)
        res += self.get_line_elem(UP, state)
        res += self.get_line_elem((UP[0], RIGHT[1]), state)

        if len(self.previous_moves) == 0:
            res += [0, 0]
        elif len(self.previous_moves) == 1:  # previous previous move
            res += [
                self.previous_moves[0][0] / 2,
                self.previous_moves[0][1] / 2,
            ]
        else:
            res += [
                self.previous_moves[0][0] + self.previous_moves[1][0] / 2,
                self.previous_moves[0][1] + self.previous_moves[1][1] / 2,
            ]

        return res

    def get_line_elem(self, direction, state):
        """
        returns a list of all elements in a straight line in a certain direction
        from the head of the snake
        """
        grid, score, alive, snake = state
        res = [0, 0, 0]  # food, snake, wall
        current = (snake[0][0] + direction[0], snake[0][1] + direction[1])
        distance = 1  # Distance between the snake head and current position

        while 0 in res:
            if FOOD_CHAR == grid[current[0]][current[1]]:
                res[0] = 1 / distance
            elif not res[1] and SNAKE_CHAR == grid[current[0]][current[1]]:
                res[1] = 1 / distance
            elif not res[2] and WALL_CHAR == grid[current[0]][current[1]]:
                res[2] = 1 / distance

            current = (current[0] + direction[0], current[1] + direction[1])
            distance += 1

        # For the border of the board (!= WALL_CHAR)
        if res[2] == 0:
            res[2] = 1 / distance

        return res


    def compute_fitness(self, gameScore):
        """
        Compute the fitness of the snake based on the number of moves done and its game score.
        Return the snake's fitness (float)
        """
        self.fitness = self.nbrMove * gameScore ** 2

        return self.fitness
    
            
    def mate(self, other, mutationRate=0.01):
        """ Mate with another state to create a new pilot """
        newDna = self.dna.mix(other.dna, mutationRate)
        return Pilot(newDna)
    

    def reset_state(self):
        
        self.nbrMove = 0
