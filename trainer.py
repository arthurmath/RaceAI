from pilot import Pilot
from adn import Adn
from main import Session






if __name__ == "__main__":
    ses = Session()
    

    population = 1000
    layers = [20, 10]
    mutation = 0.01
    hunger = 150
    elitism = 0.12
    snakesManager = SnakesManager(
        ses,
        population,
        layersSize=layers,
        mutationRate=mutation,
        hunger=hunger,
        survivalProportion=elitism,
    )
    snakesManager.train()