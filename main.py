import pygame as pg
import pickle
from game import Session
from trainer import GeneticAlgo
from pathlib import Path
from os import listdir




if __name__ == '__main__':
    
    pg.init()
    WHITE = (255, 255, 255)
    WIDTH, HEIGHT = 1200, 900 
    screen = pg.display.set_mode((WIDTH, HEIGHT))
    pg.display.set_caption("Start Screen")

    # Chargement de l'image de fond
    background = pg.image.load("media/start.png")  # Remplace avec ton image
    background = pg.transform.scale_by(background, 0.6)
    
    title = pg.font.SysFont('forte', 60)
    text = pg.font.SysFont('forte', 30)

    title_text = title.render('Welcome on Race AI!', True, WHITE)
    start_text = text.render('Press any key to start', True, WHITE)

    

    # Définition des zones cliquables (x, y, largeur, hauteur)
    zones = {
        "Human": pg.Rect(150, 200, 200, 50),
        "AI": pg.Rect(150, 300, 200, 50),
        "Train": pg.Rect(150, 400, 200, 50),
        "Quit": pg.Rect(150, 500, 200, 50)
        }



    running = True

    while running:
        screen.blit(background, (0, -30))
        screen.blit(title_text, (WIDTH/2-250, HEIGHT/2-320))
        screen.blit(start_text, (WIDTH/2-100, HEIGHT/2 + 180))
        
        # Dessiner les zones (pour visualiser les boutons)
        for zone in zones.values():
            pg.draw.rect(screen, (0, 0, 255), zone, 2)  # Contour rouge
        
        pg.display.flip()
        
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            elif event.type == pg.MOUSEBUTTONDOWN:
                for key, zone in zones.items():
                    if zone.collidepoint(event.pos):
                        
                        if key == "Human":
                            running = False
                            
                            agent = None
                            train = False
                            display = True
                            training_time = None
                            
                            ses = Session(train, agent, display, training_time)
                            ses.run()
                            
                            
                            
                        elif key == "AI":
                            running = False
                            
                            agent = "4.376.pilot"
                            train = False
                            display = True
                            training_time = None
                            
                            ses = Session(train, agent, display, training_time)
                            ses.run()
                            
                            
                            
                        elif key == "Train":
                            running = False
                            
                            population = 10 #100
                            maxGenerations = 10 #50 
                            mutation_rate = 0.01
                            survival_rate = 0.1
                            
                            # Autres paramètres :
                            # nombre de layers NN (adn)
                            # parametre 0.7 pour la sélection des neurones (pilot)
                            # temps d'entrainement de chaque pilote (main)
                            # fps acceleration training
                            
                            
                            algo = GeneticAlgo(population, maxGenerations, mutation_rate, survival_rate)
                            algo.train()
                            
                            
                            # Save the weights and biases of the snakes for the new game scores
                            files = listdir(Path("weights"))
                            
                            with open(Path("weights") / Path(f"{algo.bestScore:.2f}.pilot"), "wb") as f: # write binary
                                pickle.dump((algo.bestPilotEver.adn.weights, algo.bestPilotEver.adn.bias), f)

                            
                            
                        elif key == "Quit":
                            running = False


    pg.quit()
    exit()




# Play game 
# Launch AI player
# Train AI Genetic
# Quit OU Train AI Reinforcement







    title = pg.font.SysFont('forte', 60)
    text = pg.font.SysFont('forte', 30)

    title_text = title.render('Welcome on Flappy Bird!', True, WHITE)
    start_text = text.render('Press any key to start', True, WHITE)

    screen.blit(title_text, (WIDTH/2-250, HEIGHT/2-80))
    screen.blit(start_text, (WIDTH/2-100, HEIGHT/2 + 180))