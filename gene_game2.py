import sys
import os
import math
import numpy as np
import library as lib
from pathlib import Path
from gene_pilot import Pilot
import pickle
import pygame as pg



FPS = 50
WIDTH = 1300
HEIGHT = 900
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)



    
class Car:
    def __init__(self, ses): 
        self.initial_pos = 230, 275
        self.x, self.y = self.initial_pos
        self.angle = 0
        self.speed = 0
        self.acceleration = 0.2
        self.rotation_speed = 9
        self.max_speed = 10
        self.progression = 0
        self.old_progression = 0
        self.alive = True
        
        self.car_img = ses.car_img
        self.car_rotated = pg.transform.rotate(self.car_img, self.angle)
        self.car_rect = self.car_rotated.get_rect()
        self.font = pg.font.Font(pg.font.match_font('arial'), 16)
        
        self.collision = 0
        self.compteur = 0 # pour les collisions
        self.nbCollisions = 0
        
        
        self.checkpoints = [(239, 273), (239, 130), (300, 75), (360, 130), (360, 392), (420, 451), (479, 389), (479, 126), 
                            (531, 80), (941, 80), (988, 127), (988, 240), (940, 278), (680, 278), (614, 341), (681, 400), 
                            (941, 400), (986, 440), (986, 750), (941, 800), (890, 800), (840, 751), (840, 583), (780, 532), 
                            (680, 532), (620, 582), (620, 760), (540, 810), (238, 515), (238, 352)]

        self.total_distance = sum([math.dist(self.checkpoints[i], self.checkpoints[i + 1]) for i in range(len(self.checkpoints)-1)])
        
        self.validate_checkpoint = 0 #validation du checkpoint 0 / checkpoint 0 = position initiale i.e ligne d'arrivée
        self.checkpoints_radius = 45 #rayon d'entrée dans la zone du checkpoint 20 pixels
        self.traveled_distance = 0
        self.last_traveled_distance = 0
        self.last_position = (self.x, self.y)
        self.previous_pos = (self.x, self.y)
        self.ortho_sys_y = ((300, 40), (300, 130))
        self.ortho_sys_x = ((300, 130), (400, 130))
        self.behind_cp = False
        self.in_checkpoint = False


        
    def update(self, actions: list[int]):
    
        moved = False  
        self.progression = self.get_progression_old()
        self.previous_pos = (self.x, self.y) 
        
        moves = ['U', 'D', 'L', 'R']      
        
        # actions : [1, 0, 0, 1]
        actions = [j for j, act in enumerate(actions) if act] # [0, 3]
        actions = [moves[action] for action in actions] # ['U', 'R']
        
        if 'L' in actions:
            self.angle = (self.angle + self.rotation_speed) % 360
        if 'R' in actions:
            self.angle = (self.angle - self.rotation_speed) % 360
        if 'U' in actions:
            self.speed = min(self.speed + self.acceleration, self.max_speed)
            moved = True
        if 'D' in actions:
            self.speed = max(self.speed - self.acceleration, -self.max_speed / 2)
            moved = True
                

        if not moved: # inertie
            if self.speed > 0:
                self.speed = max(self.speed - self.acceleration, 0)
            else:
                self.speed = min(self.speed + self.acceleration, 0)

        if self.collision != 0:
            self.nbCollisions += 1
            self.alive = False
            if self.compteur < 0: # permet d'éviter de detecter les collisions trop rapidement (= 30 fois/sec), sinon bug
                self.speed = - self.speed / 2
                self.compteur = 4
        self.compteur -= 1
            
        rad = math.radians(self.angle)
        self.y -= self.speed * math.cos(rad) 
        self.x -= self.speed * math.sin(rad) 
        

    def get_progression_old(self):
        """ Calcule l'avancée de la voiture sur le circuit : distance parcourue et checkpoints"""

        #La variable validate checkpoint permet, sur un cercle approximé centré sur le checkpoint, de valider la prise de checkpoint peu importe le sens
        current_position = (self.x, self.y)
        checkpoint_progress = False

        for i, checkpoint in enumerate(self.checkpoints) :
            
            #Calcul de la distance entre la voiture et le checkpoint d'avant et d'après
            dist = math.dist((self.x, self.y), checkpoint)

            "Note pour l'aide à la compréhension de la fonction"
            """A un instant t :
                self.validate_checkpoint - 1 : checkpoint actuel
                self.validate_checkpoint : checkpoint suivant
                self.validate_checkpoint - 2 : checkpoint précédent"""

            if dist <= self.checkpoints_radius and i == self.validate_checkpoint - 1 : #On ne veut pas revalider un checkpoint déjà pris
                # print(f'Warning ! Going past the current cp {i+1} ')
                self.in_checkpoint = True
                break

            if (dist <= self.checkpoints_radius and i == self.validate_checkpoint) : #Validation du checkpoint d'après

                self.validate_checkpoint += 1 #le prochain checkpoint à valider sera le self.validate_checkpoint + 1 
                self.validate_checkpoint = min(self.validate_checkpoint, len(self.checkpoints)) #Pour éviter une éventuelle erreur dans la boucle
                checkpoint_progress = True
                self.in_checkpoint = False
                break
                
            if (dist <= self.checkpoints_radius and i == self.validate_checkpoint -2) : #Dans le cas où la voiture recule au checkpoint d'avant, elle régressera d'un checkpoint
                #Pour par qu'il y ait de décrémentation si la voiture ne recule pas (distance approximée), on met une condition
                self.validate_checkpoint -= 1
                self.validate_checkpoint = max(self.validate_checkpoint, 0)
                checkpoint_progress = True
                self.in_checkpoint = False
                break
            else :
                self.in_checkpoint = False
        
        if self.validate_checkpoint > 0:

            current_checkpoint = self.checkpoints[self.validate_checkpoint- 1]
            next_checkpoint = self.checkpoints[self.validate_checkpoint]
            prev_checkpoint = self.checkpoints[self.validate_checkpoint - 2]
            "POUR FORWARD "
            #vecteur entre current et previous pos ((prevpos, currentpos))
            vect_curr_prev = np.array([current_position[0] - self.previous_pos[0], current_position[1] - self.previous_pos[1]])

            #vect entre cp actuel et suivant 
            vect_curr_next_cp = - np.array([current_checkpoint[0] - next_checkpoint[0], current_checkpoint[1] - next_checkpoint[1]])
            "FIN POUR FORWARD "

            "POUR CALCUL DE LA DISTANCE PARCOURUE"
            # Position projetée de la voiture sur la droite current cp -> next cp
            projected_x_current = lib.projection_betw_cp (current_checkpoint, next_checkpoint, current_position)[0]
            projected_y_current = lib.projection_betw_cp (current_checkpoint, next_checkpoint, current_position)[1]
            # Position projetée de la voiture sur la droite prev cp -> current cp
            projected_x_prev = lib.projection_betw_cp(prev_checkpoint, current_checkpoint, current_position)[0]
            projected_y_prev = lib.projection_betw_cp(prev_checkpoint, current_checkpoint, current_position)[1]
            "FIN POUR CALCUL DE LA DISTANCE PARCOURUE"
            #voyons maintenant si on est devant ou derrière le checkpoint actuel

            "POUR SAVOIR SI ON EST DERRIERE LE DERNIER CHECKPOINT VALIDE"
            # Positions projetées
            #projection sur x et y de la voiture
            #Sur x
            coord_curr_pos_x = lib.ortho_projection (self.ortho_sys_x, self.ortho_sys_y, current_position)[0]
            #Sur y 
            coord_curr_pos_y = lib.ortho_projection (self.ortho_sys_x, self.ortho_sys_y, current_position)[1]
            #Projection du current cp
            #Sur x
            coord_curr_cp_x = lib.ortho_projection (self.ortho_sys_x, self.ortho_sys_y, current_checkpoint)[0]
            #Sur y
            coord_curr_cp_y = lib.ortho_projection (self.ortho_sys_x, self.ortho_sys_y, current_checkpoint)[1]
            #Projection du next cp
            coord_next_cp_x = lib.ortho_projection (self.ortho_sys_x, self.ortho_sys_y, next_checkpoint)[0]
            #Sur y
            coord_next_cp_y = lib.ortho_projection (self.ortho_sys_x, self.ortho_sys_y, next_checkpoint)[1]
            #Maintenant que nous avons les projections de la voiture, du current cp et du next cp dans la base, traçons les vecteurs
            #Voiture avec curr cp
            vect_car_curr_cp_x = np.array([coord_curr_pos_x[0] - coord_curr_cp_x[0], coord_curr_pos_x[1] - coord_curr_cp_x[1]])
            vect_car_curr_cp_y = np.array([coord_curr_pos_y[0] - coord_curr_cp_y[0], coord_curr_pos_y[1] - coord_curr_cp_y[1]])
            projected_syst_car_curr_cp = (vect_car_curr_cp_x,vect_car_curr_cp_y)
            #Curr cp avec next cp
            vect_curr_next_cp_x = np.array([coord_next_cp_x[0] - coord_curr_cp_x[0], coord_next_cp_x[1] - coord_curr_cp_x[1]])
            vect_curr_next_cp_y = np.array([coord_next_cp_y[0] - coord_curr_cp_y[0], coord_next_cp_y[1] - coord_curr_cp_y[1]])
            projected_syst_curr_next_cp = (vect_curr_next_cp_x,vect_curr_next_cp_y)
            #Comparons maintenant les signes des coordonnées de ces vecteurs projetés. S'ils sont égaux, alors ils sont orientés dans le même sens
            self.behind_cp = False
            for axis in range (0, 2) :
                if self.in_checkpoint == True :
                    break
                if (projected_syst_car_curr_cp[axis][0] * projected_syst_curr_next_cp[axis][0] < 0) or (projected_syst_car_curr_cp[axis][1] * projected_syst_curr_next_cp[axis][1] < 0) :
                    self.behind_cp = True
                    break
            "FIN POUR SAVOIR SI ON EST DERRIERE LE DERNIER CHECKPOINT VALIDE"

            if self.speed != 0 :
                forward = np.dot(vect_curr_next_cp, vect_curr_prev) > 0
                #print(forward)
                dist_to_current_cp = math.dist(current_checkpoint, (projected_x_current, projected_y_current))
                
                # print(out_of_reach_cp, dist_to_current_cp, behind_cp, np.dot(vect_car_curr_cp, vect_curr_next_cp))
                if (self.behind_cp == True) and ((self.validate_checkpoint - 1 > 0) == True) :
                    
                    segment_distance = abs(math.dist(prev_checkpoint, (projected_x_prev, projected_y_prev)))
                    self.traveled_distance = abs(sum([math.dist(self.checkpoints[checked], self.checkpoints[checked + 1]) for checked in range(self.validate_checkpoint -2 )]) + segment_distance)
                else :
                    segment_distance = abs(math.dist(current_checkpoint, (projected_x_current, projected_y_current)))

                    self.traveled_distance = abs(sum([math.dist(self.checkpoints[checked], self.checkpoints[checked + 1]) for checked in range(self.validate_checkpoint -1 )]) + segment_distance)

                    # Mettre à jour la distance parcourue

                self.last_traveled_distance = self.traveled_distance
            else :
                forward = None
                self.traveled_distance = self.last_traveled_distance
            
        # if checkpoint_progress == True :
        #     print(self.validate_checkpoint)
        #     print(f"Checkpoint {self.validate_checkpoint} atteint!")
        #     print(behind_cp)

        progression = (self.traveled_distance/self.total_distance)*100
        #print(progression)
        return min(max(progression, 0), 100)
    
    
    def get_progression(self):
        
        # Trouver la projection du point sur le tracé
        min_dist = float('inf')
        car_pos = self.x, self.y
        for i in range(len(self.checkpoints) - 1):
            proj = lib.project_point_on_segment(car_pos, self.checkpoints[i], self.checkpoints[i + 1])
            dist = lib.distance_squared(car_pos, proj)
            if dist < min_dist:
                min_dist = dist
                self.closest_projection = proj
                self.last_cp = i
        
        traveled_distance = sum(lib.distance(self.checkpoints[i], self.checkpoints[i + 1]) for i in range(self.last_cp))
        traveled_distance += lib.distance(self.checkpoints[self.last_cp], self.closest_projection)
        
        return (traveled_distance / self.total_distance) * 100 
        


    def draw(self, ses, i):
        self.car_rotated = pg.transform.rotate(self.car_img, self.angle)
        self.car_rect = self.car_rotated.get_rect(center=self.car_img.get_rect(topleft=(self.x, self.y)).center)
        ses.screen.blit(self.car_rotated, self.car_rect.topleft)
        
        # pg.draw.rect(ses.screen, (255, 0, 0), self.car_rect, 2) # heatbox
        # ses.screen.blit(show_mask(self.car_rotated), (self.car_rect.x, self.car_rect.y)) # mask 
        
        # Affichage progression 
        if i in range(51):
            text_surface1 = self.font.render(f"{ses.scores[i]:.2f}", True, WHITE)
        elif i in range(51, ses.algo.threshold + 1):
           text_surface1 = self.font.render(f"{ses.scores[i]:.2f}", True, BLUE)
        elif i in range(ses.algo.threshold + 1, 501):
            text_surface1 = self.font.render(f"{ses.scores[i]:.2f}", True, GREEN)
        ses.screen.blit(text_surface1, (self.x, self.y))
    
        


class Background:
    def __init__(self, ses):
        self.back = ses.background_img
        self.track = ses.track_img
        self.border = ses.border_img
        self.finish = ses.finish_img
        self.border_pos = (170, -10)
        self.border_mask = pg.mask.from_surface(self.border)
        # self.border_mask_img = show_mask(self.border)
        self.finish_rect = self.finish.get_rect(topleft=(200, 330)) # on crée un rect pour la ligne d'arrivée
        
    def update(self, car_list):
        for car in car_list:
            self.car_mask = pg.mask.from_surface(car.car_rotated)
            offset = (int(car.x - self.border_pos[0] - 10), int(car.y - self.border_pos[1])) # correspond à la différence des coordonnées des 2 masques.
            booleen = self.border_mask.overlap(self.car_mask, offset)
            car.collision = 0 if booleen == None else 1
        
    def draw(self, ses, car):
        ses.screen.blit(self.back, (0, 0))
        ses.screen.blit(self.track, self.border_pos)
        ses.screen.blit(self.border, self.border_pos)
        ses.screen.blit(self.finish, (200, 330))
        
        for checkpoint in car.checkpoints:
            pg.draw.circle(ses.screen, (0, 255, 0), checkpoint, 5)
        
        # ses.screen.blit(self.border_mask_img, self.border_pos) # mask
        
        
    def collision_finish(self, car):
        car_rect = car.car_rect
        return car_rect.colliderect(self.finish_rect)
        
        
        


class Score:
    def __init__(self, background, ses):
        self.start_ticks = pg.time.get_ticks() 
        self.background = background
        self.ses = ses
        self.update_high_score()
        self.font = pg.font.Font(pg.font.match_font('arial'), 20)
        
        
    def update_high_score(self):
        """ Met à jour high_score avec le meilleur temps du fichier """
        with open("results_gene/times.txt", "r") as f:
            self.high_score = min(float(line) for line in f)
    
            
    def update(self, car_list):
        self.temps_ecoule = (pg.time.get_ticks() - self.start_ticks) / 1000
        for car in car_list:
            if self.background.collision_finish(car):
                car.alive = False
                if self.temps_ecoule < self.high_score: # si le temps realisé est meilleur que l'ancien record
                    if self.temps_ecoule > 15: # pour ne pas sauvegarder les marches arrières sur le finish
                        self.high_score = self.temps_ecoule 
                        with open("results_gene/times.txt", "a") as file: # append
                            file.write(f"{self.high_score:.3f}\n")

                self.update_high_score()
    

    def draw(self, ses): 
        
        # affichage du timer
        text = f"Temps écoulé : {self.temps_ecoule:.2f}s"
        text_surface = self.font.render(text, True, WHITE)
        text_rect = text_surface.get_rect()
        text_rect.topleft = (60, 770)
        ses.screen.blit(text_surface, text_rect)
        
        # affichage high score
        text1 = f"Generation : {self.ses.generation+1}"
        text_surface1 = self.font.render(text1, True, WHITE)
        text_rect1 = text_surface1.get_rect()
        text_rect1.topleft = (60, 750)
        ses.screen.blit(text_surface1, text_rect1)
        
        # affichage population
        text2 = f"Population : {self.ses.nb_alive}"
        text_surface2 = self.font.render(text2, True, WHITE)
        text_rect2 = text_surface2.get_rect() 
        text_rect2.topleft = (60, 730)
        ses.screen.blit(text_surface2, text_rect2)
        




class Session:        
    def __init__(self, algo, nb_cars, display=True):
        self.display = display
        self.nb_cars = nb_cars
        self.quit = False
        self.algo = algo
        
        pg.init()
        self.clock = pg.time.Clock()
        self.screen = pg.display.set_mode((WIDTH, HEIGHT))
        pg.display.set_caption('Race AI')

        self.load_images()

    def load_images(self):
        self.car_img = pg.image.load('media/car.png').convert_alpha()
        img_width, img_height = self.car_img.get_size()
        self.car_img = pg.transform.scale(self.car_img, (img_width // 11, img_height // 11))
        self.car_img = pg.transform.rotate(self.car_img, 270)
        
        self.track_img = pg.image.load('media/track.png').convert_alpha()
        img_width, img_height = self.track_img.get_size()
        self.track_img = pg.transform.scale(self.track_img, (img_width, img_height))
        
        self.border_img = pg.image.load('media/track-border.png').convert_alpha()
        img_width, img_height = self.border_img.get_size()
        self.border_img = pg.transform.scale(self.border_img, (img_width, img_height))
        
        self.background_img = pg.image.load('media/background.jpg').convert()
        self.background_img = pg.transform.scale(self.background_img, (WIDTH, HEIGHT))

        self.finish_img = pg.image.load('media/finish.png').convert_alpha()
        img_width, img_height = self.finish_img.get_size()
        self.finish_img = pg.transform.scale(self.finish_img, (img_width * 0.78 , img_height * 0.78))
        
    def reset(self, gen=1, nn=None):
        self.nb_pilots = self.nb_cars
        self.nb_alive = self.nb_cars
        self.done = False
        self.generation = gen
        self.scores = [0] * self.nb_pilots
        if nn is not None:
            self.best_nn = nn
        
        self.generate_objects()
    
    def generate_objects(self):
        self.car_list = [Car(self) for _ in range(self.nb_pilots)]
        self.background = Background(self)
        self.score = Score(self.background, self)        
        
    def update(self, actions):
        for idx, car in enumerate(self.car_list):
            if len(actions) != 0 and car.alive:
                car.update(actions[idx])
        self.nb_alive = sum([car.alive for car in self.car_list])
        self.background.update(self.car_list)
        self.score.update(self.car_list)
        self.clock.tick(FPS)
    
    def draw(self):
        self.background.draw(self, self.car_list[0])
        for i, car in enumerate(self.car_list):
            if car.alive:
                car.draw(self, i)
        self.score.draw(self)
        try:
            self.draw_nn(self.best_nn)
        except:
            pass
        pg.display.flip()


    def step(self, actions):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.done = True
                self.quit = True
        
        self.update(actions)
        if self.display:
            self.draw()
        
        if not any([car.alive for car in self.car_list]):
            self.done = True
            
        self.states = self.get_states()
        self.scores = self.get_scores()

        return self.states

     
    
    def get_states(self):
        self.states = []
        for car in self.car_list:
            dist_to_center_line = lib.distance(car.closest_projection, (car.x, car.y))
            dist_to_center_line *= lib.position_relative(car.checkpoints[car.last_cp], car.checkpoints[car.last_cp + 1], (car.x, car.y)) # pour savoir si la voiture est à droite ou à gauche de la center line
            dist_to_next_cp = lib.distance(car.checkpoints[car.last_cp + 1], (car.x, car.y))
            # direction_next_curve = lib.distance(car.checkpoints[car.last_cp + 1], car.checkpoints[car.last_cp + 2]) 
            direction_next_curve = lib.position_relative(car.checkpoints[car.last_cp], car.checkpoints[car.last_cp + 1], car.checkpoints[car.last_cp + 2]) # pour savoir si le prochain virage est à droite (1) ou gauche (-1)
            angle_to_center_line = lib.center_angle(car.angle - (360 - lib.angle_segment(car.checkpoints[car.last_cp], car.checkpoints[car.last_cp + 1])) % 360)
            self.states.append([car.speed, dist_to_center_line, dist_to_next_cp, direction_next_curve, angle_to_center_line])
        self.states = lib.normalisation2(self.states) 
        return self.states    

        
    def get_scores(self):
        for i, car in enumerate(self.car_list):
            self.scores[i] = car.progression
            
            # if car.x > 370 and car.y > 410:
            #     self.scores[i] += 2
            #     # print("reward")
            
            # if not car.alive:
            #     self.scores[i] -= 1
            

        return self.scores
    
    def draw_nn(self, network):
        network = network.weights

        # Nombre de neurones par couche
        layer_sizes = [network[0].shape[0]] + [w.shape[1] for w in network]

        # Positions des neurones
        x_spacing = 70 
        neuron_positions = []
        for i, layer_size in enumerate(layer_sizes):
            y_spacing = 150 // (layer_size + 1)
            neuron_positions.append([(1000 + x_spacing * (i + 1), y_spacing * (j + 1)) for j in range(layer_size)])

        # Dessiner les connexions
        for i in range(len(network)):
            for j, neuron1 in enumerate(neuron_positions[i]):
                for k, neuron2 in enumerate(neuron_positions[i + 1]):
                    weight = network[i][j, k]
                    color = (255, 0, 0) if weight > 0 else (0, 0, 255)  # Rouge pour positif, bleu pour négatif
                    thickness = int(abs(weight) * 3)  # Épaisseur proportionnelle au poids
                    pg.draw.line(self.screen, color, neuron1, neuron2, thickness)

        # Dessiner les neurones
        for layer in neuron_positions:
            for x, y in layer:
                pg.draw.circle(self.screen, WHITE, (x, y), 5)  # Neurones en noir







if __name__ == '__main__':
    
    agent = True
    
    if agent:
        ses = Session(nb_cars=1)
        ses.reset()
        states = ses.get_states()
        
        PATH = Path("results_gene/weights")
        n_train = len(os.listdir(PATH)) # nb de fichiers dans dossier weights
        with open(PATH / Path(f"2.weights"), "rb") as f:
            weights, bias = pickle.load(f)
            agent = Pilot(weights, bias)
        
        
    else:
        nb_cars = 100
        ses = Session(nb_cars)
        ses.reset()
        states = ses.get_states()
    
    
    while not ses.done:
        if agent:
            actions = [agent.predict(states[0])]
        else:
            actions = [np.random.choice(4, p=[3/6, 1/6, 1/6, 1/6]) for _ in range(nb_cars)] # [2, 0]
            actions = [[1 if i == action else 0 for i in range(4)] for action in actions] # [[0, 0, 1, 0], [1, 0, 0, 0]]]
        
        states = ses.step(actions)
        # print([round(x, 2) for x in states[0]])
        
        for event in pg.event.get():
            if event.type == pg.MOUSEBUTTONDOWN:
                print(pg.mouse.get_pos())
    
    print(ses.get_scores()[0], "\n")
    
    
    pg.quit()
    sys.exit(0)
    
    
    
    


