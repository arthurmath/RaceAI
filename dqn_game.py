import sys
import os
import math
import time
from pilot import Pilot, Adn
from pathlib import Path
import pickle
import numpy as np
import library as lib

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame as pg

    
class Car:
    def __init__(self, ses): 
        self.initial_pos = 230, 275
        self.x, self.y = self.initial_pos
        self.angle = 0
        self.speed = 0
        self.acceleration = 0.2
        self.rotation_speed = 9
        self.max_speed = 10
        
        self.car_img = ses.car_img
        self.car_rotated = pg.transform.rotate(self.car_img, self.angle)
        self.car_rect = self.car_rotated.get_rect()
        
        self.collision = 0
        self.compteur = 0 # pour les collisions
        self.nbCollisions = 0
        
        self.checkpoints = [(239, 273), (239, 130), (300, 75), (360, 130), (370, 392), (420, 451), (479, 389), (482, 126), 
                            (531, 74), (941, 80), (988, 127), (989, 240), (940, 278), (680, 277), (614, 341), (681, 386), 
                            (941, 399), (986, 440), (987, 750), (941, 800), (890, 800), (840, 751), (831, 583), (780, 532), 
                            (680, 533), (634, 582), (611, 760), (570, 797), (301, 585), (236, 436)]

        self.total_distance = sum([math.dist(self.checkpoints[i], self.checkpoints[i + 1]) for i in range(len(self.checkpoints)-1)])
        
        self.validate_checkpoint = 0
        self.checkpoints_radius = 45
        self.traveled_distance = 0
        self.last_traveled_distance = 0
        self.last_position = (self.x, self.y)
        self.previous_pos = (self.x, self.y)
        self.ortho_sys_y = ((300, 40), (300, 130))
        self.ortho_sys_x = ((300, 130), (400, 130))
        self.behind_cp = False
        self.in_checkpoint = False

        
        
    def update(self, moves):
    
        moved = False  
        self.progression = self.get_progression()
        self.previous_pos = (self.x, self.y) 
           
                    
        if 'L' in moves:
            self.angle = (self.angle + self.rotation_speed) % 360
        if 'R' in moves:
            self.angle = (self.angle - self.rotation_speed) % 360
        if 'U' in moves:
            moved = True
            self.speed = min(self.speed + self.acceleration, self.max_speed)
        if 'D' in moves:
            moved = True
            self.speed = max(self.speed - self.acceleration, -self.max_speed / 2)
                

        if not moved: # inertie
            if self.speed > 0:
                self.speed = max(self.speed - self.acceleration, 0)
            else:
                self.speed = min(self.speed + self.acceleration, 0)

        if self.collision != 0:
            self.nbCollisions += 1
            if self.compteur < 0: # permet d'éviter de detecter les collisions trop rapidement (= 30 fois/sec), sinon bug
                self.speed = - self.speed / 2
                self.compteur = 4
        self.compteur -= 1
            
        rad = math.radians(self.angle)
        self.y -= self.speed * math.cos(rad) 
        self.x -= self.speed * math.sin(rad) 
        
        # print([self.x, self.y, self.speed, self.angle, self.collision, self.progression])


    def draw(self, ses):
        self.car_rotated = pg.transform.rotate(self.car_img, self.angle)
        self.car_rect = self.car_rotated.get_rect(center=self.car_img.get_rect(topleft=(self.x, self.y)).center)
        ses.screen.blit(self.car_rotated, self.car_rect.topleft)
        
        for checkpoint in self.checkpoints:
            pg.draw.circle(ses.screen, (0, 255, 0), checkpoint, 5)
        
        # pg.draw.rect(ses.screen, (255, 0, 0), self.car_rect, 2) # heatbox
        # ses.screen.blit(show_mask(self.car_rotated), (self.car_rect.x, self.car_rect.y)) # mask 
        
    
    
    def get_progression(self):
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
                self.in_checkpoint = False
                break
                
            if (dist <= self.checkpoints_radius and i == self.validate_checkpoint -2) : #Dans le cas où la voiture recule au checkpoint d'avant, elle régressera d'un checkpoint
                #Pour par qu'il y ait de décrémentation si la voiture ne recule pas (distance approximée), on met une condition
                self.validate_checkpoint -= 1
                self.validate_checkpoint = max(self.validate_checkpoint, 0)
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
                forward = 1 if np.dot(vect_curr_next_cp, vect_curr_prev) > 0 else -1
                #print(forward)
                dist_to_current_cp = math.dist(current_checkpoint, (projected_x_current, projected_y_current))
                
                # print(out_of_reach_cp, dist_to_current_cp, behind_cp, np.dot(vect_car_curr_cp, vect_curr_next_cp))
                if (self.behind_cp == True) and ((self.validate_checkpoint - 1 > 0) == True) :
                    
                    segment_distance = abs(math.dist(prev_checkpoint, (projected_x_prev, projected_y_prev)))
                    #print('OK1')
                    self.traveled_distance = abs(sum([math.dist(self.checkpoints[checked], self.checkpoints[checked + 1]) for checked in range(self.validate_checkpoint -2 )]) + segment_distance)
                else :
                    segment_distance = abs(math.dist(current_checkpoint, (projected_x_current, projected_y_current)))

                    self.traveled_distance = abs(sum([math.dist(self.checkpoints[checked], self.checkpoints[checked + 1]) for checked in range(self.validate_checkpoint -1 )]) + segment_distance)

                    # Mettre à jour la distance parcourue

                self.last_traveled_distance = self.traveled_distance
            else :
                forward = 0
                self.traveled_distance = self.last_traveled_distance

        # print(forward)
        progression = (self.traveled_distance/self.total_distance)*100
        return min(max(progression, 0) ,100)
    

    def reset(self):
        self.x, self.y = self.initial_pos
        self.speed = 0
        self.angle = 0
        
        
        
    



class Background:
    def __init__(self, ses):
        self.back = ses.background_img
        self.track = ses.track_img
        self.border = ses.border_img
        self.finish = ses.finish_img
        self.border_pos = (170, -10)
        self.border_mask = pg.mask.from_surface(self.border)   
        self.finish_rect = self.finish.get_rect(topleft=(200, 330)) 
        
    def update(self, car):
        self.car_mask = pg.mask.from_surface(car.car_rotated)
        offset = (int(car.x - self.border_pos[0] - 10), int(car.y - self.border_pos[1]))
        booleen = self.border_mask.overlap(self.car_mask, offset)
        car.collision = 0 if booleen == None else 1
        
    def draw(self, ses):
        ses.screen.blit(self.back, (0, 0))
        ses.screen.blit(self.track, self.border_pos)
        ses.screen.blit(self.border, self.border_pos)
        ses.screen.blit(self.finish, (200, 330))
        
    def collision_finish(self, car):
        car_rect = car.car_rect
        return car_rect.colliderect(self.finish_rect)
        
        
        


class Score:
    def __init__(self, background, car):
        self.start_ticks = pg.time.get_ticks() 
        self.background = background
        self.car = car
        self.update_high_score()
        self.font = pg.font.Font(pg.font.match_font('arial'), 20)
        
        
    def update_high_score(self):
        """ Met à jour high_score avec le meilleur temps du fichier """
        with open("results_dqn/times.txt", "r") as f:
            self.high_score = min(float(line) for line in f)
    
            
    def update(self, ses, car):
        self.temps_ecoule = (pg.time.get_ticks() - self.start_ticks) / 1000
        
        if self.background.collision_finish(self.car):
            if self.temps_ecoule < self.high_score: # si le temps realisé est meilleur que l'ancien record
                if self.temps_ecoule > 15: # pour ne pas sauvegarder les marches arrières sur le finish
                    self.high_score = self.temps_ecoule 
                    with open("results_dqn/times.txt", "a") as file:
                        file.write(f"{self.high_score:.3f}\n")
                
            ses.done = True
            self.start_ticks = pg.time.get_ticks() # reset timer
            car.reset()
            self.update_high_score()
    

    def draw(self, ses, car): 
        WHITE = (255, 255, 255)
        
        # affichage du timer
        text = f"Temps écoulé : {self.temps_ecoule:.3f}s"
        text_surface = self.font.render(text, True, WHITE)
        text_rect = text_surface.get_rect()
        text_rect.topleft = (10, 800)
        ses.screen.blit(text_surface, text_rect)
        
        # affichage high score
        text1 = f"Meilleur temps : {self.high_score:.3f}s"
        text_surface1 = self.font.render(text1, True, WHITE)
        text_rect1 = text_surface1.get_rect()
        text_rect1.topleft = (10, 780)
        ses.screen.blit(text_surface1, text_rect1)
        
        # affichage progression
        text2 = f"Progression : {car.progression:.3f}%"
        text_surface2 = self.font.render(text2, True, WHITE)
        text_rect2 = text_surface2.get_rect() 
        text_rect2.topleft = (10, 760)
        ses.screen.blit(text_surface2, text_rect2)
        




class Session:        
    def __init__(self, train, agent, display, training_time):
        self.train = train
        self.agent = agent
        self.display = display
        self.training_time = training_time
        
        self.width = 1200
        self.height = 900
        self.fps = 30
        self.done = False
                
        pg.init()
        self.clock = pg.time.Clock()
        self.screen = pg.display.set_mode((self.width, self.height))
        pg.display.set_caption('Race AI')
        
        if train:
            self.start_train = time.time()
            self.fps = 70 # faster training
            
        self.load_images()
        self.generate_objects()

        
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
        self.background_img = pg.transform.scale(self.background_img, (self.width, self.height))

        self.finish_img = pg.image.load('media/finish.png').convert_alpha()
        img_width, img_height = self.finish_img.get_size()
        self.finish_img = pg.transform.scale(self.finish_img, (img_width * 0.78 , img_height * 0.78))
        
    def generate_objects(self):
        self.car = Car(self)
        self.background = Background(self)
        self.score = Score(self.background, self.car)
            
        
    def update(self, moves):
        self.car.update(moves)
        self.background.update(self.car)
        self.score.update(self, self.car)
        self.clock.tick(self.fps) # TODO garder ?
    
    def draw(self):
        self.background.draw(self)
        self.car.draw(self)
        self.score.draw(self, self.car)
        pg.display.flip()

    def run(self):
        running = True
        while running:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
        
            self.update()
            
            if self.train:
                if time.time() - self.start_train > self.training_time:
                    running = False
            if self.display:
                self.draw()
    
    def step(self, moves):
        self.update(moves)
        if self.display:
                self.draw()
                
        next_state = [self.car.x, self.car.y, self.car.speed, self.car.angle, self.car.collision, self.car.nbCollisions, self.car.progression] # TODO supprimer collision
        next_state = self.normalize_state(next_state)
        
        if self.car.nbCollisions > 10:
            self.done = True
        
        old_fitness = 1
        new_reward = self.get_reward(self.car, reward)
        
        reward = new_reward - reward
        
        return next_state, reward, self.done
    
            
    def normalize_state(self, state):
        # Il faut que les entrées soient dans [-1, 1] pour converger
        list_ranges = [[0, 1200], [0, 900], [-10, 10], [0, 360], [0, 1], [0, 500], [0, 100], [0, 3], [0, 3]]
        for idx, ranges in enumerate(list_ranges):
            state[idx] = self.scale(state[idx], *ranges)
        return state

    def scale(self, x, a, b):
        """Transforme la valeur x initialement comprise dans l'intervalle [a, b]
            en une valeur comprise dans l'intervalle [-1, 1]."""
        return 2 * (x - a) / (b - a) - 1
    
    
    def compute_fitness(self, car): # TODO
        self.fitness = car.progression ** 2 / car.nbCollisions if car.nbCollisions else car.progression ** 2
        return self.fitness
    
    def reset(self):
        state = [self.car.x, self.car.y, self.car.speed, self.car.angle, self.car.collision, self.car.nbCollisions, self.car.progression]
        state = self.normalize_state(state)
        
        return state
    
    def close(self):
        pg.quit()

        


    
    
    

        
