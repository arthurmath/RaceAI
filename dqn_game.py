import os, sys
import math
import pickle
import numpy as np
import library as lib
import pygame as pg
from pathlib import Path
from dqn_trainer import DeepQNetwork



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
        self.moves = ['U', 'D', 'L', 'R']
        
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
        self.closest_projection = self.initial_pos
        self.last_cp = 0

        
        
    def update(self, actions: list[int]):
    
        moved = False  
        self.progression = self.get_progression()
        self.previous_pos = (self.x, self.y) 
             
        
        # actions : [1, 0, 0, 1]
        actions = [j for j, act in enumerate(actions) if act] # [0, 3]
        actions = [self.moves[action] for action in actions] # ['U', 'R']
        
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
        
        text_surface1 = self.font.render(f"{ses.progressions[i]:.2f}", True, WHITE)
        ses.screen.blit(text_surface1, (self.x, self.y))

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
        with open("results_dqn/times.txt", "r") as f:
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
    def __init__(self, nb_cars, display=True):
        self.display = display
        self.nb_cars = nb_cars
        self.done = False
        self.quit = False
        
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
        
    def reset(self, gen=1):
        self.nb_pilots = self.nb_cars
        self.nb_alive = self.nb_cars
        self.generation = gen
        self.done = False
        self.progressions = [0] * self.nb_pilots
        self.rewards = [0] * self.nb_pilots
        self.dones = [False] * self.nb_pilots
        self.generate_objects()
        return self.get_states()
        
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
        pg.display.flip()


    def step(self, actions, nb_step):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.done = True
                self.quit = True
        
        self.update(actions)
        if self.display:
            self.draw()
            
        self.states = self.get_states()
        self.progressions = self.get_progressions()
        self.rewards = self.get_rewards(nb_step)
        
        if not any([car.alive for car in self.car_list]):
            self.done = True

        return self.states, self.progressions, self.rewards, self.dones
    
    
    def get_states(self):
        states = []
        for car in self.car_list:
            dist_to_center_line = lib.distance(car.closest_projection, (car.x, car.y))
            dist_to_center_line *= lib.position_relative(car.checkpoints[car.last_cp], car.checkpoints[car.last_cp + 1], (car.x, car.y)) # pour savoir si la voiture est à droite ou à gauche de la center line
            dist_to_next_cp = lib.distance(car.checkpoints[car.last_cp + 1], (car.x, car.y))
            # direction_next_curve = lib.distance(car.checkpoints[car.last_cp + 1], car.checkpoints[car.last_cp + 2]) 
            direction_next_curve = lib.position_relative(car.checkpoints[car.last_cp], car.checkpoints[car.last_cp + 1], car.checkpoints[car.last_cp + 2]) # pour savoir si le prochain virage est à droite (1) ou gauche (-1)
            angle_to_center_line = lib.center_angle(car.angle - (360 - lib.angle_segment(car.checkpoints[car.last_cp], car.checkpoints[car.last_cp + 1])) % 360)
            states.append([car.speed, dist_to_center_line, dist_to_next_cp, direction_next_curve, angle_to_center_line])
        states = lib.normalisation2(states) 
        return states  


    def get_progressions(self):
        for i, car in enumerate(self.car_list):
            self.progressions[i] = car.progression
        return self.progressions


    def get_rewards(self, nb_step):
        for i, car in enumerate(self.car_list):
            self.rewards[i] = self.progressions[i] - 0.1 * nb_step
            
            if not car.alive:
                self.rewards[i] -= 10
                self.dones[i] = True
            
            # old TODO
            # self.fitness = self.car.progression
            # reward = self.fitness - self.old_fitness
            # self.old_fitness = self.fitness
            
            # if car.x > 370 and car.y > 410:
            #     self.rewards[i] += 2
            #     # print("reward")
        
        return self.rewards






if __name__ == '__main__':
    
    dqn = DeepQNetwork()
    with open(Path("results_dqn/weights/colab1.weights"), "rb") as f:
        dqn.model.set_weights(pickle.load(f))
        
    env = Session(nb_cars=1)
    states = env.reset()
    while not env.quit:
        actions = dqn.model.predict(np.array(states)[np.newaxis, :], verbose=0)[0].argmax()
        states = env.step(actions)
        

    print(env.get_progressions()[0], "\n")
    pg.quit()
    sys.exit(0)
    
    
        
        
        





# def update_scores(self):
#     for i, car in enumerate(self.car_list):
#         if car.alive:
#             diff = car.progression - car.old_progression
#             # self.scores[i] += 10 * diff
#             # car.old_progression = car.progression

#             self.scores[i] = car.progression
#             if abs(diff) <= 1e-4:
#                 self.scores[i] -= 1 # pénalise l’inaction

#             # if car.collision:
#                 # reward -= 0.5
#             # if self.car.progression >= 10:
#                 # reward += 10
                
#         else:
#             self.scores[i] -= 100




# for event in pg.event.get():
#     if event.type == pg.MOUSEBUTTONDOWN:
#         print(pg.mouse.get_pos())