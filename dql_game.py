import os, sys
import math
import numpy as np
import library as lib
import pygame as pg



FPS = 150
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
        self.finish_rect = ses.finish_img.get_rect(topleft=(200, 330))

        
        
    def update(self, actions: list[int]):
    
        moved = False  
        self.progression = self.get_progression()
        self.previous_pos = (self.x, self.y) 
             
        
        ## actions : [1, 0, 0, 1]
        #actions = [j for j, act in enumerate(actions) if act] # [0, 3]
        #actions = [self.moves[action] for action in actions] # ['U', 'R']
        
        if 1 in actions:
            self.angle = (self.angle + self.rotation_speed) % 360
        if 2 in actions:
            self.angle = (self.angle - self.rotation_speed) % 360
        if 0 in actions:
            self.speed = min(self.speed + self.acceleration, self.max_speed)
            moved = True
        if 3 in actions:
            self.speed = max(self.speed - self.acceleration, -self.max_speed / 2)
            moved = True


        if not moved: # inertie
            if self.speed > 0:
                self.speed = max(self.speed - self.acceleration, 0)
            else:
                self.speed = min(self.speed + self.acceleration, 0)
                

        if self.collision != 0 or self.car_rect.colliderect(self.finish_rect):
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
        
        
        



class Score:
    def __init__(self, background, ses):
        self.start_ticks = pg.time.get_ticks() 
        self.background = background
        self.ses = ses
        self.font = pg.font.Font(pg.font.match_font('arial'), 20)
        
        
    def draw(self, ses): 
        
        # affichage du timer
        text = f"Reward : {ses.rewards[0]:.2f}s"
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
        self.episode_done = False
        self.quit = False
        
        self.observation_space = [[0, 1200], [0, 900], [-5, 10], [0, 360]]
        self.action_space = [0, 1, 2, 3]
        
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
        self.episode_done = False
        self.rewards = [0] * self.nb_pilots
        self.prev_rewards = [0] * self.nb_pilots
        self.terminateds = [False] * self.nb_pilots
        self.generate_objects()
        return self.get_states()[0]
        
    def generate_objects(self):
        self.car_list = [Car(self) for _ in range(self.nb_pilots)]
        self.background = Background(self)
        self.score = Score(self.background, self)        
        
    def update(self, actions):
        actions = [[actions]]
        for idx, car in enumerate(self.car_list):
            if len(actions) != 0 and car.alive:
                car.update(actions[idx])
        self.nb_alive = sum([car.alive for car in self.car_list])
        self.background.update(self.car_list)
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
                self.episode_done = True
                self.quit = True
        
        self.update(actions)
        if self.display:
            self.draw()
            
        self.states = self.get_states()
        self.rewards = self.get_rewards(nb_step)
        
        if not any([car.alive for car in self.car_list]):
            self.episode_done = True

        return self.states[0], self.rewards[0], self.terminateds[0]
    
    
    def get_states(self):
        self.states = [[car.x, car.y, car.speed, car.angle] for car in self.car_list]
        self.states = lib.normalisation(self.states) 
        return self.states


    def get_rewards(self, nb_step):
        step_reward = 0
        for i, car in enumerate(self.car_list):
            self.rewards[i] -= 0.01
            
            step_reward = self.rewards[i] - self.prev_reward[i]
            self.prev_reward[i] = self.rewards[i]
            
            if not car.alive:
                step_reward = -10
                self.terminateds[i] = True
            
            # reward pour chaque cp atteint
        
        return step_reward
    
    def close(self):
        pg.quit()





    
    
# Gym Car Racing :      
# The reward is -0.1 every frame and +1000/N for every track tile visited, where N is the number of tiles visited in the track.
# The episode finishes when all the tiles are visited. If the car goes off the track, it will receive -100 reward and die.
        


