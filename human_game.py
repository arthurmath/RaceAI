import sys
import math
import numpy as np
import library as lib
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
        

        
        
    def update(self, ses):
    
        moved = False  
        self.progression = self.get_progression()
        self.previous_pos = (self.x, self.y) 
        
        moves = []
        
        keys = pg.key.get_pressed()
        if keys[pg.K_LEFT]:
            moves.append('L')
        if keys[pg.K_RIGHT]:
            moves.append('R')
        if keys[pg.K_UP]:
            moves.append('U')
        if keys[pg.K_DOWN]:
            moves.append('D')
                
                    
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
        for i in range(len(self.checkpoints) - 1):
            pg.draw.line(ses.screen, (0, 0, 255), self.checkpoints[i], self.checkpoints[i + 1], 2)
            
        for checkpoint in self.checkpoints:
            pg.draw.circle(ses.screen, (0, 255, 0), checkpoint, 4)
            
        self.car_rotated = pg.transform.rotate(self.car_img, self.angle)
        self.car_rect = self.car_rotated.get_rect(center=self.car_img.get_rect(topleft=(self.x, self.y)).center)
        ses.screen.blit(self.car_rotated, self.car_rect.topleft)
        
        pg.draw.circle(ses.screen, (255, 255, 255), (self.x, self.y), 3)
        
        # pg.draw.rect(ses.screen, (255, 0, 0), self.car_rect, 2) # heatbox
        # ses.screen.blit(lib.show_mask(self.car_rotated), (self.car_rect.x, self.car_rect.y)) # mask 
        

        
    
    
    
    
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
        # self.border_mask_img = lib.show_mask(self.border)
        self.finish_rect = self.finish.get_rect(topleft=(200, 330)) # on crée un rect pour la ligne d'arrivée
        
    def update(self, car):
        self.car_mask = pg.mask.from_surface(car.car_rotated)
        offset = (int(car.x - self.border_pos[0] - 10), int(car.y - self.border_pos[1])) # correspond à la différence des coordonnées des 2 masques.
        booleen = self.border_mask.overlap(self.car_mask, offset)
        car.collision = 0 if booleen == None else 1
        
    def draw(self, ses):
        ses.screen.blit(self.back, (0, 0))
        ses.screen.blit(self.track, self.border_pos)
        ses.screen.blit(self.border, self.border_pos)
        ses.screen.blit(self.finish, (200, 330))
        
        # ses.screen.blit(self.border_mask_img, self.border_pos) # mask
        
        
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
        with open("results_gene/times.txt", "r") as f:
            self.high_score = min(float(line) for line in f)
    
            
    def update(self, car):
        self.temps_ecoule = (pg.time.get_ticks() - self.start_ticks) / 1000
        
        if self.background.collision_finish(self.car):
            if self.temps_ecoule < self.high_score: # si le temps realisé est meilleur que l'ancien record
                if self.temps_ecoule > 15: # pour ne pas sauvegarder les marches arrières sur le finish
                    self.high_score = self.temps_ecoule 
                    with open("times.txt", "a") as file:
                        file.write(f"{self.high_score:.3f}\n")
                
            self.start_ticks = pg.time.get_ticks() # reset timer
            car.reset()
            self.update_high_score()
    

    def draw(self, ses, car): 
        WHITE = (255, 255, 255)
        
        # affichage du timer
        text = f"Temps écoulé : {self.temps_ecoule:.2f}s"
        text_surface = self.font.render(text, True, WHITE)
        text_rect = text_surface.get_rect()
        text_rect.topleft = (10, 800)
        ses.screen.blit(text_surface, text_rect)
        
        # affichage high score
        text1 = f"Meilleur temps : {self.high_score:.2f}s"
        text_surface1 = self.font.render(text1, True, WHITE)
        text_rect1 = text_surface1.get_rect()
        text_rect1.topleft = (10, 780)
        ses.screen.blit(text_surface1, text_rect1)
        
        # affichage progression
        text2 = f"Progression : {car.progression:.2f}%"
        text_surface2 = self.font.render(text2, True, WHITE)
        text_rect2 = text_surface2.get_rect() 
        text_rect2.topleft = (10, 760)
        ses.screen.blit(text_surface2, text_rect2)
        




class Session:        
    def __init__(self):
        
        self.width = 1200
        self.height = 900
        self.fps = 30
                
        pg.init()
        self.clock = pg.time.Clock()
        self.screen = pg.display.set_mode((self.width, self.height))
        pg.display.set_caption('Race AI')
            
        self.music()
        self.load_images()
        self.generate_objects()

        
    def music(self):
        self.music = pg.mixer.music.load('media/BandeOrganise.mp3')
        pg.mixer.music.set_volume(0.3)
        pg.mixer.music.play(-1)
        
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
        
            
        
        
    def update(self):
        self.car.update(self)
        self.background.update(self.car)
        self.score.update(self.car)
        self.clock.tick(self.fps)
    
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
            self.draw()
            states = self.get_states()
            print([round(x, 2) for x in states[0]])
            
            
    def get_states(self):
        self.states = []
        dist_to_center_line = lib.distance(self.car.closest_projection, (self.car.x, self.car.y))
        dist_to_center_line *= lib.position_relative(self.car.checkpoints[self.car.last_cp], self.car.checkpoints[self.car.last_cp + 1], (self.car.x, self.car.y)) # pour savoir si la voiture est à droite ou à gauche de la center line
        dist_to_next_cp = lib.distance(self.car.checkpoints[self.car.last_cp + 1], (self.car.x, self.car.y))
        # direction_next_curve = lib.distance(self.car.checkpoints[self.car.last_cp + 1], self.car.checkpoints[self.car.last_cp + 2]) 
        direction_next_curve = lib.position_relative(self.car.checkpoints[self.car.last_cp], self.car.checkpoints[self.car.last_cp + 1], self.car.checkpoints[self.car.last_cp + 2]) # pour savoir si le prochain virage est à droite (1) ou gauche (-1)
        angle_to_center_line = lib.center_angle(self.car.angle - (360 - lib.angle_segment(self.car.checkpoints[self.car.last_cp], self.car.checkpoints[self.car.last_cp + 1])) % 360)
        self.states.append([self.car.speed, dist_to_center_line, dist_to_next_cp, direction_next_curve, angle_to_center_line])
        self.states = lib.normalisation2(self.states) 
        return self.states    
            

if __name__ == '__main__':
    
    ses = Session()
    ses.run()
    
    pg.quit()
    sys.exit()
    
    
    
    
    



















#if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
    #pos = pg.mouse.get_pos()
    #self.checkpoints_fin.append(pos)
    #with open("checkpoints", "a") as file:
    #    file.write(f"{self.checkpoints_fin}\n")
    #print({pos})
    
    
    
    


# def get_progression(self):
#     """ Calcule l'avancée de la voiture sur le circuit """

#     distance_parcourue = 0
#     for i in range(len(self.checkpoints) - 1):
#         if math.dist((self.x, self.y), self.checkpoints[i]) < math.dist(self.checkpoints[i], self.checkpoints[i + 1]):
#             # Si la voiture est entre deux checkpoints
#             distance_parcourue += math.dist(self.checkpoints[i], (self.x, self.y))
#             break
#         else:
#             # Sinon on ajoute la distance entre les 2 derniers checkpoints
#             distance_parcourue += math.dist(self.checkpoints[i], self.checkpoints[i + 1])

#     progression = (distance_parcourue / self.total_distance) * 100
#     return min(max(progression, 0), 100)




# self.lines = [((300, 40), (300, 650), (0, 0, 255)),
#             ((200, 130), (400, 130), (255, 0, 255)),
#             ((320, 390), (520, 390), (0, 0, 255)),
#             ((420, 150), (420, 500), (0, 0, 255)),
#             ((530, 30), (530, 400), (0, 0, 255)),
#             ((440, 125), (1030, 125), (0, 0, 255)),
#             ((940, 40), (940, 320), (0, 0, 255)),
#             ((620, 240), (1050, 240), (0, 0, 255)),
#             ((680, 240), (680, 440), (0, 0, 255)),
#             ((570, 340), (1000, 340), (0, 0, 255)),
#             ((600, 440), (1030, 440), (0, 0, 255)),
#             ((940, 360), (940, 840), (0, 0, 255)),
#             ((800, 750), (1030, 750), (0, 0, 255)),
#             ((890, 600), (890, 840), (0, 0, 255)),
#             ((580, 580), (870, 580), (0, 0, 255)),
#             ((780, 490), (780, 770), (0, 0, 255)),
#             ((680, 490), (680, 770), (0, 0, 255)),
#             ((570, 570), (570, 840), (0, 0, 255)),
#             ((170, 370), (620, 820), (0, 0, 255))]


# for start, end, color in self.lines:
#     pg.draw.line(ses.screen, color, start, end, 2)