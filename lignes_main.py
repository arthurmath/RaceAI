import pygame as pg
import random
import sys
import time
import math
import numpy as np

    
class Car:
    def __init__(self, ses): 
        self.car_img = ses.car_img
        self.initial_pos = 230, 275
        self.x, self.y = self.initial_pos
        self.angle = 0
        self.speed = 0
        self.acceleration = 0.2
        self.rotation_speed = 9
        self.max_speed = 10
        self.collision = None
        
        self.car_rect = self.car_img.get_rect()
        self.car_rect.left = 200
        self.car_rect.top = HEIGHT / 2 - 50
        self.car_rect.width = 45
        self.collision_finish = None
        
        self.compteur = 0
        
        
    def update(self):
               
        moved = False     
        keys = pg.key.get_pressed()

        if keys[pg.K_LEFT]:
            self.angle += self.rotation_speed 
        if keys[pg.K_RIGHT]:
            self.angle -= self.rotation_speed
        if keys[pg.K_UP]:
            moved = True
            self.speed = min(self.speed + self.acceleration, self.max_speed)
        if keys[pg.K_DOWN]:
            moved = True
            self.speed = max(self.speed - self.acceleration, -self.max_speed / 2)

        if not moved:
            if self.speed > 0:
                self.speed = max(self.speed - self.acceleration, 0)
            else:
                self.speed = min(self.speed + self.acceleration, 0)


        if self.collision != None:
            if self.compteur < 0:
                self.speed = - self.speed / 2
                self.compteur = 10
        self.compteur -= 1
            

        rad = math.radians(self.angle)
        self.y -= self.speed * math.cos(rad) 
        self.x -= self.speed * math.sin(rad) 


    def draw(self):
        self.car_rotated = pg.transform.rotate(self.car_img, self.angle)
        self.car_rect = self.car_rotated.get_rect(center=self.car_img.get_rect(topleft=(self.x, self.y)).center)
        ses.screen.blit(self.car_rotated, self.car_rect.topleft)
        
        pg.draw.rect(ses.screen, (255, 0, 0), self.car_rect, 2) # heatbox
        # ses.screen.blit(show_mask(self.car_rotated), (self.car_rect.x, self.car_rect.y)) # mask
        
    
    
    def progression(self):
        """
        Calcule l'avancée de la voiture sur le circuit en pourcentage.
        
        :param checkpoints: Liste des points clés (x, y) définissant le circuit.
        :return: Pourcentage de progression (0 à 100).
        """
        
        checkpoints = [(240, 275), (300, 400), (400, 500)]
        
        for checkpoint in checkpoints:
            pg.draw.circle(ses.screen, (0, 255, 0), checkpoint, 5)
        
        total_distance = 0
        for i in range(len(checkpoints) - 1):
            total_distance += math.dist(checkpoints[i], checkpoints[i + 1])
        
        # Calculer la distance parcourue par la voiture
        distance_parcourue = 0
        for i in range(len(checkpoints) - 1):
            if math.dist((self.x, self.y), checkpoints[i]) < math.dist(checkpoints[i], checkpoints[i + 1]):
                # Si la voiture est entre deux checkpoints
                distance_parcourue += math.dist(checkpoints[i], (self.x, self.y))
                break
            else:
                distance_parcourue += math.dist(checkpoints[i], checkpoints[i + 1])
        
        # Calculer la progression en pourcentage
        progression = (distance_parcourue / total_distance) * 100
        return min(max(progression, 0), 100)

        
        
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
        self.border_mask_img = show_mask(self.border)
        self.finish_rect = self.finish.get_rect(topleft=(200, 330)) #on crée un masque pour la ligne d'arrivée
        
    def update(self, car):
        self.car_mask = pg.mask.from_surface(car.car_img)
        offset = (int(car.x - self.border_pos[0]), int(car.y - self.border_pos[1])) # Correspond à la différence des coordonnées des 2 masques.
        car.collision = self.border_mask.overlap(self.car_mask, offset)
        

    def draw(self):
        ses.screen.blit(self.back, (0, 0))
        ses.screen.blit(self.track, self.border_pos)
        ses.screen.blit(self.border, self.border_pos)
        ses.screen.blit(self.finish, (200, 330))
        
        # ses.screen.blit(self.border_mask_img, self.border_pos) # mask
        
        
    def collision_finish(self, car):
        car_rect = car.car_rect #cette ligne sert a récupérer les coordonnées de la voiture
        return car_rect.colliderect(self.finish_rect) #on change l'état de collision
        
        


class Score:
    def __init__(self, background, car):
        self.start_ticks = pg.time.get_ticks() 
        self.timer_running = True
        self.background = background
        self.car = car
        self.high_time = np.inf
        self.timer_running = 0
        self.time_saved = False
            
            
    def update(self, car):
        self.temps_ecoule = (pg.time.get_ticks() - self.start_ticks) / 1000
        
        if self.background.collision_finish(self.car):
            if self.temps_ecoule < self.high_time:
                self.high_time = self.temps_ecoule 
                
                if self.high_time > 20: # pour ne pas sauvegarder les marches arrières sur le finish
                    with open("times.txt", "a") as file:
                        file.write(f"{self.high_time:.3f}\n")
                
            self.start_ticks = pg.time.get_ticks() # reset timer
            car.reset()
            


    def draw(self, car): #pour affichage du timer
        font = pg.font.Font(pg.font.match_font('arial'), 20) #police d'écriture
        text = f"Temps écoulé : {self.temps_ecoule:.3f}s"
        text_surface = font.render(text, True, WHITE)   #création de la surface de texte
        text_rect = text_surface.get_rect() #récupération du rectangle de la surface de texte
        text_rect.topleft = (10, 800)
        ses.screen.blit(text_surface, text_rect)
        
        #affichage high score
        if self.high_time == np.inf:
            text1 = f"Meilleur temps : Aucun"
        else:
            text1 = f"Meilleur temps : {self.high_time}"
        text_surface1 = font.render(text1, True, WHITE)
        text_rect1 = text_surface1.get_rect()
        text_rect1.topleft = (10, 780)
        ses.screen.blit(text_surface1, text_rect1)
        
        text2 = f"Progression : {car.progression():.3f}%"
        font = pg.font.Font(pg.font.match_font('arial'), 20) #police d'écriture
        text_surface2 = font.render(text2, True, WHITE)   #création de la surface de texte
        text_rect2 = text_surface2.get_rect() #récupération du rectangle de la surface de texte
        text_rect2.topleft = (10, 760)
        ses.screen.blit(text_surface2, text_rect2)
        



class Session:
    def __init__(self):
        pg.init()
        self.clock = pg.time.Clock()
        self.screen = pg.display.set_mode((WIDTH, HEIGHT))
        pg.display.set_caption('Race AI')

        self.music()
        self.load_images()
        self.generate_objects()
        
    def music(self):
        self.music = pg.mixer.music.load('BandeOrganise.mp3')
        pg.mixer.music.set_volume(0.3)
        pg.mixer.music.play(-1)
        
    def load_images(self):
        self.car_img = pg.image.load('images/car.png').convert_alpha()
        img_width, img_height = self.car_img.get_size()
        self.car_img = pg.transform.scale(self.car_img, (img_width // 9, img_height // 9))
        self.car_img = pg.transform.rotate(self.car_img, 270)
        
        a = 1.0 # scale factor for the track
        self.track_img = pg.image.load('images/track.png').convert_alpha()
        img_width, img_height = self.track_img.get_size()
        self.track_img = pg.transform.scale(self.track_img, (img_width // a, img_height // a))
        
        self.border_img = pg.image.load('images/track-border.png').convert_alpha()
        img_width, img_height = self.border_img.get_size()
        self.border_img = pg.transform.scale(self.border_img, (img_width // a, img_height // a))
        
        self.background_img = pg.image.load('images/background.jpg').convert()
        self.background_img = pg.transform.scale(self.background_img, (WIDTH, HEIGHT))

        self.finish_img = pg.image.load('images/finish.png').convert_alpha()
        img_width, img_height = self.finish_img.get_size()
        self.finish_img = pg.transform.scale(self.finish_img, (img_width * 0.78 , img_height * 0.78))
        
    def generate_objects(self):
        self.car = Car(self)
        self.background = Background(self)
        self.score = Score(self.background, self.car)
        
        
    def update(self):
        self.background.update(self.car)
        self.car.update()
        self.score.update(self.car)
        self.clock.tick(FPS)
    
        
    def draw(self):
        self.background.draw()
        self.car.draw()
        self.score.draw(self.car)
        pg.display.flip()


    def run(self):
        running = True
        while running:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
        
            self.update()
            self.draw()
            
    
    
    
    
        


if __name__ == '__main__':
    
    def show_mask(img): #(chatgpt)
        # Créer une surface noire de la taille du masque
        mask = pg.mask.from_surface(img)
        mask_surface = pg.Surface(mask.get_size(), flags=pg.SRCALPHA)
        mask_surface.fill((0, 0, 0, 0))  # Transparence

        # Parcourir les pixels du masque et colorier en blanc les pixels actifs
        for x in range(mask.get_size()[0]):
            for y in range(mask.get_size()[1]):
                if mask.get_at((x, y)):  # Si le pixel est actif (valeur 1)
                    mask_surface.set_at((x, y), (255, 255, 255, 255))  # Blanc

        return mask_surface

    
    WIDTH = 1200
    HEIGHT = 900
    WHITE = (255, 255, 255)
    FPS = 30 
    
    
    ses = Session()
    ses.run()
    
    pg.quit()
    sys.exit(0)
    
    
    
    
    

# a faire : niveau d'avancee sur le circuit







# lines = [((300, 40), (300, 650), (0, 0, 255)),
#                  ((200, 130), (400, 130), (0, 0, 255)),
#                  ((320, 390), (520, 390), (0, 0, 255)),
#                  ((420, 150), (420, 500), (0, 0, 255)),
#                  ((530, 30), (530, 400), (0, 0, 255)),
#                  ((440, 125), (1030, 125), (0, 0, 255)),
#                  ((940, 40), (940, 320), (0, 0, 255)),
#                  ((620, 240), (1050, 240), (0, 0, 255)),
#                  ((680, 240), (680, 440), (0, 0, 255)),
#                  ((570, 340), (1000, 340), (0, 0, 255)),
#                  ((600, 440), (1030, 440), (0, 0, 255)),
#                  ((940, 360), (940, 840), (0, 0, 255)),
#                  ((800, 750), (1030, 750), (0, 0, 255)),
#                  ((890, 600), (890, 840), (0, 0, 255)),
#                  ((580, 580), (870, 580), (0, 0, 255)),
#                  ((780, 490), (780, 770), (0, 0, 255)),
#                  ((680, 490), (680, 770), (0, 0, 255)),
#                  ((570, 570), (570, 840), (0, 0, 255)),
#                  ((170, 370), (620, 820), (0, 0, 255)),
#                  ]

#         for start, end, color in lines:
#             pg.draw.line(ses.screen, color, start, end, 2)