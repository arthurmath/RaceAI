import math
import numpy as np
import time



liste = [3, 4]
vision = [1, 2, *liste]
# print(vision)





print(time.time())
time.sleep(1)
print(time.time())









































# import pygame as pg
# import sys
# import math

    
# class Car:
#     def __init__(self, ses): 
#         self.car_img = ses.car_img
#         self.initial_pos = 230, 275
#         self.x, self.y = self.initial_pos
#         self.angle = 0
#         self.speed = 0
#         self.acceleration = 0.2 # 0.1 permet de ralentir quand on est bloqué et ne pas marcher sur le bord trop vite
#         self.rotation_speed = 8
#         self.max_speed = 10
        
#         self.collision = 0
#         self.compteur = 0 # pour les collisions
        
#         self.car_rotated = pg.transform.rotate(self.car_img, self.angle)
#         self.car_mask2 = pg.mask.from_surface(self.car_rotated)
        
#     def update(self):
    
#         moved = False  
        
#         keys = pg.key.get_pressed()
#         if keys[pg.K_LEFT]:
#             self.angle += self.rotation_speed 
#         if keys[pg.K_RIGHT]:
#             self.angle -= self.rotation_speed
#         if keys[pg.K_UP]:
#             moved = True
#             self.speed = min(self.speed + self.acceleration, self.max_speed)
#             # print(self.car_rect, "\n")
#         if keys[pg.K_DOWN]:
#             moved = True
#             self.speed = max(self.speed - self.acceleration, -self.max_speed / 2)
            

#         # # voiture se déplace sans tourner
#         # if keys[pg.K_LEFT]:
#         #     self.x -= self.acceleration
#         # if keys[pg.K_RIGHT]:
#         #     self.x += self.acceleration
#         # if keys[pg.K_UP]:
#         #     moved = True
#         #     self.y -= self.acceleration
#         # if keys[pg.K_DOWN]:
#         #     self.y += self.acceleration

#         if not moved: # inertie
#             if self.speed > 0:
#                 self.speed = max(self.speed - self.acceleration, 0)
#             else:
#                 self.speed = min(self.speed + self.acceleration, 0)

#         if self.collision != None:
#             if self.compteur < 0: # permet d'éviter de detecter les collisions trop rapidement (= 30 fois/sec), sinon bug
#                 print("Collision", self.collision, self.compteur)
#                 # print(self.car_rect, "\n")
#                 self.speed = - self.speed / 2
#                 self.compteur = 5
#         self.compteur -= 1
            
#         rad = math.radians(self.angle)
#         self.y -= self.speed * math.cos(rad) 
#         self.x -= self.speed * math.sin(rad) 


#     def draw(self):
#         self.car_rotated = pg.transform.rotate(self.car_img, self.angle)
#         self.car_rect = self.car_rotated.get_rect(center=self.car_img.get_rect(topleft=(self.x, self.y)).center)
#         ses.screen.blit(self.car_rotated, self.car_rect.topleft)

#         # pg.draw.rect(ses.screen, (255, 0, 0), self.car_rect, 2) # heatbox
#         # ses.screen.blit(show_mask(self.car_rotated), (self.car_rect.x, self.car_rect.y)) # mask 
        
#         # self.car_mask2 = pg.mask.from_surface(self.car_rotated)
#         # self.car_mask_img2 = self.car_mask2.to_surface().convert_alpha()
#         # ses.screen.blit(self.car_mask_img2, self.car_rect)


# class Background:
#     def __init__(self, ses):
#         self.back = ses.background_img
#         self.track = ses.track_img
#         self.border = ses.border_img
#         self.border_pos = (170, -10)
#         # self.border_pos = (370, 300) # petit circuit au milieu
#         self.border_rect = self.border.get_rect()
#         self.border_mask = pg.mask.from_surface(self.border)
#         self.border_mask_img = show_mask(self.border)
#         self.border_mask_img2 = self.border_mask.to_surface()
        
#     def update(self, car):
#         self.car_mask = pg.mask.from_surface(car.car_rotated)
#         offset = (int(car.x - self.border_pos[0] - 10), int(car.y - self.border_pos[1])) # correspond à la différence des coordonnées des 2 masques.
#         # booleen = self.border_mask.overlap(self.car_mask, offset)
#         # car.collision = 0 if booleen == None else 1
#         car.collision = self.border_mask.overlap(self.car_mask, offset)
        
#     def draw(self):
#         ses.screen.blit(self.back, (0, 0))
#         ses.screen.blit(self.track, self.border_pos)
#         ses.screen.blit(self.border, self.border_pos)
        
#         ses.screen.blit(self.border_mask_img, self.border_pos) # mask
#         #ses.screen.blit(self.border_mask_img2, self.border_pos) # mask



# class Session:
#     def __init__(self):
#         pg.init()
#         self.clock = pg.time.Clock()
#         self.screen = pg.display.set_mode((WIDTH, HEIGHT))
#         pg.display.set_caption('Race AI')

#         self.load_images()
#         self.generate_objects()
       
        
#     def load_images(self):
#         self.car_img = pg.image.load('images/car.png').convert_alpha()
#         img_width, img_height = self.car_img.get_size()
#         b = 10 # scale factor for the track
#         self.car_img = pg.transform.scale(self.car_img, (img_width // b, img_height // b))
#         self.car_img = pg.transform.rotate(self.car_img, 270)
        
#         a = 1 # scale factor for the track
#         self.track_img = pg.image.load('images/track.png').convert_alpha()
#         img_width, img_height = self.track_img.get_size()
#         self.track_img = pg.transform.scale(self.track_img, (img_width // a, img_height // a))
        
#         self.border_img = pg.image.load('images/track-border.png').convert_alpha()
#         img_width, img_height = self.border_img.get_size()
#         self.border_img = pg.transform.scale(self.border_img, (img_width // a, img_height // a))
        
#         self.background_img = pg.image.load('images/background.jpg').convert()
#         self.background_img = pg.transform.scale(self.background_img, (WIDTH, HEIGHT))

#         self.finish_img = pg.image.load('images/finish.png').convert_alpha()
#         img_width, img_height = self.finish_img.get_size()
#         self.finish_img = pg.transform.scale(self.finish_img, (img_width * 0.78 , img_height * 0.78))
        
#     def generate_objects(self):
#         self.car = Car(self)
#         self.background = Background(self)
            
#     def update(self):
#         self.car.update()
#         self.background.update(self.car)
#         self.clock.tick(FPS)
    
#     def draw(self):
#         self.background.draw()
#         self.car.draw()
#         pg.display.flip()

#     def run(self):
#         running = True
#         while running:
#             for event in pg.event.get():
#                 if event.type == pg.QUIT:
#                     running = False


#                 self.update()
#                 self.draw()
        
#             self.update()
#             self.draw()
            

# if __name__ == '__main__':
    
#     def show_mask(img): #(chatgpt)
#         # Créer une surface blanche de la taille du masque
#         mask = pg.mask.from_surface(img)
#         mask_surface = pg.Surface(mask.get_size(), flags=pg.SRCALPHA)
#         mask_surface.fill((0, 0, 0, 0))  # Transparence
#         # Parcourir les pixels du masque et colorier en blanc les pixels actifs
#         for x in range(mask.get_size()[0]):
#             for y in range(mask.get_size()[1]):
#                 if mask.get_at((x, y)):  # Si le pixel est actif (valeur 1)
#                     mask_surface.set_at((x, y), (255, 255, 255, 255))  # Blanc
#         return mask_surface

    
#     WIDTH = 1200
#     HEIGHT = 900
#     WHITE = (255, 255, 255)
#     FPS = 30 
    
    
#     ses = Session()
#     ses.run()
    
#     pg.quit()
#     sys.exit(0)
    
    
    
    
# # PETIT CIRCUIT (a = 10)
# # les collisions sont bonnes dans l'axe verticales mais mauvaises sur l'axe horizontal.
# # On dirait que le masque du circuit a été comprimé horizontalement.

# # GRAND CRICUIT 
# # collisions verticales OK mais horizontales la voiture dépasse sur les bords.

# # Ca n'est pas une histoire de compression car le probleme est vrai à toutes les échelles

# # Théorie : pour detecter les collisions, le code utilise l'image de la voiture 
# # mais elle n'est pas tournée. Effectivement, quand la voiture ne tourne pas, 
# # les collisions sont bonnes. 

# # le problème semble changer avec la scale de la voiture