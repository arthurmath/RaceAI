import pygame as pg
import numpy as np
import sys
import math
from pilot import Pilot
from adn import Adn
    
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
        
        self.car_rect = self.car_img.get_rect()
        self.car_rect.left = 200
        self.car_rect.top = HEIGHT / 2 - 50
        self.car_rect.width = 45
        
        self.collision = 0
        self.compteur = 0 # pour les collisions
        

        self.checkpoints = [(239, 273), (239, 130), (300, 75), (360, 130), (370, 392), (420, 451), (479, 389),
                            (482, 126), (531, 74), (941, 80), (988, 127), (989, 240), (940, 278), (680, 277), (614, 341),
                            (681, 386), (941, 399), (986, 440), (987, 750), (941, 800), (890, 800), (840, 751),
                            (831, 583), (780, 532), (680, 533), (634, 582), (611, 760), (570, 797), (301, 585), (236, 436)]

        self.total_distance = 0
        for i in range(len(self.checkpoints) - 1):
            self.total_distance += math.dist(self.checkpoints[i], self.checkpoints[i + 1])
        
        
    def update(self):
    
        moved = False  
        self.progression = self.get_progression()   
        
        moves = []
        
        if ses.player == 1: 
            keys = pg.key.get_pressed()
            if keys[pg.K_LEFT]:
                moves.append('L')
            if keys[pg.K_RIGHT]:
                moves.append('R')
            if keys[pg.K_UP]:
                moves.append('U')
            if keys[pg.K_DOWN]:
                moves.append('D')
        else:
            moves = ses.agent.choose_next_move(self)
                    
                    
        if 'L' in moves:
            self.angle += self.rotation_speed 
        if 'R' in moves:
            self.angle -= self.rotation_speed
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
            if self.compteur < 0: # permet d'éviter de detecter les collisions trop rapidement (= 30 fois/sec), sinon bug
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
        
        for checkpoint in self.checkpoints:
            pg.draw.circle(ses.screen, (0, 255, 0), checkpoint, 5)


        
        # pg.draw.rect(ses.screen, (255, 0, 0), self.car_rect, 2) # heatbox
        # ses.screen.blit(show_mask(self.car_rotated), (self.car_rect.x, self.car_rect.y)) # mask  # TODO self.x ??
        
        lines = [((300, 40), (300, 650), (0, 0, 255)),
                 ((200, 130), (400, 130), (255, 0, 255)),
                 ((320, 390), (520, 390), (0, 0, 255)),
                 ((420, 150), (420, 500), (0, 0, 255)),
                 ((530, 30), (530, 400), (0, 0, 255)),
                 ((440, 125), (1030, 125), (0, 0, 255)),
                 ((940, 40), (940, 320), (0, 0, 255)),
                 ((620, 240), (1050, 240), (0, 0, 255)),
                 ((680, 240), (680, 440), (0, 0, 255)),
                 ((570, 340), (1000, 340), (0, 0, 255)),
                 ((600, 440), (1030, 440), (0, 0, 255)),
                 ((940, 360), (940, 840), (0, 0, 255)),
                 ((800, 750), (1030, 750), (0, 0, 255)),
                 ((890, 600), (890, 840), (0, 0, 255)),
                 ((580, 580), (870, 580), (0, 0, 255)),
                 ((780, 490), (780, 770), (0, 0, 255)),
                 ((680, 490), (680, 770), (0, 0, 255)),
                 ((570, 570), (570, 840), (0, 0, 255)),
                 ((170, 370), (620, 820), (0, 0, 255)),
                 ]
        
        # for start, end, color in lines:
        #     pg.draw.line(ses.screen, color, start, end, 2)
    
    
    def get_progression(self):
        """ Calcule l'avancée de la voiture sur le circuit """

        distance_parcourue = 0
        for i in range(len(self.checkpoints) - 1):
            if math.dist((self.x, self.y), self.checkpoints[i]) < math.dist(self.checkpoints[i], self.checkpoints[i + 1]):
                # Si la voiture est entre deux checkpoints
                distance_parcourue += math.dist(self.checkpoints[i], (self.x, self.y))
                break
            else:
                # Sinon on ajoute la distance entre les 2 derniers checkpoints
                distance_parcourue += math.dist(self.checkpoints[i], self.checkpoints[i + 1])

        progression = (distance_parcourue / self.total_distance) * 100
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
        self.finish_rect = self.finish.get_rect(topleft=(200, 330)) # on crée un rect pour la ligne d'arrivée
        
    def update(self, car):
        self.car_mask = pg.mask.from_surface(car.car_img)
        offset = (int(car.x - self.border_pos[0]), int(car.y - self.border_pos[1])) # correspond à la différence des coordonnées des 2 masques.
        booleen = self.border_mask.overlap(self.car_mask, offset)
        car.collision = 0 if booleen == None else 1
        
    def draw(self):
        ses.screen.blit(self.back, (0, 0))
        ses.screen.blit(self.track, self.border_pos)
        ses.screen.blit(self.border, self.border_pos)
        ses.screen.blit(self.finish, (200, 330))
        
        # ses.screen.blit(self.border_mask_img, self.border_pos) # mask
        
        
    def collision_finish(self, car):
        car_rect = car.car_rect # récupérer le rectangle de la voiture
        return car_rect.colliderect(self.finish_rect) # detecte la collision
        
        
        


class Score:
    def __init__(self, background, car):
        self.start_ticks = pg.time.get_ticks() 
        self.background = background
        self.car = car
        self.update_high_score()
        self.font = pg.font.Font(pg.font.match_font('arial'), 20) # police d'écriture
        
    def update_high_score(self):
        """ Met à jour high_score avec le meilleur temps du fichier """
        with open("times.txt", "r") as f:
            self.high_score = min(float(line) for line in f)
    
            
    def update(self, car):
        self.temps_ecoule = (pg.time.get_ticks() - self.start_ticks) / 1000
        
        if self.background.collision_finish(self.car):
            if self.temps_ecoule < self.high_score: # si le temps realisé est meilleur que l'ancien record
                if self.temps_ecoule > 20: # pour ne pas sauvegarder les marches arrières sur le finish
                    self.high_score = self.temps_ecoule 
                    with open("times.txt", "a") as file:
                        file.write(f"{self.high_score:.3f}\n")
                
            self.start_ticks = pg.time.get_ticks() # reset timer
            car.reset()
            self.update_high_score()
    

    def draw(self, car): 
        # affichage du timer
        text = f"Temps écoulé : {self.temps_ecoule:.3f}s"
        text_surface = self.font.render(text, True, WHITE)   # création de la surface du texte
        text_rect = text_surface.get_rect() # récupération du rectangle de la surface du texte
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
    def __init__(self, player):
        pg.init()
        self.clock = pg.time.Clock()
        self.screen = pg.display.set_mode((WIDTH, HEIGHT))
        pg.display.set_caption('Race AI')
        self.player = player

        # self.music()
        self.load_images()
        self.generate_objects()
       
        self.editing_checkpoints = True # curseur
        #self.checkpoints_fin = []

        
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
        if self.player == 2:
            # with open(Path(args.genetic), "rb") as f:
            #     weights, bias = pickle.load(f)
            self.agent = Pilot(Adn())
            
        
        
    def update(self):
        self.car.update()
        self.background.update(self.car)
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
                #if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                     #pos = pg.mouse.get_pos()
                     #self.checkpoints_fin.append(pos)
                     #with open("checkpoints", "a") as file:
                     #    file.write(f"{self.checkpoints_fin}\n")
                     #print({pos})


                self.update()
                self.draw()
        
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
    
    
    print("\nQui joue au jeu ? \n 1 : Humain \n 2 : IA\n")
    player = int(input("Entrez votre choix (1 ou 2) : "))
    
    ses = Session(player)
    ses.run()
    
    pg.quit()
    sys.exit(0)
    
    
    
    
    
# refaire collisions avec border 







