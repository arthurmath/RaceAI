#%%
import pygame as pg
import random
import sys
import time
import math



class Car:
    def __init__(self, ses):
        self.car_img = ses.car_img
        self.x = 200
        self.y = 275
        self.angle = 0
        self.speed = 0
        self.acceleration = 0.5
        self.rotation_speed = 8
        self.max_speed = 10
        self.collision = None
        self.has_crossed_finish = False #pour pas que le timer s'arrête lors du départ

        self.car_rect = self.car_img.get_rect()
        self.car_rect.left = 200
        self.car_rect.top = HEIGHT / 2 - 50
        self.car_rect.width = 45


    def update(self):

        keys = pg.key.get_pressed()
        moved = False

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
                self.speed = max(self.speed - self.acceleration / 2, 0)
            else:
                self.speed = min(self.speed + self.acceleration / 2, 0)

        if self.collision != None:
            if self.speed < 3:
                self.speed = 1
            else:
                self.speed = - self.speed / 2

        rad = math.radians(self.angle)
        self.y -= self.speed * math.cos(rad)
        self.x -= self.speed * math.sin(rad)


    def draw(self):
        self.car_rotated = pg.transform.rotate(self.car_img, self.angle)

        #car_rect = self.car_rotated.get_rect(center=self.car_rotated.get_rect(center=(self.x, self.y)).center)
        self.car_rect = self.car_rotated.get_rect(center=self.car_img.get_rect(topleft=(self.x, self.y)).center)
        ses.screen.blit(self.car_rotated, self.car_rect.topleft)

        # pg.draw.rect(ses.screen, (255, 0, 0), car_rect, 2) # heatbox
        # ses.screen.blit(show_mask(self.car_rotated), (car_rect.x, car_rect.y)) # mask







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
        offset = (int(car.x - self.border_pos[0]), int(car.y - self.border_pos[1])) # Correspond à la différence entre les coordonnées des rectangles associés aux deux masques.
        car.collision = self.border_mask.overlap(self.car_mask, offset)


    def draw(self):
        ses.screen.blit(self.back, (0, 0))
        ses.screen.blit(self.track, self.border_pos)
        ses.screen.blit(self.border, self.border_pos)
        ses.screen.blit(self.finish, (200, 330))

        # ses.screen.blit(self.border_mask_img, self.border_pos) # mask

    def collision_finish(self, car): #
        car_rect = car.car_rect #cette ligne sert a récupérer les coordonnées de la voiture
        return car_rect.colliderect(self.finish_rect) #on change l'état de collision

       # if car_rect.colliderect(self.finish_rect):
        #    car.has_crossed_finish = True
        #    return True
        #return False #on change l'état de collision
        #on change l'état de collision
        #colliderect : collision rectangle rectangle
        #retourne True ou False si les deux rectangles se touchent ou non









class Session:
    def __init__(self):
        pg.init()
        self.timer_running = True #pour le timer
        self.clock = pg.time.Clock()
        self.screen = pg.display.set_mode((WIDTH, HEIGHT))
        pg.display.set_caption('Race AI')

        #self.music()
        self.load_images()
        self.generate_objects()
        self.start_ticks = pg.time.get_ticks() #initialisation du départ en ms
        self.finish_collision = 0 #pour pouvoir ignorer la première collision avec la ligne d'arrivée

    def music(self):
        self.music = pg.mixer.music.load('BandeOrganise.mp3')
        pg.mixer.music.set_volume(0.3)
        pg.mixer.music.play(-1)

    def load_images(self):
        self.car_img = pg.image.load('images/car.png').convert_alpha()
        img_width, img_height = self.car_img.get_size()
        self.car_img = pg.transform.scale(self.car_img, (img_width // 9, img_height // 9))
        self.car_img = pg.transform.rotate(self.car_img, 270)

        a = 1.0
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


    def update(self):
        self.times = []
        self.background.update(self.car)
        self.car.update()
        self.clock.tick(FPS)
        if self.timer_running and self.background.collision_finish(self.car):
            self.finish_collision += 1

            if self.finish_collision > 1:
                self.timer_running = False

                self.final_time = (pg.time.get_ticks() - self.start_ticks) / 1000
                self.times.append(self.final_time)
                with open("times.txt", "a") as file:
                    file.write(f"{self.final_time:.3f}\n")

    def draw(self): #c'est ici que l'affichage d'objet se fait
        self.background.draw()
        self.car.draw()
        temps_ecoule = (pg.time.get_ticks() - self.start_ticks) / 1000  # temps écoulé en secondes
        if self.timer_running:

            self.draw_text(f"Temps écoulé : {temps_ecoule:.3f}s", 20, 10, 800) #affichage du temps écoulé avec 2 chiffres après la virgule (.2f
        else:
            self.draw_text(f"Temps écoulé : {temps_ecoule:.3f}s", 20, 10, 800)
        pg.display.flip() #mettre a jour l'affichage


    def run(self):
        running = True
        while running:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False

            self.update()
            self.draw()

    def draw_text(self, text, size, x,y): #pour affichage du timer
        font = pg.font.Font(pg.font.match_font('arial'), size) #police d'écriture
        text_surface = font.render(text, True, WHITE)   #création de la surface de texte
        text_rect = text_surface.get_rect() #récupération du rectangle de la surface de texte
        text_rect.topleft = (x, y)
        ses.screen.blit(text_surface, text_rect) #affichage de la surface de texte







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


    WIDTH = 1200 #largeur
    HEIGHT = 840 #hauteur
    WHITE = (255, 255, 255)
    FPS = 30


    ses = Session()
    ses.run()

    pg.quit()
    sys.exit(0)



# a faire : , niveau d'avancee sur le circuit, collision avec ligne arrivée = game over*
#pour que le timer s'arrête a la fin, il faut uniquement prendre les coordonnées de l'image correspondants a l'avant de la voiture (seulement une des 2 au final)
# pour pouvoir validerun temps