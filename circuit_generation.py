import numpy as np
import random as rd
import pygame as pg



class Circuit:
    def __init__(self,):
        #ajouter width et height

        self.width = 1200
        self.height = 900 #a relier a session dans game
        self.screen = pg.display.set_mode((self.width, self.height))
        self.pixel_size = 100
        self.number_of_pixel_height = int(self.height/self.pixel_size)
        self.number_of_pixel_width = int(self.width/self.pixel_size)
        self.coord = np.zeros((self.number_of_pixel_height, self.number_of_pixel_width), dtype = tuple)
        print(self.coord.shape)
        self.angle = [0, 90, 180, 270]
        self.difficulty = 10
        self.diff_straight = 1
        self.diff_curved = 2
        self.diff_counter = 0
        self.straight_road = pg.image.load('media/new_track/route.png').convert()
        self.straight_road = pg.transform.scale(self.straight_road, (self.pixel_size, self.pixel_size))
        self.straight_road_rect = self.straight_road.get_rect()
        #self.car_img = pg.transform.rotate(self.car_img, 270)

        self.curved_road_right = pg.image.load('media/new_track/corner_droite.png').convert()
        self.curved_road_right = pg.transform.scale(self.curved_road_right, (self.pixel_size, self.pixel_size))
        self.elements = []

        for i in range(0,self.number_of_pixel_height):
            for j in range(0,self.number_of_pixel_width):
                self.coord[i,j] = (j*self.pixel_size, i*self.pixel_size)

        self.first_pos = (rd.randint(1, np.shape(self.coord)[1] - 1), rd.randint(1, np.shape(self.coord)[0] - 1))
        self.first_angle = rd.choice(self.angle)
        self.elements.append([self.first_pos, self.first_angle,0])

        # while self.diff_counter < self.difficulty :



    def draw(self, pos_x, pos_y ,angle,type):
        #self.straight_road = pg.transform.rotate(self.car_img, self.angle)
        #self.straight_road_display = self.straight_road(center=self.straight_road_rect(topleft=(50, 50)).center)
        if type == 0: #straight road

            straight_road_rotated = pg.transform.rotate(self.straight_road, angle)
            self.screen.blit(straight_road_rotated, (pos_x, pos_y, self.pixel_size ,self.pixel_size))
            self.screen.blit(straight_road_rotated, (pos_x, pos_y, self.pixel_size, self.pixel_size))
        elif type == 1: #curved road
            curved_road_rotated = pg.transform.rotate(self.curved_road_right, angle)
            self.screen.blit(curved_road_rotated, (pos_x, pos_y, self.pixel_size ,self.pixel_size))
            self.screen.blit(curved_road_rotated, (pos_x, pos_y, self.pixel_size, self.pixel_size))

        pg.display.flip()


    def run(self):
        running = True
        while running:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
            self.draw()



circ = Circuit()
circ.run()
    #def rotate(image, angle):
     #   return pg.transform.rotate(image, angle)
        #def draw(self, ses):
         #  self.car_rotated = pg.transform.rotate(self.car_img, self.angle)
          #  self.car_rect = self.car_rotated.get_rect(center=self.car_img.get_rect(topleft=(self.x, self.y)).center)
           # ses.screen.blit(self.car_rotated, self.car_rect.topleft)

            #for checkpoint in self.checkpoints:
             #   pg.draw.circle(ses.screen, (0, 255, 0), checkpoint, 5)



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


