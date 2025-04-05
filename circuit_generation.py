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
        # Dictionnaire pour connaître les connexions 
        # Pour chaque type et orientation, on définit la direction de sortie (ici représentée par delta (di, dj))
        self.connections = {
            0 : { #ligne droite (image de base verticale)
                0 : (-1, 0), #angle 0 : sortie vers le haut inext = iprev + 1
                90 : (0, 1), #sortie vers la droite 
                180 : (1,0), #sortie vers le bas
                270 : (0, -1) #sortie vers la gauche
            },

            1 : { #virage
                0 : (0,1), #sortie vers la droite
                90 : (1,0), #sortie vers le bas
                180 : (0, -1), #sortie vers la gauche
                270 : (-1, 0) #sortie vers le haut
            }
        }

        
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

        #Initialisation : première route posée (ligne droite), en excluant les bords
        self.actual_coord_ij =  (rd.randint(2, np.shape(self.coord)[0] - 2),rd.randint(2, np.shape(self.coord)[1] - 2))
        self.actual_pos = self.coord[self.actual_coord_ij[0], self.actual_coord_ij[1]]
        self.actual_angle = rd.choice(self.angle)
        self.actual_road_type = 0
        self.elements.append([self.actual_pos, self.actual_angle,self.actual_road_type, self.actual_coord_ij])
        
        self.occupied = {self.actual_coord_ij :True}

        def get_entry_direction (actual_piece_type, actual_piece_angle) :
            if  actual_piece_type == 0 :
                if actual_piece_angle == 0 :
                    return 'down'
                elif actual_piece_angle == 90 :
                    return 'left'
                elif actual_piece_angle == 180 :
                    return 'up'
                elif actual_piece_angle == 270 :
                    return 'right'
            else :
                if actual_piece_angle  == 0 :
                    return 'down'
                elif actual_piece_angle == 90 :
                    return 'left'
                elif actual_piece_angle == 180 :
                    return 'up'
                elif actual_piece_angle == 270 :
                    return 'right'

        def get_exit_direction(actual_piece_type, actual_piece_angle) :
            if  actual_piece_type == 0 :
                if actual_piece_angle == 0 :
                    return 'up'
                elif actual_piece_angle == 90 :
                    return 'right'
                elif actual_piece_angle == 180 :
                    return 'down'
                elif actual_piece_angle == 270 :
                    return 'left'
            else :
                if actual_piece_angle == 0 :
                    return 'right'
                elif actual_piece_angle == 90 :
                    return 'down'
                elif actual_piece_angle == 180 :
                    return 'left'
                elif actual_piece_angle == 270 :
                    return 'up'
        
        def get_opposite_direction (direction) :
            opposites = {
                'up': 'down',
                'down': 'up',
                'left': 'right',
                'right': 'left'
            }
            return opposites[direction]
        
        def is_connection_valid (prev_piece_type, prev_piece_angle, next_piece_type, next_piece_angle) :

            prev_out = get_exit_direction (prev_piece_type, prev_piece_angle)
            next_in = get_entry_direction (next_piece_type, next_piece_angle)

            return get_opposite_direction(prev_out) == next_in
        

        def generate_circuit(self,remaining) :
            if remaining <= 0 :
                return self.elements
        
            #Determination des directions possibles en fonction de la pièce précédente

            if self.elements[-1][2] == 1 : #si la dernière route posée est un virage
                first_try = rd.choices([0,1], weights = [70, 30])[0]
            else : 
                first_try = rd.choices([0,1], weights = [50,50])[0]
            second_try = 1 - first_try

            for road_type in [first_try, second_try] :
                for angle in self.angle :
                    #vérifions que la pièce choisie peut-être connectée à celle d'avant
                    if is_connection_valid(self.elements[-1][2], self.elements[-1][1], road_type, angle) == False :
                        print(False)
                        continue


                    
                    delta = self.connections[self.elements[-1][2]][self.elements[-1][1]]
                    new_i = self.elements[-1][3][0] + delta[0]
                    new_j = self.elements[-1][3][1] + delta [1]

                    #vérifions que la nouvelle position est dans la grille
                    if not (0 <= new_i < self.number_of_pixel_height and 0 <= new_j < self.number_of_pixel_width):
                        continue  # hors grille, on passe à l'option suivante
                        
                    if (new_i, new_j) in self.occupied:
                        continue  # collision

                    # Calculer la nouvelle position sur l'écran
                    new_pos = self.coord[new_i, new_j]
                    
                    # Enregistrer la pièce (on peut stocker la position, l'angle, le type et la coordonnée de la case)
                    self.elements.append([new_pos, angle, road_type, (new_i, new_j)])
                    self.occupied[(new_i, new_j)] = True
                    
                    # Appel récursif en diminuant remaining
                    result = generate_circuit(self, remaining - 1)
                    if result:
                        return result  # Si la suite fonctionne, on renvoie le chemin complet
                    
                    # Backtracking : retirer la pièce placée et libérer la case
                    self.elements.pop()
                    del self.occupied[(new_i, new_j)]
            
            # Si aucune option n'a mené à une solution, renvoyer None
            return None

        solution = generate_circuit(self, 10)
        if solution:
            print("Circuit généré avec succès.")
        else:
            print("Échec de la génération du circuit.")         



    def draw(self, position ,angle,type):
    #self.straight_road = pg.transform.rotate(self.car_img, self.angle)
    #self.straight_road_display = self.straight_road(center=self.straight_road_rect(topleft=(50, 50)).center)
        pos_x = position[0]
        pos_y = position[1]
        if type == 0: #straight road

            straight_road_rotated = pg.transform.rotate(self.straight_road, angle)
            self.screen.blit(straight_road_rotated, (pos_x, pos_y, self.pixel_size ,self.pixel_size))
            self.screen.blit(straight_road_rotated, (pos_x, pos_y, self.pixel_size, self.pixel_size))
        elif type == 1: #curved road
            curved_road_rotated = pg.transform.rotate(self.curved_road_right, angle)
            self.screen.blit(curved_road_rotated, (pos_x, pos_y, self.pixel_size ,self.pixel_size))
            self.screen.blit(curved_road_rotated, (pos_x, pos_y, self.pixel_size, self.pixel_size))

        #pg.display.flip()
    



    """def run(self):
        running = True
        while running:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
            
            #self.draw(self.elements[0][0], self.elements[0][1], self.elements[0][2])
            for el in self.elements :
                self.draw(el[0], el[1], el[2])
        print(self.elements[0][0], self.elements[0][1], self.elements[0][2])"""
    def run(self):
        running = True
        clock = pg.time.Clock()  # pour contrôler la vitesse
        current_index = 0  # Combien d'éléments déjà dessinés

        while running:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False

            self.screen.fill((0, 0, 0))  # Efface l'écran à chaque frame (sinon ça va redessiner par-dessus)

            # On dessine les routes déjà posées
            for i in range(current_index):
                pos, angle, road_type, _ = self.elements[i]
                self.draw(pos, angle, road_type)

            # Ajouter une nouvelle tuile progressivement
            if current_index < len(self.elements):
                pos, angle, road_type, _ = self.elements[current_index]
                self.draw(pos, angle, road_type)
                current_index += 1
                pg.time.delay(1000)  # délai en ms pour ralentir l'apparition

            pg.display.flip()
            clock.tick(60)  # Limite à 60 FPS

    print("Circuit terminé !")



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


