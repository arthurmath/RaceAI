import math
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
        self.angle = 0
        self.speed = 0
        self.x, self.y = 230, 275
        self.closest_projection = (self.x, self.y)
        self.acceleration = 0.2
        self.rotation_speed = 9
        self.max_speed = 10
        self.alive = True
        self.collision = 0
        self.compteur = 0 # pour les collisions
        self.last_cp = 0
        self.current_cp = 0
        
        self.car_img = ses.car_img
        self.car_rotated = pg.transform.rotate(self.car_img, self.angle)
        self.car_rect = self.car_rotated.get_rect()
        
        self.finish_rect = ses.finish_img.get_rect(topleft=(200, 330))
        self.total_distance = ses.total_distance
        self.border_pos = ses.border_pos
        self.border_mask = ses.border_mask

        
        
    def update(self, actions: int):
        moved = False               
        
        # Apply action to car's physics
        if actions == 1: # left
            self.angle = (self.angle + self.rotation_speed) % 360
        if actions == 2: # right
            self.angle = (self.angle - self.rotation_speed) % 360
        if actions == 0: # up
            self.speed = min(self.speed + self.acceleration, self.max_speed)
            moved = True
        if actions == 3: # down
            self.speed = max(self.speed - self.acceleration, -self.max_speed / 2)
            moved = True

        # Change speed
        if not moved: # inertia
            if self.speed > 0:
                self.speed = max(self.speed - self.acceleration, 0)
            else:
                self.speed = min(self.speed + self.acceleration, 0)
                
        # Kill car if it collided with borders or finished track
        if self.collision != 0 or self.car_rect.colliderect(self.finish_rect):
            self.alive = False
            if self.compteur < 0: # don't detect collisions too quickly (=30 times/sec), else bug
                self.speed = - self.speed / 2
                self.compteur = 4
        self.compteur -= 1
        
        # Detect collisions with borders
        car_mask = pg.mask.from_surface(self.car_rotated)
        offset = (int(self.x - self.border_pos[0] - 10), int(self.y - self.border_pos[1])) # correspond à la différence des coordonnées des 2 masques.
        booleen = self.border_mask.overlap(car_mask, offset)
        self.collision = 0 if booleen == None else 1
            
        # Change car position
        rad = math.radians(self.angle)
        self.y -= self.speed * math.cos(rad) 
        self.x -= self.speed * math.sin(rad) 
        
    
    def get_progression(self, checkpoints):
        # Finds the car's position projection on the center line
        min_dist = float('inf')
        car_pos = self.x, self.y
        for i in range(len(checkpoints) - 1):
            # Project car position on all segments and find the smallest projection
            proj = lib.project_point_on_segment(car_pos, checkpoints[i], checkpoints[i + 1])
            dist = lib.distance_squared(car_pos, proj)
            if dist < min_dist:
                min_dist = dist
                self.closest_projection = proj
                self.last_cp = i
        traveled_distance = sum(lib.distance(checkpoints[i], checkpoints[i + 1]) for i in range(self.last_cp))
        traveled_distance += lib.distance(checkpoints[self.last_cp], self.closest_projection)
        return (traveled_distance / self.total_distance) * 100 
    
    
    def draw(self, ses):
        self.car_rotated = pg.transform.rotate(self.car_img, self.angle)
        self.car_rect = self.car_rotated.get_rect(center=self.car_img.get_rect(topleft=(self.x, self.y)).center)
        ses.screen.blit(self.car_rotated, self.car_rect.topleft)
        
        





class Session:        
    def __init__(self, nb_cars, display=True):
        self.display = display
        self.nb_cars = nb_cars
        self.episode_done = False
        self.quit = False
        
        self.observation_space = [[0, 1200], [0, 900], [-5, 10], [0, 360], [-60, 30], [0, 400], [-1, 1], [-180, 180]]
        self.action_space = [0, 1, 2, 3]
        
        self.checkpoints = [(240, 274), (240, 213), (239, 131), (300, 76), (364, 131), (365, 194), (365, 297), (366, 392), 
                            (421, 459), (482, 398), (481, 316), (481, 226), (480, 128), (532, 78), (627, 78), (721, 78), 
                            (824, 79), (942, 80), (991, 128), (991, 239), (941, 282), (859, 281), (766, 280), (678, 280), 
                            (613, 341), (681, 406), (764, 404), (859, 402), (940, 403), (986, 443), (988, 510), (987, 587), 
                            (987, 672), (988, 753), (948, 810), (878, 803), (833, 752), (836, 675), (833, 589), (780, 535), 
                            (680, 534), (626, 587), (626, 681), (620, 763), (571, 811), (494, 785), (444, 736), (390, 683), 
                            (316, 610), (262, 553), (239, 487), (239, 422), (239, 353),(240, 274)]
        self.total_distance = sum([math.dist(self.checkpoints[i], self.checkpoints[i + 1]) for i in range(len(self.checkpoints)-1)])
        
        pg.init()
        self.score_font = pg.font.Font(pg.font.match_font('arial'), 20)
        self.border_pos = (170, -10)
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
        self.border_mask = pg.mask.from_surface(self.border_img)
        
        self.background_img = pg.image.load('media/background.jpg').convert()
        self.background_img = pg.transform.scale(self.background_img, (WIDTH, HEIGHT))

        self.finish_img = pg.image.load('media/finish.png').convert_alpha()
        img_width, img_height = self.finish_img.get_size()
        self.finish_img = pg.transform.scale(self.finish_img, (img_width * 0.78 , img_height * 0.78))
        
    def reset(self, episode=1):
        self.episode = episode
        self.episode_done = False
        self.rewards = [0] * self.nb_cars #les rewards sont initialisés a 0 pour toute la liste de longueur nb_cars
        self.prev_rewards = [0] * self.nb_cars
        self.terminateds = [False] * self.nb_cars
        self.car_list = [Car(self) for _ in range(self.nb_cars)] #on crée N voitures
        return self.get_states()#[0] #on retourne l'ensemble des états sous une forme de liste imbriquée, et non pas seulement l'état de la première voiture
        
    def draw(self):
        # Draw background
        self.screen.blit(self.background_img, (0, 0))
        self.screen.blit(self.track_img, self.border_pos)
        self.screen.blit(self.border_img, self.border_pos)
        self.screen.blit(self.finish_img, (200, 330))
        
        # Draw checkpoints
        for checkpoint in self.checkpoints:
            pg.draw.circle(self.screen, (0, 255, 0), checkpoint, 5)
        
        # Draw cars
        for car in self.car_list:
            if car.alive:
                car.draw(self)    
        
        # Draw reward
        text = f"Reward : {self.rewards[0]:.2f}s"
        text_surface = self.score_font.render(text, True, WHITE)
        text_rect = text_surface.get_rect()
        text_rect.topleft = (60, 770)
        self.screen.blit(text_surface, text_rect)
        
        # Draw generation number
        text1 = f"Generation : {self.episode}"
        text_surface1 = self.score_font.render(text1, True, WHITE)
        text_rect1 = text_surface1.get_rect()
        text_rect1.topleft = (60, 750)
        self.screen.blit(text_surface1, text_rect1)
        
        # Draw car number
        text2 = f"Population : {self.nb_alive}"
        text_surface2 = self.score_font.render(text2, True, WHITE)
        text_rect2 = text_surface2.get_rect() 
        text_rect2.topleft = (60, 730)
        self.screen.blit(text_surface2, text_rect2)
        
        self.clock.tick(FPS)  
        pg.display.flip()


    def step(self, actions):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.episode_done = True
                self.quit = True
        
        # Apply action to each cars
        #actions = [[actions]]
        for idx, car in enumerate(self.car_list):
            if len(actions) != 0 and car.alive:
                car.update(actions[idx]) #IL FALLAIT ENCAPSULER LA PUTAIN D4ACTION DANS UNE LISTE
        
        # Check if all cars are dead
        self.nb_alive = sum([car.alive for car in self.car_list])
        if self.nb_alive == 0:
            self.episode_done = True
        
        # Rendering
        if self.display:
            self.draw()
            
        self.states = self.get_states()
        step_rewards = self.get_rewards()
        
        return self.states, step_rewards, self.terminateds

    def get_states(self): #a l'air de fonctionner lorsque display = False
        states = []
        for car in self.car_list:
            dist_to_center_line = lib.distance(car.closest_projection, (car.x, car.y))
            dist_to_center_line *= lib.position_relative(self.checkpoints[car.last_cp],
                                                         self.checkpoints[(car.last_cp + 1) % len(self.checkpoints)],
                                                         (car.x, car.y))
            dist_to_next_cp = lib.distance(self.checkpoints[(car.last_cp + 1) % len(self.checkpoints)], (car.x, car.y))

            # Utiliser modulo pour rendre les indices cycliques
            next_cp = (car.last_cp + 1) % len(self.checkpoints)
            next_next_cp = (car.last_cp + 2) % len(self.checkpoints)

            direction_next_curve = lib.position_relative(self.checkpoints[car.last_cp], self.checkpoints[next_cp],
                                                         self.checkpoints[next_next_cp])
            angle_to_center_line = lib.center_angle(
                car.angle - (360 - lib.angle_segment(self.checkpoints[car.last_cp], self.checkpoints[next_cp])) % 360)

            states.append(
                [car.x, car.y, car.speed, car.angle, dist_to_center_line, dist_to_next_cp, direction_next_curve,
                 angle_to_center_line])
        return states


    # def get_states(self): #erreur lorsque display = False
    #     states = []
    #     for car in self.car_list:
    #         dist_to_center_line = lib.distance(car.closest_projection, (car.x, car.y))
    #         dist_to_center_line *= lib.position_relative(self.checkpoints[car.last_cp],self.checkpoints[car.last_cp + 1], (car.x,car.y))  # pour savoir si la voiture est à droite ou à gauche de la center line
    #         dist_to_next_cp = lib.distance(self.checkpoints[car.last_cp + 1], (car.x, car.y))
    #         # direction_next_curve = lib.distance(self.checkpoints[car.last_cp + 1], self.checkpoints[car.last_cp + 2])
    #         direction_next_curve = lib.position_relative(self.checkpoints[car.last_cp],self.checkpoints[car.last_cp + 1], self.checkpoints[car.last_cp + 2])  # pour savoir si le prochain virage est à droite (1) ou gauche (-1)
    #         angle_to_center_line = lib.center_angle(car.angle - (360 - lib.angle_segment(self.checkpoints[car.last_cp],self.checkpoints[car.last_cp + 1])) % 360)
    #         states.append([car.x, car.y, car.speed, car.angle, dist_to_center_line, dist_to_next_cp, direction_next_curve,angle_to_center_line])
    #     return states


    def get_rewards(self):
        step_rewards = []
        for i, car in enumerate(self.car_list):
            
            car.progression = car.get_progression(self.checkpoints)
            if car.last_cp > car.current_cp:
                self.rewards[i] += 1000 / len(self.checkpoints)
                car.current_cp = car.last_cp
                
            self.rewards[i] -= 0.1
            
            step_reward = self.rewards[i] - self.prev_rewards[i]
            self.prev_rewards[i] = self.rewards[i]
            
            if not car.alive:
                step_reward = -100
                self.terminateds[i] = True
            step_rewards.append(step_reward)
        
        return step_rewards
    
    def close(self):
        pg.quit()





