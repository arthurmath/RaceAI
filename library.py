import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame as pg
import numpy as np
import math


def get_progression(self):
    """ Calcule l'avancée de la voiture sur le circuit : distance parcourue et checkpoints (Orlane)"""

    #La variable validate checkpoint permet, sur un cercle approximé centré sur le checkpoint, de valider la prise de checkpoint peu importe le sens
    # self.checkpoints = [(240, 275), (302, 75), (425, 450), (500, 95), (970, 95), (970, 270)]
    current_position = (self.x, self.y)
    checkpoint_progress = False

    for i, checkpoint in enumerate(self.checkpoints) :
        
        #Calcul de la distance entre la voiture et le checkpoint d'avant et d'après
        dist = math.dist((self.x, self.y), checkpoint)

        "Note pour l'aide à la compréhension de la fonction"
        """A un instant t :
            self.validate_checkpoint - 1 : checkpoint actuel
            self.validate_checkpoint : checkpoint suivant
            self.validate_checkpoint - 2 : checkpoint précédent"""

        if dist <= self.checkpoints_radius and i == self.validate_checkpoint - 1 : #On ne veut pas revalider un checkpoint déjà pris
            # print(f'Warning ! Going past the current cp {i+1} ')
            self.in_checkpoint = True
            break

        if (dist <= self.checkpoints_radius and i == self.validate_checkpoint) : #Validation du checkpoint d'après

            self.validate_checkpoint += 1 #le prochain checkpoint à valider sera le self.validate_checkpoint + 1 
            self.validate_checkpoint = min(self.validate_checkpoint, len(self.checkpoints)) #Pour éviter une éventuelle erreur dans la boucle
            checkpoint_progress = True
            self.in_checkpoint = False
            break
            
        if (dist <= self.checkpoints_radius and i == self.validate_checkpoint -2) : #Dans le cas où la voiture recule au checkpoint d'avant, elle régressera d'un checkpoint
            #Pour par qu'il y ait de décrémentation si la voiture ne recule pas (distance approximée), on met une condition
            self.validate_checkpoint -= 1
            self.validate_checkpoint = max(self.validate_checkpoint, 0)
            checkpoint_progress = True
            self.in_checkpoint = False
            break
        else :
            self.in_checkpoint = False
    
    if self.validate_checkpoint > 0:

        current_checkpoint = self.checkpoints[self.validate_checkpoint- 1]
        next_checkpoint = self.checkpoints[self.validate_checkpoint]
        prev_checkpoint = self.checkpoints[self.validate_checkpoint - 2]
        "POUR FORWARD "
        #vecteur entre current et previous pos ((prevpos, currentpos))
        vect_curr_prev = np.array([current_position[0] - self.previous_pos[0], current_position[1] - self.previous_pos[1]])

        #vect entre cp actuel et suivant 
        vect_curr_next_cp = - np.array([current_checkpoint[0] - next_checkpoint[0], current_checkpoint[1] - next_checkpoint[1]])
        "FIN POUR FORWARD "

        "POUR CALCUL DE LA DISTANCE PARCOURUE"
        # Position projetée de la voiture sur la droite current cp -> next cp
        projected_x_current = lib.projection_betw_cp (current_checkpoint, next_checkpoint, current_position)[0]
        projected_y_current = lib.projection_betw_cp (current_checkpoint, next_checkpoint, current_position)[1]
        # Position projetée de la voiture sur la droite prev cp -> current cp
        projected_x_prev = lib.projection_betw_cp(prev_checkpoint, current_checkpoint, current_position)[0]
        projected_y_prev = lib.projection_betw_cp(prev_checkpoint, current_checkpoint, current_position)[1]
        "FIN POUR CALCUL DE LA DISTANCE PARCOURUE"
        #voyons maintenant si on est devant ou derrière le checkpoint actuel

        "POUR SAVOIR SI ON EST DERRIERE LE DERNIER CHECKPOINT VALIDE"
        # Positions projetées
        #projection sur x et y de la voiture
        #Sur x
        coord_curr_pos_x = lib.ortho_projection (self.ortho_sys_x, self.ortho_sys_y, current_position)[0]
        #Sur y 
        coord_curr_pos_y = lib.ortho_projection (self.ortho_sys_x, self.ortho_sys_y, current_position)[1]
        #Projection du current cp
        #Sur x
        coord_curr_cp_x = lib.ortho_projection (self.ortho_sys_x, self.ortho_sys_y, current_checkpoint)[0]
        #Sur y
        coord_curr_cp_y = lib.ortho_projection (self.ortho_sys_x, self.ortho_sys_y, current_checkpoint)[1]
        #Projection du next cp
        coord_next_cp_x = lib.ortho_projection (self.ortho_sys_x, self.ortho_sys_y, next_checkpoint)[0]
        #Sur y
        coord_next_cp_y = lib.ortho_projection (self.ortho_sys_x, self.ortho_sys_y, next_checkpoint)[1]
        #Maintenant que nous avons les projections de la voiture, du current cp et du next cp dans la base, traçons les vecteurs
        #Voiture avec curr cp
        vect_car_curr_cp_x = np.array([coord_curr_pos_x[0] - coord_curr_cp_x[0], coord_curr_pos_x[1] - coord_curr_cp_x[1]])
        vect_car_curr_cp_y = np.array([coord_curr_pos_y[0] - coord_curr_cp_y[0], coord_curr_pos_y[1] - coord_curr_cp_y[1]])
        projected_syst_car_curr_cp = (vect_car_curr_cp_x,vect_car_curr_cp_y)
        #Curr cp avec next cp
        vect_curr_next_cp_x = np.array([coord_next_cp_x[0] - coord_curr_cp_x[0], coord_next_cp_x[1] - coord_curr_cp_x[1]])
        vect_curr_next_cp_y = np.array([coord_next_cp_y[0] - coord_curr_cp_y[0], coord_next_cp_y[1] - coord_curr_cp_y[1]])
        projected_syst_curr_next_cp = (vect_curr_next_cp_x,vect_curr_next_cp_y)
        #Comparons maintenant les signes des coordonnées de ces vecteurs projetés. S'ils sont égaux, alors ils sont orientés dans le même sens
        self.behind_cp = False
        for axis in range (0, 2) :
            if self.in_checkpoint == True :
                break
            if (projected_syst_car_curr_cp[axis][0] * projected_syst_curr_next_cp[axis][0] < 0) or (projected_syst_car_curr_cp[axis][1] * projected_syst_curr_next_cp[axis][1] < 0) :
                self.behind_cp = True
                break
        "FIN POUR SAVOIR SI ON EST DERRIERE LE DERNIER CHECKPOINT VALIDE"

        if self.speed != 0 :
            forward = np.dot(vect_curr_next_cp, vect_curr_prev) > 0
            #print(forward)
            dist_to_current_cp = math.dist(current_checkpoint, (projected_x_current, projected_y_current))
            
            # print(out_of_reach_cp, dist_to_current_cp, behind_cp, np.dot(vect_car_curr_cp, vect_curr_next_cp))
            if (self.behind_cp == True) and ((self.validate_checkpoint - 1 > 0) == True) :
                
                segment_distance = abs(math.dist(prev_checkpoint, (projected_x_prev, projected_y_prev)))
                #print('OK1')
                self.traveled_distance = abs(sum([math.dist(self.checkpoints[checked], self.checkpoints[checked + 1]) for checked in range(self.validate_checkpoint -2 )]) + segment_distance)
            else :
                segment_distance = abs(math.dist(current_checkpoint, (projected_x_current, projected_y_current)))

                self.traveled_distance = abs(sum([math.dist(self.checkpoints[checked], self.checkpoints[checked + 1]) for checked in range(self.validate_checkpoint -1 )]) + segment_distance)

                # Mettre à jour la distance parcourue

            self.last_traveled_distance = self.traveled_distance
        else :
            forward = None
            self.traveled_distance = self.last_traveled_distance
        
    if checkpoint_progress == True :
        print(self.validate_checkpoint)
        print(f"Checkpoint {self.validate_checkpoint} atteint!")
        # print(behind_cp)

    progression = (self.traveled_distance/self.total_distance)*100
    #print(progression)
    return min(max(progression, 0), 100)
    

def projection_betw_cp (starting_cp, ending_cp, position_to_proj) :
    "starting_cp, ending_cp et position_to_proj sont des tuples de coordonnées de la forme (x,y)"
    dx = ending_cp[0] - starting_cp[0]
    dy = ending_cp[1] - starting_cp[1]
    # Calcul du paramètre t qui représente la projection de la voiture sur la droite
    t = ((position_to_proj[0] - starting_cp[0]) * dx + (position_to_proj[1] - starting_cp[1])*dy)/(dx ** 2 + dy ** 2)
    # On s'assure que t reste entre 0 et 1 (pour qu'il soit entre les deux checkpoints)
    t = max(0, min(1,t))
    # Position projetée de la voiture sur la droite
    projected_position_x = starting_cp[0] + t * dx
    projected_position_y = starting_cp [1] + t * dy
    return(projected_position_x, projected_position_y)

def ortho_projection (ortho_sys_x, ortho_sys_y, position_to_proj) :
    "ortho_sys_x et ortho_sys_y sont des tuples de tuples sous la forme ((x1,y1), (x2,y2)) et représentent les droites du repère orthogonale"
    "position_to_proj est un tuple de coordonnées"
    dx_x = ortho_sys_x[1][0] - ortho_sys_x[0][0]
    dy_x = ortho_sys_x[1][1] - ortho_sys_x[0][1]

    dx_y = ortho_sys_y[1][0] - ortho_sys_y[0][0]
    dy_y = ortho_sys_y[1][1] - ortho_sys_y[0][1]

    if dx_x == 0 and dy_x == 0:
        print("⚠️ Erreur: dx_x et dy_x sont nuls après le checkpoint (library)")

    if dx_y == 0 and dy_y == 0:
        print("⚠️ Erreur: dx_y et dy_y sont nuls après le checkpoint (library)")

    t_x = ((position_to_proj[0] - ortho_sys_x [0][0]) * dx_x + (position_to_proj[1] - ortho_sys_x[0][1])*dy_x) / (dx_x **2 + dy_x **2)
    t_y = ((position_to_proj[0] - ortho_sys_y[0][0]) * dx_y + (position_to_proj[1] - ortho_sys_y[0][1]) * dy_y) / (dx_y ** 2 + dy_y ** 2)
    
    #Projection sur l'axe x
    projected_position_x_x = ortho_sys_x[0][0] + t_x * dx_x #coordonnée x de la projection sur x
    projected_position_y_x = ortho_sys_x[0][1] + t_x * dy_x #coordonnée y de la projection sur x
    coord_proj_pos_x = (projected_position_x_x, projected_position_y_x)
    #Projection sur l'axe y
    projected_position_x_y = ortho_sys_y[0][0] + t_y * dx_y
    projected_position_y_y = ortho_sys_y[0][1] + t_y * dy_y
    coord_proj_pos_y = (projected_position_x_y, projected_position_y_y)

    return(coord_proj_pos_x, coord_proj_pos_y)

def show_mask(img): 
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

def distance_squared(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

def distance(p1, p2):
    return math.sqrt(distance_squared(p1, p2))

def distance_signed(p1, p2):
    return (p1[0] - p2[0]) + (p1[1] - p2[1])

def project_point_on_segment(p, a, b):
    ax, ay = a
    bx, by = b
    px, py = p
    
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    ab_len_squared = abx**2 + aby**2
    
    if ab_len_squared == 0:
        return a  # a et b sont le même point
    
    t = (apx * abx + apy * aby) / ab_len_squared
    t = max(0, min(1, t))  # Clamp t entre 0 et 1
    
    proj_x = ax + t * abx
    proj_y = ay + t * aby
    
    return (int(proj_x), int(proj_y))

def normalisation(states):
    """ Il faut que les entrées soient dans [-1, 1] pour converger """
    list_ranges = [[0, 1200], [0, 900], [-5, 10], [0, 360], [0, 100]]
    states = [[scale(state[i], *list_ranges[i]) for i in range(len(state))] for state in states]
    return states

def normalisation2(states):
    """ Il faut que les entrées soient dans [-1, 1] pour converger """
    list_ranges = [[-5, 10], [-60, 30], [0, 400], [-1, 1], [-180, 180]]
    states = [[scale(state[i], *list_ranges[i]) for i in range(len(state))] for state in states]
    return states

def scale(x, a, b):
    """Transforme la valeur x initialement comprise dans l'intervalle [a, b]
        en une valeur comprise dans l'intervalle [-1, 1]."""
    return 2 * (x - a) / (b - a) - 1

def angle_segment(a, b):
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    angle = math.degrees(math.atan2(dx, -dy))  # -dy pour considérer la verticale
    return angle
    
def position_relative(A, B, C):
    """ Détermine si le point C est à gauche ou à droite par rapport au segment orienté de A vers B. """
    
    ABx, ABy = B[0] - A[0], B[1] - A[1]
    ACx, ACy = C[0] - A[0], C[1] - A[1]
    
    # Calcul du déterminant (produit vectoriel en 2D)
    det = ABx * ACy - ABy * ACx

    if det >= 0:
        return 1 # gauche
    else:
        return -1 # droite
    
def center_angle(angle):
    """ Normalise les angles pour qu'ils restent entre -180 et 180. """
    if -180 <= angle <= 180:
        return angle
    elif angle < -180:
        return angle + 360
    else:
        return angle - 360
