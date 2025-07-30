import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame as pg
import numpy as np
import math
    

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
    """ Il faut que les entrées du NN soient dans [-1, 1] pour converger """
    list_ranges = [[0, 1200], [0, 900], [-5, 10], [0, 360]]
    states = [[scale(state[i], *list_ranges[i]) for i in range(len(state))] for state in states]
    return states

def normalisation2(states):
    """ Il faut que les entrées du NN soient dans [-1, 1] pour converger (pour get_states_new)"""
    list_ranges = [[-5, 10], [-60, 30], [0, 400], [-1, 1], [-180, 180]]
    states = [[scale(state[i], *list_ranges[i]) for i in range(len(state))] for state in states]
    return states

def scale(x, a, b):
    """Transforme la valeur x initialement comprise dans l'intervalle [a, b]
        en une valeur comprise dans l'intervalle [-1, 1]."""
    return 2 * (x - a) / (b - a) - 1

def angle_segment(a, b):
    """ Renvoie l'angle du segment [a, b] par rapport à la verticale """
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

def window_average(y, win=100):
    return [sum(y[i : i+win]) / win for i in range(len(y) - win)]

def moving_average(rewards):
    len_window = 10
    moyenne_mobile = []
    for i in range(len(rewards)):
        if i < len_window:
            start_index = 0
        else:
            start_index = i - len_window + 1
        window = rewards[start_index: i + 1]
        moyenne_mobile.append(sum(window) / len(window))
    return moyenne_mobile
def is_focus_transition(car):
    """ pour le double batch des meilleurs perfs """
    return car.progression > 40.0