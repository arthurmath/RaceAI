import pygame as pg
import math
import os, sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
import library as lib

# Initialisation de Pygame
pg.init()

# Dimensions de la fenêtre
WIDTH, HEIGHT = 1200, 900
screen = pg.display.set_mode((WIDTH, HEIGHT))
pg.display.set_caption("Checkpoints et Projection")

# Liste des checkpoints
checkpoints = [(239, 273), (239, 130), (300, 75), (360, 130), (360, 392), (420, 451), (479, 389), (479, 126), 
               (531, 80), (941, 80), (988, 127), (988, 240), (940, 278), (680, 278), (614, 341), (681, 386), 
               (941, 386), (986, 440), (986, 750), (941, 800), (890, 800), (840, 751), (840, 583), (780, 532), 
               (680, 532), (620, 582), (620, 760), (570, 797), (301, 585), (238, 505), (238, 352)]


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

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


def get_progression(pos):
    
    # Trouver la projection du point sur le tracé
    closest_projection = (0, 0)
    min_dist = float('inf')
    last_cp = 0
    for i in range(len(checkpoints) - 1):
        proj = project_point_on_segment(pos, checkpoints[i], checkpoints[i + 1])
        dist = distance(pos, proj) ** 2
        if dist < min_dist:
            min_dist = dist
            closest_projection = proj
            last_cp = i
    
    traveled_distance = sum(distance(checkpoints[i], checkpoints[i + 1]) for i in range(last_cp))
    traveled_distance += distance(checkpoints[last_cp], closest_projection)
    
    return (traveled_distance / total_distance) * 100, closest_projection, last_cp





def get_states(projection, last_cp, mouse_pos):
    states = []
    dist_to_center_line = lib.distance(projection, mouse_pos)
    dist_to_next_cp = lib.distance(checkpoints[last_cp + 1], mouse_pos)
    try:
        direction_next_curve = lib.distance(checkpoints[last_cp + 1], checkpoints[last_cp + 2])
    except:
        direction_next_curve = 100 # Dernier cp
    # angle_to_center_line = self.car.angle - lib.angle_segment(checkpoints[last_cp], checkpoints[last_cp + 1])
    states.append([dist_to_center_line, dist_to_next_cp, direction_next_curve]) # speed, 
    # self.states = lib.normalisation2(self.states) 
    return states 
    
            
            



total_distance = sum(distance(checkpoints[i], checkpoints[i + 1]) for i in range(len(checkpoints) - 1))
running = True


while running:
    screen.fill((0, 0, 0))
    
    mouse_pos = pg.mouse.get_pos()
    progression, projection, last_cp = get_progression(mouse_pos)
    states = get_states(projection, last_cp, mouse_pos)
    #print([round(x, 2) for x in states[0]]) # [[,], [0, 400], [0, 400]]
    
    
    # Dessiner le point de la souris
    pg.draw.circle(screen, (0, 255, 0), mouse_pos, 7)
    
    # Dessiner les lignes entre checkpoints
    for i in range(len(checkpoints) - 1):
        pg.draw.line(screen, (0, 0, 255), checkpoints[i], checkpoints[i + 1], 2)
    
    # Dessiner les checkpoints
    for checkpoint in checkpoints:
        pg.draw.circle(screen, (255, 0, 0), checkpoint, 5)
    
    # Dessiner le dernier checkpoint validé
    pg.draw.circle(screen, (255, 255, 255), checkpoints[last_cp], 5)
    
    # Dessiner la projection
    pg.draw.circle(screen, (255, 255, 0), projection, 7)
    
    # Afficher la progression
    text = pg.font.Font(None, 36).render(f"Progression: {progression:.2f}%", True, (255, 255, 255))
    screen.blit(text, (50, 50))
    
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
    
    pg.display.flip()
    
pg.quit()


















    
# def compute_progression(proj, checkpoints):
    
#     # Trouver la projection du point sur le tracé
#     closest_projection = None
#     min_dist = float('inf')
#     last_cp = 0
#     car_pos = pg.mouse.get_pos()
#     for i in range(len(checkpoints) - 1):
#         proj = lib.project_point_on_segment(car_pos, checkpoints[i], checkpoints[i + 1])
#         dist = lib.distance_squared(car_pos, proj)
#         if dist < min_dist:
#             min_dist = dist
#             closest_projection = proj
#             last_cp = i
    
#     traveled_distance = sum(lib.distance(checkpoints[i], checkpoints[i + 1]) for i in range(last_cp))
#     traveled_distance += lib.distance(checkpoints[last_cp], closest_projection)
    
#     return (traveled_distance / total_distance) * 100 

# def projection():
#     # Trouver la projection du point sur le tracé
#     closest_projection = None
#     min_dist = float('inf')
#     last_cp = 0
#     for i in range(len(checkpoints) - 1):
#         proj = project_point_on_segment(mouse_pos, checkpoints[i], checkpoints[i + 1])
#         dist = distance_squared(mouse_pos, proj)
#         if dist < min_dist:
#             min_dist = dist
#             closest_projection = proj
#             last_cp = i
#     return closest_projection, last_cp