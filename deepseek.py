import pygame
import sys

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# Colors
WHITE = (255, 255, 255)

# Load images
background = pygame.image.load("images/background.jpg")
track = pygame.image.load("images/track.png")
border = pygame.image.load("images/track-border.png")
car = pygame.image.load("images/car.png")

# Resize images to fit the screen
background = pygame.transform.scale(background, (SCREEN_WIDTH, SCREEN_HEIGHT))
track = pygame.transform.scale(track, (SCREEN_WIDTH, SCREEN_HEIGHT))
border = pygame.transform.scale(border, (SCREEN_WIDTH, SCREEN_HEIGHT))
car = pygame.transform.scale(car, (50, 30))  # Resize car to a reasonable size

# Create masks for collision detection
border_mask = pygame.mask.from_surface(border)
car_mask = pygame.mask.from_surface(car)

# Set up the screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Car Racing Game")

# Clock for controlling frame rate
clock = pygame.time.Clock()

# Car initial position and speed
car_x = 300
car_y = 530
car_speed = 5

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Get keyboard input
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        car_y -= car_speed
    if keys[pygame.K_DOWN]:
        car_y += car_speed
    if keys[pygame.K_LEFT]:
        car_x -= car_speed
    if keys[pygame.K_RIGHT]:
        car_x += car_speed

    # Collision detection
    car_rect = car.get_rect(topleft=(car_x, car_y))
    offset = (car_rect.x, car_rect.y)
    if border_mask.overlap(car_mask, offset):
        # If collision occurs, revert the car's position
        if keys[pygame.K_UP]:
            car_y += car_speed
        if keys[pygame.K_DOWN]:
            car_y -= car_speed
        if keys[pygame.K_LEFT]:
            car_x += car_speed
        if keys[pygame.K_RIGHT]:
            car_x -= car_speed

    # Draw everything
    screen.blit(background, (0, 0))
    screen.blit(track, (0, 0))
    screen.blit(border, (0, 0))
    screen.blit(car, (car_x, car_y))

    # Update the display
    pygame.display.flip()

    # Cap the frame rate
    clock.tick(60)

# Quit Pygame
pygame.quit()
sys.exit()