import pygame
import random
import sys

WIDTH, HEIGHT = 800, 600
FPS = 60
PLAYER_SPEED = 5
TEXT_SPEED = 3

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Hello Codex Game")
    clock = pygame.time.Clock()

    font = pygame.font.Font(None, 72)
    text_surface = font.render("Hello Codex!", True, (255, 255, 255))
    text_rect = text_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    text_velocity = [TEXT_SPEED, TEXT_SPEED]

    player_rect = pygame.Rect(WIDTH // 2 - 20, HEIGHT - 60, 40, 40)
    score = 0
    score_font = pygame.font.Font(None, 36)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            player_rect.x -= PLAYER_SPEED
        if keys[pygame.K_RIGHT]:
            player_rect.x += PLAYER_SPEED
        if keys[pygame.K_UP]:
            player_rect.y -= PLAYER_SPEED
        if keys[pygame.K_DOWN]:
            player_rect.y += PLAYER_SPEED

        player_rect.clamp_ip(screen.get_rect())

        text_rect.x += text_velocity[0]
        text_rect.y += text_velocity[1]
        if text_rect.left <= 0 or text_rect.right >= WIDTH:
            text_velocity[0] = -text_velocity[0]
        if text_rect.top <= 0 or text_rect.bottom >= HEIGHT:
            text_velocity[1] = -text_velocity[1]

        if player_rect.colliderect(text_rect):
            score += 1
            text_rect.center = (
                random.randint(100, WIDTH - 100),
                random.randint(100, HEIGHT - 100),
            )

        screen.fill((30, 30, 30))
        screen.blit(text_surface, text_rect)
        pygame.draw.rect(screen, (0, 255, 0), player_rect)
        score_surface = score_font.render(f"Score: {score}", True, (255, 255, 255))
        screen.blit(score_surface, (10, 10))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()
if __name__ == "__main__":
    main()
