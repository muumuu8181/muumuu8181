import pygame
import random

# Game configuration
CELL_SIZE = 30
COLS = 10
ROWS = 20
WIDTH = CELL_SIZE * COLS
HEIGHT = CELL_SIZE * ROWS
FPS = 60

# Tetromino shapes
SHAPES = [
    [[1, 1, 1, 1]],  # I
    [[1, 1], [1, 1]],  # O
    [[0, 1, 0], [1, 1, 1]],  # T
    [[1, 0, 0], [1, 1, 1]],  # J
    [[0, 0, 1], [1, 1, 1]],  # L
    [[1, 1, 0], [0, 1, 1]],  # S
    [[0, 1, 1], [1, 1, 0]],  # Z
]

COLORS = [
    (0, 255, 255),
    (255, 255, 0),
    (128, 0, 128),
    (0, 0, 255),
    (255, 165, 0),
    (0, 255, 0),
    (255, 0, 0),
]

class Piece:
    def __init__(self, shape):
        self.shape = shape
        self.color = random.choice(COLORS)
        self.x = COLS // 2 - len(shape[0]) // 2
        self.y = 0

    def rotate(self):
        self.shape = [list(row) for row in zip(*self.shape[::-1])]


def create_grid(locked):
    grid = [[(0, 0, 0) for _ in range(COLS)] for _ in range(ROWS)]
    for y in range(ROWS):
        for x in range(COLS):
            if (x, y) in locked:
                grid[y][x] = locked[(x, y)]
    return grid


def valid_space(piece, grid):
    for y, row in enumerate(piece.shape):
        for x, cell in enumerate(row):
            if cell:
                px = piece.x + x
                py = piece.y + y
                if px < 0 or px >= COLS or py >= ROWS:
                    return False
                if py >= 0 and grid[py][px] != (0, 0, 0):
                    return False
    return True


def clear_rows(grid, locked):
    cleared = 0
    for y in range(ROWS - 1, -1, -1):
        if (0, 0, 0) not in grid[y]:
            cleared += 1
            for x in range(COLS):
                locked.pop((x, y), None)
            for key in sorted(list(locked.keys()), key=lambda k: k[1]):
                x, y_locked = key
                if y_locked < y:
                    locked[(x, y_locked + 1)] = locked.pop(key)
    return cleared


def draw_grid(surface, grid):
    for y in range(ROWS):
        for x in range(COLS):
            pygame.draw.rect(
                surface,
                grid[y][x],
                (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE),
            )
            pygame.draw.rect(
                surface,
                (40, 40, 40),
                (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE),
                1,
            )


def draw_window(surface, grid, score):
    surface.fill((0, 0, 0))
    draw_grid(surface, grid)
    font = pygame.font.Font(None, 36)
    score_surf = font.render(f"Score: {score}", True, (255, 255, 255))
    surface.blit(score_surf, (10, 10))
    pygame.display.flip()


def main():
    pygame.init()
    surface = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Tetris")
    clock = pygame.time.Clock()

    locked = {}
    grid = create_grid(locked)
    current_piece = Piece(random.choice(SHAPES))
    fall_time = 0
    fall_speed = 0.5
    score = 0

    running = True
    while running:
        grid = create_grid(locked)
        fall_time += clock.get_rawtime()
        clock.tick(FPS)

        if fall_time / 1000 >= fall_speed:
            fall_time = 0
            current_piece.y += 1
            if not valid_space(current_piece, grid):
                current_piece.y -= 1
                for y, row in enumerate(current_piece.shape):
                    for x, cell in enumerate(row):
                        if cell:
                            locked[(current_piece.x + x, current_piece.y + y)] = current_piece.color
                current_piece = Piece(random.choice(SHAPES))
                cleared = clear_rows(grid, locked)
                score += cleared * 100
                if not valid_space(current_piece, grid):
                    running = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    current_piece.x -= 1
                    if not valid_space(current_piece, grid):
                        current_piece.x += 1
                elif event.key == pygame.K_RIGHT:
                    current_piece.x += 1
                    if not valid_space(current_piece, grid):
                        current_piece.x -= 1
                elif event.key == pygame.K_DOWN:
                    current_piece.y += 1
                    if not valid_space(current_piece, grid):
                        current_piece.y -= 1
                elif event.key == pygame.K_UP:
                    current_piece.rotate()
                    if not valid_space(current_piece, grid):
                        for _ in range(3):
                            current_piece.rotate()

        draw_window(surface, grid, score)

    pygame.quit()

if __name__ == "__main__":
    main()
