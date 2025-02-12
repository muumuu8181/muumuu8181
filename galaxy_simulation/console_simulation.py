import math
import random
import time
import os

# シミュレーションの定数
NUM_PARTICLES = 100  # コンソールモードでは粒子数を減らして見やすくする
G = 0.05
TIME_STEP = 1
WIDTH = 80  # コンソール幅
HEIGHT = 24  # コンソール高さ

class Particle:
    def __init__(self, x, y, mass, vx, vy):
        self.x = x
        self.y = y
        self.mass = mass
        self.vx = vx
        self.vy = vy
        self.ax = 0
        self.ay = 0

    def update(self, particles):
        self.ax = 0
        self.ay = 0
        for other in particles:
            if other is self:
                continue
            dx = other.x - self.x
            dy = other.y - self.y
            dist = math.sqrt(dx**2 + dy**2)
            if dist < 5:
                dist = 5
            force = G * self.mass * other.mass / (dist**2)
            self.ax += force * dx / (dist * self.mass)
            self.ay += force * dy / (dist * self.mass)
        
        self.vx += self.ax * TIME_STEP
        self.vy += self.ay * TIME_STEP

    def move(self):
        self.x += self.vx * TIME_STEP
        self.y += self.vy * TIME_STEP
        if self.x < 0 or self.x > WIDTH:
            self.vx *= -1
        if self.y < 0 or self.y > HEIGHT:
            self.vy *= -1

def create_particles():
    particles = []
    for cluster in range(2):
        if cluster == 0:
            cx, cy = WIDTH * 0.3, HEIGHT / 2
            vx_center, vy_center = 1, 0
        else:
            cx, cy = WIDTH * 0.7, HEIGHT / 2
            vx_center, vy_center = -1, 0
        
        for _ in range(NUM_PARTICLES // 2):
            x = cx + random.uniform(-5, 5)
            y = cy + random.uniform(-3, 3)
            mass = random.uniform(5, 15)
            vx = vx_center + random.uniform(-0.5, 0.5)
            vy = vy_center + random.uniform(-0.5, 0.5)
            particles.append(Particle(x, y, mass, vx, vy))
    return particles

def draw_particles(particles):
    # 空の画面バッファを作成
    screen = [[' ' for _ in range(WIDTH)] for _ in range(HEIGHT)]
    
    # 各粒子を画面バッファに描画
    for p in particles:
        x, y = int(p.x), int(p.y)
        if 0 <= x < WIDTH and 0 <= y < HEIGHT:
            screen[y][x] = '*'
    
    # 画面をクリアしてバッファを描画
    os.system('clear' if os.name == 'posix' else 'cls')
    for row in screen:
        print(''.join(row))

def main():
    print("銀河衝突シミュレーション - コンソールモード")
    print("Ctrl+C で終了")
    particles = create_particles()
    
    try:
        while True:
            for p in particles:
                p.update(particles)
            for p in particles:
                p.move()
            draw_particles(particles)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nシミュレーションを終了します")

if __name__ == "__main__":
    main()
