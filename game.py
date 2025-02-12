from ursina import *
import time
import random

app = Ursina()

# グローバル変数（スコア、残機）
score = 0
lives = 5

# スコアと残機の表示
score_text = Text(text=f'Score: {score}', position=(-0.85, 0.45), origin=(0,0), scale=1.5)
lives_text = Text(text=f'Lives: {lives}', position=(-0.85, 0.40), origin=(0,0), scale=1.5)

# 地面（プレイヤーが動くフィールド）
ground = Entity(
    model='plane',
    scale=(20,1,20),
    texture='white_cube',
    texture_scale=(20,20),
    collider='box',
    color=color.green
)

# プレイヤークラス
class Player(Entity):
    def __init__(self, **kwargs):
        super().__init__(
            model='cube',
            color=color.azure,
            scale=1,
            collider='box',
            position=(0,0.5,0),
            **kwargs
        )
        self.jump_velocity = 0
        self.is_jumping = False

    def update(self):
        move_speed = 5 * time.dt
        # 矢印キー（または WASD ）による移動
        if held_keys['left arrow'] or held_keys['a']:
            self.x -= move_speed
        if held_keys['right arrow'] or held_keys['d']:
            self.x += move_speed
        if held_keys['up arrow'] or held_keys['w']:
            self.z -= move_speed
        if held_keys['down arrow'] or held_keys['s']:
            self.z += move_speed

        # ジャンプ処理（スペースキー）
        if held_keys['space'] and not self.is_jumping:
            self.jump_velocity = 5
            self.is_jumping = True

        if self.is_jumping:
            self.y += self.jump_velocity * time.dt
            self.jump_velocity -= 9.81 * time.dt  # 重力の適用
            # 地面に戻ったらジャンプ終了
            if self.y <= 0.5:
                self.y = 0.5
                self.is_jumping = False

player = Player()

# カメラを上から見下ろすように配置（上方20の位置、真下を向く）
camera.position = (0, 20, 0)
camera.rotation_x = 90

# 敵リストを保持
enemies = []

# 敵クラス（シンプルな移動とプレイヤーとの接触でダメージ）
class Enemy(Entity):
    def __init__(self, position, **kwargs):
        super().__init__(
            model='cube',
            color=color.red,
            scale=1,
            collider='box',
            position=position,
            **kwargs
        )
        self.health = 3

    def update(self):
        # プレイヤーに向かって移動（シンプルな追尾）
        direction = (player.position - self.position).normalized()
        self.position += direction * time.dt * 2  # 移動速度
        # プレイヤーとの接触をチェック
        if self.intersects(player).hit:
            damage_player()

# 敵のスポーン関数（ランダムな位置に生成）
def spawn_enemy():
    x = random.uniform(-9, 9)
    z = random.uniform(-9, 9)
    enemy = Enemy(position=(x, 0.5, z))
    enemies.append(enemy)

# プレイヤーがダメージを受けた場合の処理
def damage_player():
    global lives, lives_text
    # ※連続でダメージを受けすぎないようにクールダウンを実装するのも検討してください
    lives -= 1
    lives_text.text = f'Lives: {lives}'
    print('Player hit! Lives remaining:', lives)
    if lives <= 0:
        game_over()

# ゲームオーバー時の処理
def game_over():
    print("Game Over!")
    for e in enemies:
        e.disable()
    player.disable()
    Text(text='Game Over', origin=(0,0), scale=3, background=True)

# プレイヤーの攻撃入力（ここでは 'x' キーでパンチとする）
def input(key):
    if key == 'x':
        # プレイヤーの周囲の敵にダメージを与える（距離が近い場合）
        for enemy in enemies.copy():
            if (player.position - enemy.position).length() < 2:
                enemy.health -= 1
                print("Enemy hit! Health:", enemy.health)
                if enemy.health <= 0:
                    enemies.remove(enemy)
                    enemy.disable()
                    global score, score_text
                    score += 10
                    score_text.text = f'Score: {score}'

# 定期的に敵を生成するためのタイマー処理
enemy_spawn_time = time.time()
def update():
    global enemy_spawn_time
    # 他の update 処理（各 Entity の update は自動呼び出し）
    # 約3秒ごとに新しい敵をスポーン
    if time.time() - enemy_spawn_time > 3:
        spawn_enemy()
        enemy_spawn_time = time.time()

app.run()
