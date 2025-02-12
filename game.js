import * as THREE from 'https://unpkg.com/three@0.157.0/build/three.module.js';

let scene, camera, renderer, player, ground;
let enemies = [];
let score = 0;
let lives = 5;
let isGameOver = false;
let lastEnemySpawn = Date.now();

// Scene setup
function init() {
    // Reset game state
    score = 0;
    lives = 5;
    isGameOver = false;
    enemies = [];
    document.getElementById('score').textContent = `Score: ${score}`;
    document.getElementById('lives').textContent = `Lives: ${lives}`;
    document.getElementById('gameOver').style.display = 'none';

    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    renderer = new THREE.WebGLRenderer();
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);

    // Ground
    const groundGeometry = new THREE.PlaneGeometry(20, 20);
    const groundMaterial = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
    ground = new THREE.Mesh(groundGeometry, groundMaterial);
    ground.rotation.x = -Math.PI / 2;
    scene.add(ground);

    // Player
    const playerGeometry = new THREE.BoxGeometry(1, 1, 1);
    const playerMaterial = new THREE.MeshBasicMaterial({ color: 0x0000ff });
    player = new THREE.Mesh(playerGeometry, playerMaterial);
    player.position.y = 0.5;
    scene.add(player);

    // Camera position
    camera.position.set(0, 20, 0);
    camera.lookAt(0, 0, 0);

    // Event listeners
    document.addEventListener('keydown', onKeyDown);
    document.addEventListener('keyup', onKeyUp);
    window.addEventListener('resize', onWindowResize);
}

// Movement controls
const keys = { left: false, right: false, up: false, down: false, space: false };
const moveSpeed = 0.1;
let isJumping = false;
let jumpVelocity = 0;

function onKeyDown(event) {
    switch(event.key.toLowerCase()) {
        case 'arrowleft':
        case 'a': keys.left = true; break;
        case 'arrowright':
        case 'd': keys.right = true; break;
        case 'arrowup':
        case 'w': keys.up = true; break;
        case 'arrowdown':
        case 's': keys.down = true; break;
        case ' ': keys.space = true; break;
        case 'x': attack(); break;
    }
}

function onKeyUp(event) {
    switch(event.key.toLowerCase()) {
        case 'arrowleft':
        case 'a': keys.left = false; break;
        case 'arrowright':
        case 'd': keys.right = false; break;
        case 'arrowup':
        case 'w': keys.up = false; break;
        case 'arrowdown':
        case 's': keys.down = false; break;
        case ' ': keys.space = false; break;
    }
}

function updatePlayer() {
    if (keys.left) player.position.x -= moveSpeed;
    if (keys.right) player.position.x += moveSpeed;
    if (keys.up) player.position.z -= moveSpeed;
    if (keys.down) player.position.z += moveSpeed;

    if (keys.space && !isJumping) {
        isJumping = true;
        jumpVelocity = 0.2;
    }

    if (isJumping) {
        player.position.y += jumpVelocity;
        jumpVelocity -= 0.01;
        if (player.position.y <= 0.5) {
            player.position.y = 0.5;
            isJumping = false;
        }
    }
}

// Enemy spawning and management
function spawnEnemy() {
    const enemyGeometry = new THREE.BoxGeometry(1, 1, 1);
    const enemyMaterial = new THREE.MeshBasicMaterial({ color: 0xff0000 });
    const enemy = new THREE.Mesh(enemyGeometry, enemyMaterial);
    
    enemy.position.x = Math.random() * 18 - 9;
    enemy.position.z = Math.random() * 18 - 9;
    enemy.position.y = 0.5;
    enemy.health = 3;
    
    scene.add(enemy);
    enemies.push(enemy);
}

function updateEnemies() {
    const now = Date.now();
    if (now - lastEnemySpawn > 3000) {
        spawnEnemy();
        lastEnemySpawn = now;
    }

    enemies.forEach(enemy => {
        const direction = new THREE.Vector3();
        direction.subVectors(player.position, enemy.position).normalize();
        enemy.position.x += direction.x * 0.05;
        enemy.position.z += direction.z * 0.05;

        // Check collision with player
        if (enemy.position.distanceTo(player.position) < 1) {
            damagePlayer();
        }
    });
}

function attack() {
    if (isGameOver) return;
    
    enemies.forEach((enemy, index) => {
        if (enemy.position.distanceTo(player.position) < 2) {
            enemy.health--;
            if (enemy.health <= 0) {
                scene.remove(enemy);
                enemies.splice(index, 1);
                updateScore(10);
            }
        }
    });
}

function damagePlayer() {
    if (isGameOver) return;
    
    const now = Date.now();
    if (!player.lastDamage || now - player.lastDamage > 1000) {  // 1 second invulnerability
        lives--;
        player.lastDamage = now;
        document.getElementById('lives').textContent = `Lives: ${lives}`;
        
        if (lives <= 0) {
            gameOver();
        }
    }
}

function updateScore(points) {
    score += points;
    document.getElementById('score').textContent = `Score: ${score}`;
}

function gameOver() {
    isGameOver = true;
    document.getElementById('gameOver').style.display = 'block';
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

// Animation loop
function animate() {
    requestAnimationFrame(animate);
    
    if (!isGameOver) {
        updatePlayer();
        updateEnemies();
    }
    
    renderer.render(scene, camera);
}

// Start the game
window.addEventListener('load', () => {
    init();
    animate();
});
