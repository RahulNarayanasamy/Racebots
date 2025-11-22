import pygame
import sys
import math
import random
import copy
# ------------ VISIBILITY TOGGLES -------------
SHOW_CHECKPOINTS = False
SHOW_SENSORS = False

# ---------------- PYGAME SETUP ----------------
pygame.init()

WIDTH, HEIGHT = 1000, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI Box Race - Checkpoints")
clock = pygame.time.Clock()

# --------------- LOAD TRACK IMAGE ---------------
try:
    track_img = pygame.image.load("track.png").convert()
except pygame.error as e:
    print("Error loading track.png:", e)
    pygame.quit()
    sys.exit()

track_rect = track_img.get_rect()


# --------------- ROAD / BOUNDARY CHECK ---------------
def is_on_road(x, y):
    if x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT:
        return False

    color = track_img.get_at((int(x), int(y)))
    # White = road
    return color == pygame.Color(255, 255, 255)


# --------------- SENSORS ---------------
SENSOR_DIRECTIONS = [-0.6, -0.3, 0.0, 0.3, 0.6]  # radians offset
SENSOR_LENGTH = 200


# --------------- CHECKPOINTS ---------------
# NOTE: These are APPROX guesses for your S-shaped track.
# You will SEE them as red circles; if some are off the road,
# just tweak the (x, y) numbers slightly.
CHECKPOINTS = [
    (100, 650), 
    (300, 660),
    (530, 630),
    (530, 470), 
    (380, 390), 
    (380, 330),
    (480, 300),
    (650, 300),
    (780, 280),
    (840, 180),
    (650, 100),
    (450, 90), 
    (250, 90)
]
CHECKPOINT_RADIUS = 30  # how close car must be to count
FINISH_INDEX = len(CHECKPOINTS) - 1      # last checkpoint is finish
FINISH_RADIUS = CHECKPOINT_RADIUS + 10      # same size as CP radius
FINISH_BONUS = 5000.0

# --------------- SIMPLE NEURAL NETWORK ---------------
def relu(x):
    return x if x > 0 else 0


class Brain:
    def __init__(self, input_size=6, hidden_size=8, output_size=2, weights=None):
        if weights is None:
            self.w1 = [[random.uniform(-1, 1) for _ in range(input_size)] for _ in range(hidden_size)]
            self.b1 = [0.0 for _ in range(hidden_size)]
            self.w2 = [[random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(output_size)]
            self.b2 = [0.0 for _ in range(output_size)]
        else:
            self.w1, self.b1, self.w2, self.b2 = weights

    def forward(self, inputs):
        h = []
        for i in range(len(self.w1)):
            s = self.b1[i]
            for j in range(len(inputs)):
                s += self.w1[i][j] * inputs[j]
            h.append(relu(s))

        out = []
        for i in range(len(self.w2)):
            s = self.b2[i]
            for j in range(len(h)):
                s += self.w2[i][j] * h[j]
            out.append(s)

        return out  # [steering_raw, accel_raw]


def get_weights(brain):
    return (
        copy.deepcopy(brain.w1),
        copy.deepcopy(brain.b1),
        copy.deepcopy(brain.w2),
        copy.deepcopy(brain.b2),
    )


def mutate_weights(weights, mutation_rate=0.1, mutation_strength=0.5):
    w1, b1, w2, b2 = weights

    for layer in (w1, w2):
        for i in range(len(layer)):
            for j in range(len(layer[i])):
                if random.random() < mutation_rate:
                    layer[i][j] += random.uniform(-mutation_strength, mutation_strength)

    for biases in (b1, b2):
        for i in range(len(biases)):
            if random.random() < mutation_rate:
                biases[i] += random.uniform(-mutation_strength, mutation_strength)

    return (w1, b1, w2, b2)

def crossover(weightsA, weightsB):
    w1A, b1A, w2A, b2A = weightsA
    w1B, b1B, w2B, b2B = weightsB

    # New child weights
    new_w1 = []
    for rowA, rowB in zip(w1A, w1B):
        new_w1.append([random.choice([a, b]) for a, b in zip(rowA, rowB)])

    new_b1 = [random.choice([a, b]) for a, b in zip(b1A, b1B)]

    new_w2 = []
    for rowA, rowB in zip(w2A, w2B):
        new_w2.append([random.choice([a, b]) for a, b in zip(rowA, rowB)])

    new_b2 = [random.choice([a, b]) for a, b in zip(b2A, b2B)]

    return (new_w1, new_b1, new_w2, new_b2)

# --------------- CAR CLASS ---------------
POP_SIZE = 40
MAX_LIFE_STEPS = 800  # hard limit per car


class Car:
    def __init__(self, x, y, brain=None):
        self.x = x
        self.y = y
        self.angle = 0.0

        self.speed = 0.0
        self.max_speed = 3.0

        self.brain = brain if brain is not None else Brain()
        self.alive = True
        self.distance_travelled = 0.0
        self.time_alive = 0

        self.max_x_reached = x

        # checkpoint progress
        self.next_checkpoint = 1  # 0 is start, so aim for CP1 first
        self.checkpoints_passed = 0

        self.finished = False
        self.finish_time = None

    # ---- FITNESS FUNCTION ----
    def fitness(self):
        # Big reward per checkpoint reached
        score = self.checkpoints_passed * 1000.0

        # SIMPLE anti-Uturn shaping:
        # closer to next checkpoint = higher score
        score -= self.dist_to_next_cp() * 4.0   # tune 1.0–5.0 if needed

        # Small survival bonus (keeps stable drivers)
        score += self.time_alive * 0.05

        if self.finished:
            score += FINISH_BONUS + (MAX_LIFE_STEPS - self.finish_time) * 2.0

        return score

    def get_sensors(self):
        readings = []
        step = 3

        for d in SENSOR_DIRECTIONS:
            angle = self.angle + d
            dist = 0

            while dist < SENSOR_LENGTH:
                sx = self.x + math.cos(angle) * dist
                sy = self.y + math.sin(angle) * dist

                if not is_on_road(sx, sy):
                    break

                dist += step

            readings.append(dist / SENSOR_LENGTH)
        return readings

    def car_is_safe(self, new_x, new_y):
        car_width = 10
        car_length = 25
        half_w = car_width / 2
        half_l = car_length / 2

        local_points = [
            (half_l, 0),
            (-half_l, 0),
            (0, half_w),
            (0, -half_w),
            (half_l, half_w),
            (half_l, -half_w),
            (-half_l, half_w),
            (-half_l, -half_w),
        ]

        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)

        for lx, ly in local_points:
            wx = new_x + lx * cos_a - ly * sin_a
            wy = new_y + lx * sin_a + ly * cos_a

            if not is_on_road(wx, wy):
                return False

        return True
	
    def dist_to_next_cp(self):
        if self.next_checkpoint >= len(CHECKPOINTS):
            return 0.0
        cx, cy = CHECKPOINTS[self.next_checkpoint]
        return math.hypot(self.x - cx, self.y - cy)

    def check_checkpoint(self):
        if self.finished:
            return

        # If we've already passed all checkpoints except finish,
        # then the next required CP is the finish.
        if self.next_checkpoint == FINISH_INDEX:
            fx, fy = CHECKPOINTS[FINISH_INDEX]
            if math.hypot(self.x - fx, self.y - fy) < FINISH_RADIUS:
                self.checkpoints_passed += 1
                self.next_checkpoint += 1

                self.finished = True
                self.finish_time = self.time_alive

                # stop this car (optional, but common)
                self.alive = False
                self.speed = 0.0
            return

        # Normal checkpoints (must be reached in order)
        if self.next_checkpoint < FINISH_INDEX:
            cx, cy = CHECKPOINTS[self.next_checkpoint]
            if math.hypot(self.x - cx, self.y - cy) < CHECKPOINT_RADIUS:
                self.checkpoints_passed += 1
                self.next_checkpoint += 1

    def update(self):
        if not self.alive:
            return

        self.time_alive += 1

        if self.time_alive > MAX_LIFE_STEPS:
            self.alive = False
            self.speed = 0.0
            return

        # slow & stuck
        if self.time_alive > 300 and self.speed < 0.2:
            self.alive = False
            self.speed = 0.0
            return

        # --- sense ---
        sensors = self.get_sensors()
        inputs = sensors + [self.speed / self.max_speed]

        # --- brain decides ---
        steering_raw, accel_raw = self.brain.forward(inputs)

        steering = max(-0.45, min(0.45, steering_raw))
        accel    = max(-0.2,  min(0.2,  accel_raw))

        self.angle += steering
        self.speed += accel
        self.speed = max(0.0, min(self.max_speed, self.speed))

        dx = math.cos(self.angle) * self.speed
        dy = math.sin(self.angle) * self.speed

        dist = math.hypot(dx, dy)
        if dist == 0:
            return

        step_size = 2.0
        steps = int(dist / step_size) + 1

        for i in range(steps):
            t = (i + 1) / steps
            new_x = self.x + dx * t
            new_y = self.y + dy * t

            if is_on_road(new_x, new_y) and self.car_is_safe(new_x, new_y):
                self.x = new_x
                self.y = new_y
                self.distance_travelled += self.speed

                if self.x > self.max_x_reached:
                    self.max_x_reached = self.x

                # check checkpoint after a safe move
                self.check_checkpoint()
            else:
                self.alive = False
                self.speed = 0.0
                break

    def draw(self, surface):
        car_width = 10
        car_length = 25
        color = (0, 0, 255)

        car_surface = pygame.Surface((car_length, car_width), pygame.SRCALPHA)
        pygame.draw.rect(car_surface, color, (0, 0, car_length, car_width))

        rotated = pygame.transform.rotate(car_surface, -math.degrees(self.angle))
        rotated_rect = rotated.get_rect(center=(self.x, self.y))
        surface.blit(rotated, rotated_rect)

        # sensors
        # sensors (visible only if enabled)
        if SHOW_SENSORS:
            readings = self.get_sensors()
            for d, dist_norm in zip(SENSOR_DIRECTIONS, readings):
                angle = self.angle + d
                dist = dist_norm * SENSOR_LENGTH
                sx = self.x + math.cos(angle) * dist
                sy = self.y + math.sin(angle) * dist
                pygame.draw.line(surface, (0, 255, 0), (self.x, self.y), (sx, sy), 2)


# --------------- GENETIC ALGORITHM ---------------
def create_population(start_x, start_y):
    return [Car(start_x, start_y) for _ in range(POP_SIZE)]


def all_dead(cars):
    return all(not c.alive for c in cars)


def evolve(cars, start_x, start_y):
    # Sort by fitness (descending)
    cars_sorted = sorted(cars, key=lambda c: c.fitness(), reverse=True)

    print(
        f"Best fitness: {cars_sorted[0].fitness():.1f}, "
        f"CPs: {cars_sorted[0].checkpoints_passed}, "
        f"maxX: {cars_sorted[0].max_x_reached:.1f}"
    )

    num_elites = 5
    num_parents = 8

    # -------------------------------------------
    # 1) Identify finishers and sort by finish_time
    # -------------------------------------------
    finishers = [c for c in cars if c.finished]
    finishers.sort(key=lambda c: c.finish_time or 10**9)

    elites = []
    parents = []

    # -------------------------------------------
    # CASE A: At least 1 finisher exists
    # -------------------------------------------
    if len(finishers) >= 1:

        # (i) Elites = fastest finishers, up to 5
        elites = finishers[:num_elites]

        # (ii) Parents also must include all finishers (up to 8)
        parents = finishers[:num_parents]

        # (iii) If fewer than needed, fill from best fitness cars
        if len(parents) < num_parents:
            for c in cars_sorted:
                if c in parents:
                    continue
                parents.append(c)
                if len(parents) == num_parents:
                    break

        # Fill elites if less than 5
        if len(elites) < num_elites:
            for c in cars_sorted:
                if c in elites:
                    continue
                elites.append(c)
                if len(elites) == num_elites:
                    break

    # -------------------------------------------
    # CASE B: No finishers → use top fitness
    # -------------------------------------------
    else:
        parents = cars_sorted[:num_parents]
        elites  = cars_sorted[:num_elites]

    # -------------------------------------------
    # 2) Build new generation
    # -------------------------------------------
    new_cars = []

    # clone elites unchanged
    for e in elites:
        brain_copy = Brain(weights=get_weights(e.brain))
        new_cars.append(Car(start_x, start_y, brain_copy))

    # mutation parameters
    if len(finishers) >= 5:
        mutation_rate = 0.05
        mutation_strength = 0.03
    else:
        mutation_rate = 0.10
        mutation_strength = 0.08


    # Children = crossover + mutation
    while len(new_cars) < POP_SIZE:
        pA, pB = random.sample(parents, 2)

        w_child = crossover(
            get_weights(pA.brain),
            get_weights(pB.brain)
        )
        w_child = mutate_weights(
            w_child, mutation_rate, mutation_strength
        )

        child_brain = Brain(weights=w_child)
        new_cars.append(Car(start_x, start_y, child_brain))

    return new_cars


# --------------- MAIN LOOP ---------------
def main():
    start_x = 100
    start_y = 650  # your working start

    best_finishers_ever = 0

    generation = 1
    cars = create_population(start_x, start_y)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # ---- KILL SWITCH: press N to skip to next generation ----
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    for c in cars:
                        c.alive = False
                    print("Manual skip -> next generation")

        for c in cars:
            c.update()

        if all_dead(cars):
            finishers_gen = sum(1 for c in cars if c.finished)
            best_finishers_ever = max(best_finishers_ever, finishers_gen)

            print(
                f"Generation {generation} finished. "
                f"Finishers: {finishers_gen}  "
                f"(Best ever: {best_finishers_ever})"
            )

            cars = evolve(cars, start_x, start_y)
            generation += 1
            pygame.display.set_caption = f"AI Box Race - Generation {generation}"

        # draw world
        screen.blit(track_img, track_rect)

        # draw checkpoints (red circles)
        if SHOW_CHECKPOINTS:
            for i, (cx, cy) in enumerate(CHECKPOINTS):
                color = (255, 0, 0) if i > 0 else (0, 0, 0)  # start CP0 black
                pygame.draw.circle(screen, color, (int(cx), int(cy)), CHECKPOINT_RADIUS, 2)

        # draw finish marker (green line) at last checkpoint
        if SHOW_CHECKPOINTS:
            fx, fy = CHECKPOINTS[-1]
            pygame.draw.circle(screen, (0, 200, 0), (int(fx), int(fy)), FINISH_RADIUS + 8, 3)
            pygame.draw.line(screen, (0, 200, 0), (fx - 25, fy), (fx + 25, fy), 4)

        # draw cars
        for c in cars:
            c.draw(screen)

        # HUD text
        font = pygame.font.SysFont(None, 24)
        text_gen = font.render(f"Generation: {generation}", True, (0, 0, 0))
        screen.blit(text_gen, (10, 10))

        best_cp = max(c.checkpoints_passed for c in cars)
        text_cp = font.render(f"Best CP: {best_cp}", True, (0, 0, 0))
        screen.blit(text_cp, (10, 40))

        finishers_now = sum(1 for c in cars if c.finished)
        text_fin = font.render(f"Finishers this gen: {finishers_now}", True, (0, 0, 0))
        screen.blit(text_fin, (10, 70))

        text_bestfin = font.render(f"Best finishers ever: {best_finishers_ever}", True, (0, 0, 0))
        screen.blit(text_bestfin, (10, 100))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
