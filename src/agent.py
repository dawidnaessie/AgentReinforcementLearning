import math
import random
import pygame
from typing import List, Tuple, Optional
from src.entities import Food, Hazard


class Agent:
    """Klasa reprezentująca agenta sterowanego przez sieć neuronową NEAT."""

    def __init__(
        self,
        net,
        genome,
        width: int = 800,
        height: int = 600,
        start_pos: Optional[Tuple[float, float]] = None
    ):
        self.net = net          # Sieć neuronowa z NEAT
        self.genome = genome    # Referencja do genomu (bezpośrednie przypisywanie punktów fitness)

        # Fizyka i położenie oparte na wektorach 2D
        if start_pos is not None:
            self.pos = pygame.math.Vector2(start_pos[0], start_pos[1])
        else:
            self.pos = pygame.math.Vector2(
                random.uniform(50, width - 50),
                random.uniform(50, height - 50)
            )
        self.vel = pygame.math.Vector2(0.0, 0.0)
        self.max_speed = 4.0
        self.radius = 6.0

        # Witalność agenta
        self.max_energy = 100.0
        self.energy = 100.0
        self.is_alive = True
        self.foods_eaten = 0

        # Inicjalizacja fitnessu genomu
        self.genome.fitness = 0.0

    def _get_sensory_inputs(
        self,
        foods: List[Food],
        hazards: List[Hazard],
        all_agents: List["Agent"],
        width: int,
        height: int
    ) -> Tuple[float, ...]:
        """
        Oblicza 16 znormalizowanych wejść sensorycznych radaru i percepcji relatywnej:
        Wszystkie wejścia są skalowane do przedziałów [0.0, 1.0] lub [-1.0, 1.0],
        co zapobiega nasycaniu wag i funkcji aktywacji tanh.
        """
        max_dist = math.hypot(width, height)

        # 1-2. Własna prędkość znormalizowana [-1.0, 1.0]
        norm_vx = self.vel.x / self.max_speed if self.max_speed > 0 else 0.0
        norm_vy = self.vel.y / self.max_speed if self.max_speed > 0 else 0.0

        # 3-8. Dwa najbliższe punkty pożywienia (dystanse [0..1] oraz wektory kierunkowe dx, dy [-1..1])
        food_distances = []
        for food in foods:
            to_food = food.pos - self.pos
            dist = to_food.length()
            direction = (to_food / dist) if dist > 0 else pygame.math.Vector2(0, 0)
            food_distances.append((dist, direction))

        # Sortowanie pożywienia od najbliższego
        food_distances.sort(key=lambda item: item[0])

        # Pożywienie #1 (najbliższe)
        if len(food_distances) > 0:
            dist1, dir1 = food_distances[0]
            norm_food1_dist = min(dist1 / max_dist, 1.0)
            norm_food1_dx = dir1.x
            norm_food1_dy = dir1.y
        else:
            norm_food1_dist = 1.0
            norm_food1_dx, norm_food1_dy = 0.0, 0.0

        # Pożywienie #2 (drugi cel dla płynności planowania trasy)
        if len(food_distances) > 1:
            dist2, dir2 = food_distances[1]
            norm_food2_dist = min(dist2 / max_dist, 1.0)
            norm_food2_dx = dir2.x
            norm_food2_dy = dir2.y
        else:
            norm_food2_dist = norm_food1_dist
            norm_food2_dx, norm_food2_dy = norm_food1_dx, norm_food1_dy

        # 9-11. Najbliższe zagrożenie (dystans [0..1] oraz kierunek dx, dy [-1..1])
        nearest_hazard_dist = max_dist
        hazard_dir = pygame.math.Vector2(0, 0)
        for hazard in hazards:
            to_hazard = hazard.pos - self.pos
            dist = to_hazard.length()
            if dist < nearest_hazard_dist:
                nearest_hazard_dist = dist
                if dist > 0:
                    hazard_dir = to_hazard / dist

        norm_hazard_dist = min(nearest_hazard_dist / max_dist, 1.0)
        norm_hazard_dx = hazard_dir.x
        norm_hazard_dy = hazard_dir.y

        # 12-14. Najbliższy inny żywy agent (dystans [0..1] oraz kierunek dx, dy [-1..1])
        nearest_agent_dist = max_dist
        agent_dir = pygame.math.Vector2(0, 0)
        for other in all_agents:
            if other is not self and other.is_alive:
                to_other = other.pos - self.pos
                dist = to_other.length()
                if dist < nearest_agent_dist:
                    nearest_agent_dist = dist
                    if dist > 0:
                        agent_dir = to_other / dist

        norm_agent_dist = min(nearest_agent_dist / max_dist, 1.0)
        norm_agent_dx = agent_dir.x
        norm_agent_dy = agent_dir.y

        # 15. Dystans do najbliższej ściany (0.0 przy samej ścianie, 1.0 w centrum planszy)
        dist_to_wall = min(self.pos.x, width - self.pos.x, self.pos.y, height - self.pos.y)
        max_possible_wall_dist = min(width, height) / 2.0
        norm_wall_dist = max(0.0, min(dist_to_wall / max_possible_wall_dist, 1.0))

        # 16. Poziom energii [0.0, 1.0]
        norm_energy = max(0.0, min(self.energy / self.max_energy, 1.0))

        return (
            norm_vx, norm_vy,
            norm_food1_dist, norm_food1_dx, norm_food1_dy,
            norm_food2_dist, norm_food2_dx, norm_food2_dy,
            norm_hazard_dist, norm_hazard_dx, norm_hazard_dy,
            norm_agent_dist, norm_agent_dx, norm_agent_dy,
            norm_wall_dist,
            norm_energy
        )

    def think_and_act(
        self,
        foods: List[Food],
        hazards: List[Hazard],
        all_agents: List["Agent"],
        width: int,
        height: int
    ):
        """Pobiera wejścia, aktywuje sieć neuronową i wykonuje ruch oraz interakcje z reward shapingiem."""
        if not self.is_alive:
            return

        # Obliczenie odległości do najbliższego pożywienia przed ruchem (do reward shapingu)
        prev_min_food_dist = float('inf')
        for food in foods:
            d = (self.pos - food.pos).length()
            if d < prev_min_food_dist:
                prev_min_food_dist = d

        # 1. Zbieranie zmysłów
        inputs = self._get_sensory_inputs(foods, hazards, all_agents, width, height)

        # 2. Aktywacja sieci NEAT (2 wyjścia: przyspieszenie X, przyspieszenie Y)
        outputs = self.net.activate(inputs)
        accel = pygame.math.Vector2(outputs[0], outputs[1])

        # 3. Aktualizacja fizyki i prędkości
        self.vel += accel * 0.8
        speed = self.vel.length()
        if speed > self.max_speed:
            self.vel.scale_to_length(self.max_speed)
            speed = self.max_speed

        self.pos += self.vel

        # Ograniczenie pozycji do granic ekranu
        margin = int(self.radius)
        hit_wall = False
        if self.pos.x < margin:
            self.pos.x = margin
            self.vel.x = 0
            hit_wall = True
        elif self.pos.x > width - margin:
            self.pos.x = width - margin
            self.vel.x = 0
            hit_wall = True

        if self.pos.y < margin:
            self.pos.y = margin
            self.vel.y = 0
            hit_wall = True
        elif self.pos.y > height - margin:
            self.pos.y = height - margin
            self.vel.y = 0
            hit_wall = True

        # Drobna kara za uderzenie w ścianę
        if hit_wall:
            self.genome.fitness -= 0.05

        # 4. Spadek energii z czasem i w zależności od prędkości
        energy_cost = 0.10 + (speed / self.max_speed) * 0.06
        self.energy -= energy_cost

        # Niewielka nagroda za przetrwanie kroku
        self.genome.fitness += 0.03

        # 5. Reward Shaping: nagroda za zmniejszanie dystansu do jedzenia
        if foods and prev_min_food_dist != float('inf'):
            new_min_food_dist = float('inf')
            for food in foods:
                d = (self.pos - food.pos).length()
                if d < new_min_food_dist:
                    new_min_food_dist = d
            dist_delta = prev_min_food_dist - new_min_food_dist
            # Zbliżanie się nagradza, oddalanie nieznacznie obniża fitness
            self.genome.fitness += dist_delta * 0.08

        if self.energy <= 0.0:
            self.is_alive = False
            return

        # 6. Kolizje z pożywieniem
        for food in foods:
            if (self.pos - food.pos).length_squared() <= (self.radius + food.radius) ** 2:
                food.respawn(width, height)
                self.energy = min(self.max_energy, self.energy + 45.0)
                self.genome.fitness += 15.0
                self.foods_eaten += 1

        # 7. Kolizje z zagrożeniami
        for hazard in hazards:
            if (self.pos - hazard.pos).length_squared() <= (self.radius + hazard.radius) ** 2:
                self.energy -= 20.0
                self.genome.fitness -= 5.0
                if self.energy <= 0.0:
                    self.is_alive = False
                    return

    def draw(self, screen: pygame.Surface):
        """Rysuje agenta w sposób zoptymalizowany pod kątem wydajności."""
        if not self.is_alive:
            return

        # Rysowanie ciała agenta
        center = (int(self.pos.x), int(self.pos.y))
        pygame.draw.circle(screen, (52, 152, 219), center, int(self.radius))

        # Rysowanie wskaźnika kierunku prędkości
        if self.vel.length_squared() > 0.01:
            dir_end = self.pos + self.vel.normalize() * (self.radius + 3)
            pygame.draw.line(screen, (241, 196, 15), center, (int(dir_end.x), int(dir_end.y)), 2)