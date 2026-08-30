import math
import random
import pygame
from typing import List, Tuple
from src.entities import Food, Hazard


class Agent:
    """Klasa reprezentująca agenta sterowanego przez sieć neuronową NEAT."""

    def __init__(self, net, genome, width: int = 800, height: int = 600):
        self.net = net          # Sieć neuronowa z NEAT
        self.genome = genome    # Referencja do genomu (bezpośrednie przypisywanie punktów fitness)

        # Fizyka i położenie oparte na wektorach 2D
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
        Oblicza 14 znormalizowanych wejść sensorycznych dla sieci neuronowej:
        Wszystkie wejścia są skalowane do przedziałów [0.0, 1.0] lub [-1.0, 1.0],
        aby zapobiec przedwczesnemu nasyceniu wag i funkcji aktywacji tanh.
        """
        max_dist = math.hypot(width, height)

        # 1-2. Własna pozycja znormalizowana [0.0, 1.0]
        norm_x = self.pos.x / width
        norm_y = self.pos.y / height

        # 3-4. Własna prędkość znormalizowana [-1.0, 1.0]
        norm_vx = self.vel.x / self.max_speed if self.max_speed > 0 else 0.0
        norm_vy = self.vel.y / self.max_speed if self.max_speed > 0 else 0.0

        # 5-7. Najbliższe pożywienie (Dystans [0..1] oraz znormalizowany wektor kierunku dx, dy [-1..1])
        nearest_food_dist = max_dist
        food_dir = pygame.math.Vector2(0, 0)
        for food in foods:
            to_food = food.pos - self.pos
            dist = to_food.length()
            if dist < nearest_food_dist:
                nearest_food_dist = dist
                if dist > 0:
                    food_dir = to_food / dist

        norm_food_dist = min(nearest_food_dist / max_dist, 1.0)
        norm_food_dir_x = food_dir.x
        norm_food_dir_y = food_dir.y

        # 8-10. Najbliższe zagrożenie (Dystans [0..1] oraz znormalizowany wektor kierunku dx, dy [-1..1])
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
        norm_hazard_dir_x = hazard_dir.x
        norm_hazard_dir_y = hazard_dir.y

        # 11-13. Najbliższy inny żywy agent (Dystans [0..1] oraz kierunek dx, dy [-1..1])
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
        norm_agent_dir_x = agent_dir.x
        norm_agent_dir_y = agent_dir.y

        # 14. Poziom energii/zdrowia [0.0, 1.0]
        norm_energy = max(0.0, min(self.energy / self.max_energy, 1.0))

        return (
            norm_x, norm_y,
            norm_vx, norm_vy,
            norm_food_dist, norm_food_dir_x, norm_food_dir_y,
            norm_hazard_dist, norm_hazard_dir_x, norm_hazard_dir_y,
            norm_agent_dist, norm_agent_dir_x, norm_agent_dir_y,
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
        """Pobiera wejścia, aktywuje sieć neuronową i wykonuje ruch oraz interakcje."""
        if not self.is_alive:
            return

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
        if self.pos.x < margin:
            self.pos.x = margin
            self.vel.x = 0
        elif self.pos.x > width - margin:
            self.pos.x = width - margin
            self.vel.x = 0

        if self.pos.y < margin:
            self.pos.y = margin
            self.vel.y = 0
        elif self.pos.y > height - margin:
            self.pos.y = height - margin
            self.vel.y = 0

        # 4. Spadek energii z czasem i w zależności od prędkości
        energy_cost = 0.12 + (speed / self.max_speed) * 0.08
        self.energy -= energy_cost

        # Niewielka nagroda za przetrwanie kolejnego kroku symulacji
        self.genome.fitness += 0.05

        if self.energy <= 0.0:
            self.is_alive = False
            return

        # 5. Kolizje z pożywieniem
        for food in foods:
            if (self.pos - food.pos).length_squared() <= (self.radius + food.radius) ** 2:
                food.respawn(width, height)
                self.energy = min(self.max_energy, self.energy + 45.0)
                self.genome.fitness += 15.0
                self.foods_eaten += 1

        # 6. Kolizje z zagrożeniami
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