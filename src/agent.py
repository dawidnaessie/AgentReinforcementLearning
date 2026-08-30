import math
import random
import pygame
from typing import List, Tuple, Optional
from src.entities import Food, Hazard, Poison


class Agent:
    """Klasa reprezentująca agenta sterowanego przez sieć neuronową NEAT (Faza 2)."""

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

        # Witalność i metabolizm agenta (System Energii)
        self.max_energy = 100.0
        self.energy = 100.0
        self.is_alive = True
        self.foods_eaten = 0
        self.poisons_hit = 0
        self.allies_saved = 0

        # Inicjalizacja fitnessu genomu
        self.genome.fitness = 0.0

    def _get_sensory_inputs(
        self,
        foods: List[Food],
        poisons: List[Poison],
        hazards: List[Hazard],
        all_agents: List["Agent"],
        width: int,
        height: int
    ) -> Tuple[float, ...]:
        """
        Oblicza 20 znormalizowanych wejść sensorycznych dla Fazy 2:
        Wszystkie wejścia są skalowane do przedziałów [0.0, 1.0] lub [-1.0, 1.0].
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

        food_distances.sort(key=lambda item: item[0])

        if len(food_distances) > 0:
            dist1, dir1 = food_distances[0]
            norm_food1_dist = min(dist1 / max_dist, 1.0)
            norm_food1_dx = dir1.x
            norm_food1_dy = dir1.y
        else:
            norm_food1_dist = 1.0
            norm_food1_dx, norm_food1_dy = 0.0, 0.0

        if len(food_distances) > 1:
            dist2, dir2 = food_distances[1]
            norm_food2_dist = min(dist2 / max_dist, 1.0)
            norm_food2_dx = dir2.x
            norm_food2_dy = dir2.y
        else:
            norm_food2_dist = norm_food1_dist
            norm_food2_dx, norm_food2_dy = norm_food1_dx, norm_food1_dy

        # 9-11. Najbliższa trucizna (dystans [0..1] oraz kierunek dx, dy [-1..1])
        nearest_poison_dist = max_dist
        poison_dir = pygame.math.Vector2(0, 0)
        for poison in poisons:
            to_poison = poison.pos - self.pos
            dist = to_poison.length()
            if dist < nearest_poison_dist:
                nearest_poison_dist = dist
                if dist > 0:
                    poison_dir = to_poison / dist

        norm_poison_dist = min(nearest_poison_dist / max_dist, 1.0)
        norm_poison_dx = poison_dir.x
        norm_poison_dy = poison_dir.y

        # 12-14. Najbliższe zagrożenie ruchome (dystans [0..1] oraz kierunek dx, dy [-1..1])
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

        # 15-17. Najbliższy inny żywy agent (dystans [0..1] oraz kierunek dx, dy [-1..1])
        # 18. Stan krytyczny najbliższego sojusznika (1.0 jeśli ma < 20% energii, 0.0 wpp)
        nearest_agent_dist = max_dist
        agent_dir = pygame.math.Vector2(0, 0)
        nearest_ally_critical = 0.0

        for other in all_agents:
            if other is not self and other.is_alive:
                to_other = other.pos - self.pos
                dist = to_other.length()
                if dist < nearest_agent_dist:
                    nearest_agent_dist = dist
                    if dist > 0:
                        agent_dir = to_other / dist
                    nearest_ally_critical = 1.0 if other.energy < 20.0 else 0.0

        norm_agent_dist = min(nearest_agent_dist / max_dist, 1.0)
        norm_agent_dx = agent_dir.x
        norm_agent_dy = agent_dir.y
        norm_agent_critical = nearest_ally_critical

        # 19. Dystans do najbliższej ściany (0.0 przy samej ścianie, 1.0 w centrum)
        dist_to_wall = min(self.pos.x, width - self.pos.x, self.pos.y, height - self.pos.y)
        max_possible_wall_dist = min(width, height) / 2.0
        norm_wall_dist = max(0.0, min(dist_to_wall / max_possible_wall_dist, 1.0))

        # 20. Poziom energii własnej [0.0, 1.0]
        norm_energy = max(0.0, min(self.energy / self.max_energy, 1.0))

        return (
            norm_vx, norm_vy,
            norm_food1_dist, norm_food1_dx, norm_food1_dy,
            norm_food2_dist, norm_food2_dx, norm_food2_dy,
            norm_poison_dist, norm_poison_dx, norm_poison_dy,
            norm_hazard_dist, norm_hazard_dx, norm_hazard_dy,
            norm_agent_dist, norm_agent_dx, norm_agent_dy,
            norm_agent_critical,
            norm_wall_dist,
            norm_energy
        )

    def think_and_act(
        self,
        foods: List[Food],
        poisons: List[Poison],
        hazards: List[Hazard],
        all_agents: List["Agent"],
        width: int,
        height: int
    ):
        """Pobiera wejścia, aktywuje sieć neuronową, obsługuje metabolizm, kolizje i altruizm."""
        if not self.is_alive:
            return

        # Obliczenie odległości do najbliższego pożywienia przed ruchem (reward shaping)
        prev_min_food_dist = float('inf')
        for food in foods:
            d = (self.pos - food.pos).length()
            if d < prev_min_food_dist:
                prev_min_food_dist = d

        # 1. Zmysły i aktywacja sieci
        inputs = self._get_sensory_inputs(foods, poisons, hazards, all_agents, width, height)
        outputs = self.net.activate(inputs)
        accel = pygame.math.Vector2(outputs[0], outputs[1])

        # 2. Aktualizacja prędkości i pozycji
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

        if hit_wall:
            self.genome.fitness -= 0.05

        # 3. Metabolizm: stały koszt życia + koszt ruchu
        energy_cost = 0.10 + (speed / self.max_speed) * 0.06
        self.energy -= energy_cost

        # Premia za przetrwanie kroku
        self.genome.fitness += 0.03

        # 4. Reward Shaping za zbliżanie się do jedzenia
        if foods and prev_min_food_dist != float('inf'):
            new_min_food_dist = float('inf')
            for food in foods:
                d = (self.pos - food.pos).length()
                if d < new_min_food_dist:
                    new_min_food_dist = d
            dist_delta = prev_min_food_dist - new_min_food_dist
            self.genome.fitness += dist_delta * 0.08

        # Sprawdzenie śmierci z głodu / braku energii
        if self.energy <= 0.0:
            self.is_alive = False
            return

        # 5. Kolizje z pożywieniem (odnowienie energii + nagroda fitness)
        for food in foods:
            if (self.pos - food.pos).length_squared() <= (self.radius + food.radius) ** 2:
                food.respawn(width, height)
                self.energy = min(self.max_energy, self.energy + 45.0)
                self.genome.fitness += 15.0
                self.foods_eaten += 1

        # 6. Kolizje z trucizną (utrata dużej części energii + kara fitness)
        for poison in poisons:
            if (self.pos - poison.pos).length_squared() <= (self.radius + poison.radius) ** 2:
                poison.respawn(width, height)
                self.energy = max(0.0, self.energy - 35.0)
                self.genome.fitness -= 10.0
                self.poisons_hit += 1
                if self.energy <= 0.0:
                    self.is_alive = False
                    return

        # 7. Kolizje z ruchomymi zagrożeniami
        for hazard in hazards:
            if (self.pos - hazard.pos).length_squared() <= (self.radius + hazard.radius) ** 2:
                self.energy -= 20.0
                self.genome.fitness -= 5.0
                if self.energy <= 0.0:
                    self.is_alive = False
                    return

        # 8. Mechanika Altruizmu: transfer energii od agenta silnego (>50%) do agenta w stanie krytycznym (<20%)
        if self.energy > 50.0:
            for other in all_agents:
                if other is not self and other.is_alive and other.energy < 20.0:
                    if (self.pos - other.pos).length_squared() <= (self.radius + other.radius) ** 2:
                        transfer_amount = 20.0
                        self.energy -= transfer_amount
                        other.energy = min(other.max_energy, other.energy + transfer_amount)
                        # Potężna premia za altruizm i pomoc sojusznikowi
                        self.genome.fitness += 50.0
                        self.allies_saved += 1
                        break

    def draw(self, screen: pygame.Surface):
        """Rysuje agenta z kolorem uzależnionym od poziomu energii (wizualizacja stanu witalnego)."""
        if not self.is_alive:
            return

        center = (int(self.pos.x), int(self.pos.y))

        # Kolorystyka: niebieski przy pełnym zdrowiu, pomarańczowo-czerwony przy skrajnym wyczerpaniu
        if self.energy < 20.0:
            agent_color = (230, 126, 34)  # Stan krytyczny - pomarańcz
        else:
            agent_color = (52, 152, 219)  # Zdrowy - niebieski

        pygame.draw.circle(screen, agent_color, center, int(self.radius))

        # Rysowanie wskaźnika kierunku prędkości
        if self.vel.length_squared() > 0.01:
            dir_end = self.pos + self.vel.normalize() * (self.radius + 3)
            pygame.draw.line(screen, (241, 196, 15), center, (int(dir_end.x), int(dir_end.y)), 2)