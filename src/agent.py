import math
import random
import pygame
from typing import List, Tuple, Optional
from src.entities import Food, Hazard, Poison


TRIBE_COLORS = {
    1: (0, 245, 212),    # Plemię 1: Jaskrawy Cyjan (Neon Cyan)
    2: (255, 0, 128),    # Plemię 2: Jaskrawa Magenta (Neon Magenta)
    3: (255, 230, 0),    # Plemię 3: Jaskrawy Żółty (Electric Yellow)
    4: (240, 246, 255),  # Plemię 4: Czysty Biały (Pure White)
}


# Stałe stref brzegowych areny (Faza 8: Eliminacja Corner Exploitu)
DEADLY_ZONE_MARGIN = 20  # Śmiertelna Strefa Śmierci: -2.0 energii / klatkę
TOXIC_ZONE_MARGIN = 50   # Ostrzegawcza strefa buforowa: -0.5 energii / klatkę


class Agent:
    """Klasa reprezentująca agenta sterowanego przez sieć neuronową NEAT (Faza 8: Balans Plemion i Strefa Śmierci)."""

    def __init__(
        self,
        net,
        genome,
        width: int = 1280,
        height: int = 720,
        start_pos: Optional[Tuple[float, float]] = None,
        tribe_id: Optional[int] = None
    ):
        self.net = net          # Sieć neuronowa z NEAT
        self.genome = genome    # Referencja do genomu (bezpośrednie przypisywanie punktów fitness)

        # Faza 7: Przynależność plemienna (Tribe ID od 1 do 4)
        if tribe_id is not None:
            self.tribe_id = tribe_id
        else:
            self.tribe_id = random.randint(1, 4)

        # Fizyka i położenie oparte na wektorach 2D
        if start_pos is not None:
            self.pos = pygame.math.Vector2(start_pos[0], start_pos[1])
        else:
            self.pos = pygame.math.Vector2(
                random.uniform(80, width - 80),
                random.uniform(80, height - 80)
            )
        self.vel = pygame.math.Vector2(0.0, 0.0)
        self.max_speed = 4.0
        self.radius = 6.0

        # Witalność i metabolizm agenta (System Energii)
        self.max_energy = 150.0
        self.energy = 150.0
        self.is_alive = True
        self.frames_alive = 0
        self.is_shouting = False
        self.death_cause: Optional[str] = None  # Faza 6: "combat", "poison", "toxic_edge", "starvation", "survived"

        # Liczniki zachowań i specjalizacji ekologicznych
        self.foods_eaten = 0
        self.poisons_hit = 0
        self.allies_saved = 0
        self.attacks_made = 0
        self.defenses_made = 0
        self.herd_defenses = 0
        self.shouts_made = 0

        # Inicjalizacja fitnessu genomu (holistyczny system Faza 6)
        self.genome.fitness = 0.0

    def get_action_fitness(self) -> float:
        """
        Oblicza F_akcje: sumę ważoną pożytecznych akcji podjętych za życia:
        - Zjedzenie Jabłka (+1)
        - Odparcie Ataku / Obrona Stadna (+1)
        - Udane Polowanie (+2)
        - Akt Altruizmu (+3)
        """
        return (
            1.0 * self.foods_eaten +
            1.0 * (self.defenses_made + self.herd_defenses) +
            2.0 * self.attacks_made +
            3.0 * self.allies_saved
        )

    def finalize_fitness(self) -> float:
        """
        Oblicza i przypisuje ostateczny holistyczny fitness:
        F_total = ((frames_alive * F_akcje) / 25.0) * M_death
        Jeśli F_akcje wynosi 0, F_total wynosi 0.0.
        """
        f_actions = self.get_action_fitness()
        if f_actions <= 0.0:
            self.genome.fitness = 0.0
            return 0.0

        if self.death_cause == "survived":
            m_death = 1.2
        elif self.death_cause == "combat":
            m_death = 1.0
        elif self.death_cause == "starvation":
            m_death = 0.7
        elif self.death_cause in ("toxic_edge", "poison"):
            m_death = 0.3
        else:
            m_death = 1.0 if not self.is_alive else 1.2

        f_total = ((self.frames_alive * f_actions) / 25.0) * m_death
        self.genome.fitness = f_total
        return f_total

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
        Oblicza 22 znormalizowane wejścia sensoryczne dla Fazy 4:
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

        # 15-17. Najbliższy wrogi agent (z INNEGO plemienia) (dystans [0..1] oraz kierunek dx, dy [-1..1])
        # 18. Stan krytyczny najbliższego sojusznika (z TEGO SAMEGO plemienia) (1.0 jeśli ma < 20% energii, 0.0 wpp)
        # 19. Relatywny zwrot prędkości wroga [-1..1]
        # 20. Gęstość stada SWOJEGO plemienia wokół agenta [0..1]
        nearest_enemy_dist = max_dist
        enemy_dir = pygame.math.Vector2(0, 0)
        nearest_enemy_heading = 0.0

        nearest_ally_dist = max_dist
        nearest_ally_critical = 0.0
        allies_in_herd = 0

        for other in all_agents:
            if other is not self and other.is_alive:
                to_other = other.pos - self.pos
                dist = to_other.length()
                other_tribe = getattr(other, 'tribe_id', None)

                if other_tribe == self.tribe_id:
                    # Sojusznik z własnego plemienia:
                    # Zliczanie gęstości stada własnego plemienia w promieniu 60px
                    if dist <= 60.0:
                        allies_in_herd += 1
                    # Stan krytyczny najbliższego sojusznika
                    if dist < nearest_ally_dist:
                        nearest_ally_dist = dist
                        nearest_ally_critical = 1.0 if other.energy < 20.0 else 0.0
                else:
                    # Wróg z innego plemienia:
                    if dist < nearest_enemy_dist:
                        nearest_enemy_dist = dist
                        if dist > 0:
                            enemy_dir = to_other / dist

                        if self.vel.length_squared() > 0.01 and other.vel.length_squared() > 0.01:
                            heading_dot = self.vel.normalize().dot(other.vel.normalize())
                            nearest_enemy_heading = max(-1.0, min(1.0, heading_dot))
                        else:
                            nearest_enemy_heading = 0.0

        norm_agent_dist = min(nearest_enemy_dist / max_dist, 1.0)
        norm_agent_dx = enemy_dir.x
        norm_agent_dy = enemy_dir.y
        norm_agent_critical = nearest_ally_critical
        norm_agent_rel_heading = nearest_enemy_heading
        norm_herd_density = min(1.0, allies_in_herd / 4.0)

        # 21. Dystans do najbliższej ściany (0.0 przy samej ścianie, 1.0 w centrum)
        dist_to_wall = min(self.pos.x, width - self.pos.x, self.pos.y, height - self.pos.y)
        max_possible_wall_dist = min(width, height) / 2.0
        norm_wall_dist = max(0.0, min(dist_to_wall / max_possible_wall_dist, 1.0))

        # 22. Poziom energii własnej [0.0, 1.0]
        norm_energy = max(0.0, min(self.energy / self.max_energy, 1.0))

        # 23 - 25. Krzyk i Słuch: Zmysł akustyczny wykrywający najbliższego krzyczącego agenta
        closest_shouter = None
        min_shout_dist = float('inf')
        for other in all_agents:
            if other is not self and other.is_alive and other.is_shouting:
                d = (self.pos - other.pos).length()
                if d < min_shout_dist:
                    min_shout_dist = d
                    closest_shouter = other

        if closest_shouter is not None and min_shout_dist > 0.0001:
            norm_shout_dist = min(min_shout_dist / max_dist, 1.0)
            shout_dir = (closest_shouter.pos - self.pos).normalize()
            norm_shout_dx = shout_dir.x
            norm_shout_dy = shout_dir.y
        else:
            norm_shout_dist = 0.0
            norm_shout_dx = 0.0
            norm_shout_dy = 0.0

        return (
            norm_vx, norm_vy,
            norm_food1_dist, norm_food1_dx, norm_food1_dy,
            norm_food2_dist, norm_food2_dx, norm_food2_dy,
            norm_poison_dist, norm_poison_dx, norm_poison_dy,
            norm_hazard_dist, norm_hazard_dx, norm_hazard_dy,
            norm_agent_dist, norm_agent_dx, norm_agent_dy,
            norm_agent_critical,
            norm_agent_rel_heading,
            norm_herd_density,
            norm_wall_dist,
            norm_energy,
            norm_shout_dist,
            norm_shout_dx,
            norm_shout_dy
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
        """Pobiera wejścia, aktywuje sieć neuronową, obsługuje metabolizm, kolizje, walkę i altruizm."""
        if not self.is_alive:
            return

        self.frames_alive += 1

        # 1. Zmysły i aktywacja sieci (Faza 5: 25 wejść, 3 wyjścia)
        inputs = self._get_sensory_inputs(foods, poisons, hazards, all_agents, width, height)
        outputs = self.net.activate(inputs)
        accel = pygame.math.Vector2(outputs[0], outputs[1])

        # Wyjście #3: Aktywacja sygnału krzyku (komunikacja)
        if len(outputs) > 2:
            self.is_shouting = (outputs[2] > 0.0)
            if self.is_shouting:
                self.shouts_made += 1
        else:
            self.is_shouting = False

        # 2. Aktualizacja prędkości i pozycji
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

        # 3. Bezwzględny Metabolizm Głodu (0.20 bazowo + sprint + krzyk)
        speed_ratio = speed / self.max_speed if self.max_speed > 0 else 0.0
        energy_cost = 0.20 + (speed_ratio ** 2) * 0.08
        if self.is_shouting:
            energy_cost += 0.20
        self.energy -= energy_cost

        # Faza 8: Eliminacja "Corner Exploitu" - Strefa Śmierci (Deadly Margin = 20px) oraz strefa ostrzegawcza (50px)
        in_deadly_zone = (
            self.pos.x < DEADLY_ZONE_MARGIN or self.pos.x > width - DEADLY_ZONE_MARGIN or
            self.pos.y < DEADLY_ZONE_MARGIN or self.pos.y > height - DEADLY_ZONE_MARGIN
        )
        in_toxic_zone = (
            self.pos.x < TOXIC_ZONE_MARGIN or self.pos.x > width - TOXIC_ZONE_MARGIN or
            self.pos.y < TOXIC_ZONE_MARGIN or self.pos.y > height - TOXIC_ZONE_MARGIN
        )

        # Kara za przebywanie w strefach brzegowych (po Grace Period >= 60 klatek / 1.0s)
        # Drastyczny drenaż 2.0 energii/klatkę w Strefie Śmierci eliminuje ukrywanie się w rogach mapy.
        if self.frames_alive >= 60:
            if in_deadly_zone:
                self.energy -= 2.0
            elif in_toxic_zone:
                self.energy -= 0.5

        # Sprawdzenie śmierci z głodu / braku energii / drenażu Strefy Śmierci
        if self.energy <= 0.0:
            self.is_alive = False
            self.death_cause = "toxic_edge" if (in_toxic_zone or in_deadly_zone) else "starvation"
            self.finalize_fitness()
            return

        # 5. Kolizje z pożywieniem (odnowienie energii + licznik jabłek)
        for food in foods:
            if (self.pos - food.pos).length_squared() <= (self.radius + food.radius) ** 2:
                food.respawn(width, height)
                self.energy = min(self.max_energy, self.energy + 65.0)
                self.foods_eaten += 1

        # 6. Kolizje z trucizną (utrata dużej części energii)
        for poison in poisons:
            if (self.pos - poison.pos).length_squared() <= (self.radius + poison.radius) ** 2:
                poison.respawn(width, height)
                self.energy = max(0.0, self.energy - 35.0)
                self.poisons_hit += 1
                if self.energy <= 0.0:
                    self.is_alive = False
                    self.death_cause = "poison"
                    self.finalize_fitness()
                    return

        # 7. Kolizje z ruchomymi zagrożeniami
        for hazard in hazards:
            if (self.pos - hazard.pos).length_squared() <= (self.radius + hazard.radius) ** 2:
                self.energy -= 20.0
                if self.energy <= 0.0:
                    self.is_alive = False
                    self.death_cause = "toxic_edge" if (in_toxic_zone or in_deadly_zone) else "poison"
                    self.finalize_fitness()
                    return

        # 8. Interakcje Między Agentami: Altruizm, Obrona Stadna, Drapieżnictwo i Zderzenia Czołowe
        # Dostępne wyłącznie po zakończeniu Grace Period (>= 60 klatek / 1.0s) dla obu agentów
        if self.frames_alive >= 60:
            for other in all_agents:
                if other is not self and other.is_alive and other.frames_alive >= 60:
                    dist_sq = (self.pos - other.pos).length_squared()
                    if dist_sq <= (self.radius + other.radius) ** 2:
                        other_tribe = getattr(other, 'tribe_id', None)
                        is_same_tribe = (other_tribe == self.tribe_id)

                        if is_same_tribe:
                            # A. ALTRUIZM: Przekazywanie energii głodującemu sojusznikowi TYLKO z tego samego plemienia
                            if self.energy > 50.0 and other.energy < 20.0:
                                transfer_amount = 20.0
                                self.energy -= transfer_amount
                                other.energy = min(other.max_energy, other.energy + transfer_amount)
                                self.allies_saved += 1
                                break
                            # Kanibalizm i ataki wewnątrz tego samego plemienia są zablokowane
                        else:
                            # B. WALKA I DRAPIEŻNICTWO: Dozwolone TYLKO przeciwko agentom z INNEGO plemienia
                            v_self_len = self.vel.length()
                            v_other_len = other.vel.length()
                            if v_self_len > 0.1 and v_other_len > 0.1:
                                dot_prod = (self.vel / v_self_len).dot(other.vel / v_other_len)
                            else:
                                dot_prod = 0.0

                            if dot_prod <= -0.2:
                                # Zderzenie Czołowe z wrogiem (Obrona)
                                self.vel = -self.vel * 0.5
                                other.vel = -other.vel * 0.5
                                if self.energy >= other.energy:
                                    self.defenses_made += 1
                                self.energy = max(0.0, self.energy - 3.0)
                                if self.energy <= 0.0:
                                    self.is_alive = False
                                    self.death_cause = "combat"
                                    self.finalize_fitness()
                                    return
                            elif dot_prod > 0.0 and v_self_len >= 0.5:
                                # Próba ataku od tyłu / flanki na wroga
                                # Obrona Stadna: Sojusznicy ofiary z TEGO SAMEGO plemienia co ofiara (w promieniu 45px)
                                herd_radius_sq = 45.0 ** 2
                                allies_in_herd = [
                                    a for a in all_agents
                                    if a is not self and a is not other and a.is_alive and a.frames_alive >= 60 and
                                    getattr(a, 'tribe_id', None) == other_tribe and
                                    (a.pos - other.pos).length_squared() <= herd_radius_sq
                                ]

                                if len(allies_in_herd) >= 1:
                                    # OBRONA STADNA WROGA AKTYWOWANA!
                                    predator_damage = 15.0
                                    self.energy = max(0.0, self.energy - predator_damage)
                                    other.herd_defenses += 1
                                    for ally in allies_in_herd:
                                        ally.herd_defenses += 1
                                    if self.energy <= 0.0:
                                        self.is_alive = False
                                        self.death_cause = "combat"
                                        self.finalize_fitness()
                                        return
                                    break
                                else:
                                    # SAMOTNY WRÓG: Udany atak drapieżnika
                                    stolen_energy = min(25.0, other.energy)
                                    self.energy = min(self.max_energy, self.energy + stolen_energy)
                                    other.energy = max(0.0, other.energy - 25.0)
                                    self.attacks_made += 1
                                    if other.energy <= 0.0:
                                        other.is_alive = False
                                        other.death_cause = "combat"
                                        other.finalize_fitness()
                                    break

        # 9. Aktualizacja bieżącego fitnessu dla telemetrii UI (podczas trwania życia)
        f_actions = self.get_action_fitness()
        if f_actions > 0.0:
            self.genome.fitness = (self.frames_alive * f_actions) / 25.0
        else:
            self.genome.fitness = 0.0

    def draw(self, screen: pygame.Surface):
        """Rysuje agenta w barwach plemiennych z obwódką zależną od witalności, roli i stanu Grace Period."""
        if not self.is_alive:
            return

        center = (int(self.pos.x), int(self.pos.y))

        # Faza 7: Unikalna kolorystyka dla każdego plemienia (1: Cyjan, 2: Magenta, 3: Żółty, 4: Biały)
        agent_color = TRIBE_COLORS.get(self.tribe_id, (52, 152, 219))
        pygame.draw.circle(screen, agent_color, center, int(self.radius))

        # Wskaźnik stanu krytycznego (głód < 20.0) - pomarańczowy pierścień ostrzegawczy
        if self.energy < 20.0:
            pygame.draw.circle(screen, (230, 126, 34), center, int(self.radius + 1), 1)

        # W trybie Grace Period (< 60 klatek / 1s) rysujemy błękitno-białą poświatę nietykalności
        if self.frames_alive < 60:
            pygame.draw.circle(screen, (220, 240, 255), center, int(self.radius + 3), 1)
        elif self.attacks_made > 0:
            # Karmazynowa obwódka dla drapieżników z udanymi atakami
            pygame.draw.circle(screen, (231, 76, 60), center, int(self.radius + 2), 1)

        # Komunikacja (Faza 5): Fala dźwiękowa / krzyk (jaskrawy turkus)
        if self.is_shouting:
            pygame.draw.circle(screen, (0, 245, 212), center, int(self.radius + 6), 1)

        # Rysowanie wskaźnika kierunku prędkości
        if self.vel.length_squared() > 0.01:
            dir_end = self.pos + self.vel.normalize() * (self.radius + 3)
            pygame.draw.line(screen, (241, 196, 15), center, (int(dir_end.x), int(dir_end.y)), 2)