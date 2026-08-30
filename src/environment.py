import math
import random
import pygame
from typing import List
from src.agent import Agent
from src.entities import Food, Hazard, Poison


class SimulationExit(Exception):
    """Wyjątek rzucany przy żądaniu zakończenia symulacji przez użytkownika (ESC lub zamknięcie okna)."""
    pass


class Environment:
    """Środowisko symulacji 2D zarządzające cyklem życia Pygame, encjami oraz pętlą generacji."""

    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        food_count: int = 40,
        poison_count: int = 15,
        hazard_count: int = 5
    ):
        pygame.init()
        pygame.font.init()

        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Agent Reinforcement Learning - ALife Simulation")
        self.clock = pygame.time.Clock()

        # Inicjalizacja czcionki jednokrotnie w konstruktorze (zapobiega wyciekom pamięci w pętli)
        self.font = pygame.font.SysFont("Arial", 16)

        self.generation = 0
        self.fast_mode = False  # Przełączanie prędkości symulacji (klawisz SPACE)

        # Stała pula encji w świecie (brak alokacji pamięci w każdej klatce)
        self.food_count = food_count
        self.poison_count = poison_count
        self.hazard_count = hazard_count

        self.foods: List[Food] = [
            Food(random.randint(30, self.width - 30), random.randint(30, self.height - 30))
            for _ in range(self.food_count)
        ]
        self.poisons: List[Poison] = [
            Poison(random.randint(30, self.width - 30), random.randint(30, self.height - 30))
            for _ in range(self.poison_count)
        ]
        self.hazards: List[Hazard] = [
            Hazard(random.randint(40, self.width - 40), random.randint(40, self.height - 40))
            for _ in range(self.hazard_count)
        ]

    def _reset_world_entities(self):
        """Resetuje pozycje pożywienia, trucizn i zagrożeń na start nowej generacji."""
        for food in self.foods:
            food.respawn(self.width, self.height)
        for poison in self.poisons:
            poison.respawn(self.width, self.height)
        for hazard in self.hazards:
            hazard.pos = pygame.math.Vector2(
                random.randint(40, self.width - 40),
                random.randint(40, self.height - 40)
            )

    def eval_generation(self, nets, genomes, max_frames: int = 900):
        """Ewaluuje pojedynczą generację 50 agentów NEAT."""
        self.generation += 1
        self._reset_world_entities()

        # Równomierny rozkład startowy agentów wokół centrum planszy (zapobiega losowemu faworyzowaniu)
        num_agents = len(genomes)
        center_x, center_y = self.width / 2, self.height / 2
        spawn_radius = 180.0

        agents: List[Agent] = []
        for i, (net, genome) in enumerate(zip(nets, genomes)):
            angle = (2.0 * math.pi * i) / max(1, num_agents)
            spawn_x = center_x + spawn_radius * math.cos(angle)
            spawn_y = center_y + spawn_radius * math.sin(angle)
            agents.append(Agent(net, genome, self.width, self.height, start_pos=(spawn_x, spawn_y)))

        frames_lived = 0
        running = True

        while running and frames_lived < max_frames:
            # 1. Obsługa zdarzeń Pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise SimulationExit("Użytkownik zamknął okno symulacji.")
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        # Przełączanie trybu przyspieszonego
                        self.fast_mode = not self.fast_mode
                    elif event.key == pygame.K_ESCAPE:
                        raise SimulationExit("Wciśnięto klawisz ESC.")

            # 2. Aktualizacja zagrożeń
            for hazard in self.hazards:
                hazard.update(self.width, self.height)

            # 3. Aktualizacja agentów
            alive_count = 0
            best_current_fitness = -999999.0

            for agent in agents:
                if agent.is_alive:
                    agent.think_and_act(self.foods, self.poisons, self.hazards, agents, self.width, self.height)
                    if agent.is_alive:
                        alive_count += 1
                if agent.genome.fitness > best_current_fitness:
                    best_current_fitness = agent.genome.fitness

            # Wczesne zakończenie generacji po wyginięciu całej populacji
            if alive_count == 0:
                break

            # 4. Renderowanie świata
            self.screen.fill((25, 28, 36))  # Ciemne, estetyczne tło

            # Rysowanie pożywienia (zielone okręgi)
            for food in self.foods:
                food.draw(self.screen)

            # Rysowanie trucizny (fioletowe kwadraty)
            for poison in self.poisons:
                poison.draw(self.screen)

            # Rysowanie zagrożeń ruchomych (czerwone okręgi)
            for hazard in self.hazards:
                hazard.draw(self.screen)

            # Rysowanie agentów
            for agent in agents:
                agent.draw(self.screen)

            # 5. Renderowanie HUD (informacje o generacji, populacji i wydajności)
            fps_val = int(self.clock.get_fps())
            total_foods = sum(a.foods_eaten for a in agents)
            total_poisons = sum(a.poisons_hit for a in agents)
            total_saved = sum(a.allies_saved for a in agents)

            hud_text_1 = f"Gen: {self.generation} | Zywi: {alive_count}/{len(agents)} | Klatka: {frames_lived}/{max_frames}"
            hud_text_2 = f"Max Fitness: {best_current_fitness:.1f} | FPS: {fps_val} | [SPACJA]: {'TURBO' if self.fast_mode else 'NORMAL'}"
            hud_text_3 = f"Jablka: {total_foods} | Trucizny: {total_poisons} | Uratowani: {total_saved}"

            surface_1 = self.font.render(hud_text_1, True, (240, 240, 240))
            surface_2 = self.font.render(hud_text_2, True, (180, 200, 220))
            surface_3 = self.font.render(hud_text_3, True, (160, 230, 160))
            self.screen.blit(surface_1, (10, 10))
            self.screen.blit(surface_2, (10, 30))
            self.screen.blit(surface_3, (10, 50))

            pygame.display.flip()

            # Kontrola FPS: 60 FPS w trybie normalnym, nielimitowany w trybie turbo
            if self.fast_mode:
                self.clock.tick(0)
            else:
                self.clock.tick(60)

            frames_lived += 1

        return {
            "foods_eaten": sum(a.foods_eaten for a in agents),
            "poisons_hit": sum(a.poisons_hit for a in agents),
            "allies_saved": sum(a.allies_saved for a in agents)
        }