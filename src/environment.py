import random
import pygame
from typing import List
from src.agent import Agent
from src.entities import Food, Hazard


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
        self.hazard_count = hazard_count
        self.foods: List[Food] = [
            Food(random.randint(30, self.width - 30), random.randint(30, self.height - 30))
            for _ in range(self.food_count)
        ]
        self.hazards: List[Hazard] = [
            Hazard(random.randint(40, self.width - 40), random.randint(40, self.height - 40))
            for _ in range(self.hazard_count)
        ]

    def _reset_world_entities(self):
        """Resetuje pozycje pożywienia i zagrożeń na start nowej generacji."""
        for food in self.foods:
            food.respawn(self.width, self.height)
        for hazard in self.hazards:
            hazard.pos = pygame.math.Vector2(
                random.randint(40, self.width - 40),
                random.randint(40, self.height - 40)
            )

    def eval_generation(self, nets, genomes, max_frames: int = 600):
        """Ewaluuje pojedynczą generację 50 agentów NEAT."""
        self.generation += 1
        self._reset_world_entities()

        # Inicjalizacja agentów dla bieżącej populacji
        agents: List[Agent] = [
            Agent(net, genome, self.width, self.height)
            for net, genome in zip(nets, genomes)
        ]

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
                    agent.think_and_act(self.foods, self.hazards, agents, self.width, self.height)
                    if agent.is_alive:
                        alive_count += 1
                if agent.genome.fitness > best_current_fitness:
                    best_current_fitness = agent.genome.fitness

            # Wczesne zakończenie generacji po wyginięciu całej populacji
            if alive_count == 0:
                break

            # 4. Renderowanie świata
            self.screen.fill((25, 28, 36))  # Ciemne, estetyczne tło

            # Rysowanie pożywienia
            for food in self.foods:
                food.draw(self.screen)

            # Rysowanie zagrożeń
            for hazard in self.hazards:
                hazard.draw(self.screen)

            # Rysowanie agentów
            for agent in agents:
                agent.draw(self.screen)

            # 5. Renderowanie HUD (informacje o generacji, populacji i wydajności)
            fps_val = int(self.clock.get_fps())
            hud_text_1 = f"Gen: {self.generation} | Żywi: {alive_count}/{len(agents)} | Klatka: {frames_lived}/{max_frames}"
            hud_text_2 = f"Max Fitness: {best_current_fitness:.1f} | FPS: {fps_val} | [SPACJA]: {'TURBO' if self.fast_mode else 'NORMAL'}"

            surface_1 = self.font.render(hud_text_1, True, (240, 240, 240))
            surface_2 = self.font.render(hud_text_2, True, (180, 200, 220))
            self.screen.blit(surface_1, (10, 10))
            self.screen.blit(surface_2, (10, 30))

            pygame.display.flip()

            # Kontrola FPS: 60 FPS w trybie normalnym, nielimitowany w trybie turbo
            if self.fast_mode:
                self.clock.tick(0)
            else:
                self.clock.tick(60)

            frames_lived += 1