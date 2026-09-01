import math
import random
import pygame
from typing import List, Dict, Any, Optional, Tuple
from src.agent import Agent
from src.entities import Food, Hazard, Poison


class SimulationExit(Exception):
    """Wyjątek rzucany przy żądaniu zakończenia symulacji przez użytkownika (ESC lub zamknięcie okna)."""
    pass


class Environment:
    """Środowisko symulacji 2D zarządzające cyklem życia Pygame, encjami, panelem UI oraz wizualizatorem sieci NEAT."""

    def __init__(
        self,
        width: int = 1600,
        height: int = 720,
        arena_width: int = 1280,
        food_count: int = 50,
        poison_count: int = 20,
        hazard_count: int = 6
    ):
        pygame.init()
        pygame.font.init()

        self.width = width
        self.height = height
        self.arena_width = arena_width
        self.sidebar_width = self.width - self.arena_width

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Agent Reinforcement Learning - ALife Research Dashboard")
        self.clock = pygame.time.Clock()

        # Inicjalizacja czcionek o stałej szerokości (monospace) jednokrotnie w konstruktorze
        self.title_font = pygame.font.SysFont("Consolas, Courier, monospace", 14, bold=True)
        self.font = pygame.font.SysFont("Consolas, Courier, monospace", 13)
        self.small_font = pygame.font.SysFont("Consolas, Courier, monospace", 11)

        self.generation = 0
        self.fast_mode = False  # Przełączanie prędkości symulacji (klawisz SPACE)

        # Referencje do 4 najlepszych genomów z poprzedniej generacji (Wizualizator Top 4)
        self.top_genomes: List[Any] = []

        # Stała pula encji w świecie w granicach areny (1280 x 720)
        self.food_count = food_count
        self.poison_count = poison_count
        self.hazard_count = hazard_count

        self.foods: List[Food] = [
            Food(random.randint(60, self.arena_width - 60), random.randint(60, self.height - 60))
            for _ in range(self.food_count)
        ]
        self.poisons: List[Poison] = [
            Poison(random.randint(60, self.arena_width - 60), random.randint(60, self.height - 60))
            for _ in range(self.poison_count)
        ]
        self.hazards: List[Hazard] = [
            Hazard(random.randint(60, self.arena_width - 60), random.randint(60, self.height - 60))
            for _ in range(self.hazard_count)
        ]

    def _reset_world_entities(self):
        """Resetuje pozycje pożywienia, trucizn i zagrożeń na start nowej generacji w granicach areny."""
        for food in self.foods:
            food.respawn(self.arena_width, self.height)
        for poison in self.poisons:
            poison.respawn(self.arena_width, self.height)
        for hazard in self.hazards:
            hazard.pos = pygame.math.Vector2(
                random.randint(60, self.arena_width - 60),
                random.randint(60, self.height - 60)
            )

    def _draw_arena_grid(self):
        """Rysuje ciemne tło badawcze (#0b0c10) oraz subtelną siatkę na arenie symulacji."""
        # Wypełnienie areny
        pygame.draw.rect(self.screen, (11, 12, 16), (0, 0, self.arena_width, self.height))

        # Subtelna siatka co 64 piksele
        grid_color = (18, 22, 28)
        for x in range(0, self.arena_width, 64):
            pygame.draw.line(self.screen, grid_color, (x, 0), (x, self.height), 1)
        for y in range(0, self.height, 64):
            pygame.draw.line(self.screen, grid_color, (0, y), (self.arena_width, y), 1)

        # Subtelny obrys bezpiecznej strefy (50px od toksycznych krawędzi)
        pygame.draw.rect(self.screen, (30, 38, 48), (50, 50, self.arena_width - 100, self.height - 100), 1)

    def _draw_network_graph(self, genome, slot_rect: pygame.Rect, title: str):
        """Renderuje uproszczony graf sieci neuronowej genomu w dedykowanym slocie panelu bocznego."""
        # Tło i obramowanie slotu
        pygame.draw.rect(self.screen, (15, 18, 24), slot_rect)
        pygame.draw.rect(self.screen, (40, 50, 65), slot_rect, 1)

        # Nagłówek slotu
        title_surf = self.small_font.render(title, True, (0, 245, 212))
        self.screen.blit(title_surf, (slot_rect.x + 8, slot_rect.y + 4))

        if not hasattr(genome, 'connections') or not genome.connections:
            placeholder = self.small_font.render("Brak polaczen", True, (120, 130, 140))
            self.screen.blit(placeholder, (slot_rect.x + 10, slot_rect.y + 35))
            return

        # Ekstrakcja węzłów
        # Wejścia: klucze ujemne (-1 do -25), Wyjścia: klucze (0, 1, 2), Ukryte: klucze > 2
        active_inputs = set()
        active_outputs = {0, 1, 2}
        hidden_nodes = set()

        for (in_k, out_k), conn in genome.connections.items():
            if conn.enabled:
                if in_k < 0:
                    active_inputs.add(in_k)
                else:
                    hidden_nodes.add(in_k)
                if out_k in active_outputs:
                    pass
                elif out_k >= 0:
                    hidden_nodes.add(out_k)

        # Pozycjonowanie węzłów w przestrzeni slotu
        node_positions: Dict[int, Tuple[int, int]] = {}
        in_x = slot_rect.x + 25
        out_x = slot_rect.x + slot_rect.width - 25
        mid_x = slot_rect.x + slot_rect.width // 2

        # Wejścia (posortowane, lewa kolumna)
        in_list = sorted(list(active_inputs)) if active_inputs else [-1, -2, -3]
        in_spacing = (slot_rect.height - 30) / max(1, len(in_list))
        for idx, k in enumerate(in_list):
            node_positions[k] = (in_x, int(slot_rect.y + 22 + idx * in_spacing + in_spacing / 2))

        # Wyjścia (Ax, Ay, Shout, prawa kolumna)
        out_list = [0, 1, 2]
        out_spacing = (slot_rect.height - 30) / 3
        for idx, k in enumerate(out_list):
            node_positions[k] = (out_x, int(slot_rect.y + 22 + idx * out_spacing + out_spacing / 2))

        # Węzły ukryte (środkowa kolumna)
        hid_list = sorted(list(hidden_nodes))
        if hid_list:
            hid_spacing = (slot_rect.height - 30) / len(hid_list)
            for idx, k in enumerate(hid_list):
                node_positions[k] = (mid_x, int(slot_rect.y + 22 + idx * hid_spacing + hid_spacing / 2))

        # Rysowanie połączeń (synaps)
        for (in_k, out_k), conn in genome.connections.items():
            if conn.enabled and in_k in node_positions and out_k in node_positions:
                start_pos = node_positions[in_k]
                end_pos = node_positions[out_k]
                w = conn.weight
                # Kolor: zielony dla wag dodatnich, czerwony dla ujemnych
                line_color = (46, 204, 113) if w >= 0 else (231, 76, 60)
                line_width = max(1, min(3, int(abs(w) * 1.2)))
                pygame.draw.line(self.screen, line_color, start_pos, end_pos, line_width)

        # Rysowanie węzłów (kółek)
        for k, pos in node_positions.items():
            if k < 0:
                # Wejścia: błękitne
                pygame.draw.circle(self.screen, (52, 152, 219), pos, 3)
            elif k in active_outputs:
                # Wyjścia: turkusowe
                pygame.draw.circle(self.screen, (0, 245, 212), pos, 4)
            else:
                # Ukryte: żółte
                pygame.draw.circle(self.screen, (241, 196, 15), pos, 3)

    def _draw_sidebar(self, agents: List[Agent], frames_lived: int, max_frames: int, alive_count: int, best_current_fitness: float):
        """Renderuje panel boczny z metrykami badawczymi oraz wizualizatorem sieci Top 4."""
        sidebar_rect = pygame.Rect(self.arena_width, 0, self.sidebar_width, self.height)
        # Tło panelu bocznego (#161b22)
        pygame.draw.rect(self.screen, (22, 27, 34), sidebar_rect)
        # Linia oddzielająca areny od panelu (#2d3748)
        pygame.draw.line(self.screen, (45, 55, 72), (self.arena_width, 0), (self.arena_width, self.height), 2)

        # 1. Nagłówek i Statystyki Główne
        title_surf = self.title_font.render("=== NEURAL RESEARCH DASHBOARD ===", True, (240, 246, 252))
        self.screen.blit(title_surf, (self.arena_width + 12, 12))

        fps_val = int(self.clock.get_fps())
        total_foods = sum(a.foods_eaten for a in agents)
        total_poisons = sum(a.poisons_hit for a in agents)
        total_saved = sum(a.allies_saved for a in agents)
        total_attacks = sum(a.attacks_made for a in agents)
        total_defenses = sum(a.defenses_made for a in agents)
        total_herd = sum(a.herd_defenses for a in agents)
        active_shouts = sum(1 for a in agents if a.is_alive and a.is_shouting)

        stats_lines = [
            f"GENERACJA:   {self.generation:<4d} | KLATKA: {frames_lived:3d}/{max_frames}",
            f"POPULACJA:   {alive_count:2d}/{len(agents):<2d} | TRYB: {'TURBO' if self.fast_mode else '60 FPS'}",
            f"MAX FITNESS: {best_current_fitness:<6.1f} | FPS:  {fps_val:2d}",
            "---------------------------------------",
            f"• Zebrane Jablka:       {total_foods:4d}",
            f"• Trucizny:             {total_poisons:4d}",
            f"• Altruizm (Ratunek):   {total_saved:4d}",
            f"• Ataki Drapieznikow:   {total_attacks:4d}",
            f"• Obrony Czolowe:       {total_defenses:4d}",
            f"• Obrony Stadne:        {total_herd:4d}",
            f"• Aktywne Krzyki:       {active_shouts:4d}",
            "---------------------------------------"
        ]

        y_offset = 36
        for line in stats_lines:
            color = (201, 209, 217)
            if "MAX FITNESS" in line:
                color = (241, 196, 15)
            elif "POPULACJA" in line:
                color = (88, 166, 255)
            elif "Altruizm" in line or "Jablka" in line:
                color = (46, 204, 113)
            elif "Ataki" in line or "Trucizny" in line:
                color = (231, 76, 60)
            elif "Krzyki" in line:
                color = (0, 245, 212)

            line_surf = self.font.render(line, True, color)
            self.screen.blit(line_surf, (self.arena_width + 14, y_offset))
            y_offset += 18

        # 2. Wizualizator Sieci Top 4 (4 sloty ułożone pionowo)
        section_surf = self.title_font.render("--- TOP 4 NEURAL BRAINS ---", True, (139, 148, 158))
        self.screen.blit(section_surf, (self.arena_width + 12, y_offset + 4))

        slot_y = y_offset + 26
        slot_w = self.sidebar_width - 24
        slot_h = 88

        for i in range(4):
            rect = pygame.Rect(self.arena_width + 12, slot_y, slot_w, slot_h)
            if i < len(self.top_genomes):
                genome = self.top_genomes[i]
                fit_val = getattr(genome, 'fitness', 0.0) or 0.0
                title = f"[#{i+1} ELITA] Fitness: {fit_val:.1f}"
                self._draw_network_graph(genome, rect, title)
            else:
                pygame.draw.rect(self.screen, (15, 18, 24), rect)
                pygame.draw.rect(self.screen, (35, 45, 55), rect, 1)
                ph = self.small_font.render(f"[#{i+1} PUSTY] Oczekiwanie na epoke...", True, (80, 90, 100))
                self.screen.blit(ph, (rect.x + 10, rect.y + 35))

            slot_y += slot_h + 8

    def eval_generation(self, nets, genomes, max_frames: int = 900) -> Dict[str, Any]:
        """Ewaluuje pojedynczą generację 50 agentów NEAT."""
        self.generation += 1
        self._reset_world_entities()

        # Równomierny rozkład startowy agentów wokół centrum areny (1280 x 720)
        num_agents = len(genomes)
        center_x, center_y = self.arena_width / 2, self.height / 2
        spawn_radius = 240.0

        agents: List[Agent] = []
        for i, (net, genome) in enumerate(zip(nets, genomes)):
            angle = (2.0 * math.pi * i) / max(1, num_agents)
            spawn_x = center_x + spawn_radius * math.cos(angle)
            spawn_y = center_y + spawn_radius * math.sin(angle)
            agents.append(Agent(net, genome, self.arena_width, self.height, start_pos=(spawn_x, spawn_y)))

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

            # 2. Aktualizacja zagrożeń w granicach areny
            for hazard in self.hazards:
                hazard.update(self.arena_width, self.height)

            # 3. Aktualizacja agentów
            alive_count = 0
            best_current_fitness = -999999.0

            for agent in agents:
                if agent.is_alive:
                    agent.think_and_act(self.foods, self.poisons, self.hazards, agents, self.arena_width, self.height)
                    if agent.is_alive:
                        alive_count += 1
                if agent.genome.fitness > best_current_fitness:
                    best_current_fitness = agent.genome.fitness

            # Wczesne zakończenie generacji po wyginięciu całej populacji
            if alive_count == 0:
                break

            # 4. Renderowanie Areny i Siatki
            self._draw_arena_grid()

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

            # 5. Renderowanie Panelu Bocznego i Wizualizatora Top 4
            self._draw_sidebar(agents, frames_lived, max_frames, alive_count, best_current_fitness)

            pygame.display.flip()

            # Kontrola FPS: 60 FPS w trybie normalnym, nielimitowany w trybie turbo
            if self.fast_mode:
                self.clock.tick(0)
            else:
                self.clock.tick(60)

            frames_lived += 1

        # Finalizacja holistycznego fitnessu dla agentów, którzy przetrwali całą epokę (M_death = 1.2)
        for agent in agents:
            if agent.is_alive:
                agent.death_cause = "survived"
                agent.finalize_fitness()

        # Zapisanie 4 najlepszych genomów na koniec generacji dla wizualizatora
        self.top_genomes = sorted(
            genomes,
            key=lambda g: getattr(g, 'fitness', -999999.0) if getattr(g, 'fitness', None) is not None else -999999.0,
            reverse=True
        )[:4]

        return {
            "foods_eaten": sum(a.foods_eaten for a in agents),
            "poisons_hit": sum(a.poisons_hit for a in agents),
            "allies_saved": sum(a.allies_saved for a in agents),
            "attacks_made": sum(a.attacks_made for a in agents),
            "defenses_made": sum(a.defenses_made for a in agents),
            "herd_defenses": sum(a.herd_defenses for a in agents),
            "shouts_made": sum(a.shouts_made for a in agents)
        }