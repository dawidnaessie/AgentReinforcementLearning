import copy
import math
import random
import pygame
from typing import List, Dict, Any, Optional, Tuple
from src.agent import Agent, DEADLY_ZONE_MARGIN
from src.entities import Food, Hazard, Poison


# Szczegółowe metadane 25 znormalizowanych wejść sensorycznych zgodnie z README.md
SENSORY_DETAILS: Dict[int, Dict[str, str]] = {
    0: {
        "name": "Velocity (Vel X)",
        "short": "Vel X",
        "desc": "Predkosc pozioma agenta [-1.0..1.0]",
        "range": "[-1.0, 1.0]",
        "role": "Orientacja pedu i bezwladnosci w osi X"
    },
    1: {
        "name": "Velocity (Vel Y)",
        "short": "Vel Y",
        "desc": "Predkosc pionowa agenta [-1.0..1.0]",
        "range": "[-1.0, 1.0]",
        "role": "Orientacja pedu i bezwladnosci w osi Y"
    },
    2: {
        "name": "Nearest Food #1 Dist",
        "short": "Food #1 Dist",
        "desc": "Dystans euklidesowy do najblizszego jablka [0..1]",
        "range": "[0.0, 1.0]",
        "role": "Glowny cel zerowania (+15 fit, +65 energii)"
    },
    3: {
        "name": "Nearest Food #1 Dir X",
        "short": "Food #1 Dir X",
        "desc": "Wektor kierunku X do najblizszego jablka [-1..1]",
        "range": "[-1.0, 1.0]",
        "role": "Nawigacja pozioma w strone jablka #1"
    },
    4: {
        "name": "Nearest Food #1 Dir Y",
        "short": "Food #1 Dir Y",
        "desc": "Wektor kierunku Y do najblizszego jablka [-1..1]",
        "range": "[-1.0, 1.0]",
        "role": "Nawigacja pionowa w strone jablka #1"
    },
    5: {
        "name": "Secondary Food #2 Dist",
        "short": "Food #2 Dist",
        "desc": "Dystans do 2. najblizszego jablka [0..1]",
        "range": "[0.0, 1.0]",
        "role": "Planowanie alternatywnej trasy zerowania"
    },
    6: {
        "name": "Secondary Food #2 Dir X",
        "short": "Food #2 Dir X",
        "desc": "Wektor kierunku X do 2. najblizszego jablka [-1..1]",
        "range": "[-1.0, 1.0]",
        "role": "Nawigacja pozioma do jablka #2"
    },
    7: {
        "name": "Secondary Food #2 Dir Y",
        "short": "Food #2 Dir Y",
        "desc": "Wektor kierunku Y do 2. najblizszego jablka [-1..1]",
        "range": "[-1.0, 1.0]",
        "role": "Nawigacja pionowa do jablka #2"
    },
    8: {
        "name": "Nearest Poison Dist",
        "short": "Poison Dist",
        "desc": "Dystans do najblizszej fioletowej trucizny [0..1]",
        "range": "[0.0, 1.0]",
        "role": "Unikanie toksyn (-10 fit, -35 energii)"
    },
    9: {
        "name": "Nearest Poison Dir X",
        "short": "Poison Dir X",
        "desc": "Wektor kierunku X do najblizszej trucizny [-1..1]",
        "range": "[-1.0, 1.0]",
        "role": "Repulsja pozioma od fioletowych toksyn"
    },
    10: {
        "name": "Nearest Poison Dir Y",
        "short": "Poison Dir Y",
        "desc": "Wektor kierunku Y do najblizszej trucizny [-1..1]",
        "range": "[-1.0, 1.0]",
        "role": "Repulsja pionowa od fioletowych toksyn"
    },
    11: {
        "name": "Nearest Hazard Dist",
        "short": "Hazard Dist",
        "desc": "Dystans do wedrujacego drapiezcy [0..1]",
        "range": "[0.0, 1.0]",
        "role": "Ucieczka przed zagrozeniem (-5 fit, -20 energii)"
    },
    12: {
        "name": "Nearest Hazard Dir X",
        "short": "Hazard Dir X",
        "desc": "Wektor kierunku X do ruchomego zagrozenia [-1..1]",
        "range": "[-1.0, 1.0]",
        "role": "Unik poziomy przed ruchomym zagrozeniem"
    },
    13: {
        "name": "Nearest Hazard Dir Y",
        "short": "Hazard Dir Y",
        "desc": "Wektor kierunku Y do ruchomego zagrozenia [-1..1]",
        "range": "[-1.0, 1.0]",
        "role": "Unik pionowy przed ruchomym zagrozeniem"
    },
    14: {
        "name": "Nearest Enemy Dist",
        "short": "Enemy Dist",
        "desc": "Dystans do najblizszego wroga z innego plemienia [0..1]",
        "range": "[0.0, 1.0]",
        "role": "Percepcja miedzyplemienna: namierzanie wrogow i ofiar"
    },
    15: {
        "name": "Nearest Enemy Dir X",
        "short": "Enemy Dir X",
        "desc": "Wektor kierunku X do najblizszego wroga [-1..1]",
        "range": "[-1.0, 1.0]",
        "role": "Kierunek poziomy w strone wrogiego plemienia"
    },
    16: {
        "name": "Nearest Enemy Dir Y",
        "short": "Enemy Dir Y",
        "desc": "Wektor kierunku Y do najblizszego wroga [-1..1]",
        "range": "[-1.0, 1.0]",
        "role": "Kierunek pionowy w strone wrogiego plemienia"
    },
    17: {
        "name": "Nearest Ally Critical State",
        "short": "Ally Critical",
        "desc": "1.0 jesli sojusznik z plemienia ma <20% energii, wpp 0.0",
        "range": "{0.0, 1.0}",
        "role": "Wyzwalacz altruizmu plemiennego (+50 fit za pomoc swojemu)"
    },
    18: {
        "name": "Nearest Enemy Rel Heading",
        "short": "Enemy Heading",
        "desc": "Zwrot predkosci wroga: >0 ucieka tylem, <0 szarzuje czolowo",
        "range": "[-1.0, 1.0]",
        "role": "Taktyka walki: atak na wroga od tylu (+25 fit) vs parowanie"
    },
    19: {
        "name": "Local Tribe Herd Density",
        "short": "Tribe Density",
        "desc": "Gestosc sojusznikow z wlasnego plemienia w 60px [0..1]",
        "range": "[0.0, 1.0]",
        "role": "Obrona stadna plemienia (+15 nagrody za wspolna obrone)"
    },
    20: {
        "name": "Proximity to Nearest Wall",
        "short": "Wall Dist",
        "desc": "Dystans do krawedzi areny (0 przy scianie, 1 w centrum)",
        "range": "[0.0, 1.0]",
        "role": "Unikanie toksycznej sciany (-0.5 energii/klatke)"
    },
    21: {
        "name": "Current Energy Level",
        "short": "Energy Level",
        "desc": "Poziom wlasnej energii zyciowej [0.0..1.0]",
        "range": "[0.0, 1.0]",
        "role": "Naped witalny: zjadanie, polowanie lub bezpieczny spoczynek"
    },
    22: {
        "name": "Nearest Shout Dist",
        "short": "Shout Dist",
        "desc": "Dystans do agenta emitujacego krzyk akustyczny [0..1]",
        "range": "[0.0, 1.0]",
        "role": "Zmysl sluchu: lokalizacja wzywajacego pomocy lub zbiorki"
    },
    23: {
        "name": "Nearest Shout Dir X",
        "short": "Shout Dir X",
        "desc": "Wektor kierunku X w strone krzyczacego agenta [-1..1]",
        "range": "[-1.0, 1.0]",
        "role": "Orientacja sluchowa X w strone zrodla dzwieku"
    },
    24: {
        "name": "Nearest Shout Dir Y",
        "short": "Shout Dir Y",
        "desc": "Wektor kierunku Y w strone krzyczacego agenta [-1..1]",
        "range": "[-1.0, 1.0]",
        "role": "Orientacja sluchowa Y w strone zrodla dzwieku"
    },
}

# Szczegółowe metadane 3 wyjść motorycznych i komunikacyjnych
ACTION_DETAILS: Dict[int, Dict[str, str]] = {
    0: {
        "name": "Acceleration (Accel X)",
        "short": "Accel X",
        "desc": "Pozioma sila napedowa: Lewo (-1.0) / Prawo (+1.0)",
        "range": "[-1.0, 1.0] (aktywacja tanh)",
        "role": "Sterowanie ruchem horyzontalnym agenta na arenie"
    },
    1: {
        "name": "Acceleration (Accel Y)",
        "short": "Accel Y",
        "desc": "Pionowa sila napedowa: Gora (-1.0) / Dol (+1.0)",
        "range": "[-1.0, 1.0] (aktywacja tanh)",
        "role": "Sterowanie ruchem wertykalnym agenta na arenie"
    },
    2: {
        "name": "Acoustic Shout (Komunikacja)",
        "short": "Shout",
        "desc": "Emisja fali dzwiekowej gdy > 0.0 (-0.2 energii/klatke)",
        "range": "[-1.0, 1.0] (aktywacja > 0.0)",
        "role": "Ostrzeganie stada, wezwanie do obrony lub sygnal glodu"
    }
}

# Słowniki kompatybilności wstecznej dla modułów zewnętrznych i testów
SENSORY_INPUT_LABELS: Dict[int, str] = {k: v["name"] for k, v in SENSORY_DETAILS.items()}
ACTION_OUTPUT_LABELS: Dict[int, str] = {k: v["name"] for k, v in ACTION_DETAILS.items()}


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
        self.inspector_title_font = pygame.font.SysFont("Consolas, Courier, monospace", 16, bold=True)
        self.title_font = pygame.font.SysFont("Consolas, Courier, monospace", 14, bold=True)
        self.font = pygame.font.SysFont("Consolas, Courier, monospace", 13)
        self.small_font = pygame.font.SysFont("Consolas, Courier, monospace", 11)

        # Powierzchnia nakładki dla przyciemnienia (tworzona raz dla optymalizacji pamięci)
        self.overlay_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        self.frozen_screen: Optional[pygame.Surface] = None

        # Faza 8: Pre-renderowana półprzezroczysta czerwona ramka Strefy Śmierci (Deadly Margin = 20px)
        self.deadly_margin = DEADLY_ZONE_MARGIN
        self.deadly_zone_surface = pygame.Surface((self.arena_width, self.height), pygame.SRCALPHA)
        deadly_fill_color = (231, 76, 60, 50)
        pygame.draw.rect(self.deadly_zone_surface, deadly_fill_color, (0, 0, self.arena_width, self.height), self.deadly_margin)
        deadly_border_color = (231, 76, 60, 140)
        pygame.draw.rect(
            self.deadly_zone_surface,
            deadly_border_color,
            (self.deadly_margin - 1, self.deadly_margin - 1, self.arena_width - 2 * self.deadly_margin + 2, self.height - 2 * self.deadly_margin + 2),
            1
        )

        self.generation = 0
        self.fast_mode = False  # Przełączanie prędkości symulacji (klawisz SPACE)

        # Stan interaktywnego Inspektora Sieci (Neural Inspector)
        self.inspector_active: bool = False
        self.inspected_genome: Optional[Any] = None
        self.inspector_show_all: bool = False  # Przełącznik TAB: False = tylko aktywne, True = wszystkie 25 zmysłów

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

        # Faza 8: Renderowanie Strefy Śmierci (czerwona półprzezroczysta ramka 20px)
        self.screen.blit(self.deadly_zone_surface, (0, 0))

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

    def get_top_genome_slot_rects(self) -> List[pygame.Rect]:
        """Zwraca prostokąty (Rect) dla 4 slotów wizualizatora Top 4 w panelu bocznym."""
        rects = []
        slot_x = self.arena_width + 12
        slot_y = 278
        slot_w = self.sidebar_width - 24
        slot_h = 88
        for i in range(4):
            rects.append(pygame.Rect(slot_x, slot_y + i * (slot_h + 8), slot_w, slot_h))
        return rects

    def handle_event(self, event: pygame.event.Event) -> None:
        """Obsługuje pojedyncze zdarzenie Pygame (klawisze, kliknięcia myszy, wyjście)."""
        if event.type == pygame.QUIT:
            raise SimulationExit("Użytkownik zamknął okno symulacji.")
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                # Przełączanie trybu przyspieszonego
                self.fast_mode = not self.fast_mode
            elif event.key == pygame.K_TAB:
                # Przełączanie widoku w Inspektorze (tylko aktywne vs wszystkie zmysły)
                if self.inspector_active:
                    self.inspector_show_all = not self.inspector_show_all
            elif event.key == pygame.K_ESCAPE:
                if self.inspector_active:
                    self.inspector_active = False
                    self.inspected_genome = None
                    self.frozen_screen = None
                else:
                    raise SimulationExit("Wciśnięto klawisz ESC.")
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and not self.inspector_active:
                for i, rect in enumerate(self.get_top_genome_slot_rects()):
                    if rect.collidepoint(event.pos) and i < len(self.top_genomes):
                        self.inspected_genome = copy.deepcopy(self.top_genomes[i])
                        self.inspector_active = True
                        if hasattr(self, 'screen') and self.screen is not None:
                            self.frozen_screen = self.screen.copy()
                        break

    def _draw_neural_inspector(self):
        """Renderuje pełnoekranowy, szczegółowy schemat sieci neuronowej z bogatymi opisami każdego węzła."""
        if self.inspected_genome is None:
            return

        genome = self.inspected_genome
        mouse_pos = pygame.mouse.get_pos()

        # 1. Półprzezroczyste przyciemnienie tła całej areny i dashboardu (1600 x 720)
        self.overlay_surface.fill((10, 14, 20, 235))
        self.screen.blit(self.overlay_surface, (0, 0))

        # 2. Główna ramka modalna
        modal_rect = pygame.Rect(30, 15, self.width - 60, self.height - 30)
        pygame.draw.rect(self.screen, (13, 17, 24), modal_rect)
        pygame.draw.rect(self.screen, (35, 48, 68), modal_rect, 2)

        # Belka nagłówka
        header_h = 42
        pygame.draw.rect(self.screen, (20, 28, 40), (modal_rect.x, modal_rect.y, modal_rect.width, header_h))
        pygame.draw.line(self.screen, (0, 245, 212), (modal_rect.x, modal_rect.y + header_h), (modal_rect.right, modal_rect.y + header_h), 2)

        fit_val = getattr(genome, 'fitness', 0.0) or 0.0
        gid = getattr(genome, 'key', 'Top Genom')
        title_text = f"NEURAL INSPECTOR - ARCHITEKTURA MÓZGU AGENTA [ID: {gid} | FITNESS: {fit_val:.1f}]"
        title_surf = self.inspector_title_font.render(title_text, True, (0, 245, 212))
        self.screen.blit(title_surf, (modal_rect.x + 16, modal_rect.y + 11))

        mode_text = "[TAB] Pokaz: Wszystkie 25 zmyslow" if not self.inspector_show_all else "[TAB] Pokaz: Tylko aktywne neurony"
        tab_hint = self.small_font.render(mode_text, True, (88, 166, 255))
        esc_hint = self.title_font.render("[ESC] Zamknij", True, (241, 196, 15))
        self.screen.blit(tab_hint, (modal_rect.right - 380, modal_rect.y + 14))
        self.screen.blit(esc_hint, (modal_rect.right - 140, modal_rect.y + 12))

        # 3. Ekstrakcja i pozycjonowanie węzłów
        connections = getattr(genome, 'connections', {}) or {}

        active_inputs = set()
        active_outputs = set()
        hidden_nodes = set()
        active_conns = []

        for (in_k, out_k), conn in connections.items():
            if getattr(conn, 'enabled', True):
                active_conns.append((in_k, out_k, conn))
                if in_k < 0:
                    active_inputs.add(in_k)
                elif in_k not in (0, 1, 2):
                    hidden_nodes.add(in_k)

                if out_k in (0, 1, 2):
                    active_outputs.add(out_k)
                elif out_k not in (0, 1, 2):
                    hidden_nodes.add(out_k)

        # Decyzja, które wejścia wyświetlić
        if self.inspector_show_all:
            draw_inputs = [-(i + 1) for i in range(25)]
        else:
            draw_inputs = sorted(list(active_inputs)) if active_inputs else [-1, -2, -3]

        node_positions: Dict[int, Tuple[int, int]] = {}
        node_types: Dict[int, str] = {}

        # Granice przestrzeni roboczej grafu
        hud_height = 80
        avail_h = modal_rect.height - header_h - hud_height - 30
        start_y = modal_rect.y + header_h + 16

        in_x = modal_rect.x + 460
        out_x = modal_rect.right - 460
        mid_x = (in_x + out_x) // 2

        # Pozycje wejść sensorycznych
        in_step = avail_h / max(1, len(draw_inputs))
        for idx, k in enumerate(draw_inputs):
            pos_y = int(start_y + idx * in_step + in_step / 2)
            node_positions[k] = (in_x, pos_y)
            node_types[k] = "input"

        # Pozycje 3 wyjść motorycznych / komunikacyjnych
        out_step = avail_h / 3.0
        for idx in range(3):
            pos_y = int(start_y + idx * out_step + out_step / 2)
            node_positions[idx] = (out_x, pos_y)
            node_types[idx] = "output"

        # Pozycje węzłów ukrytych (jeśli wyewoluowały)
        hid_list = sorted(list(hidden_nodes))
        if hid_list:
            hid_step = avail_h / float(len(hid_list))
            for idx, k in enumerate(hid_list):
                pos_y = int(start_y + idx * hid_step + hid_step / 2)
                node_positions[k] = (mid_x, pos_y)
                node_types[k] = "hidden"

        # 4. Wykrywanie najechania myszą (Hover)
        hovered_node = None
        for k, pos in node_positions.items():
            if math.hypot(pos[0] - mouse_pos[0], pos[1] - mouse_pos[1]) <= 14:
                hovered_node = k
                break

        hovered_synapse = None
        if hovered_node is None:
            for (in_k, out_k, conn) in active_conns:
                if in_k in node_positions and out_k in node_positions:
                    p1 = node_positions[in_k]
                    p2 = node_positions[out_k]
                    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                    dist_sq = dx * dx + dy * dy
                    if dist_sq > 0:
                        u = max(0.0, min(1.0, ((mouse_pos[0] - p1[0]) * dx + (mouse_pos[1] - p1[1]) * dy) / dist_sq))
                        proj_x = p1[0] + u * dx
                        proj_y = p1[1] + u * dy
                        if math.hypot(mouse_pos[0] - proj_x, mouse_pos[1] - proj_y) <= 6:
                            hovered_synapse = (in_k, out_k, getattr(conn, 'weight', 0.0))
                            break

        # 5. Rysowanie linii połączeń (synaps)
        thick_connections = []
        for (in_k, out_k, conn) in active_conns:
            if in_k in node_positions and out_k in node_positions:
                start_pos = node_positions[in_k]
                end_pos = node_positions[out_k]
                w = getattr(conn, 'weight', 0.0)
                is_hovered_conn = hovered_synapse and (hovered_synapse[0] == in_k and hovered_synapse[1] == out_k)

                if is_hovered_conn:
                    line_color = (255, 255, 100)
                    line_width = 5
                else:
                    line_color = (46, 204, 113) if w >= 0 else (231, 76, 60)
                    line_width = max(1, min(6, int(abs(w) * 1.5)))

                pygame.draw.line(self.screen, line_color, start_pos, end_pos, line_width)

                if abs(w) >= 1.4 or is_hovered_conn or len(active_conns) <= 8:
                    thick_connections.append((w, start_pos, end_pos, is_hovered_conn))

        # Wypisanie wartości wag na liniach
        for w, start_pos, end_pos, is_h in thick_connections:
            mid_p = ((start_pos[0] + end_pos[0]) // 2, (start_pos[1] + end_pos[1]) // 2)
            w_text = f"{w:+.2f}"
            txt_color = (255, 255, 100) if is_h else (255, 255, 255)
            w_surf = self.small_font.render(w_text, True, txt_color)
            bg_rect = pygame.Rect(mid_p[0] - 2, mid_p[1] - 2, w_surf.get_width() + 4, w_surf.get_height() + 4)
            pygame.draw.rect(self.screen, (15, 20, 30), bg_rect)
            pygame.draw.rect(self.screen, (60, 75, 95), bg_rect, 1)
            self.screen.blit(w_surf, mid_p)

        # 6. Rysowanie węzłów i wyczerpujących opisów
        # --- A. WEJŚCIA SENSORYCZNE ---
        is_dense = len(draw_inputs) > 12
        for k in draw_inputs:
            pos = node_positions[k]
            idx_num = -(k + 1)
            detail = SENSORY_DETAILS.get(idx_num, {
                "name": f"Sensor {idx_num}",
                "short": f"In {idx_num}",
                "desc": "Zmysł percepcyjny",
                "range": "[-1..1]",
                "role": "Percepcja środowiska"
            })
            is_connected = k in active_inputs
            is_h = hovered_node == k

            circle_color = (52, 152, 219) if is_connected else (65, 80, 100)
            if is_h:
                circle_color = (100, 200, 255)
                pygame.draw.circle(self.screen, (255, 255, 255), pos, 9, 2)

            pygame.draw.circle(self.screen, circle_color, pos, 6 if is_connected else 4)

            # Opisy tekstowe po lewej stronie węzła
            title_color = (240, 246, 252) if is_connected else (130, 140, 150)
            desc_color = (0, 210, 255) if is_connected else (90, 100, 110)

            if not is_dense:
                t_surf = self.font.render(f"[{idx_num:2d}] {detail['name']}", True, title_color)
                d_surf = self.small_font.render(f"{detail['desc']} ({detail['range']})", True, desc_color)
                self.screen.blit(t_surf, (modal_rect.x + 16, pos[1] - 14))
                self.screen.blit(d_surf, (modal_rect.x + 16, pos[1] + 1))
            else:
                line_str = f"[{idx_num:2d}] {detail['short']}: {detail['desc'][:36]}..."
                t_surf = self.small_font.render(line_str, True, title_color)
                self.screen.blit(t_surf, (modal_rect.x + 16, pos[1] - 6))

        # --- B. WYJŚCIA AKCJI (3 neurony motoryczne/komunikacyjne) ---
        for k in range(3):
            pos = node_positions[k]
            detail = ACTION_DETAILS[k]
            is_connected = k in active_outputs
            is_h = hovered_node == k

            circle_color = (0, 245, 212) if is_connected else (80, 120, 110)
            if is_h:
                circle_color = (150, 255, 240)
                pygame.draw.circle(self.screen, (255, 255, 255), pos, 11, 2)

            pygame.draw.circle(self.screen, circle_color, pos, 8)

            # Wielowierszowy, bogaty opis wyjścia po prawej stronie
            out_title_surf = self.title_font.render(f"[Wyjście #{k}] {detail['name']}", True, (0, 245, 212))
            out_desc_surf = self.font.render(detail['desc'], True, (240, 246, 252))
            out_role_surf = self.small_font.render(f"Rola: {detail['role']}  |  Zakres: {detail['range']}", True, (139, 148, 158))

            text_x = pos[0] + 18
            self.screen.blit(out_title_surf, (text_x, pos[1] - 22))
            self.screen.blit(out_desc_surf, (text_x, pos[1] - 3))
            self.screen.blit(out_role_surf, (text_x, pos[1] + 15))

        # --- C. WĘZŁY UKRYTE (Żółte interneurony) ---
        for k in hid_list:
            pos = node_positions[k]
            is_h = hovered_node == k
            if is_h:
                pygame.draw.circle(self.screen, (255, 255, 255), pos, 10, 2)
            pygame.draw.circle(self.screen, (241, 196, 15), pos, 7)

            hid_title = self.font.render(f"[Neuron #{k}]", True, (241, 196, 15))
            hid_sub = self.small_font.render("Interneuron (tanh)", True, (160, 170, 180))
            self.screen.blit(hid_title, (pos[0] + 12, pos[1] - 12))
            self.screen.blit(hid_sub, (pos[0] + 12, pos[1] + 2))

        # 7. Dolny Panel Szczegółowej Inspekcji (Interactive HUD Card)
        hud_y = modal_rect.bottom - hud_height - 6
        hud_rect = pygame.Rect(modal_rect.x + 14, hud_y, modal_rect.width - 28, hud_height)
        pygame.draw.rect(self.screen, (16, 22, 32), hud_rect)
        pygame.draw.rect(self.screen, (50, 70, 95) if (hovered_node or hovered_synapse) else (30, 42, 58), hud_rect, 1)

        if hovered_node is not None:
            if hovered_node < 0:
                idx_num = -(hovered_node + 1)
                det = SENSORY_DETAILS.get(idx_num, {})
                h_title = f"🔍 SZCZEGÓŁY ZMYSŁU: [{idx_num}] {det.get('name', 'Sensor')} (Sygnał Wejściowy #{idx_num})"
                h_line1 = f"Funkcja sensoryczna: {det.get('desc', '')}"
                h_line2 = f"Zakres sygnału: {det.get('range', '')}  |  Rola w ekosystemie: {det.get('role', '')}"
                c_title = (88, 166, 255)
            elif hovered_node in (0, 1, 2):
                det = ACTION_DETAILS[hovered_node]
                h_title = f"🔍 SZCZEGÓŁY WYJŚCIA: [{hovered_node}] {det['name']} (Efektor Motoryczny / Akcja)"
                h_line1 = f"Mechanika efektora: {det['desc']}"
                h_line2 = f"Dynamika biologiczna: {det['role']}  |  Zakres sygnału: {det['range']}"
                c_title = (0, 245, 212)
            else:
                h_title = f"🔍 SZCZEGÓŁY INTERNEURONU: Neuron ukryty #{hovered_node}"
                h_line1 = "Funkcja: Pośredniczy w przetwarzaniu sygnałów sensorycznych i kształtuje emergentne zachowania agenta."
                h_line2 = "Matematyka: Nieliniowa funkcja aktywacji tanh [-1.0 .. 1.0] z sumowaniem wag wejściowych."
                c_title = (241, 196, 15)

            self.screen.blit(self.font.render(h_title, True, c_title), (hud_rect.x + 12, hud_rect.y + 8))
            self.screen.blit(self.font.render(h_line1, True, (240, 246, 252)), (hud_rect.x + 12, hud_rect.y + 28))
            self.screen.blit(self.small_font.render(h_line2, True, (160, 175, 190)), (hud_rect.x + 12, hud_rect.y + 48))

        elif hovered_synapse is not None:
            in_k, out_k, w = hovered_synapse
            src_str = SENSORY_DETAILS[-(in_k + 1)]['name'] if in_k < 0 else f"Neuron #{in_k}"
            dst_str = ACTION_DETAILS[out_k]['name'] if out_k in (0, 1, 2) else f"Neuron #{out_k}"
            syn_type = "Pobudzająca (Excitatory)" if w >= 0 else "Hamująca (Inhibitory)"
            syn_color = (46, 204, 113) if w >= 0 else (231, 76, 60)

            h_title = f"🔍 SZCZEGÓŁY SYNAPSY: {src_str}  ───( Waga: {w:+.3f} )───>  {dst_str}"
            h_line1 = f"Charakter połączenia: {syn_type} | Siła synaptyczna: |w| = {abs(w):.3f}"
            h_line2 = f"Wpływ na agenta: {'Wzmacnia wyzwalanie akcji docelowej' if w >= 0 else 'Tłumi wyzwalanie akcji docelowej'}"

            self.screen.blit(self.font.render(h_title, True, syn_color), (hud_rect.x + 12, hud_rect.y + 8))
            self.screen.blit(self.font.render(h_line1, True, (240, 246, 252)), (hud_rect.x + 12, hud_rect.y + 28))
            self.screen.blit(self.small_font.render(h_line2, True, (160, 175, 190)), (hud_rect.x + 12, hud_rect.y + 48))

        else:
            tot_in = len(active_inputs)
            tot_hid = len(hidden_nodes)
            tot_out = len(active_outputs)
            tot_syn = len(active_conns)

            h_title = f"💡 PODSUMOWANIE TOPOLOGII MÓZGU: {tot_in}/25 podłączonych zmysłów | {tot_hid} neuronów ukrytych | {tot_out}/3 aktywnych akcji | {tot_syn} synaps"
            h_line1 = f"Tryb wyświetlania: {'Tylko aktywne neurony (przejrzysty schemat)' if not self.inspector_show_all else 'Wszystkie 25 zmysłów sensorycznych'} [Klawisz TAB przełącza]"
            h_line2 = "Wskazówka: Najedź kursorem myszy na dowolny węzeł lub synapsę, aby wyświetlić szczegółowe parametry biologiczne."

            self.screen.blit(self.font.render(h_title, True, (0, 245, 212)), (hud_rect.x + 12, hud_rect.y + 8))
            self.screen.blit(self.font.render(h_line1, True, (201, 209, 217)), (hud_rect.x + 12, hud_rect.y + 28))
            self.screen.blit(self.small_font.render(h_line2, True, (139, 148, 158)), (hud_rect.x + 12, hud_rect.y + 48))

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

        for i, rect in enumerate(self.get_top_genome_slot_rects()):
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

    def eval_generation(self, nets, genomes, max_frames: int = 900) -> Dict[str, Any]:
        """Ewaluuje pojedynczą generację 50 agentów NEAT."""
        self.generation += 1
        self._reset_world_entities()

        # Równomierny rozkład startowy agentów wokół centrum areny (1280 x 720)
        # Faza 8: Zbalansowany podział na 4 równe plemiona (przy 40 agentach dokładnie 10 na plemię)
        num_agents = len(genomes)
        center_x, center_y = self.arena_width / 2, self.height / 2
        spawn_radius = 240.0

        agents: List[Agent] = []
        for i, (net, genome) in enumerate(zip(nets, genomes)):
            angle = (2.0 * math.pi * i) / max(1, num_agents)
            spawn_x = center_x + spawn_radius * math.cos(angle)
            spawn_y = center_y + spawn_radius * math.sin(angle)
            tribe_id = (i % 4) + 1
            agents.append(Agent(net, genome, self.arena_width, self.height, start_pos=(spawn_x, spawn_y), tribe_id=tribe_id))

        frames_lived = 0
        running = True

        while running and frames_lived < max_frames:
            # 1. Obsługa zdarzeń Pygame
            for event in pygame.event.get():
                self.handle_event(event)

            # Jeśli aktywny jest Inspektor Sieci – pauzujemy fizykę i ewolucję, renderujemy tylko nakładkę
            if self.inspector_active:
                if self.frozen_screen is not None:
                    self.screen.blit(self.frozen_screen, (0, 0))
                self._draw_neural_inspector()
                pygame.display.flip()
                self.clock.tick(60)
                continue

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

        # Zapisanie 4 najlepszych genomów na koniec generacji dla wizualizatora (głębokie kopie dla stabilności)
        sorted_elites = sorted(
            genomes,
            key=lambda g: getattr(g, 'fitness', -999999.0) if getattr(g, 'fitness', None) is not None else -999999.0,
            reverse=True
        )[:4]
        self.top_genomes = [copy.deepcopy(g) for g in sorted_elites]

        return {
            "foods_eaten": sum(a.foods_eaten for a in agents),
            "poisons_hit": sum(a.poisons_hit for a in agents),
            "allies_saved": sum(a.allies_saved for a in agents),
            "attacks_made": sum(a.attacks_made for a in agents),
            "defenses_made": sum(a.defenses_made for a in agents),
            "herd_defenses": sum(a.herd_defenses for a in agents),
            "shouts_made": sum(a.shouts_made for a in agents)
        }