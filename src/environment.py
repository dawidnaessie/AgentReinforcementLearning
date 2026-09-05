import copy
import math
import os
import random
import pygame
from typing import List, Dict, Any, Optional, Tuple
from src.agent import Agent, DEADLY_ZONE_MARGIN
from src.entities import Food, Hazard, Poison


# Detailed metadata for the 25 normalized sensory inputs according to README.md
SENSORY_DETAILS: Dict[int, Dict[str, str]] = {
    0: {
        "name": "Velocity (Vel X)",
        "short": "Vel X",
        "desc": "Agent horizontal velocity [-1.0..1.0]",
        "range": "[-1.0, 1.0]",
        "role": "X-axis momentum and inertia tracking"
    },
    1: {
        "name": "Velocity (Vel Y)",
        "short": "Vel Y",
        "desc": "Agent vertical velocity [-1.0..1.0]",
        "range": "[-1.0, 1.0]",
        "role": "Y-axis momentum and inertia tracking"
    },
    2: {
        "name": "Nearest Food #1 Dist",
        "short": "Food #1 Dist",
        "desc": "Euclidean distance to nearest food apple [0..1]",
        "range": "[0.0, 1.0]",
        "role": "Primary foraging target (+40 fit, +65 energy)"
    },
    3: {
        "name": "Nearest Food #1 Dir X",
        "short": "Food #1 Dir X",
        "desc": "X direction vector to nearest food [-1..1]",
        "range": "[-1.0, 1.0]",
        "role": "Horizontal navigation toward food #1"
    },
    4: {
        "name": "Nearest Food #1 Dir Y",
        "short": "Food #1 Dir Y",
        "desc": "Y direction vector to nearest food [-1..1]",
        "range": "[-1.0, 1.0]",
        "role": "Vertical navigation toward food #1"
    },
    5: {
        "name": "Secondary Food #2 Dist",
        "short": "Food #2 Dist",
        "desc": "Distance to 2nd nearest food apple [0..1]",
        "range": "[0.0, 1.0]",
        "role": "Secondary trajectory planning for foraging"
    },
    6: {
        "name": "Secondary Food #2 Dir X",
        "short": "Food #2 Dir X",
        "desc": "X direction vector to 2nd nearest food [-1..1]",
        "range": "[-1.0, 1.0]",
        "role": "Horizontal navigation toward food #2"
    },
    7: {
        "name": "Secondary Food #2 Dir Y",
        "short": "Food #2 Dir Y",
        "desc": "Y direction vector to 2nd nearest food [-1..1]",
        "range": "[-1.0, 1.0]",
        "role": "Vertical navigation toward food #2"
    },
    8: {
        "name": "Nearest Poison Dist",
        "short": "Poison Dist",
        "desc": "Distance to nearest purple poison [0..1]",
        "range": "[0.0, 1.0]",
        "role": "Toxin avoidance (-10 fit, -35 energy)"
    },
    9: {
        "name": "Nearest Poison Dir X",
        "short": "Poison Dir X",
        "desc": "X direction vector to nearest poison [-1..1]",
        "range": "[-1.0, 1.0]",
        "role": "Horizontal repulsion from purple toxins"
    },
    10: {
        "name": "Nearest Poison Dir Y",
        "short": "Poison Dir Y",
        "desc": "Y direction vector to nearest poison [-1..1]",
        "range": "[-1.0, 1.0]",
        "role": "Vertical repulsion from purple toxins"
    },
    11: {
        "name": "Nearest Hazard Dist",
        "short": "Hazard Dist",
        "desc": "Distance to roving predator hazard [0..1]",
        "range": "[0.0, 1.0]",
        "role": "Threat evasion (-5 fit, -20 energy)"
    },
    12: {
        "name": "Nearest Hazard Dir X",
        "short": "Hazard Dir X",
        "desc": "X direction vector to moving hazard [-1..1]",
        "range": "[-1.0, 1.0]",
        "role": "Horizontal evasion from roving hazard"
    },
    13: {
        "name": "Nearest Hazard Dir Y",
        "short": "Hazard Dir Y",
        "desc": "Y direction vector to moving hazard [-1..1]",
        "range": "[-1.0, 1.0]",
        "role": "Vertical evasion from roving hazard"
    },
    14: {
        "name": "Nearest Enemy Dist",
        "short": "Enemy Dist",
        "desc": "Distance to nearest enemy from different tribe [0..1]",
        "range": "[0.0, 1.0]",
        "role": "Inter-tribal perception: enemy targeting and prey tracking"
    },
    15: {
        "name": "Nearest Enemy Dir X",
        "short": "Enemy Dir X",
        "desc": "X direction vector to nearest enemy [-1..1]",
        "range": "[-1.0, 1.0]",
        "role": "Horizontal orientation toward enemy tribe"
    },
    16: {
        "name": "Nearest Enemy Dir Y",
        "short": "Enemy Dir Y",
        "desc": "Y direction vector to nearest enemy [-1..1]",
        "range": "[-1.0, 1.0]",
        "role": "Vertical orientation toward enemy tribe"
    },
    17: {
        "name": "Critical Ally Dir X",
        "short": "Ally Dir X",
        "desc": "X direction vector to nearest starving ally (<20% energy) [-1..1]",
        "range": "[-1.0, 1.0]",
        "role": "Kin altruism navigation: horizontal steering toward ally"
    },
    18: {
        "name": "Critical Ally Dir Y",
        "short": "Ally Dir Y",
        "desc": "Y direction vector to nearest starving ally (<20% energy) [-1..1]",
        "range": "[-1.0, 1.0]",
        "role": "Kin altruism navigation: vertical steering toward ally"
    },
    19: {
        "name": "Nearest Enemy Rel Heading",
        "short": "Enemy Heading",
        "desc": "Enemy heading alignment: >0 fleeing back exposed, <0 charging head-on",
        "range": "[-1.0, 1.0]",
        "role": "Combat tactics: backstab hunting (+25 fit) vs parrying"
    },
    20: {
        "name": "Local Tribe Herd Density",
        "short": "Tribe Density",
        "desc": "Density of allies from own tribe within 60px [0..1]",
        "range": "[0.0, 1.0]",
        "role": "Tribe herd defense (+15 reward for cooperative defense)"
    },
    21: {
        "name": "Proximity to Nearest Wall",
        "short": "Wall Dist",
        "desc": "Distance to arena boundary (0 at wall, 1 at center)",
        "range": "[0.0, 1.0]",
        "role": "Boundary repulsion to avoid Deadly Zone (-2.0 energy/frame)"
    },
    22: {
        "name": "Current Energy Level",
        "short": "Energy Level",
        "desc": "Current internal vital energy reserve [0.0..1.0]",
        "range": "[0.0, 1.0]",
        "role": "Vital drive: foraging, hunting, or energy conservation"
    }
}

# Detailed metadata for the 2 motor action outputs (Phase 9: Acoustic Lobotomy)
ACTION_DETAILS: Dict[int, Dict[str, str]] = {
    0: {
        "name": "Acceleration (Accel X)",
        "short": "Accel X",
        "desc": "Horizontal propulsion force: Left (-1.0) / Right (+1.0)",
        "range": "[-1.0, 1.0] (tanh activation)",
        "role": "Horizontal locomotion control in arena"
    },
    1: {
        "name": "Acceleration (Accel Y)",
        "short": "Accel Y",
        "desc": "Vertical propulsion force: Up (-1.0) / Down (+1.0)",
        "range": "[-1.0, 1.0] (tanh activation)",
        "role": "Vertical locomotion control in arena"
    }
}

# Backward-compatibility dictionaries for external modules and unit tests
SENSORY_INPUT_LABELS: Dict[int, str] = {k: v["name"] for k, v in SENSORY_DETAILS.items()}
ACTION_OUTPUT_LABELS: Dict[int, str] = {k: v["name"] for k, v in ACTION_DETAILS.items()}


def format_node_label(node_id: int, is_source: bool = False) -> str:
    """
    Translates a NEAT or inspector node ID into human-readable string labels
    matching the Neural Inspector UI.
    - Negative IDs (-1 to -23) map to sensory inputs (0 to 22) via SENSORY_INPUT_LABELS.
    - Output IDs (0, 1) map to action outputs via ACTION_OUTPUT_LABELS.
    - Hidden node IDs (>= 2 or non-standard) map to 'Node {node_id}'.
    """
    if isinstance(node_id, str):
        return node_id

    if node_id < 0:
        idx = -(node_id + 1)
        if idx in SENSORY_INPUT_LABELS:
            return SENSORY_INPUT_LABELS[idx]
        return f"Input #{idx}"
    elif node_id in ACTION_OUTPUT_LABELS:
        return ACTION_OUTPUT_LABELS[node_id]
    else:
        return f"Node {node_id}"


def export_brain_to_txt(genome: Any, logs_dir: str = "logs") -> str:
    """
    Exports the complete mathematical topology of a NEAT genome into a readable .txt file.
    Saves directly to {logs_dir}/brain_id_{genome.key}.txt.
    """
    os.makedirs(logs_dir, exist_ok=True)
    key = getattr(genome, 'key', 0)
    fitness = getattr(genome, 'fitness', None)
    fitness_str = f"{fitness}" if fitness is not None else "0.0"

    lines = [
        "--- GENERAL INFO ---",
        f"Genome ID: {key}",
        f"Fitness: {fitness_str}",
        "",
        "--- NODES ---"
    ]

    # Extract hidden nodes from genome.nodes (excluding output nodes 0, 1)
    nodes_dict = getattr(genome, 'nodes', {}) or {}
    hidden_nodes = []
    if isinstance(nodes_dict, dict):
        for nid, node in nodes_dict.items():
            if nid not in (0, 1):
                hidden_nodes.append((nid, node))
    elif isinstance(nodes_dict, (list, tuple)):
        for idx, node in enumerate(nodes_dict):
            nid = getattr(node, 'key', idx)
            if nid not in (0, 1):
                hidden_nodes.append((nid, node))

    hidden_nodes.sort(key=lambda item: item[0])
    if hidden_nodes:
        for nid, node in hidden_nodes:
            bias = getattr(node, 'bias', 0.0)
            act = getattr(node, 'activation', 'tanh')
            if isinstance(node, dict):
                bias = node.get('bias', bias)
                act = node.get('activation', act)
            lines.append(f"Node ID: {nid} | Activation: {act} | Bias: {bias:.4f}")
    else:
        lines.append("No hidden nodes")

    lines.append("")
    lines.append("--- SYNAPSES (CONNECTIONS) ---")

    connections = getattr(genome, 'connections', {}) or {}
    if connections and isinstance(connections, dict):
        sorted_conns = sorted(connections.items(), key=lambda item: (item[0][0], item[0][1]))
        for (in_k, out_k), conn in sorted_conns:
            src_label = format_node_label(in_k, is_source=True)
            dst_label = format_node_label(out_k, is_source=False)
            weight = getattr(conn, 'weight', 0.0)
            enabled = getattr(conn, 'enabled', True)
            if isinstance(conn, dict):
                weight = conn.get('weight', weight)
                enabled = conn.get('enabled', enabled)
            status_str = "Enabled" if enabled else "Disabled"
            lines.append(f"[{src_label}] -> [{dst_label}] | Weight: {weight:.4f} | Status: {status_str}")
    else:
        lines.append("No connections")

    content = "\n".join(lines) + "\n"
    filepath = os.path.join(logs_dir, f"brain_id_{key}.txt")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    return filepath


class SimulationExit(Exception):
    """Exception raised when user requests simulation termination (ESC or window close)."""
    pass


class Environment:
    """2D simulation environment managing Pygame lifecycle, world entities, UI panel, and NEAT visualizer."""

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

        # Monospace font initialization (initialized once in constructor)
        self.inspector_title_font = pygame.font.SysFont("Consolas, Courier, monospace", 16, bold=True)
        self.title_font = pygame.font.SysFont("Consolas, Courier, monospace", 14, bold=True)
        self.font = pygame.font.SysFont("Consolas, Courier, monospace", 13)
        self.small_font = pygame.font.SysFont("Consolas, Courier, monospace", 11)

        # Overlay surface for dimming (allocated once for memory efficiency)
        self.overlay_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        self.frozen_screen: Optional[pygame.Surface] = None

        # Phase 8: Pre-rendered semi-transparent red frame for Deadly Zone (Deadly Margin = 20px)
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
        self.fast_mode = False  # Toggle simulation speed (SPACE key)

        # Interactive Neural Inspector state
        self.inspector_active: bool = False
        self.inspected_genome: Optional[Any] = None
        self.inspector_show_all: bool = False  # TAB toggle: False = active only, True = all 25 senses
        self.brain_dump_feedback: Optional[str] = None
        self.brain_dump_feedback_timer: int = 0

        # References to top 4 genomes from previous generation (Top 4 Visualizer)
        self.top_genomes: List[Any] = []

        # World entity pools within arena boundaries (1280 x 720)
        self.food_count = food_count
        self.poison_count = poison_count
        self.hazard_count = hazard_count

        self.foods: List[Food] = Food.create_clustered(
            self.food_count, self.arena_width, self.height, margin=60
        )
        self.poisons: List[Poison] = [
            Poison(random.randint(60, self.arena_width - 60), random.randint(60, self.height - 60))
            for _ in range(self.poison_count)
        ]
        self.hazards: List[Hazard] = [
            Hazard(random.randint(60, self.arena_width - 60), random.randint(60, self.height - 60))
            for _ in range(self.hazard_count)
        ]

    def _reset_world_entities(self):
        """Resets food, poison, and hazard positions across the arena at the start of a new generation."""
        Food.respawn_clustered(self.foods, self.arena_width, self.height, margin=60)
        for poison in self.poisons:
            poison.respawn(self.arena_width, self.height)
        for hazard in self.hazards:
            hazard.pos = pygame.math.Vector2(
                random.randint(60, self.arena_width - 60),
                random.randint(60, self.height - 60)
            )

    def _draw_arena_grid(self):
        """Renders dark research background (#0b0c10) and subtle grid on the simulation arena."""
        # Arena background fill
        pygame.draw.rect(self.screen, (11, 12, 16), (0, 0, self.arena_width, self.height))

        # Subtle grid every 64 pixels
        grid_color = (18, 22, 28)
        for x in range(0, self.arena_width, 64):
            pygame.draw.line(self.screen, grid_color, (x, 0), (x, self.height), 1)
        for y in range(0, self.height, 64):
            pygame.draw.line(self.screen, grid_color, (0, y), (self.arena_width, y), 1)

        # Phase 8: Render Deadly Zone (red semi-transparent 20px frame)
        self.screen.blit(self.deadly_zone_surface, (0, 0))

        # Subtle boundary for safe zone (50px from boundaries)
        pygame.draw.rect(self.screen, (30, 38, 48), (50, 50, self.arena_width - 100, self.height - 100), 1)

    def _draw_network_graph(self, genome, slot_rect: pygame.Rect, title: str):
        """Renders a simplified neural network graph for a genome in a dedicated sidebar slot."""
        # Slot background and outline
        pygame.draw.rect(self.screen, (15, 18, 24), slot_rect)
        pygame.draw.rect(self.screen, (40, 50, 65), slot_rect, 1)

        # Slot header title
        title_surf = self.small_font.render(title, True, (0, 245, 212))
        self.screen.blit(title_surf, (slot_rect.x + 8, slot_rect.y + 4))

        if not hasattr(genome, 'connections') or not genome.connections:
            placeholder = self.small_font.render("No connections", True, (120, 130, 140))
            self.screen.blit(placeholder, (slot_rect.x + 10, slot_rect.y + 35))
            return

        # Node extraction
        # Inputs: negative keys (-1 to -25), Outputs: keys (0, 1, 2), Hidden: keys > 2
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

        # Node positioning within slot bounds
        node_positions: Dict[int, Tuple[int, int]] = {}
        in_x = slot_rect.x + 25
        out_x = slot_rect.x + slot_rect.width - 25
        mid_x = slot_rect.x + slot_rect.width // 2

        # Inputs (sorted, left column)
        in_list = sorted(list(active_inputs)) if active_inputs else [-1, -2, -3]
        in_spacing = (slot_rect.height - 30) / max(1, len(in_list))
        for idx, k in enumerate(in_list):
            node_positions[k] = (in_x, int(slot_rect.y + 22 + idx * in_spacing + in_spacing / 2))

        # Outputs (Ax, Ay, right column)
        out_list = [0, 1]
        out_spacing = (slot_rect.height - 30) / 2
        for idx, k in enumerate(out_list):
            node_positions[k] = (out_x, int(slot_rect.y + 22 + idx * out_spacing + out_spacing / 2))

        # Hidden nodes (middle column)
        hid_list = sorted(list(hidden_nodes))
        if hid_list:
            hid_spacing = (slot_rect.height - 30) / len(hid_list)
            for idx, k in enumerate(hid_list):
                node_positions[k] = (mid_x, int(slot_rect.y + 22 + idx * hid_spacing + hid_spacing / 2))

        # Draw synaptic connections
        for (in_k, out_k), conn in genome.connections.items():
            if conn.enabled and in_k in node_positions and out_k in node_positions:
                start_pos = node_positions[in_k]
                end_pos = node_positions[out_k]
                w = conn.weight
                # Color: green for positive weights, red for negative weights
                line_color = (46, 204, 113) if w >= 0 else (231, 76, 60)
                line_width = max(1, min(3, int(abs(w) * 1.2)))
                pygame.draw.line(self.screen, line_color, start_pos, end_pos, line_width)

        # Draw node circles
        for k, pos in node_positions.items():
            if k < 0:
                # Inputs: blue
                pygame.draw.circle(self.screen, (52, 152, 219), pos, 3)
            elif k in active_outputs:
                # Outputs: cyan
                pygame.draw.circle(self.screen, (0, 245, 212), pos, 4)
            else:
                # Hidden: yellow
                pygame.draw.circle(self.screen, (241, 196, 15), pos, 3)

    def get_top_genome_slot_rects(self) -> List[pygame.Rect]:
        """Returns Rect bounding boxes for the 4 Top-4 visualizer slots in the sidebar."""
        rects = []
        slot_x = self.arena_width + 12
        slot_y = 278
        slot_w = self.sidebar_width - 24
        slot_h = 88
        for i in range(4):
            rects.append(pygame.Rect(slot_x, slot_y + i * (slot_h + 8), slot_w, slot_h))
        return rects

    def export_brain_to_txt(self, genome: Optional[Any] = None, logs_dir: str = "logs") -> str:
        """Exports the specified genome (or currently inspected genome) to a readable .txt topology dump."""
        target = genome if genome is not None else self.inspected_genome
        if target is None:
            raise ValueError("No genome specified and no inspected genome active.")
        return export_brain_to_txt(target, logs_dir=logs_dir)

    def handle_event(self, event: pygame.event.Event) -> None:
        """Handles a single Pygame event (keys, mouse clicks, quit)."""
        if event.type == pygame.QUIT:
            raise SimulationExit("User closed simulation window.")
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                # Toggle fast simulation mode
                self.fast_mode = not self.fast_mode
            elif event.key == pygame.K_TAB:
                # Toggle view mode in Neural Inspector (active only vs all 25 senses)
                if self.inspector_active:
                    self.inspector_show_all = not self.inspector_show_all
            elif event.key == pygame.K_s:
                # Save complete mathematical topology dump of inspected genome to logs/
                if self.inspector_active and self.inspected_genome is not None:
                    out_path = self.export_brain_to_txt(self.inspected_genome)
                    file_name = os.path.basename(out_path)
                    self.brain_dump_feedback = f"[SAVED] {file_name}"
                    self.brain_dump_feedback_timer = 180
            elif event.key == pygame.K_ESCAPE:
                if self.inspector_active:
                    self.inspector_active = False
                    self.inspected_genome = None
                    self.frozen_screen = None
                else:
                    raise SimulationExit("ESC key pressed.")
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
        """Renders the fullscreen interactive Neural Inspector with rich descriptions for each node and synapse."""
        if self.inspected_genome is None:
            return

        genome = self.inspected_genome
        mouse_pos = pygame.mouse.get_pos()

        # 1. Semi-transparent background dimming across arena and dashboard (1600 x 720)
        self.overlay_surface.fill((10, 14, 20, 235))
        self.screen.blit(self.overlay_surface, (0, 0))

        # 2. Main modal frame
        modal_rect = pygame.Rect(30, 15, self.width - 60, self.height - 30)
        pygame.draw.rect(self.screen, (13, 17, 24), modal_rect)
        pygame.draw.rect(self.screen, (35, 48, 68), modal_rect, 2)

        # Header bar
        header_h = 42
        pygame.draw.rect(self.screen, (20, 28, 40), (modal_rect.x, modal_rect.y, modal_rect.width, header_h))
        pygame.draw.line(self.screen, (0, 245, 212), (modal_rect.x, modal_rect.y + header_h), (modal_rect.right, modal_rect.y + header_h), 2)

        fit_val = getattr(genome, 'fitness', 0.0) or 0.0
        gid = getattr(genome, 'key', 'Top Genome')
        title_text = f"NEURAL INSPECTOR - AGENT BRAIN ARCHITECTURE [ID: {gid} | FITNESS: {fit_val:.1f}]"
        title_surf = self.inspector_title_font.render(title_text, True, (0, 245, 212))
        self.screen.blit(title_surf, (modal_rect.x + 16, modal_rect.y + 11))

        tot_senses = len(SENSORY_DETAILS)
        mode_text = f"[TAB] Show: All {tot_senses} senses" if not self.inspector_show_all else "[TAB] Show: Active neurons only"
        tab_hint = self.small_font.render(mode_text, True, (88, 166, 255))
        esc_hint = self.title_font.render("[ESC] Close", True, (241, 196, 15))
        self.screen.blit(tab_hint, (modal_rect.right - 380, modal_rect.y + 14))
        self.screen.blit(esc_hint, (modal_rect.right - 140, modal_rect.y + 12))

        # 3. Node extraction and positioning
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
                elif in_k not in (0, 1):
                    hidden_nodes.add(in_k)

                if out_k in (0, 1):
                    active_outputs.add(out_k)
                elif out_k not in (0, 1):
                    hidden_nodes.add(out_k)

        # Decide which sensory inputs to display
        if self.inspector_show_all:
            draw_inputs = [-(i + 1) for i in range(len(SENSORY_DETAILS))]
        else:
            draw_inputs = sorted(list(active_inputs)) if active_inputs else [-1, -2, -3]

        node_positions: Dict[int, Tuple[int, int]] = {}
        node_types: Dict[int, str] = {}

        # Graph workspace boundaries
        hud_height = 80
        avail_h = modal_rect.height - header_h - hud_height - 30
        start_y = modal_rect.y + header_h + 16

        in_x = modal_rect.x + 460
        out_x = modal_rect.right - 460
        mid_x = (in_x + out_x) // 2

        # Sensory input positions
        in_step = avail_h / max(1, len(draw_inputs))
        for idx, k in enumerate(draw_inputs):
            pos_y = int(start_y + idx * in_step + in_step / 2)
            node_positions[k] = (in_x, pos_y)
            node_types[k] = "input"

        # Output positions (Ax, Ay)
        out_step = avail_h / 2.0
        for idx in range(2):
            pos_y = int(start_y + idx * out_step + out_step / 2)
            node_positions[idx] = (out_x, pos_y)
            node_types[idx] = "output"

        # Hidden node positions (if evolved)
        hid_list = sorted(list(hidden_nodes))
        if hid_list:
            hid_step = avail_h / float(len(hid_list))
            for idx, k in enumerate(hid_list):
                pos_y = int(start_y + idx * hid_step + hid_step / 2)
                node_positions[k] = (mid_x, pos_y)
                node_types[k] = "hidden"

        # 4. Mouse hover detection
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

        # 5. Draw synaptic connections
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

        # Render weight values on connection lines
        for w, start_pos, end_pos, is_h in thick_connections:
            mid_p = ((start_pos[0] + end_pos[0]) // 2, (start_pos[1] + end_pos[1]) // 2)
            w_text = f"{w:+.2f}"
            txt_color = (255, 255, 100) if is_h else (255, 255, 255)
            w_surf = self.small_font.render(w_text, True, txt_color)
            bg_rect = pygame.Rect(mid_p[0] - 2, mid_p[1] - 2, w_surf.get_width() + 4, w_surf.get_height() + 4)
            pygame.draw.rect(self.screen, (15, 20, 30), bg_rect)
            pygame.draw.rect(self.screen, (60, 75, 95), bg_rect, 1)
            self.screen.blit(w_surf, mid_p)

        # 6. Draw nodes and descriptions
        # --- A. SENSORY INPUTS ---
        is_dense = len(draw_inputs) > 12
        for k in draw_inputs:
            pos = node_positions[k]
            idx_num = -(k + 1)
            detail = SENSORY_DETAILS.get(idx_num, {
                "name": f"Sensor {idx_num}",
                "short": f"In {idx_num}",
                "desc": "Perceptual sense",
                "range": "[-1..1]",
                "role": "Environmental perception"
            })
            is_connected = k in active_inputs
            is_h = hovered_node == k

            circle_color = (52, 152, 219) if is_connected else (65, 80, 100)
            if is_h:
                circle_color = (100, 200, 255)
                pygame.draw.circle(self.screen, (255, 255, 255), pos, 9, 2)

            pygame.draw.circle(self.screen, circle_color, pos, 6 if is_connected else 4)

            # Text labels to the left of the node
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

        # --- B. ACTION OUTPUTS (2 motor neurons) ---
        for k in range(2):
            pos = node_positions[k]
            detail = ACTION_DETAILS[k]
            is_connected = k in active_outputs
            is_h = hovered_node == k

            circle_color = (0, 245, 212) if is_connected else (80, 120, 110)
            if is_h:
                circle_color = (150, 255, 240)
                pygame.draw.circle(self.screen, (255, 255, 255), pos, 11, 2)

            pygame.draw.circle(self.screen, circle_color, pos, 8)

            # Multi-line description to the right of output node
            out_title_surf = self.title_font.render(f"[Output #{k}] {detail['name']}", True, (0, 245, 212))
            out_desc_surf = self.font.render(detail['desc'], True, (240, 246, 252))
            out_role_surf = self.small_font.render(f"Role: {detail['role']}  |  Range: {detail['range']}", True, (139, 148, 158))

            text_x = pos[0] + 18
            self.screen.blit(out_title_surf, (text_x, pos[1] - 22))
            self.screen.blit(out_desc_surf, (text_x, pos[1] - 3))
            self.screen.blit(out_role_surf, (text_x, pos[1] + 15))

        # --- C. HIDDEN NODES (Yellow interneuons) ---
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

        # 7. Bottom Detailed Inspection Panel (Interactive HUD Card)
        hud_y = modal_rect.bottom - hud_height - 6
        hud_rect = pygame.Rect(modal_rect.x + 14, hud_y, modal_rect.width - 28, hud_height)
        pygame.draw.rect(self.screen, (16, 22, 32), hud_rect)
        pygame.draw.rect(self.screen, (50, 70, 95) if (hovered_node or hovered_synapse) else (30, 42, 58), hud_rect, 1)

        if hovered_node is not None:
            if hovered_node < 0:
                idx_num = -(hovered_node + 1)
                det = SENSORY_DETAILS.get(idx_num, {})
                h_title = f"🔍 SENSORY DETAILS: [{idx_num}] {det.get('name', 'Sensor')} (Input Signal #{idx_num})"
                h_line1 = f"Sensory function: {det.get('desc', '')}"
                h_line2 = f"Signal range: {det.get('range', '')}  |  Ecological role: {det.get('role', '')}"
                c_title = (88, 166, 255)
            elif hovered_node in (0, 1):
                det = ACTION_DETAILS[hovered_node]
                h_title = f"🔍 ACTION DETAILS: [{hovered_node}] {det['name']} (Motor Effector / Action)"
                h_line1 = f"Effector mechanics: {det['desc']}"
                h_line2 = f"Biological role: {det['role']}  |  Signal range: {det['range']}"
                c_title = (0, 245, 212)
            else:
                h_title = f"🔍 INTERNEURON DETAILS: Hidden Neuron #{hovered_node}"
                h_line1 = "Function: Mediates sensory processing and shapes emergent agent behavior."
                h_line2 = "Mathematics: Non-linear tanh activation [-1.0 .. 1.0] with weighted input summation."
                c_title = (241, 196, 15)

            self.screen.blit(self.font.render(h_title, True, c_title), (hud_rect.x + 12, hud_rect.y + 8))
            self.screen.blit(self.font.render(h_line1, True, (240, 246, 252)), (hud_rect.x + 12, hud_rect.y + 28))
            self.screen.blit(self.small_font.render(h_line2, True, (160, 175, 190)), (hud_rect.x + 12, hud_rect.y + 48))

        elif hovered_synapse is not None:
            in_k, out_k, w = hovered_synapse
            src_str = SENSORY_DETAILS.get(-(in_k + 1), {}).get('name', f"Input #{-(in_k + 1)}") if in_k < 0 else f"Neuron #{in_k}"
            dst_str = ACTION_DETAILS[out_k]['name'] if out_k in (0, 1) else f"Neuron #{out_k}"
            syn_type = "Excitatory" if w >= 0 else "Inhibitory"
            syn_color = (46, 204, 113) if w >= 0 else (231, 76, 60)

            h_title = f"🔍 SYNAPSE DETAILS: {src_str}  ───( Weight: {w:+.3f} )───>  {dst_str}"
            h_line1 = f"Connection nature: {syn_type} | Synaptic strength: |w| = {abs(w):.3f}"
            h_line2 = f"Agent impact: {'Excites target action trigger' if w >= 0 else 'Inhibits target action trigger'}"

            self.screen.blit(self.font.render(h_title, True, syn_color), (hud_rect.x + 12, hud_rect.y + 8))
            self.screen.blit(self.font.render(h_line1, True, (240, 246, 252)), (hud_rect.x + 12, hud_rect.y + 28))
            self.screen.blit(self.small_font.render(h_line2, True, (160, 175, 190)), (hud_rect.x + 12, hud_rect.y + 48))

        else:
            tot_in = len(active_inputs)
            tot_hid = len(hidden_nodes)
            tot_out = len(active_outputs)
            tot_syn = len(active_conns)

            tot_senses = len(SENSORY_DETAILS)
            h_title = f"💡 BRAIN TOPOLOGY SUMMARY: {tot_in}/{tot_senses} connected senses | {tot_hid} hidden neurons | {tot_out}/2 active actions | {tot_syn} synapses"
            h_line1 = f"Display mode: {'Active neurons only (clean graph)' if not self.inspector_show_all else f'All {tot_senses} sensory inputs'} [Press TAB to toggle]"
            h_line2 = "Hint: Hover mouse cursor over any node or synapse to inspect detailed biological parameters."

            self.screen.blit(self.font.render(h_title, True, (0, 245, 212)), (hud_rect.x + 12, hud_rect.y + 8))
            self.screen.blit(self.font.render(h_line1, True, (201, 209, 217)), (hud_rect.x + 12, hud_rect.y + 28))
            self.screen.blit(self.small_font.render(h_line2, True, (139, 148, 158)), (hud_rect.x + 12, hud_rect.y + 48))

        # 8. Bottom text hint for brain dump export
        if self.brain_dump_feedback and self.brain_dump_feedback_timer > 0:
            hint_text = self.brain_dump_feedback
            hint_color = (0, 245, 212)
            self.brain_dump_feedback_timer -= 1
        else:
            hint_text = "[S] Save brain dump to logs/"
            hint_color = (139, 148, 158)

        hint_surf = self.small_font.render(hint_text, True, hint_color)
        self.screen.blit(hint_surf, (hud_rect.right - hint_surf.get_width() - 16, hud_rect.y + 50))

    def _draw_sidebar(self, agents: List[Agent], frames_lived: int, max_frames: int, alive_count: int, best_current_fitness: float):
        """Renders the telemetry sidebar and Top-4 neural visualizer."""
        sidebar_rect = pygame.Rect(self.arena_width, 0, self.sidebar_width, self.height)
        # Sidebar background (#161b22)
        pygame.draw.rect(self.screen, (22, 27, 34), sidebar_rect)
        # Boundary line separating arena from sidebar (#2d3748)
        pygame.draw.line(self.screen, (45, 55, 72), (self.arena_width, 0), (self.arena_width, self.height), 2)

        # 1. Main Header and Stats
        title_surf = self.title_font.render("=== NEURAL RESEARCH DASHBOARD ===", True, (240, 246, 252))
        self.screen.blit(title_surf, (self.arena_width + 12, 12))

        fps_val = int(self.clock.get_fps())
        total_foods = sum(a.foods_eaten for a in agents)
        total_poisons = sum(a.poisons_hit for a in agents)
        total_saved = sum(a.allies_saved for a in agents)
        total_attacks = sum(a.attacks_made for a in agents)
        total_defenses = sum(a.defenses_made for a in agents)
        total_herd = sum(a.herd_defenses for a in agents)

        stats_lines = [
            f"GENERATION:  {self.generation:<4d} | FRAME: {frames_lived:3d}/{max_frames}",
            f"POPULATION:  {alive_count:2d}/{len(agents):<2d} | MODE: {'TURBO' if self.fast_mode else '60 FPS'}",
            f"MAX FITNESS: {best_current_fitness:<6.1f} | FPS:  {fps_val:2d}",
            "---------------------------------------",
            f"• Eaten Apples:        {total_foods:4d}",
            f"• Poisons Hit:         {total_poisons:4d}",
            f"• Altruism (Rescues):  {total_saved:4d}",
            f"• Predator Attacks:    {total_attacks:4d}",
            f"• Frontal Defenses:    {total_defenses:4d}",
            f"• Herd Defenses:       {total_herd:4d}",
            "---------------------------------------"
        ]

        y_offset = 36
        for line in stats_lines:
            color = (201, 209, 217)
            if "MAX FITNESS" in line:
                color = (241, 196, 15)
            elif "POPULATION" in line:
                color = (88, 166, 255)
            elif "Altruism" in line or "Apples" in line:
                color = (46, 204, 113)
            elif "Attacks" in line or "Poisons" in line:
                color = (231, 76, 60)

            line_surf = self.font.render(line, True, color)
            self.screen.blit(line_surf, (self.arena_width + 14, y_offset))
            y_offset += 18

        # 2. Top-4 Neural Visualizer (4 vertically stacked slots)
        section_surf = self.title_font.render("--- TOP 4 NEURAL BRAINS ---", True, (139, 148, 158))
        self.screen.blit(section_surf, (self.arena_width + 12, y_offset + 4))

        for i, rect in enumerate(self.get_top_genome_slot_rects()):
            if i < len(self.top_genomes):
                genome = self.top_genomes[i]
                fit_val = getattr(genome, 'fitness', 0.0) or 0.0
                title = f"[#{i+1} ELITE] Fitness: {fit_val:.1f}"
                self._draw_network_graph(genome, rect, title)
            else:
                pygame.draw.rect(self.screen, (15, 18, 24), rect)
                pygame.draw.rect(self.screen, (35, 45, 55), rect, 1)
                ph = self.small_font.render(f"[#{i+1} EMPTY] Awaiting epoch...", True, (80, 90, 100))
                self.screen.blit(ph, (rect.x + 10, rect.y + 35))

    def eval_generation(self, nets, genomes, max_frames: int = 900) -> Dict[str, Any]:
        """Evaluates a single generation of NEAT agents."""
        self.generation += 1
        self._reset_world_entities()

        # Uniform agent spawn distribution around arena center (1280 x 720)
        # Phase 8: Balanced division across 4 equal tribes (precisely 10 per tribe for 40 agents)
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
            # 1. Event handling
            for event in pygame.event.get():
                self.handle_event(event)

            # If Neural Inspector is active – pause physics and evolution, render overlay only
            if self.inspector_active:
                if self.frozen_screen is not None:
                    self.screen.blit(self.frozen_screen, (0, 0))
                self._draw_neural_inspector()
                pygame.display.flip()
                self.clock.tick(60)
                continue

            # 2. Update dynamic hazards
            for hazard in self.hazards:
                hazard.update(self.arena_width, self.height)

            # 3. Update agents
            alive_count = 0
            best_current_fitness = -999999.0

            for agent in agents:
                if agent.is_alive:
                    agent.think_and_act(self.foods, self.poisons, self.hazards, agents, self.arena_width, self.height)
                    if agent.is_alive:
                        alive_count += 1
                if agent.genome.fitness > best_current_fitness:
                    best_current_fitness = agent.genome.fitness

            # Early generation termination if entire population dies
            if alive_count == 0:
                break

            # 4. Render arena and grid
            self._draw_arena_grid()

            # Render food entities (green circles)
            for food in self.foods:
                food.draw(self.screen)

            # Render poison entities (purple squares)
            for poison in self.poisons:
                poison.draw(self.screen)

            # Render roving hazards (red circles)
            for hazard in self.hazards:
                hazard.draw(self.screen)

            # Render agents
            for agent in agents:
                agent.draw(self.screen)

            # 5. Render sidebar and Top-4 visualizer
            self._draw_sidebar(agents, frames_lived, max_frames, alive_count, best_current_fitness)

            pygame.display.flip()

            # Control FPS: 60 FPS normal mode, uncapped in turbo mode
            if self.fast_mode:
                self.clock.tick(0)
            else:
                self.clock.tick(60)

            frames_lived += 1

        # Finalize holistic fitness for agents that survived the entire epoch (M_death = 1.2)
        for agent in agents:
            if agent.is_alive:
                agent.death_cause = "survived"
                agent.finalize_fitness()

        # Preserve Top 4 genomes at generation completion (deepcopies for isolation)
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