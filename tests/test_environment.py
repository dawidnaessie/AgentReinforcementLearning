import os
import unittest

# Ustawienie sterownika dummy przed importem pygame w celach testowych
os.environ['SDL_VIDEODRIVER'] = 'dummy'

from src.environment import Environment
from tests.test_agent import DummyGenome, DummyNetwork


class TestEnvironment(unittest.TestCase):
    """Testy jednostkowe środowiska i pętli symulacji w trybie headless."""

    def setUp(self):
        self.env = Environment()

    def test_environment_initialization(self):
        self.assertEqual(self.env.width, 1600)
        self.assertEqual(self.env.height, 720)
        self.assertEqual(self.env.arena_width, 1280)
        self.assertEqual(len(self.env.foods), 50)
        self.assertEqual(len(self.env.poisons), 20)
        self.assertEqual(len(self.env.hazards), 6)
        self.assertEqual(self.env.generation, 0)
        self.assertEqual(len(self.env.top_genomes), 0)

    def test_eval_generation_cycle(self):
        nets = [DummyNetwork() for _ in range(5)]
        genomes = [DummyGenome() for _ in range(5)]
        for i, g in enumerate(genomes):
            g.fitness = float(i * 10)

        # Uruchomienie krótkiej generacji na 20 klatek
        self.env.eval_generation(nets, genomes, max_frames=20)

        self.assertEqual(self.env.generation, 1)
        # Zapisano 4 najlepsze genomy (Top 4)
        self.assertEqual(len(self.env.top_genomes), 4)
        # Wszystkie genomy powinny mieć zaktualizowany fitness (będący liczbą float/int)
        for genome in genomes:
            self.assertIsInstance(genome.fitness, (int, float))


    def test_inspector_initial_state(self):
        """Upewnia się, że inspektor jest domyślnie wyłączony."""
        self.assertFalse(self.env.inspector_active)
        self.assertIsNone(self.env.inspected_genome)

    def test_sensory_and_action_labels(self):
        """Weryfikuje poprawność mapowania 25 wejść i 3 wyjść na czytelne etykiety."""
        from src.environment import SENSORY_INPUT_LABELS, ACTION_OUTPUT_LABELS
        self.assertEqual(len(SENSORY_INPUT_LABELS), 25)
        self.assertEqual(len(ACTION_OUTPUT_LABELS), 3)

        # Sprawdzenie kluczowych wejść z README
        self.assertIn("Vel X", SENSORY_INPUT_LABELS[0])
        self.assertIn("Vel Y", SENSORY_INPUT_LABELS[1])
        self.assertIn("Food", SENSORY_INPUT_LABELS[2])
        self.assertIn("Shout", SENSORY_INPUT_LABELS[22])

        # Sprawdzenie 3 wyjść
        self.assertIn("Accel X", ACTION_OUTPUT_LABELS[0])
        self.assertIn("Accel Y", ACTION_OUTPUT_LABELS[1])
        self.assertIn("Shout", ACTION_OUTPUT_LABELS[2])

    def test_sensory_and_action_details_metadata(self):
        """Weryfikuje obecność pełnych opisów funkcjonalnych, zakresów i ról dla każdego węzła."""
        from src.environment import SENSORY_DETAILS, ACTION_DETAILS
        self.assertEqual(len(SENSORY_DETAILS), 25)
        self.assertEqual(len(ACTION_DETAILS), 3)

        for i, det in SENSORY_DETAILS.items():
            self.assertIn("name", det)
            self.assertIn("desc", det)
            self.assertIn("range", det)
            self.assertIn("role", det)
            self.assertTrue(len(det["desc"]) > 5)

        for i, det in ACTION_DETAILS.items():
            self.assertIn("name", det)
            self.assertIn("desc", det)
            self.assertIn("range", det)
            self.assertIn("role", det)
            self.assertTrue(len(det["desc"]) > 5)

    def test_inspector_tab_toggle(self):
        """Klawisz TAB w aktywnym inspektorze przełącza widok między tylko aktywnymi a wszystkimi zmysłami."""
        import pygame
        self.env.inspector_active = True
        self.assertFalse(self.env.inspector_show_all)

        tab_event = pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_TAB})
        self.env.handle_event(tab_event)
        self.assertTrue(self.env.inspector_show_all)

        self.env.handle_event(tab_event)
        self.assertFalse(self.env.inspector_show_all)

    def test_get_top_slot_rects(self):
        """Weryfikuje poprawność geometrii 4 slotów w panelu bocznym."""
        rects = self.env.get_top_genome_slot_rects()
        self.assertEqual(len(rects), 4)
        for rect in rects:
            self.assertGreaterEqual(rect.x, self.env.arena_width)
            self.assertLess(rect.right, self.env.width)
            self.assertGreaterEqual(rect.y, 0)
            self.assertLessEqual(rect.bottom, self.env.height)

    def test_mouse_click_top_slot_activates_inspector_with_deepcopy(self):
        """Kliknięcie w slot Top 4 powinno aktywować inspektora z kopią genomu (deepcopy)."""
        import pygame
        dummy_g = DummyGenome()
        dummy_g.fitness = 123.4
        dummy_g.custom_marker = [1, 2, 3]
        self.env.top_genomes = [dummy_g]

        rects = self.env.get_top_genome_slot_rects()
        slot_pos = (rects[0].centerx, rects[0].centery)
        click_event = pygame.event.Event(pygame.MOUSEBUTTONDOWN, {'button': 1, 'pos': slot_pos})

        self.env.handle_event(click_event)

        self.assertTrue(self.env.inspector_active)
        self.assertIsNotNone(self.env.inspected_genome)
        # Deepcopy weryfikacja: obiekty nie są tożsame referencyjnie, ale mają te same wartości
        self.assertIsNot(self.env.inspected_genome, dummy_g)
        self.assertEqual(self.env.inspected_genome.fitness, 123.4)
        self.assertEqual(self.env.inspected_genome.custom_marker, [1, 2, 3])
        self.assertIsNot(self.env.inspected_genome.custom_marker, dummy_g.custom_marker)

    def test_mouse_click_empty_slot_does_nothing(self):
        """Kliknięcie w pusty slot nie powinno aktywować inspektora."""
        import pygame
        self.env.top_genomes = []
        rects = self.env.get_top_genome_slot_rects()
        click_event = pygame.event.Event(pygame.MOUSEBUTTONDOWN, {'button': 1, 'pos': rects[0].center})

        self.env.handle_event(click_event)
        self.assertFalse(self.env.inspector_active)
        self.assertIsNone(self.env.inspected_genome)

    def test_esc_closes_inspector_without_exiting(self):
        """Wciśnięcie ESC przy aktywnym inspektorze powinno go zamknąć bez rzucania SimulationExit."""
        import pygame
        from src.environment import SimulationExit

        self.env.inspector_active = True
        self.env.inspected_genome = DummyGenome()

        esc_event = pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_ESCAPE})
        # Nie powinno rzucić wyjątku
        self.env.handle_event(esc_event)

        self.assertFalse(self.env.inspector_active)
        self.assertIsNone(self.env.inspected_genome)

        # Kolejne wciśnięcie ESC (gdy inspektor nieaktywny) powinno zakończyć symulację
        with self.assertRaises(SimulationExit):
            self.env.handle_event(esc_event)

    def test_draw_neural_inspector_headless(self):
        """Weryfikuje renderowanie Inspektora Sieci z węzłami ukrytymi i etykietami wag."""
        class DummyConn:
            def __init__(self, weight: float, enabled: bool = True):
                self.weight = weight
                self.enabled = enabled

        g = DummyGenome()
        g.fitness = 350.5
        g.key = 42
        g.connections = {
            (-1, 0): DummyConn(weight=2.35, enabled=True),    # grube połączenie wejście-wyjście (tekst wagi)
            (-2, 3): DummyConn(weight=-1.80, enabled=True),   # grube połączenie do ukrytego węzła
            (3, 1): DummyConn(weight=0.75, enabled=True),     # ukryty do wyjścia
            (-23, 2): DummyConn(weight=-0.50, enabled=True),  # krzyk do wyjścia shout
            (-5, 0): DummyConn(weight=1.0, enabled=False),    # wyłączone połączenie
        }

        self.env.inspected_genome = g
        # Powinno wyrenderować się bez rzucania błędów
        self.env._draw_neural_inspector()

    def test_top_genomes_stabilization(self):
        """Upewnia się, że Top 4 zawiera genomy z ukończonej epoki i nie jest nadpisywane w trakcie."""
        import unittest.mock
        # Początkowo puste
        self.assertEqual(len(self.env.top_genomes), 0)

        nets_g1 = [DummyNetwork() for _ in range(5)]
        genomes_g1 = [DummyGenome() for _ in range(5)]
        for i, g in enumerate(genomes_g1):
            g.tag = i + 1

        def mock_finalize(agent_self):
            agent_self.genome.fitness = float(getattr(agent_self.genome, 'tag', 0) * 10.0)

        with unittest.mock.patch('src.agent.Agent.finalize_fitness', mock_finalize):
            self.env.eval_generation(nets_g1, genomes_g1, max_frames=5)

        # Po ukończeniu gen 1 mamy 4 najlepsze genomy z gen 1
        self.assertEqual(len(self.env.top_genomes), 4)
        top_fitnesses_g1 = [g.fitness for g in self.env.top_genomes]
        self.assertEqual(top_fitnesses_g1, [50.0, 40.0, 30.0, 20.0])

        # W generacji 2 agenci wchodzą do nowej epoki (gdzie Agent.__init__ resetuje fitness do 0.0)
        # top_genomes musi zachować elitarne wartości z gen 1!
        # Tworzymy agenta z genomem z genomes_g1, aby upewnić się, że Agent.__init__ resetuje fitness
        from src.agent import Agent
        a = Agent(nets_g1[4], genomes_g1[4], 1280, 720)
        self.assertEqual(genomes_g1[4].fitness, 0.0)

        # A w self.env.top_genomes genomy są niezmienne i stabilne:
        self.assertEqual([g.fitness for g in self.env.top_genomes], [50.0, 40.0, 30.0, 20.0])

    def test_simulation_runner_eval_genomes_creates_recurrent_networks(self):
        """Weryfikuje, że SimulationRunner.eval_genomes tworzy instancje RecurrentNetwork dla przekazanych genomów."""
        import neat
        import unittest.mock
        from src.main import SimulationRunner

        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config-feedforward.txt')
        config = neat.config.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path
        )
        population = neat.Population(config)
        sample_genomes = list(population.population.items())[:3]

        runner = SimulationRunner()
        captured_nets = []

        def mock_eval(nets, genomes, **kwargs):
            captured_nets.extend(nets)
            for g in genomes:
                g.fitness = 10.0
            return {'foods_eaten': 5}

        with unittest.mock.patch.object(runner.env, 'eval_generation', side_effect=mock_eval):
            runner.eval_genomes(sample_genomes, config)

        self.assertEqual(len(captured_nets), 3)
        for net in captured_nets:
            self.assertIsInstance(net, neat.nn.RecurrentNetwork, "Sieć utworzona przez SimulationRunner musi być RecurrentNetwork.")
        self.assertEqual(len(runner.tracker.generations_data), 1)
        self.assertGreaterEqual(runner.tracker.generations_data[0]["best_synapses"], 0)
        self.assertEqual(runner.tracker.peak_synapses, runner.tracker.generations_data[0]["best_synapses"])

    def test_deadly_zone_surface_initialization(self):
        """Weryfikuje poprawną inicjalizację Strefy Śmierci (20px czerwona ramka) w Environment (Faza 8)."""
        self.assertEqual(self.env.deadly_margin, 20)
        self.assertIsNotNone(self.env.deadly_zone_surface)
        self.assertEqual(self.env.deadly_zone_surface.get_width(), self.env.arena_width)
        self.assertEqual(self.env.deadly_zone_surface.get_height(), self.env.height)

    def test_eval_generation_balanced_tribes_distribution_40_agents(self):
        """Weryfikuje, że dla 40 agentów generacja tworzy dokładnie po 10 agentów dla każdego z 4 plemion (Faza 8)."""
        import unittest.mock
        captured_agents = []

        def mock_think_and_act(agent_self, foods, poisons, hazards, all_agents, width, height):
            captured_agents.append(agent_self)
            # Kończymy życie agenta natychmiast, by pętla szybko się skończyła
            agent_self.is_alive = False

        nets = [DummyNetwork() for _ in range(40)]
        genomes = [DummyGenome() for _ in range(40)]

        with unittest.mock.patch('src.agent.Agent.think_and_act', mock_think_and_act):
            self.env.eval_generation(nets, genomes, max_frames=2)

        self.assertEqual(len(captured_agents), 40)
        tribe_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        for agent in captured_agents:
            tribe_counts[agent.tribe_id] += 1

        for tid in (1, 2, 3, 4):
            self.assertEqual(
                tribe_counts[tid],
                10,
                f"Plemię {tid} powinno mieć dokładnie 10 agentów, a miało {tribe_counts[tid]}."
            )


if __name__ == '__main__':
    unittest.main()

