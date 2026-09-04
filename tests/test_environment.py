import os
import tempfile
import unittest

# Set dummy video driver before importing pygame for headless test execution
os.environ['SDL_VIDEODRIVER'] = 'dummy'

from src.environment import Environment
from tests.test_agent import DummyGenome, DummyNetwork


class TestEnvironment(unittest.TestCase):
    """Unit tests for environment and simulation loop in headless mode."""

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

        # Run a short generation of 20 frames
        self.env.eval_generation(nets, genomes, max_frames=20)

        self.assertEqual(self.env.generation, 1)
        # 4 best genomes saved (Top 4)
        self.assertEqual(len(self.env.top_genomes), 4)
        # All genomes should have updated fitness (float or int)
        for genome in genomes:
            self.assertIsInstance(genome.fitness, (int, float))


    def test_inspector_initial_state(self):
        """Ensures that the inspector is disabled by default."""
        self.assertFalse(self.env.inspector_active)
        self.assertIsNone(self.env.inspected_genome)

    def test_sensory_and_action_labels(self):
        """Verifies correct mapping of 25 inputs and 3 outputs to human-readable labels."""
        from src.environment import SENSORY_INPUT_LABELS, ACTION_OUTPUT_LABELS
        self.assertEqual(len(SENSORY_INPUT_LABELS), 25)
        self.assertEqual(len(ACTION_OUTPUT_LABELS), 3)

        # Check key inputs from README
        self.assertIn("Vel X", SENSORY_INPUT_LABELS[0])
        self.assertIn("Vel Y", SENSORY_INPUT_LABELS[1])
        self.assertIn("Food", SENSORY_INPUT_LABELS[2])
        self.assertIn("Shout", SENSORY_INPUT_LABELS[22])

        # Check 3 outputs
        self.assertIn("Accel X", ACTION_OUTPUT_LABELS[0])
        self.assertIn("Accel Y", ACTION_OUTPUT_LABELS[1])
        self.assertIn("Shout", ACTION_OUTPUT_LABELS[2])

    def test_sensory_and_action_details_metadata(self):
        """Verifies presence of complete functional descriptions, ranges, and roles for each node."""
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
        """TAB key in active inspector toggles view between only active and all senses."""
        import pygame
        self.env.inspector_active = True
        self.assertFalse(self.env.inspector_show_all)

        tab_event = pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_TAB})
        self.env.handle_event(tab_event)
        self.assertTrue(self.env.inspector_show_all)

        self.env.handle_event(tab_event)
        self.assertFalse(self.env.inspector_show_all)

    def test_get_top_slot_rects(self):
        """Verifies geometry correctness of 4 slots in the side panel."""
        rects = self.env.get_top_genome_slot_rects()
        self.assertEqual(len(rects), 4)
        for rect in rects:
            self.assertGreaterEqual(rect.x, self.env.arena_width)
            self.assertLess(rect.right, self.env.width)
            self.assertGreaterEqual(rect.y, 0)
            self.assertLessEqual(rect.bottom, self.env.height)

    def test_mouse_click_top_slot_activates_inspector_with_deepcopy(self):
        """Clicking on a Top 4 slot should activate inspector with a genome copy (deepcopy)."""
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
        # Deepcopy verification: objects are not referentially identical, but hold equivalent values
        self.assertIsNot(self.env.inspected_genome, dummy_g)
        self.assertEqual(self.env.inspected_genome.fitness, 123.4)
        self.assertEqual(self.env.inspected_genome.custom_marker, [1, 2, 3])
        self.assertIsNot(self.env.inspected_genome.custom_marker, dummy_g.custom_marker)

    def test_mouse_click_empty_slot_does_nothing(self):
        """Clicking on an empty slot should not activate the inspector."""
        import pygame
        self.env.top_genomes = []
        rects = self.env.get_top_genome_slot_rects()
        click_event = pygame.event.Event(pygame.MOUSEBUTTONDOWN, {'button': 1, 'pos': rects[0].center})

        self.env.handle_event(click_event)
        self.assertFalse(self.env.inspector_active)
        self.assertIsNone(self.env.inspected_genome)

    def test_esc_closes_inspector_without_exiting(self):
        """Pressing ESC with active inspector should close it without raising SimulationExit."""
        import pygame
        from src.environment import SimulationExit

        self.env.inspector_active = True
        self.env.inspected_genome = DummyGenome()

        esc_event = pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_ESCAPE})
        # Should not raise an exception
        self.env.handle_event(esc_event)

        self.assertFalse(self.env.inspector_active)
        self.assertIsNone(self.env.inspected_genome)

        # Subsequent ESC press (when inspector is inactive) should terminate simulation
        with self.assertRaises(SimulationExit):
            self.env.handle_event(esc_event)

    def test_draw_neural_inspector_headless(self):
        """Verifies rendering of Neural Inspector with hidden nodes and weight labels."""
        class DummyConn:
            def __init__(self, weight: float, enabled: bool = True):
                self.weight = weight
                self.enabled = enabled

        g = DummyGenome()
        g.fitness = 350.5
        g.key = 42
        g.connections = {
            (-1, 0): DummyConn(weight=2.35, enabled=True),    # thick input-output connection (weight label)
            (-2, 3): DummyConn(weight=-1.80, enabled=True),   # thick connection to hidden node
            (3, 1): DummyConn(weight=0.75, enabled=True),     # hidden to output
            (-23, 2): DummyConn(weight=-0.50, enabled=True),  # shout to shout output
            (-5, 0): DummyConn(weight=1.0, enabled=False),    # disabled connection
        }

        self.env.inspected_genome = g
        # Should render without raising errors
        self.env._draw_neural_inspector()

    def test_top_genomes_stabilization(self):
        """Ensures that Top 4 contains genomes from the completed epoch and is not overwritten mid-generation."""
        import unittest.mock
        # Initially empty
        self.assertEqual(len(self.env.top_genomes), 0)

        nets_g1 = [DummyNetwork() for _ in range(5)]
        genomes_g1 = [DummyGenome() for _ in range(5)]
        for i, g in enumerate(genomes_g1):
            g.tag = i + 1

        def mock_finalize(agent_self):
            agent_self.genome.fitness = float(getattr(agent_self.genome, 'tag', 0) * 10.0)

        with unittest.mock.patch('src.agent.Agent.finalize_fitness', mock_finalize):
            self.env.eval_generation(nets_g1, genomes_g1, max_frames=5)

        # After completing gen 1 we have 4 best genomes from gen 1
        self.assertEqual(len(self.env.top_genomes), 4)
        top_fitnesses_g1 = [g.fitness for g in self.env.top_genomes]
        self.assertEqual(top_fitnesses_g1, [50.0, 40.0, 30.0, 20.0])

        # In generation 2 agents enter a new epoch (where Agent.__init__ resets fitness to 0.0)
        # top_genomes must retain the elite values from gen 1!
        # Create an agent with genome from genomes_g1 to verify Agent.__init__ resets fitness
        from src.agent import Agent
        a = Agent(nets_g1[4], genomes_g1[4], 1280, 720)
        self.assertEqual(genomes_g1[4].fitness, 0.0)

        # And in self.env.top_genomes genomes remain immutable and stable:
        self.assertEqual([g.fitness for g in self.env.top_genomes], [50.0, 40.0, 30.0, 20.0])

    def test_simulation_runner_eval_genomes_creates_recurrent_networks(self):
        """Verifies that SimulationRunner.eval_genomes creates RecurrentNetwork instances for supplied genomes."""
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
            self.assertIsInstance(net, neat.nn.RecurrentNetwork, "Network created by SimulationRunner must be a RecurrentNetwork.")
        self.assertEqual(len(runner.tracker.generations_data), 1)
        self.assertGreaterEqual(runner.tracker.generations_data[0]["best_synapses"], 0)
        self.assertEqual(runner.tracker.peak_synapses, runner.tracker.generations_data[0]["best_synapses"])

    def test_deadly_zone_surface_initialization(self):
        """Verifies proper initialization of Deadly Zone (20px red border) in Environment (Phase 8)."""
        self.assertEqual(self.env.deadly_margin, 20)
        self.assertIsNotNone(self.env.deadly_zone_surface)
        self.assertEqual(self.env.deadly_zone_surface.get_width(), self.env.arena_width)
        self.assertEqual(self.env.deadly_zone_surface.get_height(), self.env.height)

    def test_eval_generation_balanced_tribes_distribution_40_agents(self):
        """Verifies that for 40 agents, generation spawns exactly 10 agents for each of the 4 tribes (Phase 8)."""
        import unittest.mock
        captured_agents = []

        def mock_think_and_act(agent_self, foods, poisons, hazards, all_agents, width, height):
            captured_agents.append(agent_self)
            # End agent lifetime immediately so loop terminates quickly
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
                f"Tribe {tid} should have exactly 10 agents, but had {tribe_counts[tid]}."
            )

    def test_export_brain_to_txt_file_creation_and_topology(self):
        """Verifies export_brain_to_txt generates readable mathematical topology dump in logs/."""
        from src.environment import export_brain_to_txt

        class DummyConn:
            def __init__(self, weight: float, enabled: bool = True):
                self.weight = weight
                self.enabled = enabled

        class DummyNodeGene:
            def __init__(self, key: int, bias: float, activation: str = 'tanh'):
                self.key = key
                self.bias = bias
                self.activation = activation

        genome = DummyGenome()
        genome.key = 77
        genome.fitness = 450.25
        genome.nodes = {
            3: DummyNodeGene(key=3, bias=0.5, activation='tanh'),
            4: DummyNodeGene(key=4, bias=-1.25, activation='relu'),
            0: DummyNodeGene(key=0, bias=0.0, activation='tanh'),  # Output node 0 (must NOT be in hidden nodes)
        }
        genome.connections = {
            (-1, 0): DummyConn(weight=2.5, enabled=True),
            (-2, 3): DummyConn(weight=-1.75, enabled=True),
            (3, 1): DummyConn(weight=0.8, enabled=True),
            (-25, 2): DummyConn(weight=-0.3, enabled=True),
            (-5, 0): DummyConn(weight=1.1, enabled=False),
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = export_brain_to_txt(genome, logs_dir=tmp_dir)
            expected_path = os.path.join(tmp_dir, "brain_id_77.txt")
            self.assertEqual(out_path, expected_path)
            self.assertTrue(os.path.exists(expected_path))

            with open(expected_path, "r", encoding="utf-8") as f:
                content = f.read()

            self.assertIn("--- GENERAL INFO ---", content)
            self.assertIn("Genome ID: 77", content)
            self.assertIn("Fitness: 450.25", content)

            self.assertIn("--- NODES ---", content)
            self.assertIn("Node ID: 3 | Activation: tanh | Bias: 0.5000", content)
            self.assertIn("Node ID: 4 | Activation: relu | Bias: -1.2500", content)
            self.assertNotIn("Node ID: 0", content)

            self.assertIn("--- SYNAPSES (CONNECTIONS) ---", content)
            self.assertIn("[Velocity (Vel X)] -> [Acceleration (Accel X)] | Weight: 2.5000 | Status: Enabled", content)
            self.assertIn("[Velocity (Vel Y)] -> [Node 3] | Weight: -1.7500 | Status: Enabled", content)
            self.assertIn("[Node 3] -> [Acceleration (Accel Y)] | Weight: 0.8000 | Status: Enabled", content)
            self.assertIn("[Nearest Shout Dir Y] -> [Acoustic Shout (Communication)] | Weight: -0.3000 | Status: Enabled", content)
            self.assertIn("[Nearest Food #1 Dir Y] -> [Acceleration (Accel X)] | Weight: 1.1000 | Status: Disabled", content)

    def test_environment_key_s_triggers_brain_dump(self):
        """Verifies pressing [S] key in active Neural Inspector triggers brain dump export and UI feedback."""
        import pygame
        from unittest.mock import patch

        g = DummyGenome()
        g.key = 88
        g.fitness = 120.0
        self.env.inspector_active = True
        self.env.inspected_genome = g

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch('src.environment.export_brain_to_txt', wraps=lambda genome, logs_dir="logs": os.path.join(tmp_dir, f"brain_id_{genome.key}.txt")):
                s_event = pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_s})
                self.env.handle_event(s_event)

                self.assertIsNotNone(self.env.brain_dump_feedback)
                self.assertIn("brain_id_88.txt", self.env.brain_dump_feedback)
                self.assertGreater(self.env.brain_dump_feedback_timer, 0)

    def test_environment_key_s_ignored_when_inspector_inactive(self):
        """Verifies pressing [S] key when Neural Inspector is inactive does not trigger brain dump."""
        import pygame

        self.env.inspector_active = False
        self.env.inspected_genome = DummyGenome()
        self.env.brain_dump_feedback = None

        s_event = pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_s})
        self.env.handle_event(s_event)

        self.assertIsNone(self.env.brain_dump_feedback)

    def test_export_brain_to_txt_empty_nodes_and_connections(self):
        """Verifies clean export formatting for genomes with no hidden nodes or connections."""
        from src.environment import export_brain_to_txt

        g = DummyGenome()
        g.key = 99
        g.fitness = 0.0
        g.nodes = {}
        g.connections = {}

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = export_brain_to_txt(g, logs_dir=tmp_dir)
            with open(out_path, "r", encoding="utf-8") as f:
                content = f.read()

            self.assertIn("Genome ID: 99", content)
            self.assertIn("No hidden nodes", content)
            self.assertIn("No connections", content)


if __name__ == '__main__':
    unittest.main()

