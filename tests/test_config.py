import os
import unittest
# pyrefly: ignore [missing-import]
import neat


class TestNeatConfig(unittest.TestCase):
    """Tests verifying NEAT configuration validity in config-feedforward.txt."""

    def setUp(self):
        tests_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(tests_dir)
        self.config_path = os.path.join(self.project_root, 'config-feedforward.txt')

    def test_config_file_exists(self):
        self.assertTrue(os.path.exists(self.config_path), "config-feedforward.txt file must exist in the project root.")

    def test_neat_config_loading_and_parameters(self):
        config = neat.config.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            self.config_path
        )

        # Verification of key design assumptions (Phase 8: 25 sensory inputs, 3 outputs, pop_size=40)
        self.assertEqual(config.genome_config.num_inputs, 25, "Number of inputs must be 25.")
        self.assertEqual(config.genome_config.num_outputs, 3, "Number of outputs must be 3.")
        self.assertEqual(getattr(config, 'pop_size'), 40, "Population size must be 40 (4 tribes of 10 agents each).")
        self.assertEqual(config.reproduction_config.elitism, 4, "Elitism must be 4 (Top 4 unchanged).")
        self.assertFalse(getattr(config, 'reset_on_extinction'))

        # Phase 6 & 8: Short-Term Memory (RNN) - verification of recurrent architecture and network growth
        self.assertFalse(config.genome_config.feed_forward, "Network architecture must have feed_forward = False for RNN.")
        self.assertAlmostEqual(config.genome_config.node_add_prob, 0.15, places=2, msg="node_add_prob should be 0.15 for RNN node creation.")
        self.assertAlmostEqual(config.genome_config.conn_add_prob, 0.50, places=2)
        self.assertAlmostEqual(config.genome_config.conn_delete_prob, 0.20, places=2)
        self.assertGreater(config.genome_config.conn_add_prob, config.genome_config.conn_delete_prob)

    def test_recurrent_network_creation_and_activation(self):
        """Verifies RecurrentNetwork creation from population and activation of 25 inputs to 3 outputs."""
        config = neat.config.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            self.config_path
        )
        population = neat.Population(config)
        sample_genome = list(population.population.values())[0]

        net = neat.nn.RecurrentNetwork.create(sample_genome, config)
        self.assertIsInstance(net, neat.nn.RecurrentNetwork, "Network must be an instance of neat.nn.RecurrentNetwork.")

        # Activation test for 25 sensory inputs
        inputs = [0.1 * i for i in range(25)]
        outputs = net.activate(inputs)
        self.assertEqual(len(outputs), 3, "RecurrentNetwork must return exactly 3 action outputs.")
        for out in outputs:
            self.assertIsInstance(out, float)

    def test_recurrent_network_memory_persistence(self):
        """Verifies that the recurrent loop preserves internal memory state between steps."""
        from neat.nn.recurrent import RecurrentNetwork
        # Create a recurrent network with a self-exciting loop: output node 0 connected to input -1 (w=1.0) and to itself (w=0.5)
        # Output(t) = In(t)*1.0 + Output(t-1)*0.5
        net = RecurrentNetwork(
            inputs=[-1],
            outputs=[0],
            node_evals=[(0, lambda x: x, sum, 0.0, 1.0, [(-1, 1.0), (0, 0.5)])]
        )

        # Step 1: Provide input impulse 1.0 -> output should be 1.0
        out1 = net.activate([1.0])[0]
        self.assertAlmostEqual(out1, 1.0, places=3)

        # Step 2: Zero input (0.0) -> output due to recurrent memory is 0.5 (previous state * 0.5)
        out2 = net.activate([0.0])[0]
        self.assertAlmostEqual(out2, 0.5, places=3)

        # Step 3: Input still 0.0 -> memory decays to 0.25
        out3 = net.activate([0.0])[0]
        self.assertAlmostEqual(out3, 0.25, places=3)

        # Step 4: Network reset clears memory buffer to 0.0
        net.reset()
        out4 = net.activate([0.0])[0]
        self.assertAlmostEqual(out4, 0.0, places=3)


if __name__ == '__main__':
    unittest.main()
