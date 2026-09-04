import os
import unittest
# pyrefly: ignore [missing-import]
import neat


class TestNeatConfig(unittest.TestCase):
    """Testy weryfikujące poprawność konfiguracji NEAT w pliku config-feedforward.txt."""

    def setUp(self):
        tests_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(tests_dir)
        self.config_path = os.path.join(self.project_root, 'config-feedforward.txt')

    def test_config_file_exists(self):
        self.assertTrue(os.path.exists(self.config_path), "Plik config-feedforward.txt musi istnieć w korzeniu projektu.")

    def test_neat_config_loading_and_parameters(self):
        config = neat.config.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            self.config_path
        )

        # Weryfikacja kluczowych założeń projektowych (Faza 8: 25 wejść sensorycznych, 3 wyjścia, pop_size=40)
        self.assertEqual(config.genome_config.num_inputs, 25, "Liczba wejść musi wynosić 25.")
        self.assertEqual(config.genome_config.num_outputs, 3, "Liczba wyjść musi wynosić 3.")
        self.assertEqual(getattr(config, 'pop_size'), 40, "Wielkość populacji musi wynosić 40 (4 plemiona po 10 agentów).")
        self.assertEqual(config.reproduction_config.elitism, 4, "Elitaryzm musi wynosić 4 (Top 4 bez zmian).")
        self.assertFalse(getattr(config, 'reset_on_extinction'))

        # Faza 6 & 8: Pamięć Krótkotrwała (RNN) - weryfikacja architektury rekurencyjnej i rozrostu sieci
        self.assertFalse(config.genome_config.feed_forward, "Architektura sieci musi mieć feed_forward = False dla RNN.")
        self.assertAlmostEqual(config.genome_config.node_add_prob, 0.15, places=2, msg="node_add_prob powinno wynosić 0.15 dla tworzenia węzłów RNN.")
        self.assertAlmostEqual(config.genome_config.conn_add_prob, 0.50, places=2)
        self.assertAlmostEqual(config.genome_config.conn_delete_prob, 0.20, places=2)
        self.assertGreater(config.genome_config.conn_add_prob, config.genome_config.conn_delete_prob)

    def test_recurrent_network_creation_and_activation(self):
        """Weryfikuje tworzenie sieci RecurrentNetwork z populacji i aktywację 25 wejść na 3 wyjścia."""
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
        self.assertIsInstance(net, neat.nn.RecurrentNetwork, "Sieć musi być instancją neat.nn.RecurrentNetwork.")

        # Test aktywacji dla 25 wejść sensorycznych
        inputs = [0.1 * i for i in range(25)]
        outputs = net.activate(inputs)
        self.assertEqual(len(outputs), 3, "RecurrentNetwork musi zwracać dokładnie 3 wyjścia akcji.")
        for out in outputs:
            self.assertIsInstance(out, float)

    def test_recurrent_network_memory_persistence(self):
        """Weryfikuje, że pętla rekurencyjna zachowuje stan wewnętrzny w pamięci między krokami."""
        from neat.nn.recurrent import RecurrentNetwork
        # Tworzymy sieć rekurencyjną z samowzbudzającą się pętlą: węzeł wyjściowy 0 połączony z wejściem -1 (w=1.0) oraz ze sobą (w=0.5)
        # Wyjście(t) = In(t)*1.0 + Wyjście(t-1)*0.5
        net = RecurrentNetwork(
            inputs=[-1],
            outputs=[0],
            node_evals=[(0, lambda x: x, sum, 0.0, 1.0, [(-1, 1.0), (0, 0.5)])]
        )

        # Krok 1: Podajemy impuls wejściowy 1.0 -> wyjście powinno wynosić 1.0
        out1 = net.activate([1.0])[0]
        self.assertAlmostEqual(out1, 1.0, places=3)

        # Krok 2: Zerujemy wejście (0.0) -> wyjście dzięki pamięci rekurencyjnej wynosi 0.5 (stan poprzedni * 0.5)
        out2 = net.activate([0.0])[0]
        self.assertAlmostEqual(out2, 0.5, places=3)

        # Krok 3: Wejście nadal 0.0 -> pamięć zanika do 0.25
        out3 = net.activate([0.0])[0]
        self.assertAlmostEqual(out3, 0.25, places=3)

        # Krok 4: Reset sieci czyści pamięć podręczną do 0.0
        net.reset()
        out4 = net.activate([0.0])[0]
        self.assertAlmostEqual(out4, 0.0, places=3)


if __name__ == '__main__':
    unittest.main()
