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

        # Weryfikacja kluczowych założeń projektowych (Faza 3: 21 wejść sensorycznych)
        self.assertEqual(config.genome_config.num_inputs, 21, "Liczba wejść musi wynosić 21.")
        self.assertEqual(config.genome_config.num_outputs, 2, "Liczba wyjść musi wynosić 2.")
        self.assertEqual(config.pop_size, 50, "Wielkość populacji musi wynosić 50.")
        self.assertEqual(config.reproduction_config.elitism, 4, "Elitaryzm musi wynosić 4 (Top 4 bez zmian).")
        self.assertFalse(config.reset_on_extinction)


if __name__ == '__main__':
    unittest.main()
