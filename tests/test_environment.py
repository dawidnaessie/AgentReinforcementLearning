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


if __name__ == '__main__':
    unittest.main()
