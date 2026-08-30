import unittest
import pygame
from src.entities import Food, Hazard


class TestEntities(unittest.TestCase):
    """Testy jednostkowe encji środowiska (Food, Hazard) bez wymogu renderowania okna."""

    def test_food_initialization(self):
        food = Food(100, 200, radius=4.0)
        self.assertEqual(food.pos.x, 100)
        self.assertEqual(food.pos.y, 200)
        self.assertEqual(food.radius, 4.0)
        self.assertEqual(food.color, (46, 204, 113))

    def test_food_respawn_bounds(self):
        food = Food(100, 200)
        width, height, margin = 800, 600, 30
        for _ in range(50):
            food.respawn(width, height, margin)
            self.assertGreaterEqual(food.pos.x, margin)
            self.assertLessEqual(food.pos.x, width - margin)
            self.assertGreaterEqual(food.pos.y, margin)
            self.assertLessEqual(food.pos.y, height - margin)

    def test_hazard_initialization_and_velocity(self):
        hazard = Hazard(150, 250, radius=12.0)
        self.assertEqual(hazard.pos.x, 150)
        self.assertEqual(hazard.pos.y, 250)
        self.assertEqual(hazard.radius, 12.0)
        self.assertEqual(hazard.color, (231, 76, 60))
        self.assertGreater(hazard.vel.length(), 0.0)

    def test_hazard_movement_and_wall_bouncing(self):
        hazard = Hazard(790, 300, radius=12.0)
        hazard.vel = pygame.math.Vector2(5.0, 0.0)
        hazard.update(800, 600, margin=20)
        # Po zderzeniu z prawą ścianą (x > 800 - 20) prędkość x powinna zmienić znak na ujemny
        self.assertLess(hazard.vel.x, 0.0)


if __name__ == '__main__':
    unittest.main()
