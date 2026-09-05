import unittest
import pygame
from src.entities import Food, Hazard, Poison


class TestEntities(unittest.TestCase):
    """Unit tests for environment entities (Food, Hazard, Poison) without requiring a display window."""

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

    def test_food_create_clustered(self):
        """Verifies create_clustered spawns the exact requested number of foods within arena bounds."""
        Food.reset_cluster_state()
        foods = Food.create_clustered(25, 1280, 720, margin=60)
        self.assertEqual(len(foods), 25)
        for f in foods:
            self.assertGreaterEqual(f.pos.x, 60)
            self.assertLessEqual(f.pos.x, 1280 - 60)
            self.assertGreaterEqual(f.pos.y, 60)
            self.assertLessEqual(f.pos.y, 720 - 60)

    def test_food_respawn_clustered(self):
        """Verifies respawn_clustered repositions all existing food instances within boundaries."""
        foods = [Food(0, 0) for _ in range(15)]
        Food.respawn_clustered(foods, 1000, 800, margin=50)
        for f in foods:
            self.assertGreaterEqual(f.pos.x, 50)
            self.assertLessEqual(f.pos.x, 950)
            self.assertGreaterEqual(f.pos.y, 50)
            self.assertLessEqual(f.pos.y, 750)

    def test_food_dynamic_hotspot_clustering(self):
        """Verifies that dynamic food respawn clusters tightly around a defined hotspot."""
        Food.set_hotspot(500.0, 300.0, cluster_size=3)
        food = Food(0, 0)
        # Respawn 3 apples around hotspot (500, 300) with sigma=10
        for _ in range(3):
            food.respawn(1280, 720, margin=30, sigma=10.0)
            self.assertAlmostEqual(food.pos.x, 500.0, delta=40.0)
            self.assertAlmostEqual(food.pos.y, 300.0, delta=40.0)

        # After 3 apples, cluster is exhausted; next respawn picks a new hotspot
        self.assertEqual(Food._cluster_remaining, 0)
        food.respawn(1280, 720, margin=30)
        # New cluster remaining should be between 3 and 5 (since 1 was used)
        self.assertGreaterEqual(Food._cluster_remaining, 3)

    def test_poison_initialization_and_respawn(self):
        poison = Poison(150, 250, size=8.0)
        self.assertEqual(poison.pos.x, 150)
        self.assertEqual(poison.pos.y, 250)
        self.assertEqual(poison.size, 8.0)
        self.assertEqual(poison.radius, 4.0)  # collision radius = size / 2
        self.assertEqual(poison.color, (155, 89, 182))  # purple color

        poison.respawn(800, 600, margin=30)
        self.assertGreaterEqual(poison.pos.x, 30)
        self.assertLessEqual(poison.pos.x, 770)

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
        # After colliding with right wall (x > 800 - 20), velocity x should reverse to negative
        self.assertLess(hazard.vel.x, 0.0)


if __name__ == '__main__':
    unittest.main()
