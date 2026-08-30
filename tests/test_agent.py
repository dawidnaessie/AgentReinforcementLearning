import unittest
import pygame
from src.agent import Agent
from src.entities import Food, Hazard


class DummyGenome:
    """Prosty obiekt atrapa genomu do testów jednostkowych bez zależności od NEAT."""
    def __init__(self):
        self.fitness = 0.0


class DummyNetwork:
    """Prosta atrapa sieci neuronowej zwracająca stałe wyjścia (ruch dx, dy)."""
    def __init__(self, output_x=1.0, output_y=0.0):
        self.output_x = output_x
        self.output_y = output_y

    def activate(self, inputs):
        return (self.output_x, self.output_y)


class TestAgent(unittest.TestCase):
    """Testy jednostkowe klasy Agent (fizyka, zmysły, energia, kolizje, reward shaping)."""

    def setUp(self):
        self.net = DummyNetwork()
        self.genome = DummyGenome()
        self.agent = Agent(self.net, self.genome, width=800, height=600)

    def test_agent_initialization(self):
        self.assertTrue(self.agent.is_alive)
        self.assertEqual(self.agent.energy, 100.0)
        self.assertEqual(self.agent.genome.fitness, 0.0)
        self.assertEqual(self.agent.foods_eaten, 0)
        self.assertGreaterEqual(self.agent.pos.x, 50)
        self.assertLessEqual(self.agent.pos.x, 750)

    def test_sensory_inputs_structure_and_bounds(self):
        foods = [Food(100, 100), Food(500, 500), Food(300, 300)]
        hazards = [Hazard(200, 200)]
        other_agent = Agent(self.net, DummyGenome(), width=800, height=600)
        agents = [self.agent, other_agent]

        inputs = self.agent._get_sensory_inputs(foods, hazards, agents, 800, 600)

        # Weryfikacja dokładnej liczby 16 wejść
        self.assertEqual(len(inputs), 16)

        # Weryfikacja zakresów wartości
        for idx, val in enumerate(inputs):
            self.assertGreaterEqual(val, -1.01, f"Input {idx} value {val} is below -1.0")
            self.assertLessEqual(val, 1.01, f"Input {idx} value {val} is above 1.0")

    def test_energy_decay_and_movement(self):
        initial_pos_x = self.agent.pos.x
        initial_energy = self.agent.energy

        foods = [Food(0, 0)]  # Daleko od agenta
        hazards = [Hazard(0, 0)]
        agents = [self.agent]

        self.agent.think_and_act(foods, hazards, agents, 800, 600)

        # Agent powinien ruszyć się w prawo i zużyć energię
        self.assertGreater(self.agent.pos.x, initial_pos_x)
        self.assertLess(self.agent.energy, initial_energy)

    def test_reward_shaping_moving_towards_food(self):
        # Umieszczamy jedzenie po prawej stronie agenta, do którego agent (output_x = 1.0) się zbliża
        self.agent.pos = pygame.math.Vector2(200, 300)
        food = Food(300, 300)
        foods = [food]

        initial_fitness = self.agent.genome.fitness
        self.agent.think_and_act(foods, [], [self.agent], 800, 600)

        # Zbliżanie się do jedzenia powinno nagrodzić fitness (reward shaping > 0)
        self.assertGreater(self.agent.genome.fitness, initial_fitness)

    def test_food_consumption_reward(self):
        # Umieszczamy jedzenie w dokładnie tym samym miejscu co agent
        food = Food(self.agent.pos.x, self.agent.pos.y)
        foods = [food]
        hazards = []
        agents = [self.agent]

        self.agent.energy = 50.0
        initial_fitness = self.agent.genome.fitness

        self.agent.think_and_act(foods, hazards, agents, 800, 600)

        # Agent powinien zjeść jedzenie, zwiększyć energię i otrzymać punkty fitness (uwzględniając reward shaping i +15 za jedzenie)
        self.assertEqual(self.agent.foods_eaten, 1)
        self.assertGreater(self.agent.energy, 50.0)
        self.assertGreaterEqual(self.agent.genome.fitness, initial_fitness + 14.0)

    def test_hazard_collision_penalty(self):
        # Umieszczamy zagrożenie w miejscu agenta
        hazard = Hazard(self.agent.pos.x, self.agent.pos.y)
        hazards = [hazard]
        foods = []
        agents = [self.agent]

        initial_energy = self.agent.energy
        initial_fitness = self.agent.genome.fitness

        self.agent.think_and_act(foods, hazards, agents, 800, 600)

        # Agent powinien stracić punkty fitness i energię
        self.assertLess(self.agent.energy, initial_energy - 15.0)
        self.assertLess(self.agent.genome.fitness, initial_fitness)

    def test_agent_death_on_zero_energy(self):
        self.agent.energy = 0.05
        self.agent.think_and_act([], [], [self.agent], 800, 600)
        self.assertFalse(self.agent.is_alive)


if __name__ == '__main__':
    unittest.main()
