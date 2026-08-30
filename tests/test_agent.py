import unittest
import pygame
from src.agent import Agent
from src.entities import Food, Hazard, Poison


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
    """Testy jednostkowe klasy Agent (Faza 2: fizyka, zmysły 20D, trucizna, altruizm)."""

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
        poisons = [Poison(150, 150)]
        hazards = [Hazard(200, 200)]
        other_agent = Agent(self.net, DummyGenome(), width=800, height=600)
        agents = [self.agent, other_agent]

        inputs = self.agent._get_sensory_inputs(foods, poisons, hazards, agents, 800, 600)

        # Weryfikacja dokładnej liczby 20 wejść
        self.assertEqual(len(inputs), 20)

        # Weryfikacja zakresów wartości
        for idx, val in enumerate(inputs):
            self.assertGreaterEqual(val, -1.01, f"Input {idx} value {val} is below -1.0")
            self.assertLessEqual(val, 1.01, f"Input {idx} value {val} is above 1.0")

    def test_ally_critical_state_sensing(self):
        foods = [Food(100, 100)]
        poisons = []
        hazards = []

        # Tworzymy sojusznika w stanie krytycznym (energia < 20%)
        ally_critical = Agent(self.net, DummyGenome(), width=800, height=600, start_pos=(self.agent.pos.x + 20, self.agent.pos.y))
        ally_critical.energy = 15.0  # < 20.0

        inputs_crit = self.agent._get_sensory_inputs(foods, poisons, hazards, [self.agent, ally_critical], 800, 600)
        # Indeks 17 to norm_agent_critical (18. wejście)
        self.assertEqual(inputs_crit[17], 1.0)

        # Sojusznik w stanie zdrowym (energia >= 20%)
        ally_healthy = Agent(self.net, DummyGenome(), width=800, height=600, start_pos=(self.agent.pos.x + 20, self.agent.pos.y))
        ally_healthy.energy = 60.0
        inputs_healthy = self.agent._get_sensory_inputs(foods, poisons, hazards, [self.agent, ally_healthy], 800, 600)
        self.assertEqual(inputs_healthy[17], 0.0)

    def test_energy_decay_and_movement(self):
        initial_pos_x = self.agent.pos.x
        initial_energy = self.agent.energy

        foods = [Food(0, 0)]  # Daleko od agenta
        poisons = []
        hazards = []
        agents = [self.agent]

        self.agent.think_and_act(foods, poisons, hazards, agents, 800, 600)

        # Agent powinien ruszyć się w prawo i zużyć energię
        self.assertGreater(self.agent.pos.x, initial_pos_x)
        self.assertLess(self.agent.energy, initial_energy)

    def test_reward_shaping_moving_towards_food(self):
        self.agent.pos = pygame.math.Vector2(200, 300)
        food = Food(300, 300)
        foods = [food]

        initial_fitness = self.agent.genome.fitness
        self.agent.think_and_act(foods, [], [], [self.agent], 800, 600)

        # Zbliżanie się do jedzenia powinno nagrodzić fitness (reward shaping > 0)
        self.assertGreater(self.agent.genome.fitness, initial_fitness)

    def test_food_consumption_reward(self):
        food = Food(self.agent.pos.x, self.agent.pos.y)
        foods = [food]
        poisons = []
        hazards = []
        agents = [self.agent]

        self.agent.energy = 50.0
        initial_fitness = self.agent.genome.fitness

        self.agent.think_and_act(foods, poisons, hazards, agents, 800, 600)

        # Agent powinien zjeść jedzenie, zwiększyć energię i otrzymać punkty fitness
        self.assertEqual(self.agent.foods_eaten, 1)
        self.assertGreater(self.agent.energy, 50.0)
        self.assertGreaterEqual(self.agent.genome.fitness, initial_fitness + 14.0)

    def test_poison_collision_penalty(self):
        poison = Poison(self.agent.pos.x, self.agent.pos.y)
        foods = []
        poisons = [poison]
        hazards = []
        agents = [self.agent]

        initial_energy = self.agent.energy
        initial_fitness = self.agent.genome.fitness

        self.agent.think_and_act(foods, poisons, hazards, agents, 800, 600)

        # Agent powinien stracić energię (np. -35) oraz otrzymać karę fitness (-10)
        self.assertLessEqual(self.agent.energy, initial_energy - 30.0)
        self.assertLessEqual(self.agent.genome.fitness, initial_fitness - 8.0)
        self.assertEqual(self.agent.poisons_hit, 1)

    def test_hazard_collision_penalty(self):
        hazard = Hazard(self.agent.pos.x, self.agent.pos.y)
        foods = []
        poisons = []
        hazards = [hazard]
        agents = [self.agent]

        initial_energy = self.agent.energy
        initial_fitness = self.agent.genome.fitness

        self.agent.think_and_act(foods, poisons, hazards, agents, 800, 600)

        # Agent powinien stracić punkty fitness i energię
        self.assertLess(self.agent.energy, initial_energy - 15.0)
        self.assertLess(self.agent.genome.fitness, initial_fitness)

    def test_altruism_energy_transfer_and_reward(self):
        # Dawca A ma dużo energii (> 50%)
        agent_a = Agent(DummyNetwork(output_x=0.0, output_y=0.0), DummyGenome(), width=800, height=600, start_pos=(200, 200))
        agent_a.energy = 80.0

        # Biorca B w stanie krytycznym (< 20%) w tej samej lokalizacji
        agent_b = Agent(DummyNetwork(output_x=0.0, output_y=0.0), DummyGenome(), width=800, height=600, start_pos=(200, 200))
        agent_b.energy = 10.0

        initial_fitness_a = agent_a.genome.fitness

        # Wykonanie akcji przez dawcę A
        agent_a.think_and_act([], [], [], [agent_a, agent_b], 800, 600)

        # Sprawdzenie transferu energii (A traci 20, B zyskuje 20)
        self.assertAlmostEqual(agent_a.energy, 59.9, places=2)
        self.assertAlmostEqual(agent_b.energy, 30.0, places=2)
        # Dawca A otrzymuje potężną nagrodę altruizmu +50.0 i inkrementuje licznik uratowanych
        self.assertGreaterEqual(agent_a.genome.fitness, initial_fitness_a + 49.0)
        self.assertEqual(agent_a.allies_saved, 1)

    def test_agent_death_on_zero_energy(self):
        self.agent.energy = 0.05
        self.agent.think_and_act([], [], [], [self.agent], 800, 600)
        self.assertFalse(self.agent.is_alive)


if __name__ == '__main__':
    unittest.main()
