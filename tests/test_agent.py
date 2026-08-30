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
    """Testy jednostkowe klasy Agent (Zbalansowany metabolizm, pożywienie i obrona stadna)."""

    def setUp(self):
        self.net = DummyNetwork()
        self.genome = DummyGenome()
        self.agent = Agent(self.net, self.genome, width=800, height=600)

    def test_agent_initialization(self):
        self.assertTrue(self.agent.is_alive)
        self.assertEqual(self.agent.energy, 150.0)
        self.assertEqual(self.agent.max_energy, 150.0)
        self.assertEqual(self.agent.genome.fitness, 0.0)
        self.assertEqual(self.agent.foods_eaten, 0)
        self.assertEqual(self.agent.poisons_hit, 0)
        self.assertEqual(self.agent.allies_saved, 0)
        self.assertEqual(self.agent.attacks_made, 0)
        self.assertEqual(self.agent.defenses_made, 0)
        self.assertEqual(self.agent.herd_defenses, 0)
        self.assertGreaterEqual(self.agent.pos.x, 50)
        self.assertLessEqual(self.agent.pos.x, 750)

    def test_sensory_inputs_structure_and_bounds(self):
        foods = [Food(100, 100), Food(500, 500), Food(300, 300)]
        poisons = [Poison(150, 150)]
        hazards = [Hazard(200, 200)]
        other_agent = Agent(self.net, DummyGenome(), width=800, height=600)
        other_agent.vel = pygame.math.Vector2(2.0, 0.0)
        agents = [self.agent, other_agent]

        inputs = self.agent._get_sensory_inputs(foods, poisons, hazards, agents, 800, 600)

        # Weryfikacja 22 wejść
        self.assertEqual(len(inputs), 22)

        # Weryfikacja zakresów wartości
        for idx, val in enumerate(inputs):
            self.assertGreaterEqual(val, -1.01, f"Input {idx} value {val} is below -1.0")
            self.assertLessEqual(val, 1.01, f"Input {idx} value {val} is above 1.0")

    def test_herd_density_sensing(self):
        ally1 = Agent(self.net, DummyGenome(), width=800, height=600, start_pos=(self.agent.pos.x + 10, self.agent.pos.y))
        ally2 = Agent(self.net, DummyGenome(), width=800, height=600, start_pos=(self.agent.pos.x + 20, self.agent.pos.y))
        ally3 = Agent(self.net, DummyGenome(), width=800, height=600, start_pos=(self.agent.pos.x + 30, self.agent.pos.y))

        inputs_dense = self.agent._get_sensory_inputs([], [], [], [self.agent, ally1, ally2, ally3], 800, 600)
        self.assertGreater(inputs_dense[19], 0.4)

        inputs_lonely = self.agent._get_sensory_inputs([], [], [], [self.agent], 800, 600)
        self.assertEqual(inputs_lonely[19], 0.0)

    def test_sprint_fatigue_metabolism_cost(self):
        # Sprintujący agent
        sprinting_agent = Agent(DummyNetwork(output_x=1.0, output_y=0.0), DummyGenome(), width=800, height=600, start_pos=(200, 200))
        sprinting_agent.vel = pygame.math.Vector2(4.0, 0.0)
        initial_energy_sprint = sprinting_agent.energy
        sprinting_agent.think_and_act([], [], [], [sprinting_agent], 800, 600)
        sprint_loss = initial_energy_sprint - sprinting_agent.energy

        # Wolny agent
        slow_agent = Agent(DummyNetwork(output_x=0.0, output_y=0.0), DummyGenome(), width=800, height=600, start_pos=(200, 200))
        slow_agent.vel = pygame.math.Vector2(0.0, 0.0)
        initial_energy_slow = slow_agent.energy
        slow_agent.think_and_act([], [], [], [slow_agent], 800, 600)
        slow_loss = initial_energy_slow - slow_agent.energy

        # Sprint spala zauważalnie więcej niż powolny ruch
        self.assertGreater(sprint_loss, slow_loss * 2.0)

    def test_herd_defense_repels_predator(self):
        predator = Agent(DummyNetwork(output_x=1.0, output_y=0.0), DummyGenome(), width=800, height=600, start_pos=(195, 200))
        predator.vel = pygame.math.Vector2(3.0, 0.0)
        predator.energy = 80.0

        prey = Agent(DummyNetwork(output_x=0.5, output_y=0.0), DummyGenome(), width=800, height=600, start_pos=(200, 200))
        prey.vel = pygame.math.Vector2(1.0, 0.0)
        prey.energy = 60.0

        ally = Agent(DummyNetwork(output_x=0.0, output_y=0.0), DummyGenome(), width=800, height=600, start_pos=(215, 200))
        ally.vel = pygame.math.Vector2(1.0, 0.0)
        ally.energy = 60.0

        initial_pred_fit = predator.genome.fitness
        initial_prey_fit = prey.genome.fitness
        initial_ally_fit = ally.genome.fitness

        predator.think_and_act([], [], [], [predator, prey, ally], 800, 600)

        # Drapieżnik otrzymuje zbalansowane obrażenia (-15 energii) i karę fitness (-20)
        self.assertLessEqual(predator.energy, 66.0)
        self.assertLess(predator.genome.fitness, initial_pred_fit - 15.0)
        self.assertGreaterEqual(prey.genome.fitness, initial_prey_fit + 14.0)
        self.assertGreaterEqual(ally.genome.fitness, initial_ally_fit + 14.0)
        self.assertEqual(prey.herd_defenses, 1)

    def test_predation_on_isolated_prey(self):
        predator = Agent(DummyNetwork(output_x=1.0, output_y=0.0), DummyGenome(), width=800, height=600, start_pos=(195, 200))
        predator.vel = pygame.math.Vector2(3.0, 0.0)
        predator.energy = 50.0

        prey = Agent(DummyNetwork(output_x=0.5, output_y=0.0), DummyGenome(), width=800, height=600, start_pos=(200, 200))
        prey.vel = pygame.math.Vector2(1.0, 0.0)
        prey.energy = 50.0

        initial_pred_fit = predator.genome.fitness
        initial_prey_fit = prey.genome.fitness

        predator.think_and_act([], [], [], [predator, prey], 800, 600)

        self.assertGreaterEqual(predator.energy, 70.0)
        self.assertLessEqual(prey.energy, 26.0)
        self.assertGreaterEqual(predator.genome.fitness, initial_pred_fit + 24.0)
        self.assertLess(prey.genome.fitness, initial_prey_fit)
        self.assertEqual(predator.attacks_made, 1)

    def test_frontal_defense_clash(self):
        agent_a = Agent(DummyNetwork(output_x=1.0, output_y=0.0), DummyGenome(), width=800, height=600, start_pos=(196, 200))
        agent_a.vel = pygame.math.Vector2(2.0, 0.0)
        agent_a.energy = 80.0

        agent_b = Agent(DummyNetwork(output_x=-1.0, output_y=0.0), DummyGenome(), width=800, height=600, start_pos=(200, 200))
        agent_b.vel = pygame.math.Vector2(-2.0, 0.0)
        agent_b.energy = 40.0

        initial_fit_a = agent_a.genome.fitness

        agent_a.think_and_act([], [], [], [agent_a, agent_b], 800, 600)

        self.assertGreaterEqual(agent_a.genome.fitness, initial_fit_a + 9.0)
        self.assertEqual(agent_a.defenses_made, 1)

    def test_altruism_energy_transfer_and_reward(self):
        agent_a = Agent(DummyNetwork(output_x=0.0, output_y=0.0), DummyGenome(), width=800, height=600, start_pos=(200, 200))
        agent_a.energy = 100.0

        agent_b = Agent(DummyNetwork(output_x=0.0, output_y=0.0), DummyGenome(), width=800, height=600, start_pos=(200, 200))
        agent_b.energy = 10.0

        initial_fitness_a = agent_a.genome.fitness

        agent_a.think_and_act([], [], [], [agent_a, agent_b], 800, 600)

        self.assertAlmostEqual(agent_a.energy, 79.9, places=1)
        self.assertAlmostEqual(agent_b.energy, 30.0, places=1)
        self.assertGreaterEqual(agent_a.genome.fitness, initial_fitness_a + 49.0)
        self.assertEqual(agent_a.allies_saved, 1)

    def test_food_consumption_reward(self):
        food = Food(self.agent.pos.x, self.agent.pos.y)
        foods = [food]
        self.agent.energy = 50.0
        initial_fitness = self.agent.genome.fitness

        self.agent.think_and_act(foods, [], [], [self.agent], 800, 600)

        self.assertEqual(self.agent.foods_eaten, 1)
        # Jabłko odnawia teraz +65 energii
        self.assertGreaterEqual(self.agent.energy, 110.0)
        self.assertGreaterEqual(self.agent.genome.fitness, initial_fitness + 14.0)

    def test_poison_collision_penalty(self):
        poison = Poison(self.agent.pos.x, self.agent.pos.y)
        initial_energy = self.agent.energy
        initial_fitness = self.agent.genome.fitness

        self.agent.think_and_act([], [poison], [], [self.agent], 800, 600)

        self.assertLessEqual(self.agent.energy, initial_energy - 30.0)
        self.assertLessEqual(self.agent.genome.fitness, initial_fitness - 8.0)
        self.assertEqual(self.agent.poisons_hit, 1)

    def test_hazard_collision_penalty(self):
        hazard = Hazard(self.agent.pos.x, self.agent.pos.y)
        initial_energy = self.agent.energy
        initial_fitness = self.agent.genome.fitness

        self.agent.think_and_act([], [], [hazard], [self.agent], 800, 600)

        self.assertLess(self.agent.energy, initial_energy - 15.0)
        self.assertLess(self.agent.genome.fitness, initial_fitness)

    def test_agent_death_on_zero_energy(self):
        self.agent.energy = 0.02
        self.agent.think_and_act([], [], [], [self.agent], 800, 600)
        self.assertFalse(self.agent.is_alive)


if __name__ == '__main__':
    unittest.main()
