import unittest
import pygame
from src.agent import Agent
from src.entities import Food, Hazard, Poison


class DummyGenome:
    """Prosty obiekt atrapa genomu do testów jednostkowych bez zależności od NEAT."""
    def __init__(self):
        self.fitness = 0.0


class DummyNetwork:
    """Prosta atrapa sieci neuronowej zwracająca 3 wyjścia (ruch dx, dy oraz krzyk shout)."""
    def __init__(self, output_x=1.0, output_y=0.0, output_shout=0.0):
        self.output_x = output_x
        self.output_y = output_y
        self.output_shout = output_shout

    def activate(self, inputs):
        return (self.output_x, self.output_y, self.output_shout)


class TestAgent(unittest.TestCase):
    """Testy jednostkowe klasy Agent (Faza 5: Komunikacja, krzyk, słuch, 25 wejść, 3 wyjścia)."""

    def setUp(self):
        self.net = DummyNetwork()
        self.genome = DummyGenome()
        self.agent = Agent(self.net, self.genome, width=1280, height=720)

    def test_agent_initialization(self):
        self.assertTrue(self.agent.is_alive)
        self.assertEqual(self.agent.energy, 150.0)
        self.assertEqual(self.agent.max_energy, 150.0)
        self.assertEqual(self.agent.frames_alive, 0)
        self.assertFalse(self.agent.is_shouting)
        self.assertEqual(self.agent.genome.fitness, 0.0)
        self.assertEqual(self.agent.foods_eaten, 0)
        self.assertEqual(self.agent.poisons_hit, 0)
        self.assertEqual(self.agent.allies_saved, 0)
        self.assertEqual(self.agent.attacks_made, 0)
        self.assertEqual(self.agent.defenses_made, 0)
        self.assertEqual(self.agent.herd_defenses, 0)

    def test_shout_activation_and_deactivation(self):
        # Sieć z sygnałem krzyku > 0.0
        shouting_net = DummyNetwork(output_x=0.0, output_y=0.0, output_shout=0.8)
        shouter = Agent(shouting_net, DummyGenome(), width=1280, height=720, start_pos=(400, 300))
        shouter.think_and_act([], [], [], [shouter], 1280, 720)
        self.assertTrue(shouter.is_shouting)

        # Sieć z sygnałem krzyku <= 0.0
        silent_net = DummyNetwork(output_x=0.0, output_y=0.0, output_shout=-0.5)
        silent_agent = Agent(silent_net, DummyGenome(), width=1280, height=720, start_pos=(400, 300))
        silent_agent.is_shouting = True
        silent_agent.think_and_act([], [], [], [silent_agent], 1280, 720)
        self.assertFalse(silent_agent.is_shouting)

    def test_hearing_sensors(self):
        # 1. Przypadek, gdy w pobliżu nikt nie krzyczy
        silent_peer = Agent(DummyNetwork(output_shout=-1.0), DummyGenome(), width=1280, height=720, start_pos=(500, 300))
        silent_peer.is_shouting = False
        inputs_quiet = self.agent._get_sensory_inputs([], [], [], [self.agent, silent_peer], 1280, 720)
        # Sensory #23, #24, #25 powinny mieć 0.0
        self.assertEqual(inputs_quiet[22], 0.0)
        self.assertEqual(inputs_quiet[23], 0.0)
        self.assertEqual(inputs_quiet[24], 0.0)

        # 2. Przypadek, gdy inny agent krzyczy w odległości 100px na prawo
        shouting_peer = Agent(DummyNetwork(output_shout=1.0), DummyGenome(), width=1280, height=720, start_pos=(self.agent.pos.x + 100, self.agent.pos.y))
        shouting_peer.is_shouting = True
        inputs_heard = self.agent._get_sensory_inputs([], [], [], [self.agent, shouting_peer], 1280, 720)
        # Dystans znormalizowany > 0.0 oraz kierunek dx bliski 1.0, dy bliski 0.0
        self.assertGreater(inputs_heard[22], 0.0)
        self.assertAlmostEqual(inputs_heard[23], 1.0, places=1)
        self.assertAlmostEqual(inputs_heard[24], 0.0, places=1)

    def test_grace_period_no_combat_or_edge_penalty(self):
        # 1. Brak kary krawędziowej w czasie Grace Period (< 60 klatek / 1 sekunda)
        ghost_agent = Agent(DummyNetwork(output_x=0.0, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(30, 300))
        ghost_agent.vel = pygame.math.Vector2(0.0, 0.0)
        ghost_agent.frames_alive = 20  # < 60
        initial_energy = ghost_agent.energy
        initial_fit = ghost_agent.genome.fitness

        ghost_agent.think_and_act([], [], [], [ghost_agent], 1280, 720)

        # Traci tylko bazowy metabolizm (0.20), brak kary strefy (-0.5) i brak kary fitness (-0.1)
        self.assertAlmostEqual(initial_energy - ghost_agent.energy, 0.20, places=2)
        self.assertAlmostEqual(ghost_agent.genome.fitness - initial_fit, 0.03, places=2)

        # 2. Brak możliwości ataku i kradzieży energii w czasie Grace Period
        predator = Agent(DummyNetwork(output_x=1.0, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(195, 200))
        predator.vel = pygame.math.Vector2(3.0, 0.0)
        predator.frames_alive = 20

        prey = Agent(DummyNetwork(output_x=0.5, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(200, 200))
        prey.vel = pygame.math.Vector2(1.0, 0.0)
        prey.frames_alive = 20
        initial_prey_energy = prey.energy

        predator.think_and_act([], [], [], [predator, prey], 1280, 720)

        self.assertEqual(predator.attacks_made, 0)
        self.assertEqual(prey.energy, initial_prey_energy)

    def test_strict_hunger_metabolism(self):
        still_agent = Agent(DummyNetwork(output_x=0.0, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(400, 300))
        still_agent.vel = pygame.math.Vector2(0.0, 0.0)
        still_agent.frames_alive = 100  # Poza grace period
        initial_energy = still_agent.energy

        still_agent.think_and_act([], [], [], [still_agent], 1280, 720)

        energy_loss = initial_energy - still_agent.energy
        self.assertAlmostEqual(energy_loss, 0.20, places=2)

    def test_toxic_edge_penalty_after_grace_period(self):
        edge_agent = Agent(DummyNetwork(output_x=0.0, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(40, 300))
        edge_agent.vel = pygame.math.Vector2(0.0, 0.0)
        edge_agent.frames_alive = 100  # Poza grace period
        initial_energy = edge_agent.energy
        initial_fitness = edge_agent.genome.fitness

        edge_agent.think_and_act([], [], [], [edge_agent], 1280, 720)

        energy_loss = initial_energy - edge_agent.energy
        self.assertAlmostEqual(energy_loss, 0.70, places=2)
        self.assertAlmostEqual(edge_agent.genome.fitness - initial_fitness, -0.07, places=2)

    def test_safe_zone_no_edge_penalty(self):
        safe_agent = Agent(DummyNetwork(output_x=0.0, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(400, 300))
        safe_agent.vel = pygame.math.Vector2(0.0, 0.0)
        safe_agent.frames_alive = 100
        initial_fitness = safe_agent.genome.fitness

        safe_agent.think_and_act([], [], [], [safe_agent], 1280, 720)

        self.assertAlmostEqual(safe_agent.genome.fitness - initial_fitness, 0.03, places=2)

    def test_sensory_inputs_structure_and_bounds(self):
        foods = [Food(100, 100), Food(500, 500), Food(300, 300)]
        poisons = [Poison(150, 150)]
        hazards = [Hazard(200, 200)]
        other_agent = Agent(self.net, DummyGenome(), width=1280, height=720)
        other_agent.vel = pygame.math.Vector2(2.0, 0.0)
        agents = [self.agent, other_agent]

        inputs = self.agent._get_sensory_inputs(foods, poisons, hazards, agents, 1280, 720)

        # Faza 5: dokładnie 25 wejść sensorycznych
        self.assertEqual(len(inputs), 25)
        for idx, val in enumerate(inputs):
            self.assertGreaterEqual(val, -1.01, f"Input {idx} value {val} is below -1.0")
            self.assertLessEqual(val, 1.01, f"Input {idx} value {val} is above 1.0")

    def test_herd_density_sensing(self):
        ally1 = Agent(self.net, DummyGenome(), width=1280, height=720, start_pos=(self.agent.pos.x + 10, self.agent.pos.y))
        ally2 = Agent(self.net, DummyGenome(), width=1280, height=720, start_pos=(self.agent.pos.x + 20, self.agent.pos.y))
        ally3 = Agent(self.net, DummyGenome(), width=1280, height=720, start_pos=(self.agent.pos.x + 30, self.agent.pos.y))

        inputs_dense = self.agent._get_sensory_inputs([], [], [], [self.agent, ally1, ally2, ally3], 1280, 720)
        self.assertGreater(inputs_dense[19], 0.4)

        inputs_lonely = self.agent._get_sensory_inputs([], [], [], [self.agent], 1280, 720)
        self.assertEqual(inputs_lonely[19], 0.0)

    def test_sprint_fatigue_metabolism_cost(self):
        sprinting_agent = Agent(DummyNetwork(output_x=1.0, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(200, 200))
        sprinting_agent.vel = pygame.math.Vector2(4.0, 0.0)
        initial_energy_sprint = sprinting_agent.energy
        sprinting_agent.think_and_act([], [], [], [sprinting_agent], 1280, 720)
        sprint_loss = initial_energy_sprint - sprinting_agent.energy

        slow_agent = Agent(DummyNetwork(output_x=0.0, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(200, 200))
        slow_agent.vel = pygame.math.Vector2(0.0, 0.0)
        initial_energy_slow = slow_agent.energy
        slow_agent.think_and_act([], [], [], [slow_agent], 1280, 720)
        slow_loss = initial_energy_slow - slow_agent.energy

        self.assertGreater(sprint_loss, slow_loss)

    def test_herd_defense_repels_predator(self):
        predator = Agent(DummyNetwork(output_x=1.0, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(195, 200))
        predator.vel = pygame.math.Vector2(3.0, 0.0)
        predator.energy = 80.0
        predator.frames_alive = 100

        prey = Agent(DummyNetwork(output_x=0.5, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(200, 200))
        prey.vel = pygame.math.Vector2(1.0, 0.0)
        prey.energy = 60.0
        prey.frames_alive = 100

        ally = Agent(DummyNetwork(output_x=0.0, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(215, 200))
        ally.vel = pygame.math.Vector2(1.0, 0.0)
        ally.energy = 60.0
        ally.frames_alive = 100

        initial_pred_fit = predator.genome.fitness
        initial_prey_fit = prey.genome.fitness
        initial_ally_fit = ally.genome.fitness

        predator.think_and_act([], [], [], [predator, prey, ally], 1280, 720)

        self.assertLessEqual(predator.energy, 66.0)
        self.assertLess(predator.genome.fitness, initial_pred_fit - 15.0)
        self.assertGreaterEqual(prey.genome.fitness, initial_prey_fit + 14.0)
        self.assertGreaterEqual(ally.genome.fitness, initial_ally_fit + 14.0)
        self.assertEqual(prey.herd_defenses, 1)

    def test_predation_on_isolated_prey(self):
        predator = Agent(DummyNetwork(output_x=1.0, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(195, 200))
        predator.vel = pygame.math.Vector2(3.0, 0.0)
        predator.energy = 50.0
        predator.frames_alive = 100

        prey = Agent(DummyNetwork(output_x=0.5, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(200, 200))
        prey.vel = pygame.math.Vector2(1.0, 0.0)
        prey.energy = 50.0
        prey.frames_alive = 100

        initial_pred_fit = predator.genome.fitness
        initial_prey_fit = prey.genome.fitness

        predator.think_and_act([], [], [], [predator, prey], 1280, 720)

        self.assertGreaterEqual(predator.energy, 70.0)
        self.assertLessEqual(prey.energy, 26.0)
        self.assertGreaterEqual(predator.genome.fitness, initial_pred_fit + 24.0)
        self.assertLess(prey.genome.fitness, initial_prey_fit)
        self.assertEqual(predator.attacks_made, 1)

    def test_frontal_defense_clash(self):
        agent_a = Agent(DummyNetwork(output_x=1.0, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(196, 200))
        agent_a.vel = pygame.math.Vector2(2.0, 0.0)
        agent_a.energy = 80.0
        agent_a.frames_alive = 100

        agent_b = Agent(DummyNetwork(output_x=-1.0, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(200, 200))
        agent_b.vel = pygame.math.Vector2(-2.0, 0.0)
        agent_b.energy = 40.0
        agent_b.frames_alive = 100

        initial_fit_a = agent_a.genome.fitness

        agent_a.think_and_act([], [], [], [agent_a, agent_b], 1280, 720)

        self.assertGreaterEqual(agent_a.genome.fitness, initial_fit_a + 9.0)
        self.assertEqual(agent_a.defenses_made, 1)

    def test_altruism_energy_transfer_and_reward(self):
        agent_a = Agent(DummyNetwork(output_x=0.0, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(200, 200))
        agent_a.energy = 100.0
        agent_a.frames_alive = 100

        agent_b = Agent(DummyNetwork(output_x=0.0, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(200, 200))
        agent_b.energy = 10.0
        agent_b.frames_alive = 100

        initial_fitness_a = agent_a.genome.fitness

        agent_a.think_and_act([], [], [], [agent_a, agent_b], 1280, 720)

        self.assertAlmostEqual(agent_a.energy, 79.8, places=1)
        self.assertAlmostEqual(agent_b.energy, 30.0, places=1)
        self.assertGreaterEqual(agent_a.genome.fitness, initial_fitness_a + 49.0)
        self.assertEqual(agent_a.allies_saved, 1)

    def test_food_consumption_reward(self):
        food = Food(self.agent.pos.x, self.agent.pos.y)
        foods = [food]
        self.agent.energy = 50.0
        initial_fitness = self.agent.genome.fitness

        self.agent.think_and_act(foods, [], [], [self.agent], 1280, 720)

        self.assertEqual(self.agent.foods_eaten, 1)
        self.assertGreaterEqual(self.agent.energy, 110.0)
        self.assertGreaterEqual(self.agent.genome.fitness, initial_fitness + 14.0)

    def test_poison_collision_penalty(self):
        poison = Poison(self.agent.pos.x, self.agent.pos.y)
        initial_energy = self.agent.energy
        initial_fitness = self.agent.genome.fitness

        self.agent.think_and_act([], [poison], [], [self.agent], 1280, 720)

        self.assertLessEqual(self.agent.energy, initial_energy - 30.0)
        self.assertLessEqual(self.agent.genome.fitness, initial_fitness - 8.0)
        self.assertEqual(self.agent.poisons_hit, 1)

    def test_hazard_collision_penalty(self):
        hazard = Hazard(self.agent.pos.x, self.agent.pos.y)
        initial_energy = self.agent.energy
        initial_fitness = self.agent.genome.fitness

        self.agent.think_and_act([], [], [hazard], [self.agent], 1280, 720)

        self.assertLess(self.agent.energy, initial_energy - 15.0)
        self.assertLess(self.agent.genome.fitness, initial_fitness)

    def test_agent_death_on_zero_energy(self):
        self.agent.energy = 0.02
        self.agent.think_and_act([], [], [], [self.agent], 1280, 720)
        self.assertFalse(self.agent.is_alive)


if __name__ == '__main__':
    unittest.main()
