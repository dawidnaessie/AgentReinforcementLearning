import math
import unittest
from typing import Any, Dict, Optional
import pygame

from src.agent import Agent
from src.entities import Food, Hazard, Poison


class DummyGenome:
    """Simple dummy genome object for unit testing without NEAT dependencies."""
    fitness: float
    key: int
    connections: Dict[Any, Any]
    nodes: Dict[Any, Any]
    custom_marker: Any
    tag: int

    def __init__(self, fitness: float = 0.0, **kwargs: Any):
        self.fitness = fitness
        self.key = 0
        self.connections = {}
        self.nodes = {}
        self.custom_marker = None
        self.tag = 0
        for k, v in kwargs.items():
            setattr(self, k, v)


class DummyNetwork:
    """Simple dummy neural network returning 3 outputs (movement dx, dy, and shout)."""
    def __init__(self, output_x=1.0, output_y=0.0, output_shout=0.0):
        self.output_x = output_x
        self.output_y = output_y
        self.output_shout = output_shout

    def activate(self, inputs):
        return (self.output_x, self.output_y, self.output_shout)

    def reset(self):
        pass


class TestAgent(unittest.TestCase):
    """Unit tests for Agent class (Phase 5: Communication, shout, hearing, 25 inputs, 3 outputs)."""

    def setUp(self):
        self.net = DummyNetwork()
        self.genome = DummyGenome()
        self.agent = Agent(self.net, self.genome, width=1280, height=720, tribe_id=1)

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
        self.assertIn(self.agent.tribe_id, [1, 2, 3, 4])

    def test_shout_activation_and_deactivation(self):
        # Network with shout signal > 0.0
        shouting_net = DummyNetwork(output_x=0.0, output_y=0.0, output_shout=0.8)
        shouter = Agent(shouting_net, DummyGenome(), width=1280, height=720, start_pos=(400, 300))
        shouter.think_and_act([], [], [], [shouter], 1280, 720)
        self.assertTrue(shouter.is_shouting)

        # Network with shout signal <= 0.0
        silent_net = DummyNetwork(output_x=0.0, output_y=0.0, output_shout=-0.5)
        silent_agent = Agent(silent_net, DummyGenome(), width=1280, height=720, start_pos=(400, 300))
        silent_agent.is_shouting = True
        silent_agent.think_and_act([], [], [], [silent_agent], 1280, 720)
        self.assertFalse(silent_agent.is_shouting)

    def test_shout_energy_cost(self):
        # Agent shouting in place (cost: 0.20 base + 0.20 shout = 0.40)
        shouting_net = DummyNetwork(output_x=0.0, output_y=0.0, output_shout=1.0)
        shouter = Agent(shouting_net, DummyGenome(), width=1280, height=720, start_pos=(400, 300))
        initial_shouter_energy = shouter.energy
        shouter.think_and_act([], [], [], [shouter], 1280, 720)
        shout_loss = initial_shouter_energy - shouter.energy
        self.assertAlmostEqual(shout_loss, 0.40, places=2)

        # Silent agent in place (cost: 0.20 base)
        silent_net = DummyNetwork(output_x=0.0, output_y=0.0, output_shout=-1.0)
        silent_agent = Agent(silent_net, DummyGenome(), width=1280, height=720, start_pos=(400, 300))
        initial_silent_energy = silent_agent.energy
        silent_agent.think_and_act([], [], [], [silent_agent], 1280, 720)
        silent_loss = initial_silent_energy - silent_agent.energy
        self.assertAlmostEqual(silent_loss, 0.20, places=2)

    def test_hearing_sensors(self):
        # 1. Case where nobody nearby is shouting
        silent_peer = Agent(DummyNetwork(output_shout=-1.0), DummyGenome(), width=1280, height=720, start_pos=(500, 300))
        silent_peer.is_shouting = False
        inputs_quiet = self.agent._get_sensory_inputs([], [], [], [self.agent, silent_peer], 1280, 720)
        # Sensors #23, #24, #25 should be 0.0
        self.assertEqual(inputs_quiet[22], 0.0)
        self.assertEqual(inputs_quiet[23], 0.0)
        self.assertEqual(inputs_quiet[24], 0.0)

        # 2. Case where another agent is shouting 100px to the right
        shouting_peer = Agent(DummyNetwork(output_shout=1.0), DummyGenome(), width=1280, height=720, start_pos=(self.agent.pos.x + 100, self.agent.pos.y))
        shouting_peer.is_shouting = True
        inputs_heard = self.agent._get_sensory_inputs([], [], [], [self.agent, shouting_peer], 1280, 720)
        # Normalized distance > 0.0 and direction dx close to 1.0, dy close to 0.0
        self.assertGreater(inputs_heard[22], 0.0)
        self.assertAlmostEqual(inputs_heard[23], 1.0, places=1)
        self.assertAlmostEqual(inputs_heard[24], 0.0, places=1)

    def test_grace_period_no_combat_or_edge_penalty(self):
        # 1. No edge penalty during Grace Period (< 60 frames / 1 second)
        ghost_agent = Agent(DummyNetwork(output_x=0.0, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(30, 300))
        ghost_agent.vel = pygame.math.Vector2(0.0, 0.0)
        ghost_agent.frames_alive = 20  # < 60
        initial_energy = ghost_agent.energy

        ghost_agent.think_and_act([], [], [], [ghost_agent], 1280, 720)

        # Loses only base metabolism (0.20), no zone penalty (-0.5)
        self.assertAlmostEqual(initial_energy - ghost_agent.energy, 0.20, places=2)

        # 2. No attacking or energy theft during Grace Period
        predator = Agent(DummyNetwork(output_x=1.0, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(195, 200), tribe_id=1)
        predator.vel = pygame.math.Vector2(3.0, 0.0)
        predator.frames_alive = 20

        prey = Agent(DummyNetwork(output_x=0.5, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(200, 200), tribe_id=2)
        prey.vel = pygame.math.Vector2(1.0, 0.0)
        prey.frames_alive = 20
        initial_prey_energy = prey.energy

        predator.think_and_act([], [], [], [predator, prey], 1280, 720)

        self.assertEqual(predator.attacks_made, 0)
        self.assertEqual(prey.energy, initial_prey_energy)

    def test_strict_hunger_metabolism(self):
        still_agent = Agent(DummyNetwork(output_x=0.0, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(400, 300))
        still_agent.vel = pygame.math.Vector2(0.0, 0.0)
        still_agent.frames_alive = 100  # Outside grace period
        initial_energy = still_agent.energy

        still_agent.think_and_act([], [], [], [still_agent], 1280, 720)

        energy_loss = initial_energy - still_agent.energy
        self.assertAlmostEqual(energy_loss, 0.20, places=2)

    def test_toxic_edge_penalty_after_grace_period(self):
        edge_agent = Agent(DummyNetwork(output_x=0.0, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(40, 300))
        edge_agent.vel = pygame.math.Vector2(0.0, 0.0)
        edge_agent.frames_alive = 100  # Outside grace period
        initial_energy = edge_agent.energy

        edge_agent.think_and_act([], [], [], [edge_agent], 1280, 720)

        # Loses metabolism (0.20) + toxic zone penalty (0.50) = 0.70
        energy_loss = initial_energy - edge_agent.energy
        self.assertAlmostEqual(energy_loss, 0.70, places=2)

    def test_safe_zone_no_edge_penalty(self):
        safe_agent = Agent(DummyNetwork(output_x=0.0, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(400, 300))
        safe_agent.vel = pygame.math.Vector2(0.0, 0.0)
        safe_agent.frames_alive = 100
        initial_energy = safe_agent.energy

        safe_agent.think_and_act([], [], [], [safe_agent], 1280, 720)

        # In safe zone only base metabolism applies
        self.assertAlmostEqual(initial_energy - safe_agent.energy, 0.20, places=2)

    def test_sensory_inputs_structure_and_bounds(self):
        foods = [Food(100, 100), Food(500, 500), Food(300, 300)]
        poisons = [Poison(150, 150)]
        hazards = [Hazard(200, 200)]
        other_agent = Agent(self.net, DummyGenome(), width=1280, height=720)
        other_agent.vel = pygame.math.Vector2(2.0, 0.0)
        agents = [self.agent, other_agent]

        inputs = self.agent._get_sensory_inputs(foods, poisons, hazards, agents, 1280, 720)

        # Phase 5: exactly 25 sensory inputs
        self.assertEqual(len(inputs), 25)
        for idx, val in enumerate(inputs):
            self.assertGreaterEqual(val, -1.01, f"Input {idx} value {val} is below -1.0")
            self.assertLessEqual(val, 1.01, f"Input {idx} value {val} is above 1.0")

    def test_herd_density_sensing(self):
        ally1 = Agent(self.net, DummyGenome(), width=1280, height=720, start_pos=(self.agent.pos.x + 10, self.agent.pos.y), tribe_id=self.agent.tribe_id)
        ally2 = Agent(self.net, DummyGenome(), width=1280, height=720, start_pos=(self.agent.pos.x + 20, self.agent.pos.y), tribe_id=self.agent.tribe_id)
        ally3 = Agent(self.net, DummyGenome(), width=1280, height=720, start_pos=(self.agent.pos.x + 30, self.agent.pos.y), tribe_id=self.agent.tribe_id)

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
        predator = Agent(DummyNetwork(output_x=1.0, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(195, 200), tribe_id=1)
        predator.vel = pygame.math.Vector2(3.0, 0.0)
        predator.energy = 80.0
        predator.frames_alive = 100

        prey = Agent(DummyNetwork(output_x=0.5, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(200, 200), tribe_id=2)
        prey.vel = pygame.math.Vector2(1.0, 0.0)
        prey.energy = 60.0
        prey.frames_alive = 100

        ally = Agent(DummyNetwork(output_x=0.0, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(215, 200), tribe_id=2)
        ally.vel = pygame.math.Vector2(1.0, 0.0)
        ally.energy = 60.0
        ally.frames_alive = 100

        predator.think_and_act([], [], [], [predator, prey, ally], 1280, 720)

        # Predator receives damage
        self.assertLessEqual(predator.energy, 66.0)
        # Herd defense credited to victim and ally
        self.assertEqual(prey.herd_defenses, 1)
        self.assertEqual(ally.herd_defenses, 1)

    def test_predation_on_isolated_prey(self):
        predator = Agent(DummyNetwork(output_x=1.0, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(195, 200), tribe_id=1)
        predator.vel = pygame.math.Vector2(3.0, 0.0)
        predator.energy = 50.0
        predator.frames_alive = 100

        prey = Agent(DummyNetwork(output_x=0.5, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(200, 200), tribe_id=2)
        prey.vel = pygame.math.Vector2(1.0, 0.0)
        prey.energy = 50.0
        prey.frames_alive = 100

        predator.think_and_act([], [], [], [predator, prey], 1280, 720)

        self.assertGreaterEqual(predator.energy, 70.0)
        self.assertLessEqual(prey.energy, 26.0)
        self.assertEqual(predator.attacks_made, 1)

    def test_frontal_defense_clash(self):
        agent_a = Agent(DummyNetwork(output_x=1.0, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(196, 200), tribe_id=1)
        agent_a.vel = pygame.math.Vector2(2.0, 0.0)
        agent_a.energy = 80.0
        agent_a.frames_alive = 100

        agent_b = Agent(DummyNetwork(output_x=-1.0, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(200, 200), tribe_id=2)
        agent_b.vel = pygame.math.Vector2(-2.0, 0.0)
        agent_b.energy = 40.0
        agent_b.frames_alive = 100

        agent_a.think_and_act([], [], [], [agent_a, agent_b], 1280, 720)

        self.assertEqual(agent_a.defenses_made, 1)

    def test_altruism_energy_transfer_and_reward(self):
        agent_a = Agent(DummyNetwork(output_x=0.0, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(200, 200), tribe_id=1)
        agent_a.energy = 100.0
        agent_a.frames_alive = 100

        agent_b = Agent(DummyNetwork(output_x=0.0, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(200, 200), tribe_id=1)
        agent_b.energy = 10.0
        agent_b.frames_alive = 100

        agent_a.think_and_act([], [], [], [agent_a, agent_b], 1280, 720)

        # Agent A loses 20 energy transfer (+ metabolism), agent B gains 20 energy
        self.assertAlmostEqual(agent_a.energy, 79.8, places=1)
        self.assertAlmostEqual(agent_b.energy, 30.0, places=1)
        self.assertEqual(agent_a.allies_saved, 1)

    def test_food_consumption_reward(self):
        food = Food(self.agent.pos.x, self.agent.pos.y)
        foods = [food]
        self.agent.energy = 50.0

        self.agent.think_and_act(foods, [], [], [self.agent], 1280, 720)

        self.assertEqual(self.agent.foods_eaten, 1)
        self.assertGreaterEqual(self.agent.energy, 110.0)

    def test_poison_collision_penalty(self):
        poison = Poison(self.agent.pos.x, self.agent.pos.y)
        initial_energy = self.agent.energy

        self.agent.think_and_act([], [poison], [], [self.agent], 1280, 720)

        self.assertLessEqual(self.agent.energy, initial_energy - 30.0)
        self.assertEqual(self.agent.poisons_hit, 1)

    def test_hazard_collision_penalty(self):
        hazard = Hazard(self.agent.pos.x, self.agent.pos.y)
        initial_energy = self.agent.energy

        self.agent.think_and_act([], [], [hazard], [self.agent], 1280, 720)

        self.assertLess(self.agent.energy, initial_energy - 15.0)

    def test_agent_death_on_zero_energy(self):
        self.agent.energy = 0.02
        self.agent.think_and_act([], [], [], [self.agent], 1280, 720)
        self.assertFalse(self.agent.is_alive)

    # =========================================================================
    # PHASE 6: HOLISTIC FITNESS AND AUDIT PATCH TESTS
    # =========================================================================

    def test_holistic_fitness_zero_actions_gives_zero(self):
        """If agent collected no food and took no actions, fitness is strictly 0.0."""
        agent = Agent(DummyNetwork(), DummyGenome(), width=1280, height=720, start_pos=(400, 300))
        agent.frames_alive = 500
        agent.is_alive = False
        agent.death_cause = "starvation"

        fitness = agent.finalize_fitness()
        self.assertEqual(fitness, 0.0)
        self.assertEqual(agent.genome.fitness, 0.0)

    def test_holistic_fitness_formula_and_action_weights(self):
        """Verification of F_actions weights: Apple (+1), Defense (+1), Hunt (+2), Altruism (+3), and formula."""
        agent = Agent(DummyNetwork(), DummyGenome(), width=1280, height=720, start_pos=(400, 300))
        agent.frames_alive = 250
        agent.foods_eaten = 2          # 2 * 1 = 2
        agent.defenses_made = 1        # 1 * 1 = 1
        agent.herd_defenses = 1        # 1 * 1 = 1
        agent.attacks_made = 3         # 3 * 2 = 6
        agent.allies_saved = 1         # 1 * 3 = 3
        # F_actions = 2 + 1 + 1 + 6 + 3 = 13.0

        # Test for combat death: M_death = 1.0
        agent.death_cause = "combat"
        fitness = agent.finalize_fitness()
        # F_total = ((250 * 13.0) / 25.0) * 1.0 = (3250.0 / 25.0) = 130.0
        self.assertAlmostEqual(fitness, 130.0, places=2)
        self.assertAlmostEqual(agent.genome.fitness, 130.0, places=2)

    def test_death_multipliers(self):
        """Verification of M_death multipliers: Survived (1.2), Combat (1.0), Starvation (0.7), Toxic/Poison (0.3)."""
        agent = Agent(DummyNetwork(), DummyGenome(), width=1280, height=720, start_pos=(400, 300))
        agent.frames_alive = 100
        agent.foods_eaten = 5  # F_actions = 5.0
        # Base score before M_death = (100 * 5.0) / 25.0 = 20.0

        # 1. Survived entire epoch -> 1.2
        agent.death_cause = "survived"
        self.assertAlmostEqual(agent.finalize_fitness(), 20.0 * 1.2, places=2)

        # 2. Death in combat -> 1.0
        agent.death_cause = "combat"
        self.assertAlmostEqual(agent.finalize_fitness(), 20.0 * 1.0, places=2)

        # 3. Death from starvation -> 0.7
        agent.death_cause = "starvation"
        self.assertAlmostEqual(agent.finalize_fitness(), 20.0 * 0.7, places=2)

        # 4. Death from toxic edge -> 0.3
        agent.death_cause = "toxic_edge"
        self.assertAlmostEqual(agent.finalize_fitness(), 20.0 * 0.3, places=2)

        # 5. Death from poison -> 0.3
        agent.death_cause = "poison"
        self.assertAlmostEqual(agent.finalize_fitness(), 20.0 * 0.3, places=2)

    def test_toxic_edges_all_boundaries_set_0_3_multiplier(self):
        """Verification that death in 50px zone (top, bottom, left, right) assigns death_cause='toxic_edge' and M_death=0.3."""
        # Test 1: Top wall (y < 50)
        agent_top = Agent(DummyNetwork(output_x=0.0, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(300, 20))
        agent_top.frames_alive = 100
        agent_top.energy = 0.1
        agent_top.foods_eaten = 5
        agent_top.think_and_act([], [], [], [agent_top], 1280, 720)
        self.assertFalse(agent_top.is_alive)
        self.assertEqual(agent_top.death_cause, "toxic_edge")
        # ((101 * 5) / 25) * 0.3 = 20.2 * 0.3 = 6.06
        self.assertAlmostEqual(agent_top.genome.fitness, ((101 * 5.0) / 25.0) * 0.3, places=2)

        # Test 2: Bottom wall (y > 720 - 50 = 670)
        agent_bottom = Agent(DummyNetwork(output_x=0.0, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(300, 700))
        agent_bottom.frames_alive = 100
        agent_bottom.energy = 0.1
        agent_bottom.foods_eaten = 5
        agent_bottom.think_and_act([], [], [], [agent_bottom], 1280, 720)
        self.assertFalse(agent_bottom.is_alive)
        self.assertEqual(agent_bottom.death_cause, "toxic_edge")

        # Test 3: Left wall (x < 50)
        agent_left = Agent(DummyNetwork(output_x=0.0, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(20, 300))
        agent_left.frames_alive = 100
        agent_left.energy = 0.1
        agent_left.foods_eaten = 5
        agent_left.think_and_act([], [], [], [agent_left], 1280, 720)
        self.assertFalse(agent_left.is_alive)
        self.assertEqual(agent_left.death_cause, "toxic_edge")

        # Test 4: Right wall (x > 1280 - 50 = 1230)
        agent_right = Agent(DummyNetwork(output_x=0.0, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(1260, 300))
        agent_right.frames_alive = 100
        agent_right.energy = 0.1
        agent_right.foods_eaten = 5
        agent_right.think_and_act([], [], [], [agent_right], 1280, 720)
        self.assertFalse(agent_right.is_alive)
        self.assertEqual(agent_right.death_cause, "toxic_edge")

    def test_altruism_strict_energy_conservation_and_conditions(self):
        """Verification of strict altruism energy conservation: donor loses 20, recipient gains 20, threshold conditions."""
        # 1. Condition met: Donor > 50, Recipient < 20 (within same tribe)
        donor = Agent(DummyNetwork(output_x=0.0, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(300, 300), tribe_id=1)
        donor.energy = 80.0
        donor.frames_alive = 100

        recipient = Agent(DummyNetwork(output_x=0.0, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(300, 300), tribe_id=1)
        recipient.energy = 15.0
        recipient.frames_alive = 100

        donor.think_and_act([], [], [], [donor, recipient], 1280, 720)
        # Donor loses 20 (transfer) + 0.20 (metabolism) = 59.8
        self.assertAlmostEqual(donor.energy, 59.8, places=2)
        # Recipient gains 20
        self.assertAlmostEqual(recipient.energy, 35.0, places=2)
        self.assertEqual(donor.allies_saved, 1)

        # 2. Donor has <= 50 energy -> no transfer
        donor2 = Agent(DummyNetwork(output_x=0.0, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(300, 300), tribe_id=1)
        donor2.energy = 45.0
        donor2.frames_alive = 100
        recipient2 = Agent(DummyNetwork(output_x=0.0, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(300, 300), tribe_id=1)
        recipient2.energy = 15.0
        recipient2.frames_alive = 100

        donor2.think_and_act([], [], [], [donor2, recipient2], 1280, 720)
        self.assertAlmostEqual(donor2.energy, 44.8, places=2)
        self.assertEqual(recipient2.energy, 15.0)
        self.assertEqual(donor2.allies_saved, 0)

        # 3. Recipient has >= 20 energy -> no transfer
        donor3 = Agent(DummyNetwork(output_x=0.0, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(300, 300), tribe_id=1)
        donor3.energy = 90.0
        donor3.frames_alive = 100
        recipient3 = Agent(DummyNetwork(output_x=0.0, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(300, 300), tribe_id=1)
        recipient3.energy = 25.0
        recipient3.frames_alive = 100

        donor3.think_and_act([], [], [], [donor3, recipient3], 1280, 720)
        self.assertAlmostEqual(donor3.energy, 89.8, places=2)
        self.assertEqual(recipient3.energy, 25.0)
        self.assertEqual(donor3.allies_saved, 0)

    def test_grace_period_frame_59_to_60_transition(self):
        """Verification of precise ghost mode deactivation at frame 60 (edge penalty activation)."""
        edge_agent = Agent(DummyNetwork(output_x=0.0, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(30, 300))
        edge_agent.frames_alive = 58  # At frame 58 -> think_and_act increments to 59 (< 60)
        initial_energy = edge_agent.energy
        edge_agent.think_and_act([], [], [], [edge_agent], 1280, 720)
        # Frame 59: no edge penalty, only 0.20
        self.assertAlmostEqual(initial_energy - edge_agent.energy, 0.20, places=2)

        # Next frame -> increments to 60 (>= 60): edge penalty (-0.5) activated
        energy_before_60 = edge_agent.energy
        edge_agent.think_and_act([], [], [], [edge_agent], 1280, 720)
        self.assertAlmostEqual(energy_before_60 - edge_agent.energy, 0.70, places=2)

    def test_tribe_initialization_and_colors(self):
        """Verifies random tribe_id assignment (1-4), explicit assignment, and TRIBE_COLORS palette."""
        from src.agent import TRIBE_COLORS
        self.assertEqual(len(TRIBE_COLORS), 4)
        for tid in (1, 2, 3, 4):
            self.assertIn(tid, TRIBE_COLORS)
            self.assertEqual(len(TRIBE_COLORS[tid]), 3)

        # Random tribe assignment
        tribes_observed = {Agent(DummyNetwork(), DummyGenome(), 1280, 720).tribe_id for _ in range(50)}
        for tid in tribes_observed:
            self.assertIn(tid, (1, 2, 3, 4))
        # With 50 agents, at least 3 out of 4 tribes should be observed
        self.assertGreaterEqual(len(tribes_observed), 3)

        # Explicit assignment
        custom_agent = Agent(DummyNetwork(), DummyGenome(), 1280, 720, tribe_id=3)
        self.assertEqual(custom_agent.tribe_id, 3)

    def test_altruism_blocked_between_different_tribes(self):
        """Altruism (energy transfer) does NOT work between agents from different tribes."""
        donor = Agent(DummyNetwork(output_x=0.0, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(300, 300), tribe_id=1)
        donor.energy = 80.0
        donor.frames_alive = 100

        recipient = Agent(DummyNetwork(output_x=0.0, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(300, 300), tribe_id=2)
        recipient.energy = 10.0
        recipient.frames_alive = 100

        donor.think_and_act([], [], [], [donor, recipient], 1280, 720)

        # Donor loses only base metabolism (no transfer to different tribe)
        self.assertAlmostEqual(donor.energy, 79.8, places=2)
        self.assertEqual(recipient.energy, 10.0)
        self.assertEqual(donor.allies_saved, 0)

    def test_cannibalism_blocked_within_same_tribe(self):
        """Predator CANNOT attack or kill members of own tribe (cannibalism blocked)."""
        predator = Agent(DummyNetwork(output_x=1.0, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(195, 200), tribe_id=1)
        predator.vel = pygame.math.Vector2(3.0, 0.0)
        predator.energy = 50.0
        predator.frames_alive = 100

        prey = Agent(DummyNetwork(output_x=0.5, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(200, 200), tribe_id=1)
        prey.vel = pygame.math.Vector2(1.0, 0.0)
        prey.energy = 50.0
        prey.frames_alive = 100

        predator.think_and_act([], [], [], [predator, prey], 1280, 720)

        # No successful attack, no energy stolen
        self.assertEqual(predator.attacks_made, 0)
        self.assertEqual(prey.energy, 50.0)

    def test_herd_defense_only_protects_same_tribe(self):
        """Herd defense activates ONLY when victim's nearby allies belong to HER tribe."""
        predator = Agent(DummyNetwork(output_x=1.0, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(195, 200), tribe_id=1)
        predator.vel = pygame.math.Vector2(3.0, 0.0)
        predator.energy = 80.0
        predator.frames_alive = 100

        prey = Agent(DummyNetwork(output_x=0.5, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(200, 200), tribe_id=2)
        prey.vel = pygame.math.Vector2(1.0, 0.0)
        prey.energy = 60.0
        prey.frames_alive = 100

        # Foreign agent from tribe 3 near victim (does not belong to victim's tribe 2)
        stranger = Agent(DummyNetwork(output_x=0.0, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(215, 200), tribe_id=3)
        stranger.frames_alive = 100

        predator.think_and_act([], [], [], [predator, prey, stranger], 1280, 720)

        # Foreign agent does not defend victim -> predator attack succeeds
        self.assertEqual(predator.attacks_made, 1)
        self.assertEqual(prey.herd_defenses, 0)
        self.assertEqual(stranger.herd_defenses, 0)

    def test_nearest_enemy_sensors(self):
        """Sensors [14, 15, 16] target nearest ENEMY from another tribe, ignoring closer allies."""
        # Main agent (Tribe 1) at (200, 200)
        main_agent = Agent(DummyNetwork(), DummyGenome(), width=1280, height=720, start_pos=(200, 200), tribe_id=1)

        # Close ally (Tribe 1) at (250, 200) - distance 50px
        close_ally = Agent(DummyNetwork(), DummyGenome(), width=1280, height=720, start_pos=(250, 200), tribe_id=1)
        close_ally.energy = 15.0  # In critical state

        # Farther enemy (Tribe 2) at (350, 200) - distance 150px
        far_enemy = Agent(DummyNetwork(), DummyGenome(), width=1280, height=720, start_pos=(350, 200), tribe_id=2)
        far_enemy.vel = pygame.math.Vector2(1.0, 0.0)

        agents = [main_agent, close_ally, far_enemy]
        inputs = main_agent._get_sensory_inputs([], [], [], agents, 1280, 720)

        # Sensor 14: Distance to enemy (150px / max_dist), NOT to ally (50px / max_dist)
        max_dist = math.hypot(1280, 720)
        expected_enemy_norm_dist = 150.0 / max_dist
        self.assertAlmostEqual(inputs[14], expected_enemy_norm_dist, places=2)

        # Sensors 15, 16: Direction to enemy X close to 1.0 (rightward), Y close to 0.0
        self.assertAlmostEqual(inputs[15], 1.0, places=1)
        self.assertAlmostEqual(inputs[16], 0.0, places=1)

        # Sensor 17: Critical status of ally = 1.0 (close_ally has 15 energy)
        self.assertEqual(inputs[17], 1.0)

        # Sensor 19: Herd density of own tribe (close_ally within 60px) > 0
        self.assertGreater(inputs[19], 0.0)

    def test_draw_all_tribes_headless(self):
        """Verifies correct execution of draw() method for each of the 4 tribes on a dummy Surface."""
        surface = pygame.Surface((1280, 720))
        for tid in (1, 2, 3, 4):
            a = Agent(DummyNetwork(), DummyGenome(), width=1280, height=720, start_pos=(100 * tid, 100), tribe_id=tid)
            # Should draw without exception
            a.draw(surface)
            # Drawing in hunger state
            a.energy = 10.0
            a.draw(surface)

    def test_deadly_zone_drain_after_grace_period(self):
        """Verifies that staying in Deadly Zone (<20px) drains 2.0 energy/frame (Phase 8)."""
        # Agent at 15px from left edge (inside 20px deadly zone)
        deadly_agent = Agent(DummyNetwork(output_x=0.0, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(15, 300))
        deadly_agent.vel = pygame.math.Vector2(0.0, 0.0)
        deadly_agent.frames_alive = 100  # Outside Grace Period
        initial_energy = deadly_agent.energy

        deadly_agent.think_and_act([], [], [], [deadly_agent], 1280, 720)

        # Loses base metabolism (0.20) + severe Deadly Zone penalty (2.0) = 2.20
        energy_loss = initial_energy - deadly_agent.energy
        self.assertAlmostEqual(energy_loss, 2.20, places=2)

    def test_deadly_zone_all_four_edges_rapid_death(self):
        """Verifies that all 4 deadly edges (<20px) cause rapid death with death_cause='toxic_edge'."""
        edges = [
            (10, 360),    # Left edge (x=10 < 20)
            (1270, 360),  # Right edge (x=1270 > 1280-20)
            (640, 10),    # Top edge (y=10 < 20)
            (640, 710),   # Bottom edge (y=710 > 720-20)
        ]

        for pos in edges:
            agent = Agent(DummyNetwork(output_x=0.0, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=pos)
            agent.frames_alive = 100
            agent.energy = 1.5  # Less than 2.0 drain
            agent.think_and_act([], [], [], [agent], 1280, 720)
            self.assertFalse(agent.is_alive, f"Agent at position {pos} should immediately die in the Deadly Zone.")
            self.assertEqual(agent.death_cause, "toxic_edge")
            # M_death multiplier for edge death should be 0.3
            agent.foods_eaten = 2
            self.assertAlmostEqual(agent.finalize_fitness(), ((101 * 2.0) / 25.0) * 0.3, places=2)

    def test_deadly_zone_corner_exploit_rapid_kill(self):
        """Corner Exploit elimination: agent in corner (10, 10) with 10 energy points dies within a few frames."""
        corner_camper = Agent(DummyNetwork(output_x=0.0, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(10, 10))
        corner_camper.frames_alive = 60  # Grace Period just ended
        corner_camper.energy = 10.0

        # Loses 2.20 energy each frame (2.0 drain + 0.2 metabolism) -> dies in <= 5 frames
        frames = 0
        while corner_camper.is_alive and frames < 10:
            corner_camper.think_and_act([], [], [], [corner_camper], 1280, 720)
            frames += 1

        self.assertFalse(corner_camper.is_alive)
        self.assertLessEqual(frames, 5, "Agent in corner must die within a fraction of a second (<= 5 frames).")
        self.assertEqual(corner_camper.death_cause, "toxic_edge")


if __name__ == '__main__':
    unittest.main()

