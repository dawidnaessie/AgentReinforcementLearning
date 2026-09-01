import unittest
from typing import Any, Dict, Optional
import pygame

from src.agent import Agent
from src.entities import Food, Hazard, Poison


class DummyGenome:
    """Prosty obiekt atrapa genomu do testów jednostkowych bez zależności od NEAT."""
    fitness: float
    key: int
    connections: Dict[Any, Any]
    custom_marker: Any
    tag: int

    def __init__(self, fitness: float = 0.0, **kwargs: Any):
        self.fitness = fitness
        self.key = 0
        self.connections = {}
        self.custom_marker = None
        self.tag = 0
        for k, v in kwargs.items():
            setattr(self, k, v)


class DummyNetwork:
    """Prosta atrapa sieci neuronowej zwracająca 3 wyjścia (ruch dx, dy oraz krzyk shout)."""
    def __init__(self, output_x=1.0, output_y=0.0, output_shout=0.0):
        self.output_x = output_x
        self.output_y = output_y
        self.output_shout = output_shout

    def activate(self, inputs):
        return (self.output_x, self.output_y, self.output_shout)

    def reset(self):
        pass


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

    def test_shout_energy_cost(self):
        # Agent krzyczący w miejscu (koszt: 0.20 bazowo + 0.20 krzyk = 0.40)
        shouting_net = DummyNetwork(output_x=0.0, output_y=0.0, output_shout=1.0)
        shouter = Agent(shouting_net, DummyGenome(), width=1280, height=720, start_pos=(400, 300))
        initial_shouter_energy = shouter.energy
        shouter.think_and_act([], [], [], [shouter], 1280, 720)
        shout_loss = initial_shouter_energy - shouter.energy
        self.assertAlmostEqual(shout_loss, 0.40, places=2)

        # Agent cichy w miejscu (koszt: 0.20 bazowo)
        silent_net = DummyNetwork(output_x=0.0, output_y=0.0, output_shout=-1.0)
        silent_agent = Agent(silent_net, DummyGenome(), width=1280, height=720, start_pos=(400, 300))
        initial_silent_energy = silent_agent.energy
        silent_agent.think_and_act([], [], [], [silent_agent], 1280, 720)
        silent_loss = initial_silent_energy - silent_agent.energy
        self.assertAlmostEqual(silent_loss, 0.20, places=2)

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

        ghost_agent.think_and_act([], [], [], [ghost_agent], 1280, 720)

        # Traci tylko bazowy metabolizm (0.20), brak kary strefy (-0.5)
        self.assertAlmostEqual(initial_energy - ghost_agent.energy, 0.20, places=2)

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

        edge_agent.think_and_act([], [], [], [edge_agent], 1280, 720)

        # Traci metabolizm (0.20) + karę strefy toksycznej (0.50) = 0.70
        energy_loss = initial_energy - edge_agent.energy
        self.assertAlmostEqual(energy_loss, 0.70, places=2)

    def test_safe_zone_no_edge_penalty(self):
        safe_agent = Agent(DummyNetwork(output_x=0.0, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(400, 300))
        safe_agent.vel = pygame.math.Vector2(0.0, 0.0)
        safe_agent.frames_alive = 100
        initial_energy = safe_agent.energy

        safe_agent.think_and_act([], [], [], [safe_agent], 1280, 720)

        # W bezpiecznej strefie tylko bazowy metabolizm
        self.assertAlmostEqual(initial_energy - safe_agent.energy, 0.20, places=2)

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

        predator.think_and_act([], [], [], [predator, prey, ally], 1280, 720)

        # Drapieżnik otrzymuje obrażenia
        self.assertLessEqual(predator.energy, 66.0)
        # Obrona stadna nalicza się dla ofiary i sojusznika
        self.assertEqual(prey.herd_defenses, 1)
        self.assertEqual(ally.herd_defenses, 1)

    def test_predation_on_isolated_prey(self):
        predator = Agent(DummyNetwork(output_x=1.0, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(195, 200))
        predator.vel = pygame.math.Vector2(3.0, 0.0)
        predator.energy = 50.0
        predator.frames_alive = 100

        prey = Agent(DummyNetwork(output_x=0.5, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(200, 200))
        prey.vel = pygame.math.Vector2(1.0, 0.0)
        prey.energy = 50.0
        prey.frames_alive = 100

        predator.think_and_act([], [], [], [predator, prey], 1280, 720)

        self.assertGreaterEqual(predator.energy, 70.0)
        self.assertLessEqual(prey.energy, 26.0)
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

        agent_a.think_and_act([], [], [], [agent_a, agent_b], 1280, 720)

        self.assertEqual(agent_a.defenses_made, 1)

    def test_altruism_energy_transfer_and_reward(self):
        agent_a = Agent(DummyNetwork(output_x=0.0, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(200, 200))
        agent_a.energy = 100.0
        agent_a.frames_alive = 100

        agent_b = Agent(DummyNetwork(output_x=0.0, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(200, 200))
        agent_b.energy = 10.0
        agent_b.frames_alive = 100

        agent_a.think_and_act([], [], [], [agent_a, agent_b], 1280, 720)

        # Agent A traci 20 energii transferu (+ metabolizm), agent B zyskuje 20 energii
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
    # FAZA 6: TESTY HOLISTYCZNEGO FITNESSU ORAZ ŁATEK AUDYTU
    # =========================================================================

    def test_holistic_fitness_zero_actions_gives_zero(self):
        """Jeśli agent nie zebrał jedzenia i nie podjął żadnej akcji, fitness wynosi bezwzględnie 0.0."""
        agent = Agent(DummyNetwork(), DummyGenome(), width=1280, height=720, start_pos=(400, 300))
        agent.frames_alive = 500
        agent.is_alive = False
        agent.death_cause = "starvation"

        fitness = agent.finalize_fitness()
        self.assertEqual(fitness, 0.0)
        self.assertEqual(agent.genome.fitness, 0.0)

    def test_holistic_fitness_formula_and_action_weights(self):
        """Weryfikacja wag F_akcje: Jabłko (+1), Obrona (+1), Polowanie (+2), Altruizm (+3) oraz wzoru."""
        agent = Agent(DummyNetwork(), DummyGenome(), width=1280, height=720, start_pos=(400, 300))
        agent.frames_alive = 250
        agent.foods_eaten = 2          # 2 * 1 = 2
        agent.defenses_made = 1        # 1 * 1 = 1
        agent.herd_defenses = 1        # 1 * 1 = 1
        agent.attacks_made = 3         # 3 * 2 = 6
        agent.allies_saved = 1         # 1 * 3 = 3
        # F_akcje = 2 + 1 + 1 + 6 + 3 = 13.0

        # Test dla śmierci w walce: M_death = 1.0
        agent.death_cause = "combat"
        fitness = agent.finalize_fitness()
        # F_total = ((250 * 13.0) / 25.0) * 1.0 = (3250.0 / 25.0) = 130.0
        self.assertAlmostEqual(fitness, 130.0, places=2)
        self.assertAlmostEqual(agent.genome.fitness, 130.0, places=2)

    def test_death_multipliers(self):
        """Weryfikacja mnożników M_death: Survived (1.2), Combat (1.0), Starvation (0.7), Toxic/Poison (0.3)."""
        agent = Agent(DummyNetwork(), DummyGenome(), width=1280, height=720, start_pos=(400, 300))
        agent.frames_alive = 100
        agent.foods_eaten = 5  # F_akcje = 5.0
        # Bazowy wynik przed M_death = (100 * 5.0) / 25.0 = 20.0

        # 1. Przetrwanie całej epoki -> 1.2
        agent.death_cause = "survived"
        self.assertAlmostEqual(agent.finalize_fitness(), 20.0 * 1.2, places=2)

        # 2. Śmierć w walce -> 1.0
        agent.death_cause = "combat"
        self.assertAlmostEqual(agent.finalize_fitness(), 20.0 * 1.0, places=2)

        # 3. Śmierć głodowa -> 0.7
        agent.death_cause = "starvation"
        self.assertAlmostEqual(agent.finalize_fitness(), 20.0 * 0.7, places=2)

        # 4. Śmierć od krawędzi toksycznej -> 0.3
        agent.death_cause = "toxic_edge"
        self.assertAlmostEqual(agent.finalize_fitness(), 20.0 * 0.3, places=2)

        # 5. Śmierć od trucizny -> 0.3
        agent.death_cause = "poison"
        self.assertAlmostEqual(agent.finalize_fitness(), 20.0 * 0.3, places=2)

    def test_toxic_edges_all_boundaries_set_0_3_multiplier(self):
        """Weryfikacja, że zgon w strefie 50px (góra, dół, lewo, prawo) przypisuje death_cause='toxic_edge' i M_death=0.3."""
        # Test 1: Górna ściana (y < 50)
        agent_top = Agent(DummyNetwork(output_x=0.0, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(300, 20))
        agent_top.frames_alive = 100
        agent_top.energy = 0.1
        agent_top.foods_eaten = 5
        agent_top.think_and_act([], [], [], [agent_top], 1280, 720)
        self.assertFalse(agent_top.is_alive)
        self.assertEqual(agent_top.death_cause, "toxic_edge")
        # ((101 * 5) / 25) * 0.3 = 20.2 * 0.3 = 6.06
        self.assertAlmostEqual(agent_top.genome.fitness, ((101 * 5.0) / 25.0) * 0.3, places=2)

        # Test 2: Dolna ściana (y > 720 - 50 = 670)
        agent_bottom = Agent(DummyNetwork(output_x=0.0, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(300, 700))
        agent_bottom.frames_alive = 100
        agent_bottom.energy = 0.1
        agent_bottom.foods_eaten = 5
        agent_bottom.think_and_act([], [], [], [agent_bottom], 1280, 720)
        self.assertFalse(agent_bottom.is_alive)
        self.assertEqual(agent_bottom.death_cause, "toxic_edge")

        # Test 3: Lewa ściana (x < 50)
        agent_left = Agent(DummyNetwork(output_x=0.0, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(20, 300))
        agent_left.frames_alive = 100
        agent_left.energy = 0.1
        agent_left.foods_eaten = 5
        agent_left.think_and_act([], [], [], [agent_left], 1280, 720)
        self.assertFalse(agent_left.is_alive)
        self.assertEqual(agent_left.death_cause, "toxic_edge")

        # Test 4: Prawa ściana (x > 1280 - 50 = 1230)
        agent_right = Agent(DummyNetwork(output_x=0.0, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(1260, 300))
        agent_right.frames_alive = 100
        agent_right.energy = 0.1
        agent_right.foods_eaten = 5
        agent_right.think_and_act([], [], [], [agent_right], 1280, 720)
        self.assertFalse(agent_right.is_alive)
        self.assertEqual(agent_right.death_cause, "toxic_edge")

    def test_altruism_strict_energy_conservation_and_conditions(self):
        """Weryfikacja ścisłego bilansu energii altruizmu: dawca traci 20, biorca zyskuje 20, warunki progowe."""
        # 1. Warunek spełniony: Dawca > 50, Biorca < 20
        donor = Agent(DummyNetwork(output_x=0.0, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(300, 300))
        donor.energy = 80.0
        donor.frames_alive = 100

        recipient = Agent(DummyNetwork(output_x=0.0, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(300, 300))
        recipient.energy = 15.0
        recipient.frames_alive = 100

        donor.think_and_act([], [], [], [donor, recipient], 1280, 720)
        # Dawca traci 20 (transfer) + 0.20 (metabolizm) = 59.8
        self.assertAlmostEqual(donor.energy, 59.8, places=2)
        # Biorca zyskuje 20
        self.assertAlmostEqual(recipient.energy, 35.0, places=2)
        self.assertEqual(donor.allies_saved, 1)

        # 2. Dawca ma <= 50 energii -> brak transferu
        donor2 = Agent(DummyNetwork(output_x=0.0, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(300, 300))
        donor2.energy = 45.0
        donor2.frames_alive = 100
        recipient2 = Agent(DummyNetwork(output_x=0.0, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(300, 300))
        recipient2.energy = 15.0
        recipient2.frames_alive = 100

        donor2.think_and_act([], [], [], [donor2, recipient2], 1280, 720)
        self.assertAlmostEqual(donor2.energy, 44.8, places=2)
        self.assertEqual(recipient2.energy, 15.0)
        self.assertEqual(donor2.allies_saved, 0)

        # 3. Biorca ma >= 20 energii -> brak transferu
        donor3 = Agent(DummyNetwork(output_x=0.0, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(300, 300))
        donor3.energy = 90.0
        donor3.frames_alive = 100
        recipient3 = Agent(DummyNetwork(output_x=0.0, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(300, 300))
        recipient3.energy = 25.0
        recipient3.frames_alive = 100

        donor3.think_and_act([], [], [], [donor3, recipient3], 1280, 720)
        self.assertAlmostEqual(donor3.energy, 89.8, places=2)
        self.assertEqual(recipient3.energy, 25.0)
        self.assertEqual(donor3.allies_saved, 0)

    def test_grace_period_frame_59_to_60_transition(self):
        """Weryfikacja precyzyjnego wyłączenia trybu ducha w klatce 60 (aktywacja kary krawędziowej)."""
        edge_agent = Agent(DummyNetwork(output_x=0.0, output_y=0.0), DummyGenome(), width=1280, height=720, start_pos=(30, 300))
        edge_agent.frames_alive = 58  # W klatce 58 -> think_and_act podbija do 59 (< 60)
        initial_energy = edge_agent.energy
        edge_agent.think_and_act([], [], [], [edge_agent], 1280, 720)
        # Klatka 59: brak kary krawędziowej, tylko 0.20
        self.assertAlmostEqual(initial_energy - edge_agent.energy, 0.20, places=2)

        # Następna klatka -> podbicie do 60 (>= 60): kara krawędziowa (-0.5) aktywowana
        energy_before_60 = edge_agent.energy
        edge_agent.think_and_act([], [], [], [edge_agent], 1280, 720)
        self.assertAlmostEqual(energy_before_60 - edge_agent.energy, 0.70, places=2)


if __name__ == '__main__':
    unittest.main()
