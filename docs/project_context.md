# Project Context: AgentReinforcementLearning

## Project Objective
Creation of an Artificial Life (ALife) simulation where a balanced population of 40 AI agents (divided into 4 equal tribes of 10 individuals, governed by NEAT Recurrent Neural Networks - RNN) operates within a shared, continuous 2D environment. The primary objective is observing the emergence of social behaviors, division of civilizational roles (forager, predator, defender), and cooperation (altruistic energy transfer) enforced by an engineered environment featuring a Deadly Margin (20px) and a holistic fitness function.

## Technology Stack
- **Language:** Python 3.x (Pure Python, standard library `math`, `random`, `time`)
- **Neuroevolution:** `neat-python` (recurrent neural networks RNN, internal hidden state memory, weight and topology mutations, crossover, speciation, and Top 4 elitism).
- **Environment & Physics:** `pygame` (built-in `pygame.math.Vector2`, simulation loop, headless unit tests, pre-rendered Deadly Zone red border).
- **Testing:** `unittest` (TDD, complete separation of game logic without display window requirements, 66 tests).

## Core Evolutionary Principles
1. **Generational Cycle:** Each generation runs for a predefined frame duration or terminates early upon population extinction.
2. **Elitism:** The top 4 genomes (Top 4) advance to the next generation without mutation.
3. **Initial Minimalism & RNN:** Networks initialize with 0 hidden layers (direct input-output connections supporting recurrent feedback loops) and autonomously expand their topology via mutations (`node_add_prob = 0.15`).
4. **Metabolism & Resources:** Every step costs energy; eating food replenishes vitality, poison and predation drain energy, and the Deadly Zone (20px) rapidly drains energy (-2.0/frame), eliminating idling and corner camping exploits.
5. **Autonomy & Tribal Balance:** The ecosystem is distributed evenly across 4 tribes of 10 agents (Cyan, Magenta, Yellow, White), investigating intra-tribal cooperation and inter-tribal combat dynamics.

## Project Structure
- `/src/main.py` – NEAT initialization, entry point, concise console logging.
- `/src/environment.py` – Pygame lifecycle, rendering, HUD, world entity management (`Food`, `Poison`, `Hazard`).
- `/src/agent.py` – `Agent` class, sensory perception, metabolism, interaction mechanics (altruism, predation, defense), fitness assignment.
- `/src/entities.py` – Modular world entities (`Food`, `Poison`, `Hazard`).
- `/src/stats.py` – `EvolutionTracker` gathering generational metrics and generating the final simulation summary.
- `/logs/logs.txt` – Evolutionary telemetry run reports with automatic directory creation and size-based rotation.
- `/config-feedforward.txt` – NEAT algorithm hyperparameters.
- `/tests/` – Comprehensive headless unit test suite (TDD, 66 tests).