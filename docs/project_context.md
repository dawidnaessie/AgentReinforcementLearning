# Project Context: AgentReinforcementLearning

## Project Objective
Creation of an Artificial Life (ALife) simulation where a balanced population of 40 AI agents (divided into 4 equal tribes of 10 individuals, governed by NEAT Recurrent Neural Networks - RNN) operates within a shared, continuous 2D environment. The primary objective is observing the emergence of social behaviors, division of civilizational roles (forager, predator, defender), and cooperation (directional kin altruism) enforced by an engineered environment featuring a Deadly Margin (20px), a 30-frame Combat Cooldown (anti-micro-farming), dynamic Gaussian food clustering (patch dispersion), and a 23-input / 2-output recurrent neural architecture.

## Technology Stack
- **Language:** Python 3.x (Pure Python, standard library `math`, `random`, `time`, `shutil`, `datetime`)
- **Neuroevolution:** `neat-python` (recurrent neural networks RNN, internal hidden state memory, weight and topology mutations, structural pruning, speciation, and Top 4 elitism).
- **Environment & Physics:** `pygame` (built-in `pygame.math.Vector2`, simulation loop, headless unit tests, pre-rendered Deadly Zone red border).
- **MLOps & Telemetry Analysis:** `requests` (Google Gemini REST API client with model fallback: `gemini-3.6-flash`, standalone `.env` loader, automated raw log archiving into `logs/{timestamp}-LogsArchive/`, and compiled benchmark reports saved to `benchmarks/{timestamp}-AnalyticsSummary.md`).
- **Testing:** `unittest` (TDD, complete separation of game logic without display window requirements, 92 tests).

## Core Evolutionary Principles
1. **Generational Cycle:** Each generation runs for a predefined frame duration or terminates early upon population extinction.
2. **Elitism:** The top 4 genomes (Top 4) advance to the next generation without mutation.
3. **Initial Minimalism & RNN:** Networks initialize with 23 sensory inputs and 2 locomotive action outputs (`Accel X`, `Accel Y`) with 0 hidden layers (direct input-output connections supporting recurrent feedback loops) and expand their topology via loosened structural mutations and lower speciation threshold (`node_add_prob = 0.08`, `conn_add_prob = 0.15`, `conn_delete_prob = 0.03`, `compatibility_threshold = 2.5`).
4. **Metabolism & Clustered Resources:** Every step costs energy; eating food replenishes vitality (+40.0 Fitness), poison and predation drain energy, and the Deadly Zone (20px) rapidly drains energy (-2.0/frame), eliminating idling and corner camping exploits. Combat cooldown (30 frames) prevents parasitic collision micro-farming. Dynamic Gaussian patch dispersion clusters food in contested hotspots (4-6 apples), forcing territorial friction and rewarding directional altruism.
5. **Autonomy & Tribal Balance:** The ecosystem is distributed evenly across 4 tribes of 10 agents (Cyan, Magenta, Yellow, White), investigating intra-tribal cooperation, directional kin rescue, and inter-tribal combat dynamics.

## Project Structure
- `/src/main.py` – NEAT initialization, entry point, concise console logging.
- `/src/environment.py` – Pygame lifecycle, rendering, HUD, Neural Inspector with Brain Dump export (`logs/brain_id_{key}.txt`), world entity management with clustered food dispersion (`Food`, `Poison`, `Hazard`).
- `/src/agent.py` – `Agent` class, sensory perception (23 inputs including directional altruism), metabolism, interaction mechanics (altruism, predation, defense, combat cooldown), fitness assignment.
- `/src/entities.py` – Modular world entities (`Food` with dynamic Gaussian clustering and hotspot replenishment, `Poison`, `Hazard`).
- `/src/stats.py` – `EvolutionTracker` gathering generational metrics, final simulation summary, and `export_brain_to_txt` export.
- `/analyze.py` – Automated simulation telemetry and brain dump analysis with Google Gemini, local folder archiving (`{timestamp}-LogsArchive`), and compiled `benchmarks/{timestamp}-AnalyticsSummary.md` executive reporting.
- `/benchmarks/` – Compiled, version-controlled ALife evolutionary diagnostic reports.
- `/.env.example` – Environment variable template for `GEMINI_API_KEY` and optional model overrides.
- `/logs/` – Local evolutionary telemetry run reports (`logs.txt`), reverse engineering brain dumps (`brain_id_{key}.txt`), and timestamped archive folders (git-ignored).
- `/config-feedforward.txt` – NEAT algorithm hyperparameters (23 inputs, 2 outputs, structural pruning).
- `/tests/` – Comprehensive headless unit test suite (TDD, 92 tests including simulation and analysis tests).