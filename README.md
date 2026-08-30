# 🧬 AgentReinforcementLearning

> **Artificial Life (ALife) & Neuroevolution Simulation in 2D** powered by **NEAT-Python** and **Pygame**.

![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![NEAT](https://img.shields.io/badge/NEAT--Python-2.0.0-green.svg)
![Pygame](https://img.shields.io/badge/Pygame-2.6.1-orange.svg)
![Tests](https://img.shields.io/badge/tests-16%20passed-brightgreen.svg)

---

## 📖 Overview & Core Ideas

**AgentReinforcementLearning** is an Artificial Life (ALife) simulation where a population of **50 autonomous neural agents** coexists and evolves within a shared 2D continuous environment.

The primary objective of the project is to observe the autonomous emergence of complex survival strategies, resource competition, hazard avoidance, and social behaviors driven purely by evolutionary pressures and a carefully designed **fitness function**.

---

## 🧠 Evolutionary Mechanics (NEAT)

The simulation uses the **NEAT (NeuroEvolution of Augmenting Topologies)** algorithm to evolve both network weights and network topologies over successive generations:

- **Minimal Initial Complexity:** Networks start with **0 hidden layers** (direct input-to-output connections) and autonomously evolve new hidden nodes and synaptic pathways as mutation adds complexity.
- **Top 4 Elitism:** The 4 best-performing genomes in each generation are preserved verbatim into the next generation without mutation.
- **Species Clustering & Speciation:** Genomes are partitioned into species based on topological compatibility to protect innovative structural mutations from premature extinction.
- **Infinite Self-Sustaining Loop:** The evolutionary loop runs indefinitely, automatically advancing generations upon timer expiration or early population extinction.

---

## 👁️ Agent Sensory & Action Space

Each agent perceives its surroundings through **14 normalized sensory inputs** (scaled to `[0.0, 1.0]` or `[-1.0, 1.0]` to prevent neural saturation):

| Input Index | Sensory Signal | Range | Description |
| :---: | :--- | :---: | :--- |
| **1 – 2** | `Position (X, Y)` | `[0.0, 1.0]` | Normalized 2D coordinates on screen |
| **3 – 4** | `Velocity (VX, VY)` | `[-1.0, 1.0]` | Current movement speed normalized to maximum velocity |
| **5** | `Nearest Food Distance` | `[0.0, 1.0]` | Normalized Euclidean distance to the closest food item |
| **6 – 7** | `Nearest Food Direction (DX, DY)` | `[-1.0, 1.0]` | Normalized direction unit vector pointing toward food |
| **8** | `Nearest Hazard Distance` | `[0.0, 1.0]` | Normalized distance to the closest mobile hazard |
| **9 – 10** | `Nearest Hazard Direction (DX, DY)` | `[-1.0, 1.0]` | Normalized direction unit vector pointing toward hazard |
| **11** | `Nearest Agent Distance` | `[0.0, 1.0]` | Normalized distance to the closest alive competitor/peer |
| **12 – 13** | `Nearest Agent Direction (DX, DY)` | `[-1.0, 1.0]` | Direction unit vector pointing toward nearest agent |
| **14** | `Current Energy Level` | `[0.0, 1.0]` | Remaining vitality percentage before starvation |

### Action Outputs (2 Neurons with `tanh` activation):
- **Output 1 (`Ax`):** Horizontal acceleration force `[-1.0, 1.0]`
- **Output 2 (`Ay`):** Vertical acceleration force `[-1.0, 1.0]`

---

## ⚡ Energy & Fitness Dynamics

- **Metabolic Cost:** Every frame consumes a baseline energy amount plus movement cost proportional to speed.
- **Foraging (+15.0 Fitness, +45 Energy):** Consuming a food entity restores vital energy and grants a large fitness bonus.
- **Hazards (-5.0 Fitness, -20 Energy):** Contact with wandering red hazards causes damage and fitness penalties.
- **Lifespan Reward (+0.05 Fitness/step):** Living longer yields incremental rewards, encouraging agents to maintain energy balance.
- **Death:** When energy hits `0.0`, the agent perishes and ceases activity for the remainder of the generation.

---

## 🕹️ Controls & Features

- **🎮 Real-Time Visual HUD:** Displays Generation, Alive/Total agents, Current Frame, Peak Fitness, and live FPS.
- **⚡ Turbo Mode (`[SPACE]`):** Instantly toggles between 60 FPS capped rendering and uncapped simulation speed for rapid multi-generation training.
- **🛑 Graceful Exit (`[ESC]` or window close):** Closes the window and prints an executive summary report to the terminal.

---

## 📊 End-of-Run Summary Report

When stopping the simulation, the built-in `EvolutionTracker` outputs a structured overview of the entire evolutionary trajectory:

```text
=================================================================
          RAPORT PODSUMOWUJACY EWOLUCJE POPULACJI (ALife)
=================================================================
 * Liczba ukonczonych generacji:    16
 * Czas trwania calej symulacji:     38.45 s (2.40 s / generacja)
-----------------------------------------------------------------
 * Sredni fitness na starcie (Gen 1):  29.68 pkt
 * Sredni fitness na koncu (Gen 16):  194.45 pkt
 * Wzrost sredniej sprawnosci:       +555.2%
 * Rekordowy wynik (Gen 10):           570.00 pkt
-----------------------------------------------------------------
 HISTORIA OSTATNICH GENERACJI:
 Gen    | Max Fitness    | Sredni Fitness   | Czas (s)  
 -------------------------------------------------------
 9      | 359.55         | 110.68           | 2.23      
 10     | 570.00         | 164.00           | 2.29      
 11     | 555.00         | 142.53           | 2.25      
 12     | 405.00         | 138.99           | 2.17      
 ...
=================================================================
 Status: Ewolucja zakonczona. Wszystkie dane zostaly podsumowane.
=================================================================
```

---

## 📁 Repository Structure

```text
AgentReinforcementLearning/
├── config-feedforward.txt   # NEAT hyperparameters and genetic configuration
├── README.md                # Project documentation
├── .gitignore               # Comprehensive ignores (pycache, venv, checkpoints, IDE)
├── docs/                    # Architectural guidelines and standards
│   ├── coding_standards.md
│   ├── project_context.md
│   └── workflow_and_testing.md
├── src/                     # Application source code
│   ├── agent.py             # Agent class (sensors, physics, energy, collision)
│   ├── entities.py          # Food & Hazard entities (Vector2 math, respawn pool)
│   ├── environment.py       # Pygame lifecycle, HUD, rendering, generation loop
│   ├── main.py              # NEAT runner, CLI entrypoint, exit handlers
│   └── stats.py             # EvolutionTracker statistics aggregator & formatter
└── tests/                   # 100% headless unit test suite (16 tests)
    ├── test_agent.py
    ├── test_config.py
    ├── test_entities.py
    ├── test_environment.py
    └── test_stats.py
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/dawidnaessie/AgentReinforcmentLearning.git
cd AgentReinforcmentLearning
```

### 2. Set up virtual environment
```bash
python -m venv venv

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Linux / macOS
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install neat-python pygame
```

### 4. Run the simulation
```bash
python src/main.py
```

### 5. Run unit tests
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

---

## 🧪 Testing & Quality Assurance

The codebase strictly follows **Test-Driven Development (TDD)** and clean separation between domain logic and rendering:
- Physics, collisions, senses, and genetics are completely testable in **headless mode** without launching a Pygame window.
- All **16 unit tests** pass in under 2 seconds.

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
