# 🧬 AgentReinforcementLearning

> **Artificial Life (ALife) & Neuroevolution Simulation in 2D** powered by **NEAT-Python** and **Pygame**.

![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![NEAT](https://img.shields.io/badge/NEAT--Python-2.0.0-green.svg)
![Pygame](https://img.shields.io/badge/Pygame-2.6.1-orange.svg)
![Tests](https://img.shields.io/badge/tests-21%20passed-brightgreen.svg)

---

## 📖 Overview & Core Ideas

**AgentReinforcementLearning** is an Artificial Life (ALife) simulation where a population of **50 autonomous neural agents** coexists and evolves within a shared 2D continuous environment.

In **Phase 2 (Survival and Cooperation)**, the simulation introduces metabolic energy constraints, environmental toxins (`Poison`), and civilizational altruism mechanics (inter-agent energy sharing to save starving peers).

---

## 👁️ Agent Sensory & Action Space

Each agent perceives its surroundings through **20 normalized sensory inputs** (scaled to `[0.0, 1.0]` or `[-1.0, 1.0]` to prevent neural saturation):

| Input Index | Sensory Signal | Range | Description |
| :---: | :--- | :---: | :--- |
| **1 – 2** | `Velocity (VX, VY)` | `[-1.0, 1.0]` | Current movement speed normalized to maximum velocity |
| **3** | `Nearest Food #1 Distance` | `[0.0, 1.0]` | Normalized Euclidean distance to the closest food item |
| **4 – 5** | `Nearest Food #1 Direction (DX, DY)` | `[-1.0, 1.0]` | Direction unit vector pointing toward nearest food |
| **6** | `Secondary Food #2 Distance` | `[0.0, 1.0]` | Normalized distance to 2nd closest food (smoother multi-target route planning) |
| **7 – 8** | `Secondary Food #2 Direction (DX, DY)` | `[-1.0, 1.0]` | Direction unit vector pointing toward 2nd closest food |
| **9** | `Nearest Poison Distance` | `[0.0, 1.0]` | Normalized distance to the closest environmental toxin (`Poison`) |
| **10 – 11** | `Nearest Poison Direction (DX, DY)` | `[-1.0, 1.0]` | Direction unit vector pointing toward nearest poison |
| **12** | `Nearest Hazard Distance` | `[0.0, 1.0]` | Normalized distance to the closest mobile hazard |
| **13 – 14** | `Nearest Hazard Direction (DX, DY)` | `[-1.0, 1.0]` | Direction unit vector pointing toward hazard |
| **15** | `Nearest Agent Distance` | `[0.0, 1.0]` | Normalized distance to the closest alive competitor/peer |
| **16 – 17** | `Nearest Agent Direction (DX, DY)` | `[-1.0, 1.0]` | Direction unit vector pointing toward nearest agent |
| **18** | `Nearest Ally Critical State` | `{0.0, 1.0}` | Binary flag: `1.0` if closest ally has energy `< 20%` (starving), else `0.0` |
| **19** | `Proximity to Nearest Wall` | `[0.0, 1.0]` | Distance to nearest arena boundary (`0.0` at edge, `1.0` at center) |
| **20** | `Current Energy Level` | `[0.0, 1.0]` | Remaining vitality percentage before starvation |

### Action Outputs (2 Neurons with `tanh` activation):
- **Output 1 (`Ax`):** Horizontal acceleration force `[-1.0, 1.0]`
- **Output 2 (`Ay`):** Vertical acceleration force `[-1.0, 1.0]`

---

## ⚡ Energy, Fitness, Poison & Altruism Dynamics

- **Metabolic Cost:** Every frame consumes a baseline energy amount plus movement cost proportional to speed.
- **Foraging (+15.0 Fitness, +45 Energy):** Consuming a food entity restores vital energy and grants a fitness bonus.
- **Poison Obstacle (-10.0 Fitness, -35 Energy):** Contact with purple square toxins deals severe damage.
- **Mobile Hazards (-5.0 Fitness, -20 Energy):** Wandering red hazards penalize fitness and health.
- **Altruism & Cooperation (+50.0 Fitness):**
  - When an agent with high energy (`> 50%`) touches an ally in a critical state (`< 20%`), it transfers **20 energy** to the starving peer.
  - The donor receives a massive **+50.0 fitness bonus**, driving the genetic algorithm to favor networks capable of social cooperation.
- **Reward Shaping (Distance Closing Gradient):** In each frame, closing distance toward food provides a direct gradient reward `(prev_dist - new_dist) * 0.08`.
- **Wall Collision Penalty (-0.05 Fitness):** Discourages pinning against boundaries.
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
