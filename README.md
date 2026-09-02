# 🧬 AgentReinforcementLearning

> **Artificial Life (ALife) & Neuroevolution Simulation in 2D** powered by **NEAT-Python (RNN)** and **Pygame**.

![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![NEAT](https://img.shields.io/badge/NEAT--Python-RNN-green.svg)
![Pygame](https://img.shields.io/badge/Pygame-2.6.1-orange.svg)
![Tests](https://img.shields.io/badge/tests-58%20passed-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-purple.svg)

---

## 📖 Overview & Core Concepts

**AgentReinforcementLearning** is a rich Artificial Life (ALife) sandbox and evolutionary benchmark where a population of **50 autonomous neural agents** evolves across generations in a dynamic 2D ecosystem.

Built on a **1600 x 720 Research Dashboard** (1280px continuous arena + 320px telemetry & visualizer sidebar), the simulation models complex ecological, tribal, and neurological phenomena without relying on heavyweight scientific dependencies—powered purely by clean standard Python, NEAT, and Pygame.

### 🌟 Key Evolutionary Milestones:

- **Recurrent Neural Networks (RNN) with Short-Term Memory:**
  Agents are governed by recurrent neural topologies (`feed_forward = False`). By forming recurrent cycles and self-feedback loops, agents preserve internal states across frames—enabling temporal awareness (e.g., remembering a predator or prey even when it momentarily disappears from immediate sensory sight).
- **Kin Selection & Tribal Warfare (4 Tribes):**
  The population is segmented into **4 distinct factions (Tribes 1–4)**, each rendered in vivid signature colors:
  - **Tribe 1:** Neon Cyan `(0, 245, 212)`
  - **Tribe 2:** Vibrant Magenta `(255, 0, 128)`
  - **Tribe 3:** Electric Yellow `(255, 230, 0)`
  - **Tribe 4:** Pure White `(240, 246, 255)`
  Tribal membership enforces strict social boundaries: **intra-tribe altruism**, **prohibition of cannibalism / friendly fire**, **inter-tribe predation**, and **tribal herd defense**.
- **Acoustic Communication (Shout & Hearing):**
  Agents possess an active shouting output neuron to broadcast acoustic distress or rallying calls (costing energy per frame), alongside dedicated auditory sensory inputs that pinpoint the direction and distance to shouting peers.
- **Top 4 NEAT Brains Visualizer & Fullscreen Neural Inspector:**
  Real-time rendering of the 4 elite neural networks in the sidebar. Clicking on any elite slot pauses the simulation and opens an interactive, fullscreen **Neural Inspector** showcasing node activations, layer layouts, and synaptic weights.
- **Automated Experiment Logging (`logs.txt`):**
  Closing the simulation automatically appends an executive summary to `logs.txt`, logging the date/time of the run, overall peak fitness, the number of active synapses in the peak genome, and a per-generation progression table.
- **Safe Spawn Grace Period (60 frames / 1.0s):**
  Agents spawn with a protective invulnerability shield, allowing initial dispersion across the map without unfair spawn camping or edge penalties.
- **Strict Energy & Metabolic Pressure:**
  Base metabolic burn, sprint fatigue costs, communication energy drain, and toxic arena borders prevent passive camping and drive continuous evolution.

---

## 👁️ Agent Sensory & Action Space

Each agent evaluates its surroundings through **25 normalized sensory inputs** (scaled to `[0.0, 1.0]` or `[-1.0, 1.0]`):

| Input Index | Sensory Signal | Range | Description & Ecological Role |
| :---: | :--- | :--- | :--- |
| **1 – 2** | `Velocity (VX, VY)` | `[-1.0, 1.0]` | Current agent velocity vector normalized to max speed |
| **3** | `Nearest Food #1 Distance` | `[0.0, 1.0]` | Normalized Euclidean distance to the closest food item |
| **4 – 5** | `Nearest Food #1 Direction (DX, DY)` | `[-1.0, 1.0]` | Unit direction vector pointing toward nearest food |
| **6** | `Secondary Food #2 Distance` | `[0.0, 1.0]` | Normalized distance to 2nd closest food (enables trajectory planning) |
| **7 – 8** | `Secondary Food #2 Direction (DX, DY)` | `[-1.0, 1.0]` | Unit direction vector pointing toward 2nd closest food |
| **9** | `Nearest Poison Distance` | `[0.0, 1.0]` | Normalized distance to closest environmental toxin (`Poison`) |
| **10 – 11** | `Nearest Poison Direction (DX, DY)` | `[-1.0, 1.0]` | Unit direction vector pointing toward nearest poison |
| **12** | `Nearest Hazard Distance` | `[0.0, 1.0]` | Normalized distance to closest mobile hazard |
| **13 – 14** | `Nearest Hazard Direction (DX, DY)` | `[-1.0, 1.0]` | Unit direction vector pointing toward hazard |
| **15** | `Nearest Enemy Distance` | `[0.0, 1.0]` | Normalized distance to closest agent from a **foreign tribe** (`other.tribe_id != self.tribe_id`) |
| **16 – 17** | `Nearest Enemy Direction (DX, DY)` | `[-1.0, 1.0]` | Unit direction vector pointing toward nearest enemy |
| **18** | `Nearest Ally Critical State` | `{0.0, 1.0}` | Binary trigger: `1.0` if nearest **own tribe ally** has energy `< 20%` (starving), else `0.0` |
| **19** | `Nearest Enemy Relative Heading` | `[-1.0, 1.0]` | Heading alignment with enemy: `> 0.0` if enemy is fleeing, `< 0.0` if charging head-on |
| **20** | `Local Tribe Herd Density` | `[0.0, 1.0]` | Proximity density of **own tribe allies** within 60px (`0.0` isolated, `1.0` densely packed) |
| **21** | `Proximity to Nearest Wall` | `[0.0, 1.0]` | Proximity to arena borders (`0.0` at wall, `1.0` at center) |
| **22** | `Current Energy Level` | `[0.0, 1.0]` | Current vitality reserve percentage |
| **23** | `Nearest Shouting Agent Distance` | `[0.0, 1.0]` | Normalized distance to nearest agent currently shouting (`0.0` if none) |
| **24 – 25** | `Nearest Shout Direction (DX, DY)` | `[-1.0, 1.0]` | Unit direction vector pointing toward the shouting agent |

### Action Outputs (3 Neurons with `tanh` activation):
- **Output 1 (`Ax`):** Horizontal acceleration force in `[-1.0, 1.0]`.
- **Output 2 (`Ay`):** Vertical acceleration force in `[-1.0, 1.0]`.
- **Output 3 (`Shout`):** Acoustic call activation in `[-1.0, 1.0]`. Emits an acoustic wave when `> 0.0` (costs `-0.2` energy/frame).

---

## ⚔️ Social Dynamics, Kin Selection & Ecosystem Mechanics

### 1. Kin Selection & Tribal Rules
- **Altruism (+50.0 Fitness):**
  High-energy agents (`> 50` energy) can transfer `20.0` energy to save a starving agent (`< 20` energy). This transfer is **strictly permitted only between members of the same tribe** (`donor.tribe_id == recipient.tribe_id`).
- **Cannibalism Prohibition:**
  Agents cannot attack, siphon energy from, or kill members of their own tribe.
- **Inter-Tribal Predation (+25.0 Fitness, +25.0 Energy):**
  Predators can stalk and attack isolated enemies from behind (`dot_prod > 0.0`). Attacking an isolated enemy siphons up to 25 energy.
- **Frontal Defense & Parrying (+10.0 Fitness):**
  When two enemies collide head-on (`dot_prod <= -0.2`), the attack is parried with minor kinetic bounce and defensive fitness rewards.
- **Tribal Herd Defense (+15.0 Fitness for Defenders, -15 Energy for Predator):**
  If an enemy attempts to attack a victim that has $\ge 1$ ally from **its own tribe** within 45px, the entire herd counter-attacks, dealing damage to the predator and granting herd defense fitness to all participating allies.

### 2. Metabolism & Environmental Pressures
- **Strict Basal Metabolism:** Baseline burn ($-0.20$ energy/frame) + sprint quadratic cost $(\text{speed}/\text{max})^2 \times 0.08$ + acoustic shout cost ($-0.20$ energy/frame).
- **Foraging (+15.0 Fitness, +65.0 Energy):** Eating green apples restores energy.
- **Poison Obstacles (-10.0 Fitness, -35.0 Energy):** Consuming purple square toxins causes severe damage.
- **Toxic Edge Zones (50px Margin):** Hovering near the perimeter incurs continuous penalties ($-0.5$ energy, $-0.1$ fitness/frame).
- **Grace Period (60 frames / 1.0s):** Blue protective glow preventing early collisions, edge penalties, or predation right after spawn.

---

## 🕹️ Controls & Interactive UI

| Input | Function | Description |
| :---: | :--- | :--- |
| **`[SPACE]`** | **Toggle Turbo Mode** | Switches between 60 FPS visual rendering and uncapped simulation speed for rapid evolution. |
| **`[ESC]` / `[X]`** | **Graceful Exit & Dump** | Safely exits Pygame, prints console summary, and appends the run log to `logs.txt`. |
| **`Left Mouse Click`** | **Neural Inspector** | Click on any of the **Top 4 elite slots** in the sidebar to open the full-screen Neural Inspector. |
| **`[ESC]` (in Inspector)** | **Close Inspector** | Closes the Neural Inspector and resumes live simulation. |

---

## 📝 Automated Experiment Logging (`logs.txt`)

Every simulation run automatically appends structured diagnostic records to `logs.txt`. This allows comparing different evolutionary runs, tracking fitness growth over time, and monitoring structural neural complexity (synapses count).

Example log entry appended to `logs.txt`:
```text
==================================================================================================
SIMULATION RUN LOG - 2026-09-02 01:45:30
==================================================================================================
• Data rozpoczecia:    2026-09-02 01:41:15
• Data zakonczenia:    2026-09-02 01:45:30
• Czas trwania:        255.40 s (4.26 min)
• Ukonczone generacje: 12

NAJLEPSZY WYNIK W CALEJ SYMULACJI (PEAK PERFORMANCE):
• Najwyzszy fitness w ogole: 1420.50 pkt
• Osiagniety w generacji:    Gen 9
• Liczba aktywnych synaps:   38 polaczen

PODSUMOWANIE EKOSYSTEMU I ZACHOWAN:
• Sredni fitness startowy (Gen 1):  14.20 pkt
• Sredni fitness koncowy (Gen 12):  285.60 pkt
• Wzrost sredniej sprawnosci:       +1911.3%
• Zebrane jablka:                   485 szt.
• Zjedzone trucizny:                72 szt.
• Akty altruizmu (uratowani):       104
• Ataki drapieznikow:               135
• Obrony czolowe:                   62
• Obrony stadne:                    89
• Wyemitowane krzyki:               340

SZCZEGOLOWY PRZEBIEG GENERACJA PO GENERACJI (AVG SCORE & PEAK):
Gen   | Sr Fitness  | Max Fitness | Synapsy  | Jablka  | Trucizny | Altruizm | Ataki  | Obrony | Stado  | Czas   
--------------------------------------------------------------------------------------------------
1     | 14.20       | 52.40       | 25       | 18      | 12       | 3        | 2      | 1      | 0      | 18.20s
2     | 38.60       | 120.10      | 26       | 29      | 8        | 6        | 5      | 2      | 2      | 21.05s
...
==================================================================================================
```

---

## 📁 Repository Structure

```text
AgentReinforcementLearning/
├── config-feedforward.txt   # NEAT hyperparameters (RNN enabled, mutation probabilities)
├── logs.txt                 # Automated run logs and evolutionary telemetry (gitignored)
├── README.md                # Project documentation
├── .gitignore               # Comprehensive ignores (pycache, venv, checkpoints, logs)
├── docs/                    # Architecture and developer guidelines
│   ├── coding_standards.md  # Clean code, KISS, and standard library rules
│   ├── project_context.md   # Simulation domain context and phase breakdown
│   └── workflow_and_testing.md # TDD workflow, testing rules, and QA protocols
├── src/                     # Source code
│   ├── agent.py             # Agent class (sensors, RNN activation, tribes, physics)
│   ├── entities.py          # Food, Hazard, Poison entities
│   ├── environment.py       # Simulation loop, HUD, Neural Inspector, Top 4 slots
│   ├── main.py              # Runner entry point, eval loop, graceful exit handlers
│   └── stats.py             # EvolutionTracker statistics, summary printer, dump_to_file
└── tests/                   # Comprehensive headless unit test suite (58 tests)
    ├── test_agent.py        # Agent physics, sensors, combat, tribal rules, altruism
    ├── test_config.py       # NEAT configuration, RNN recurrent validation
    ├── test_entities.py     # Entity collisions, boundaries, respawning
    ├── test_environment.py  # Simulation lifecycle, HUD, inspector deepcopy, runner
    └── test_stats.py        # Statistics tracking, terminal summary, log dumping
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/dawidnaessie/AgentReinforcmentLearning.git
cd AgentReinforcmentLearning
```

### 2. Set up a virtual environment
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

### 5. Run the unit test suite
```bash
python -m unittest discover tests -v
```

---

## 🧪 Testing & Quality Assurance

The codebase strictly adheres to **Test-Driven Development (TDD)** and clean separation of concerns:
- **Headless Testing:** All agent mechanics, RNN outputs, tribal interactions, and telemetry are 100% executable headlessly without opening display windows.
- **Deepcopy Isolation:** Neural inspection uses isolated deepcopies to avoid mutation or state corruption during live evolution.
- **Fast Execution:** All **58 unit tests** execute in under 1.5 seconds.

---

## 📜 License

This project is open-source and licensed under the [MIT License](LICENSE).
