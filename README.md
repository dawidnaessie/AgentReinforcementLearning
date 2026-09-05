# 🧬 AgentReinforcementLearning

> **Artificial Life (ALife) & Neuroevolution Simulation in 2D** powered by **NEAT-Python (RNN)** and **Pygame**.

![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![NEAT](https://img.shields.io/badge/NEAT--Python-RNN-green.svg)
![Pygame](https://img.shields.io/badge/Pygame-2.6.1-orange.svg)
![Tests](https://img.shields.io/badge/tests-85%20passed-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-purple.svg)

---

## 📖 Overview & Core Concepts

**AgentReinforcementLearning** is a rich Artificial Life (ALife) sandbox and evolutionary benchmark where a balanced population of **40 autonomous neural agents** (divided equally into 4 tribes of 10) evolves across generations in a dynamic 2D ecosystem.

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
- **Acoustic Lobotomy & Combat Economy Rebalance (Phase 9):**
  Telemetry and reverse-engineering dumps proved that acoustic shout communication was evolutionarily unviable (suppressed to conserve energy) while agents collapsed into dense collision clusters to micro-farm points. Phase 9 excises shout inputs and outputs (down to 22 inputs and 2 outputs), establishes a **30-frame Combat Cooldown** (0.5s at 60 FPS) that blocks rapid repeated point/energy farming, and heavily buffs foraging (+40.0 Fitness) to stimulate active arena exploration.
- **Top 4 NEAT Brains Visualizer & Fullscreen Neural Inspector:**
  Real-time rendering of the 4 elite neural networks in the sidebar. Clicking on any elite slot pauses the simulation and opens an interactive, fullscreen **Neural Inspector** showcasing node activations, layer layouts, and synaptic weights. Pressing **`[S]`** exports the agent's complete mathematical topology to `logs/brain_id_{key}.txt` for reverse engineering.
- **Automated Experiment Logging (`logs/logs.txt`):**
  Closing the simulation automatically appends an executive summary to `logs/logs.txt`, logging the date/time of the run, overall peak fitness, the number of active synapses in the peak genome, and a per-generation progression table. Supports automatic directory creation, manual renaming (e.g., `logs1.txt`), and automatic log rotation.
- **Safe Spawn Grace Period (60 frames / 1.0s):**
  Agents spawn with a protective invulnerability shield, allowing initial dispersion across the map without unfair spawn camping or edge penalties.
- **Strict Energy & Metabolic Pressure:**
  Base metabolic burn, sprint fatigue costs, and toxic arena borders prevent passive camping and drive continuous evolution.
- **Deadly Margin & Elimination of Corner Exploit (Phase 8):**
  A lethal 20px perimeter border (**Strefa Śmierci**) with a brutal **-2.0 energy/frame** drain and pre-rendered semi-transparent crimson visual border. Agents pushed into or hiding in corners are terminated in fractions of a second, permanently eliminating parasitic corner collision farming.
- **Even Faction Balancing & RNN Mutation Tuning (Phase 8):**
  Deterministic allocation of **exactly 10 agents per tribe** (40 total across Cyan, Magenta, Yellow, White) ensuring symmetrical warfare, paired with `node_add_prob = 0.15` in `config-feedforward.txt` to accelerate recurrent hidden node emergence.
- **Automated AI Telemetry Analysis & Archiving (`analyze.py`):**
  Automated post-simulation evaluation powered by the Google Gemini API (`gemini-3.6-flash`). Ingests `logs/*.txt` telemetry and `brain_id_*.txt` neural topologies, produces a structured executive diagnostic report covering population health, emergence, and reverse-engineered synaptic circuits, and atomically packages all processed files into a timestamped archive (`logs/HH-MM-DD-MM-YYYY-LogsArchive/`) containing `AnalyticsSummary.md`.

---

## 👁️ Agent Sensory & Action Space

Each agent evaluates its surroundings through **22 normalized sensory inputs** (scaled to `[0.0, 1.0]` or `[-1.0, 1.0]`):

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

### Action Outputs (2 Neurons with `tanh` activation):
- **Output 1 (`Ax`):** Horizontal acceleration force in `[-1.0, 1.0]`.
- **Output 2 (`Ay`):** Vertical acceleration force in `[-1.0, 1.0]`.

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

- **Strict Basal Metabolism:** Baseline burn ($-0.20$ energy/frame) + sprint quadratic cost $(\text{speed}/\text{max})^2 \times 0.08$.
- **Combat Cooldown (30 frames / 0.5s):** Granted upon attack, frontal defense, or herd defense; subsequent collisions while on cooldown grant zero fitness and zero energy, halting micro-farming.
- **Foraging (+40.0 Fitness, +65.0 Energy):** Eating green apples restores energy and strongly rewards exploration.
- **Poison Obstacles (-10.0 Fitness, -35.0 Energy):** Consuming purple square toxins causes severe damage.
- **Deadly Death Zone (20px Margin):** Severe $-2.0$ energy drain per frame for touching the outer 20px perimeter, liquidating corner campers within fractions of a second with $M_{death} = 0.3$ penalty multiplier.
- **Toxic Edge Buffer (50px Margin):** Outer warning zone inflicting continuous moderate drain ($-0.5$ energy/frame).
- **Grace Period (60 frames / 1.0s):** Blue protective glow preventing early collisions, edge penalties, or predation right after spawn.

---

## 🕹️ Controls & Interactive UI

| Input | Function | Description |
| :---: | :--- | :--- |
| **`[SPACE]`** | **Toggle Turbo Mode** | Switches between 60 FPS visual rendering and uncapped simulation speed for rapid evolution. |
| **`[ESC]` / `[X]`** | **Graceful Exit & Dump** | Safely exits Pygame, prints console summary, and appends the run log to `logs/logs.txt`. |
| **`Left Mouse Click`** | **Neural Inspector** | Click on any of the **Top 4 elite slots** in the sidebar to open the full-screen Neural Inspector. |
| **`[TAB]` (in Inspector)** | **Toggle Senses View** | Toggles between active connected neurons only and all 22 sensory inputs. |
| **`[S]` (in Inspector)** | **Brain Dump Export** | Saves the inspected agent's complete mathematical topology into `logs/brain_id_{key}.txt` for reverse engineering. |
| **`[ESC]` (in Inspector)** | **Close Inspector** | Closes the Neural Inspector and resumes live simulation. |

---

## 🧠 Neural Inspector & "Brain Dump" Export (`logs/brain_id_{key}.txt`)

When inspecting any elite brain in the full-screen Neural Inspector, pressing **`[S]`** generates an instant mathematical topology dump in the `logs/` directory named `logs/brain_id_{genome.key}.txt`.

The exported file provides a complete, human-readable breakdown ready for reverse engineering:
- **`--- GENERAL INFO ---`**: Genome ID and current holistic fitness score.
- **`--- NODES ---`**: Hidden interneurons with ID, non-linear activation function (e.g., `tanh`), and bias value.
- **`--- SYNAPSES (CONNECTIONS) ---`**: All synaptic pathways with source and target IDs translated into the exact human-readable sensor and action labels used in the UI (e.g. `[Velocity (Vel X)] -> [Acceleration (Accel X)] | Weight: 2.3500 | Status: Enabled`).

---

## 📝 Automated Experiment Logging (`logs/logs.txt`)

Every simulation run automatically appends structured diagnostic records to `logs/logs.txt`. If the `logs/` directory does not exist, it is created automatically.
- **Manual Archiving:** You can rename existing log files at any time (e.g. `logs.txt` -> `logs1.txt`); subsequent runs will seamlessly create a clean, fresh `logs.txt`.
- **Automatic Rotation:** Built-in safeguards automatically archive the file (to `logs1.txt`, `logs2.txt`, etc.) if it exceeds 5 MB, preventing log files from growing excessively large.

Example log entry appended to `logs/logs.txt`:
```text
==================================================================================================
SIMULATION RUN LOG - 2026-09-02 01:45:30
==================================================================================================
• Start Date:           2026-09-02 01:41:15
• End Date:             2026-09-02 01:45:30
• Duration:             255.40 s (4.26 min)
• Completed Generations: 12

PEAK SIMULATION PERFORMANCE:
• All-Time Highest Fitness: 1420.50 pts
• Achieved in Generation:   Gen 9
• Active Synaptic Count:    38 connections

ECOSYSTEM & BEHAVIORAL SUMMARY:
• Initial Average Fitness (Gen 1):  14.20 pts
• Final Average Fitness (Gen 12):   285.60 pts
• Average Fitness Growth:           +1911.3%
• Collected Apples:                 485 pcs.
• Consumed Poisons:                 72 pcs.
• Altruistic Rescues:               104
• Predator Attacks:                 135
• Frontal Defenses:                 62
• Herd Defenses:                    89
• Broadcast Shouts:                 340

DETAILED PROGRESSION GENERATION BY GENERATION (AVG SCORE & PEAK):
Gen   | Avg Fitness | Max Fitness | Synapses | Apples  | Poisons  | Altruism | Attacks| Defense| Herd   | Time   
--------------------------------------------------------------------------------------------------
1     | 14.20       | 52.40       | 25       | 18      | 12       | 3        | 2      | 1      | 0      | 18.20s
2     | 38.60       | 120.10      | 26       | 29      | 8        | 6        | 5      | 2      | 2      | 21.05s
...
==================================================================================================
```

---

## 🤖 Automated AI Analysis & Archiving (`analyze.py`)

To streamline post-simulation research without manual log wrangling or subjective inspection, `analyze.py` automates the synthesis of telemetry and neural reverse-engineering dumps using **Google Gemini** (`gemini-3.6-flash` with automatic fallback cascade).

### Key Pipeline Features:
- **Zero-Friction Configuration:** Automatically loads `GEMINI_API_KEY` from a local `.env` file (copied from `.env.example`) or environment variables with zero third-party `dotenv` dependencies.
- **Intelligent File Gathering:** Identifies all `.txt` logs (`logs.txt`, `og_logs.txt`) and brain dumps (`brain_id_*.txt`) located in the root of `logs/`, safely ignoring prior archive directories.
- **Token-Aware Optimization:** Intelligently truncates large multi-thousand-generation tables while keeping run summary headers and recent generational snapshots intact.
- **Deep Architectural Reverse-Engineering:** Master prompt guides Gemini to act as a Senior AI Architect and Neuroevolution Specialist, producing an in-depth 4-part executive diagnostic:
  1. *Population Evolutionary Health & Dynamics* (fitness curves, speciation, diversity).
  2. *Behavioral Telemetry & Emergence* (foraging vs. predation vs. herd defenses).
  3. *Reverse-Engineered Neural Topologies* (excitatory and inhibitory synaptic weights, hidden layer circuits).
  4. *Architectural Recommendations* (concrete hyperparameter and environmental tuning for future runs).
- **Atomic Archiving & Cleanup Routine:**
  Once the AI response is validated, the script generates a timestamped directory and cleanly moves all processed files:
  ```text
  logs/
  └── 14-30-05-09-2026-LogsArchive/
      ├── logs.txt
      ├── brain_id_12.txt
      └── AnalyticsSummary.md   # Complete executive AI report
  ```
- **CLI Execution:**
  ```bash
  python analyze.py
  ```

---

## 📁 Repository Structure

```text
AgentReinforcementLearning/
├── config-feedforward.txt   # NEAT hyperparameters (RNN enabled, mutation probabilities)
├── analyze.py               # Automated simulation log & brain dump analysis with Google Gemini
├── .env.example             # Template for GEMINI_API_KEY configuration
├── logs/                    # Automated run logs, brain dumps, and timestamped archives (gitignored)
│   ├── logs.txt             # Primary log file (auto-rotates or accepts manual renaming to logs1.txt)
│   ├── brain_id_{key}.txt   # Reverse engineering brain dumps exported via [S] in Neural Inspector
│   └── HH-MM-DD-MM-YYYY-LogsArchive/ # Automated archives with AnalyticsSummary.md
├── README.md                # Project documentation
├── .gitignore               # Comprehensive ignores (pycache, venv, checkpoints, logs/*, .env*)
├── docs/                    # Architecture and developer guidelines
│   ├── coding_standards.md  # Clean code, KISS, and standard library rules
│   ├── documentation.md     # Production technical specification & phase evolution log (Phases 1-10)
│   ├── project_context.md   # Simulation domain context and phase breakdown
│   └── workflow_and_testing.md # TDD workflow, testing rules, and QA protocols
├── src/                     # Source code
│   ├── agent.py             # Agent class (sensors, RNN activation, tribes, physics, deadly margin)
│   ├── entities.py          # Food, Hazard, Poison entities
│   ├── environment.py       # Simulation loop, HUD, Neural Inspector, Top 4 slots, brain dump export
│   ├── main.py              # Runner entry point, eval loop, graceful exit handlers
│   └── stats.py             # EvolutionTracker statistics, summary printer, dump_to_file, export_brain_to_txt
└── tests/                   # Comprehensive headless unit test suite (85 tests)
    ├── test_agent.py        # Agent physics, sensors, combat, tribal rules, altruism, deadly zone
    ├── test_analyze.py      # Automated analysis pipeline, .env parser, API mocking, archiving
    ├── test_config.py       # NEAT configuration, RNN recurrent validation, pop_size=40, node_add_prob=0.15
    ├── test_entities.py     # Entity collisions, boundaries, respawning
    ├── test_environment.py  # Simulation lifecycle, HUD, inspector deepcopy, brain dump [S] export, balanced tribes
    └── test_stats.py        # Statistics tracking, terminal summary, log rotation & export_brain_to_txt import
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/dawidnaessie/AgentReinforcementLearning.git
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

### 4. Configure Gemini API key (Optional for automated analysis)
```bash
# Copy template and fill in your Gemini API key
cp .env.example .env
```

### 5. Run the simulation
```bash
python src/main.py
```

### 6. Run automated analysis and archiving
```bash
python analyze.py
```

### 7. Run the unit test suite
```bash
python -m unittest discover tests -v
```

---

## 🧪 Testing & Quality Assurance

The codebase strictly adheres to **Test-Driven Development (TDD)** and clean separation of concerns:
- **Headless Testing:** All agent mechanics, RNN outputs, tribal interactions, and telemetry are 100% executable headlessly without opening display windows.
- **Deepcopy Isolation:** Neural inspection uses isolated deepcopies to avoid mutation or state corruption during live evolution.
- **Fast Execution:** All **85 unit tests** execute in under 1.8 seconds.

---

## 📜 License

This project is open-source and licensed under the [MIT License](LICENSE).
