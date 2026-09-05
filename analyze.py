#!/usr/bin/env python3
"""
analyze.py - Automated NEAT Simulation Log & Brain Dump Analysis with Google Gemini.

Key Responsibilities:
1. Loads GEMINI_API_KEY from environment or .env file (zero external dependencies).
2. Scans logs/ root directory for simulation logs (*.txt) and brain dumps (brain_id_*.txt).
3. Injects collected telemetry and neural topologies into a structured prompt.
4. Queries Google Gemini API (gemini-3.6-flash / fallback models).
5. Creates a timestamped archive folder: logs/HH-MM-DD-MM-YYYY-LogsArchive/
6. Moves all processed .txt files into this archive folder using shutil.
7. Saves the AI's textual response to AnaliticsSummary.txt inside the archive folder.
"""

import os
import sys
import shutil
import datetime
from typing import List, Dict, Tuple, Optional


def load_env_file(dotenv_path: str = ".env") -> None:
    """
    Parses key=value pairs from a local .env file and sets them in os.environ
    if they are not already set. Avoids mandatory python-dotenv dependency.
    """
    if not os.path.exists(dotenv_path):
        return

    try:
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = val
    except Exception as err:
        print(f"[WARNING] Could not read '{dotenv_path}': {err}")


def get_api_key() -> Optional[str]:
    """Retrieves GEMINI_API_KEY (or GOOGLE_API_KEY) from environment or .env."""
    load_env_file()
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if key:
        return key.strip()
    return None


def gather_log_files(logs_dir: str = "logs") -> Tuple[List[str], List[str]]:
    """
    Finds all .txt files directly located in logs_dir (skips subdirectories).
    Returns a tuple of: (log_files, brain_dump_files).
    """
    if not os.path.exists(logs_dir) or not os.path.isdir(logs_dir):
        return [], []

    log_files = []
    brain_files = []

    for entry in os.scandir(logs_dir):
        if entry.is_file() and entry.name.endswith(".txt"):
            fname = entry.name
            if fname.startswith("brain_id_"):
                brain_files.append(entry.path)
            else:
                log_files.append(entry.path)

    log_files.sort()
    brain_files.sort()
    return log_files, brain_files


def read_file_content(filepath: str, max_lines: int = 1500) -> str:
    """
    Reads file content with UTF-8 encoding.
    Truncates extremely large generation tables gracefully if needed while
    preserving header summary and latest generations.
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        if len(lines) <= max_lines:
            return "".join(lines)

        # Truncate very long logs: preserve run summary header (first 50 lines) and latest 1000 lines
        head = "".join(lines[:50])
        tail = "".join(lines[-1000:])
        return (
            f"{head}\n"
            f"\n... [Truncated {len(lines) - 1050} intermediate generation rows for token optimization] ...\n\n"
            f"{tail}"
        )
    except Exception as err:
        return f"[Error reading file {os.path.basename(filepath)}: {err}]"


def construct_prompt(log_contents: Dict[str, str], brain_contents: Dict[str, str]) -> str:
    """Constructs the comprehensive master prompt for Google Gemini."""
    prompt_sections = [
        "Act as a Senior AI/IT Architect and Neuroevolution Specialist analyzing an Artificial Life (ALife) NEAT simulation.",
        "",
        "=== ARCHITECTURAL CONTEXT & RULES ===",
        "- Simulation: 2D continuous arena (1280x720) with 4 balanced tribes (Cyan, Magenta, Yellow, White) of 10 agents each (40 total).",
        "- Genetics: Recurrent Neural Networks (neat.nn.RecurrentNetwork, feed_forward=False), hidden states mutate dynamically.",
        "- Sensory Space (Phase 9: 22 normalized inputs):",
        "    0-1: Velocity VX, VY [-1..1]",
        "    2-4: Nearest Food #1 Dist [0..1], Dir X, Dir Y [-1..1]",
        "    5-7: Secondary Food #2 Dist [0..1], Dir X, Dir Y [-1..1]",
        "    8-10: Nearest Poison Dist [0..1], Dir X, Dir Y [-1..1]",
        "    11-13: Nearest Hazard Dist [0..1], Dir X, Dir Y [-1..1]",
        "    14-16: Nearest Enemy Dist [0..1], Dir X, Dir Y [-1..1]",
        "    17: Ally in Critical State (< 20% energy) [0 or 1]",
        "    18: Enemy Relative Heading [-1..1] (> 0 fleeing/back exposed, < 0 head-on charge)",
        "    19: Local Herd Density of own tribe [0..1]",
        "    20: Proximity to Nearest Wall [0..1] (0 at wall, 1 at center)",
        "    21: Self Energy Level [0..1]",
        "- Motor Actions (2 outputs): Accel X [-1..1], Accel Y [-1..1]. Note: Acoustic Shout output was lobotomized in Phase 9.",
        "- Fitness Function: F_total = ((frames_alive * F_actions) / 25.0) * M_death",
        "    F_actions = 1.0 * foods + 1.0 * defenses + 2.0 * attacks + 3.0 * altruism",
        "    M_death: Survived (1.2), Combat (1.0), Starvation (0.7), Poison/Toxic Edge (0.3).",
        "- Deadly Zone: 20px outer margin (-2.0 energy/frame) terminates corner campers within ~5 frames.",
        "- Anti-Micro-Farming: 30-frame Combat Cooldown on attack/defense prevents rapid collision point farming.",
        "- Economy: Foraging apple gives +65 energy and +40.0 nominal fitness.",
        "",
        "=== COLLECTED TELEMETRY & DATA PAYLOAD ==="
    ]

    for fname, content in log_contents.items():
        prompt_sections.append(f"\n--- [LOG FILE: {fname}] ---\n{content}\n")

    for fname, content in brain_contents.items():
        prompt_sections.append(f"\n--- [BRAIN DUMP: {fname}] ---\n{content}\n")

    prompt_sections.extend([
        "=== ANALYSIS DIRECTIVES ===",
        "Provide an in-depth, structured diagnostic report with the following sections:",
        "1. POPULATION EVOLUTIONARY HEALTH & DYNAMICS:",
        "   - Progression of maximum vs. average fitness across generations.",
        "   - Lifespan trends, stability, diversity, and speciation behavior.",
        "   - Did the population avoid local optima or fall into passive camping / point-farming traps?",
        "",
        "2. BEHAVIORAL TELEMETRY & EMERGENCE:",
        "   - Action distribution: Apples Eaten vs. Predator Attacks vs. Frontal Defenses vs. Herd Defenses vs. Altruism Rescues.",
        "   - Analysis of civilizational role specialization (foragers, predatory hunters, cooperative defenders).",
        "   - Effectiveness of the 30-frame Combat Cooldown and Deadly Margin.",
        "",
        "3. REVERSE-ENGINEERED NEURAL TOPOLOGIES (BRAIN DUMPS):",
        "   - Mathematical breakdown of the dominant brains: inspect hidden interneurons (tanh/relu, biases) and active synapses.",
        "   - Identify key excitatory (+) and inhibitory (-) control pathways (e.g. food steering vs. hazard repulsion vs. enemy flanking).",
        "   - How do these synaptic weights translate into tangible spatial survival strategies in the arena?",
        "",
        "4. ARCHITECTURAL RECOMMENDATIONS:",
        "   - Specific, concrete adjustments to NEAT hyperparameters (mutation rates, compatibility thresholds, speciation) or environmental parameters for subsequent evolutionary phases."
    ])

    return "\n".join(prompt_sections)


def call_gemini_api(prompt: str, api_key: str, requested_model: Optional[str] = None) -> str:
    """
    Sends the prompt to Google Gemini API using standard requests.
    Supports automatic fallback to active models if one is deprecated or unavailable.
    """
    candidate_models = []
    if requested_model:
        candidate_models.append(requested_model)

    env_model = os.environ.get("GEMINI_MODEL")
    if env_model and env_model not in candidate_models:
        candidate_models.append(env_model)

    # Standard fallback sequence
    for m in ["gemini-3.6-flash", "gemini-3.5-flash", "gemini-flash-latest", "gemini-2.5-flash"]:
        if m not in candidate_models:
            candidate_models.append(m)

    try:
        import requests
    except ImportError:
        raise RuntimeError("Missing required module 'requests'. Please run: pip install requests")

    last_error = None
    for model in candidate_models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 8192
            }
        }

        try:
            print(f"[API] Querying Google Gemini API (model: {model})...")
            resp = requests.post(url, headers=headers, json=payload, timeout=120)

            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts and "text" in parts[0]:
                        return parts[0]["text"]
                raise RuntimeError(f"Unexpected response payload structure from Gemini API: {data}")

            elif resp.status_code == 404:
                # Model deprecated or unavailable, try next candidate
                last_error = f"HTTP 404: {resp.text}"
                continue
            elif resp.status_code in (401, 403):
                raise RuntimeError(f"Authentication failure (HTTP {resp.status_code}): Invalid or unauthorized GEMINI_API_KEY.")
            elif resp.status_code == 429:
                raise RuntimeError("Gemini API Rate limit / quota exceeded (HTTP 429). Please try again shortly.")
            else:
                resp.raise_for_status()

        except requests.exceptions.Timeout:
            raise RuntimeError("Network timeout: Gemini API request timed out after 120 seconds.")
        except requests.exceptions.ConnectionError as ce:
            raise RuntimeError(f"Network connection failed: {ce}")
        except Exception as e:
            last_error = str(e)
            if "Authentication failure" in str(e) or "quota exceeded" in str(e):
                raise
            continue

    raise RuntimeError(f"All candidate Gemini models failed. Last error: {last_error}")


def create_archive_directory(logs_dir: str = "logs") -> str:
    """
    Generates a timestamped archive directory inside logs/ in format:
    HH-MM-DD-MM-YYYY-LogsArchive.
    Handles duplicate runs within the same minute by appending a suffix.
    """
    now = datetime.datetime.now()
    base_name = now.strftime("%H-%M-%d-%m-%Y-LogsArchive")
    archive_path = os.path.join(logs_dir, base_name)

    counter = 1
    target_path = archive_path
    while os.path.exists(target_path):
        target_path = f"{archive_path}_{counter}"
        counter += 1

    os.makedirs(target_path, exist_ok=True)
    return target_path


def main() -> int:
    """Main execution pipeline for automated log analysis and archiving."""
    print("=" * 70)
    print("NEAT AI ARCHITECT - AUTOMATED TELEMETRY & BRAIN DUMP ANALYSIS")
    print("=" * 70)

    # 1. API Key validation
    api_key = get_api_key()
    if not api_key:
        print("[ERROR] GEMINI_API_KEY not found!")
        print("   Please provide a valid API key via environment variable or in a .env file:")
        print("   GEMINI_API_KEY=your_google_gemini_api_key")
        return 1

    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        print(f"[INFO] Directory '{logs_dir}' does not exist. Nothing to analyze.")
        return 0

    # 2. File gathering
    print(f"[INFO] Scanning '{logs_dir}/' for simulation logs and brain dumps...")
    log_files, brain_files = gather_log_files(logs_dir)
    all_files = log_files + brain_files

    if not all_files:
        print(f"[INFO] No active .txt files found in the root of '{logs_dir}/' to process.")
        print("   (Subdirectories and existing archives are safely preserved).")
        return 0

    print(f"[INFO] Found {len(all_files)} file(s) to process:")
    for f in log_files:
        print(f"   * Log:   {os.path.basename(f)}")
    for f in brain_files:
        print(f"   * Brain: {os.path.basename(f)}")

    # 3. Read files into memory
    print("\n[INFO] Reading file contents into memory...")
    log_contents = {os.path.basename(f): read_file_content(f) for f in log_files}
    brain_contents = {os.path.basename(f): read_file_content(f) for f in brain_files}

    # 4. Construct prompt and query API
    prompt = construct_prompt(log_contents, brain_contents)
    print("[INFO] Synthesizing master analysis prompt...")

    try:
        summary_text = call_gemini_api(prompt, api_key)
        print("[SUCCESS] AI Analysis generated successfully!")
    except Exception as err:
        print(f"[ERROR] Failed to obtain analysis from Gemini API: {err}")
        print("[WARNING] Archiving aborted: Original log and brain dump files remain untouched.")
        return 1

    # 5. Archiving and Cleanup
    print("\n[INFO] Initializing archive and cleanup routine...")
    try:
        archive_dir = create_archive_directory(logs_dir)
        print(f"[ARCHIVE] Created archive folder: {archive_dir}")

        # Move all processed files into archive
        for fpath in all_files:
            dest_path = os.path.join(archive_dir, os.path.basename(fpath))
            shutil.move(fpath, dest_path)
            print(f"   -> Moved: {os.path.basename(fpath)}")

        # Save AI analysis report
        summary_path = os.path.join(archive_dir, "AnaliticsSummary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary_text)

        print(f"[SUCCESS] Saved executive analysis to: {summary_path}")
        print("\n" + "=" * 70)
        print("[SUCCESS] All logs processed, analyzed, and archived cleanly!")
        print("=" * 70)
        return 0

    except Exception as err:
        print(f"[ERROR] Error during file archiving: {err}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
