import os
import sys
import time
import math
import neat
import pygame

# Ensure correct path for src module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.environment import Environment, SimulationExit
from src.stats import EvolutionTracker


class SimulationRunner:
    """Manages the environment instance, collects statistics, and coordinates NEAT genome evolution."""

    def __init__(self):
        # Single environment instance – created once to prevent window reloading and memory leaks
        self.env = Environment()
        self.tracker = EvolutionTracker()

    def eval_genomes(self, genomes, config):
        """Evaluation function invoked by NEAT for each generation."""
        nets = []
        valid_genomes = []

        for genome_id, genome in genomes:
            # Create Recurrent Neural Network (RNN) based on genome and configuration
            net = neat.nn.RecurrentNetwork.create(genome, config)
            nets.append(net)
            valid_genomes.append(genome)

        start_t = time.time()

        # Run generation cycle in persistent environment
        metrics = self.env.eval_generation(nets, valid_genomes)
        if not isinstance(metrics, dict):
            metrics = {}

        duration = time.time() - start_t

        # Calculate generation statistics and identify best performer
        best_genome = max(valid_genomes, key=lambda g: getattr(g, 'fitness', -999999.0)) if valid_genomes else None
        fitnesses = [g.fitness for g in valid_genomes]
        best_fit = best_genome.fitness if best_genome else 0.0
        avg_fit = (sum(fitnesses) / len(fitnesses)) if fitnesses else 0.0
        variance = sum((f - avg_fit) ** 2 for f in fitnesses) / len(fitnesses) if fitnesses else 0.0
        stdev = math.sqrt(variance)

        # Count active synapses (connections) in the generation's best genome
        best_synapses = 0
        if best_genome and hasattr(best_genome, 'connections') and best_genome.connections:
            best_synapses = sum(1 for c in best_genome.connections.values() if getattr(c, 'enabled', True))

        foods_eaten = metrics.get('foods_eaten', 0)
        poisons_hit = metrics.get('poisons_hit', 0)
        allies_saved = metrics.get('allies_saved', 0)
        attacks_made = metrics.get('attacks_made', 0)
        defenses_made = metrics.get('defenses_made', 0)
        herd_defenses = metrics.get('herd_defenses', 0)
        shouts_made = metrics.get('shouts_made', 0)

        self.tracker.record_generation(
            generation=self.env.generation,
            best_fitness=best_fit,
            avg_fitness=avg_fit,
            stdev=stdev,
            species_count=1,
            duration_sec=duration,
            foods_eaten=foods_eaten,
            poisons_hit=poisons_hit,
            allies_saved=allies_saved,
            attacks_made=attacks_made,
            defenses_made=defenses_made,
            herd_defenses=herd_defenses,
            shouts_made=shouts_made,
            best_synapses=best_synapses
        )

        # Concise single-line terminal log with telemetry metrics
        print(f" -> [Generation {self.env.generation:3d}] Max: {best_fit:6.1f} | Avg: {avg_fit:6.1f} | Apples: {foods_eaten:3d} | Poisons: {poisons_hit:2d} | Rescued: {allies_saved:2d} | Attacks: {attacks_made:2d} | Defenses: {defenses_made:2d} | Herd: {herd_defenses:2d} | Shouts: {shouts_made:3d} | Time: {duration:4.2f}s")


def run(config_path: str):
    """Main entry point initializing and running NEAT evolution."""
    # Load NEAT configuration
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    # Initialize population (40 agents)
    population = neat.Population(config)

    # Initialize runner with persistent environment window and telemetry tracker
    runner = SimulationRunner()

    print("\n" + "=" * 65)
    print("      AgentReinforcementLearning - ALife Social Evolution")
    print("=" * 65)
    print(" • Controls:")
    print("    [SPACE]   - Toggle TURBO mode (uncapped FPS / 60 FPS)")
    print("    [ESC / X] - Terminate simulation and generate summary")
    print("=" * 65 + "\n")

    try:
        # Evolutionary loop runs indefinitely (n=None denotes open-ended evolution)
        population.run(runner.eval_genomes, n=None)
    except (KeyboardInterrupt, SimulationExit) as e:
        print(f"\n[INFO] Simulation stopped ({e})")
    finally:
        # Graceful Pygame shutdown, display console summary and dump report to logs/logs.txt
        pygame.quit()
        runner.tracker.print_summary()
        log_path = os.path.join(project_root, 'logs', 'logs.txt')
        runner.tracker.dump_to_file(log_path)
        print(f"[INFO] Simulation report successfully saved to: {log_path}\n")


if __name__ == '__main__':
    config_file = os.path.join(project_root, 'config-feedforward.txt')
    run(config_file)