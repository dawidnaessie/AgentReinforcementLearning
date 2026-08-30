import os
import sys
import time
import math
import neat
import pygame

# Zapewnienie poprawnej ścieżki do importów modułów src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.environment import Environment, SimulationExit
from src.stats import EvolutionTracker


class SimulationRunner:
    """Zarządza instancją środowiska, zbiera statystyki i koordynuje ewolucję genomów NEAT."""

    def __init__(self):
        # Pojedyncza instancja środowiska – tworzona tylko raz (brak przeładowywania okna i wycieków pamięci)
        self.env = Environment()
        self.tracker = EvolutionTracker()

    def eval_genomes(self, genomes, config):
        """Funkcja ewaluacji wywoływana przez NEAT dla każdej nowej generacji."""
        nets = []
        valid_genomes = []

        for genome_id, genome in genomes:
            # Tworzenie sieci neuronowej typu FeedForward na podstawie genomu i konfiguracji
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            nets.append(net)
            valid_genomes.append(genome)

        start_t = time.time()

        # Uruchomienie generacji w trwałym środowisku
        metrics = self.env.eval_generation(nets, valid_genomes)
        if not isinstance(metrics, dict):
            metrics = {}

        duration = time.time() - start_t

        # Obliczenie statystyk generacji
        fitnesses = [g.fitness for g in valid_genomes]
        best_fit = max(fitnesses) if fitnesses else 0.0
        avg_fit = (sum(fitnesses) / len(fitnesses)) if fitnesses else 0.0
        variance = sum((f - avg_fit) ** 2 for f in fitnesses) / len(fitnesses) if fitnesses else 0.0
        stdev = math.sqrt(variance)

        foods_eaten = metrics.get('foods_eaten', 0)
        poisons_hit = metrics.get('poisons_hit', 0)
        allies_saved = metrics.get('allies_saved', 0)

        self.tracker.record_generation(
            generation=self.env.generation,
            best_fitness=best_fit,
            avg_fitness=avg_fit,
            stdev=stdev,
            species_count=1,
            duration_sec=duration,
            foods_eaten=foods_eaten,
            poisons_hit=poisons_hit,
            allies_saved=allies_saved
        )

        # Czytelny, zwięzły log jednolinijkowy w terminalu z metrykami behawioralnymi
        print(f" -> [Generacja {self.env.generation:3d}] Max: {best_fit:6.1f} | Sr: {avg_fit:6.1f} (±{stdev:4.1f}) | Jablka: {foods_eaten:3d} | Trucizny: {poisons_hit:2d} | Uratowani: {allies_saved:2d} | Czas: {duration:4.2f}s")


def run(config_path: str):
    """Główna funkcja uruchamiająca ewolucję NEAT."""
    # Wczytanie konfiguracji NEAT
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    # Inicjalizacja populacji (50 agentów)
    population = neat.Population(config)

    # Inicjalizacja runnera z trwałym oknem środowiska i trackerem statystyk
    runner = SimulationRunner()

    print("\n" + "=" * 65)
    print("      AgentReinforcementLearning - Ewolucja Społeczna ALife")
    print("=" * 65)
    print(" • Sterowanie:")
    print("    [SPACJA]  - Przełączanie trybu TURBO (odblokowany FPS / 60 FPS)")
    print("    [ESC / X] - Zakończenie i wygenerowanie podsumowania")
    print("=" * 65 + "\n")

    try:
        # Pętla ewolucyjna działa bez ograniczeń (n=None oznacza nieskończoną ewolucję)
        population.run(runner.eval_genomes, n=None)
    except (KeyboardInterrupt, SimulationExit) as e:
        print(f"\n[INFO] Zatrzymano symulację ({e})")
    finally:
        # Czyste zamknięcie okna Pygame i wyświetlenie estetycznego podsumowania
        pygame.quit()
        runner.tracker.print_summary()


if __name__ == '__main__':
    config_file = os.path.join(project_root, 'config-feedforward.txt')
    run(config_file)