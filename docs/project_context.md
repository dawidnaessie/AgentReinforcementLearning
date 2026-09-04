# Kontekst Projektu: AgentReinforcementLearning

## Cel Projektu
Stworzenie symulacji sztucznego życia (ALife), w której zbalansowana populacja 40 agentów AI (podzielona na 4 równe plemiona po 10 osobników, sterowane przez rekurencyjne sieci neuronowe NEAT RNN) funkcjonuje we wspólnym, ciągłym środowisku 2D. Celem nadrzędnym jest zaobserwowanie emergencji zachowań społecznych, podziału na role cywilizacyjne (zbieracz, drapieżnik, obrońca) oraz współpracy (altruistyczny transfer energii) wymuszonej przez odpowiednio zaprojektowane środowisko ze Strefą Śmierci (Deadly Margin = 20px) i holistyczną funkcję dopasowania (fitness function).

## Stos Technologiczny
- **Język:** Python 3.x (czysty Python, standardowa biblioteka `math`, `random`, `time`)
- **Neuroewolucja:** `neat-python` (obsługa rekurencyjnych sieci neuronowych RNN, pamięci stanów wewnętrznych, mutacji wag i topologii, krzyżowania, speciacji oraz elitaryzmu Top 4).
- **Środowisko i Fizyka:** `pygame` (wbudowany `pygame.math.Vector2`, pętla symulacji, headless testy, pre-renderowana czerwona ramka Strefy Śmierci).
- **Testy:** `unittest` (TDD, pełna izolacja logiki bez wymogu okna graficznego, 66 testów).

## Główne Założenia Ewolucyjne
1. **Pętla Pokoleniowa:** Generacja trwa określoną liczbę klatek lub kończy się wcześniej po wymarciu populacji.
2. **Elitaryzm:** 4 najlepsze genomy (Top 4) przechodzą do kolejnego pokolenia bez żadnych mutacji.
3. **Początkowy Minimalizm & RNN:** Sieci startują z 0 warstw ukrytych (bezpośrednie połączenia wejścia-wyjścia z obsługą pętli rekurencyjnych) i samodzielnie rozbudowują topologię poprzez mutacje (`node_add_prob = 0.15`).
4. **Metabolizm i Zasoby:** Każdy krok kosztuje energię; zjedzenie pożywienia odnawia zasoby, trucizna i drapieżnictwo odbierają siły witalne, a Strefa Śmierci (20px) błyskawicznie drenuje energię (-2.0/klatkę), eliminując bierność i corner exploit.
5. **Autonomia i Równowaga Plemion:** Ekosystem dzieli się równomiernie na 4 plemiona po 10 agentów (Cyjan, Magenta, Żółty, Biały), badając dynamikę walki i kooperacji wewnątrzplemiennej.

## Struktura Projektu
- `/src/main.py` – Inicjalizacja NEAT, punkt wejścia, zwięzłe logi w konsoli.
- `/src/environment.py` – Cykl życia Pygame, renderowanie, HUD, zarządzanie encjami (`Food`, `Poison`, `Hazard`).
- `/src/agent.py` – Klasa `Agent`, zmysły percepcyjne, metabolizm, mechaniki interakcji (altruizm, drapieżnictwo, obrona), przypisywanie fitnessu.
- `/src/entities.py` – Modularne encje świata (`Food`, `Poison`, `Hazard`).
- `/src/stats.py` – `EvolutionTracker` zbierający metryki generacji i generujący raport końcowy.
- `/logs/logs.txt` – Raporty telemetryczne symulacji z obsługą auto-tworzenia i rotacji.
- `/config-feedforward.txt` – Parametry algorytmu NEAT.
- `/tests/` – Pełny zestaw testów jednostkowych (TDD, 66 testów).