# Standardy Kodu dla Asystenta AI

Pisząc lub modyfikując kod w tym projekcie, bezwzględnie przestrzegaj poniższych zasad:

## Zasady dotyczące AI i NEAT
1. **Używaj tylko `neat-python`:** Nie implementuj algorytmów genetycznych ani sieci neuronowych od zera (np. w PyTorch czy NumPy), chyba że użytkownik wyraźnie o to poprosi. Używaj gotowych mechanizmów `neat`.
2. **Wejścia (Inputs) i Wyjścia (Outputs):** Zawsze weryfikuj, czy liczba wejść i wyjść w `agent.py` zgadza się z plikiem `config-feedforward.txt`. 
3. **Funkcja Fitness:** Punkty fitness muszą być przypisywane bezpośrednio do obiektu `genome.fitness`. Unikaj lokalnych liczników punktów, które nie są przekazywane do genomu.

## Zasady dotyczące Środowiska (Pygame)
1. **Wydajność:** Ogranicz rysowanie skomplikowanych kształtów. Symulacja musi działać płynnie dla 50+ agentów. Używaj prostych `pygame.draw.rect` lub `pygame.draw.circle`.
2. **Separacja Logiki:** Kod renderowania (Pygame) musi znajdować się głównie w `environment.py` lub w dedykowanych metodach `draw()` obiektów. Nie umieszczaj logiki fizyki w `main.py`.

## Styl Kodu Python
- Stosuj PEP 8 (Type Hinting jest mile widziany, ale nie obowiązkowy dla głównych skryptów symulacji).
- Zawsze komentuj zawiłą logikę wejść do sieci neuronowej (np. tłumacząc, co oznacza dana wartość przekazywana agentowi jako "wzrok").
- Zachowaj modularność: nowe encje w świecie (np. Jedzenie, Zagrożenie) twórz jako osobne klasy.

## Jakość Kodu i Zależności (Production-Grade & Clean Code)
1. **Zasada KISS (Keep It Simple, Stupid):** Pisz kod zwięzły, modularny i gotowy na produkcję. Unikaj over-engineeringu, zbędnych warstw abstrakcji i zawiłych "sprytnych" jednolinijkowców, które utrudniają czytanie.
2. **Don't Reinvent the Wheel:** Zanim napiszesz niestandardową funkcję (np. do obliczania dystansu, kątów, czy kolizji), sprawdź, czy `pygame.math.Vector2` lub standardowe moduły Pythona (`math`, `itertools`) już tego nie robią. Używaj wbudowanych, zoptymalizowanych metod.
3. **Zero Over-Importing:** Utrzymuj absolutne minimum zależności. Do prostej matematyki 2D używaj standardowej biblioteki `math` lub wektorów z `pygame`. Kategorycznie nie importuj ciężkich zewnętrznych bibliotek (takich jak `numpy`, `pandas`, `scipy`, `matplotlib`), chyba że użytkownik o to wyraźnie poprosi.
4. **Zarządzanie Pamięcią:** W pętli głównej `pygame` (wewnątrz `while running:`) unikaj ciągłego inicjowania nowych obiektów, ładowania czcionek czy grafik. Inicjuj je raz w `__init__`, aby zapobiec wyciekom pamięci i spadkom FPS przy 50+ agentach.