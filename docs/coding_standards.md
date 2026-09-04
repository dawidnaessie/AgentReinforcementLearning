# Coding Standards for the AI Assistant

When writing or modifying code in this project, strictly adhere to the following principles:

## AI and NEAT Principles
1. **Use Only `neat-python`:** Do not implement genetic algorithms or neural networks from scratch (e.g., in PyTorch or NumPy) unless explicitly requested by the user. Rely on the established mechanisms provided by `neat`.
2. **Inputs and Outputs Alignment:** Always verify that the number of sensory inputs and motor outputs in `agent.py` precisely matches the configuration in `config-feedforward.txt`.
3. **Fitness Assignment:** Fitness points must be assigned directly to the `genome.fitness` attribute. Avoid maintaining local score counters that fail to transfer to the genome.

## Environment Principles (Pygame)
1. **Performance:** Restrict the rendering of complex geometries. The simulation must run smoothly for 50+ agents. Prefer lightweight primitives like `pygame.draw.rect` or `pygame.draw.circle`.
2. **Separation of Logic:** Rendering routines (Pygame) must reside primarily in `environment.py` or dedicated object `draw()` methods. Do not place physics or game logic in `main.py`.

## Python Code Style
- Follow PEP 8 (Type Hinting is encouraged and welcomed across the codebase).
- Always document intricate sensory inputs into the neural network (e.g., explaining what each value passed to the agent represents as "vision" or "hearing").
- Maintain modularity: encapsulate new world entities (e.g., Food, Hazard) in dedicated, standalone classes.

## Code Quality & Dependencies (Production-Grade & Clean Code)
1. **KISS Principle (Keep It Simple, Stupid):** Write concise, modular, and production-ready code. Avoid over-engineering, unnecessary abstraction layers, and convoluted "clever" one-liners that impair readability.
2. **Don't Reinvent the Wheel:** Before implementing custom helper functions (e.g., for calculating Euclidean distances, angles, or collisions), check if `pygame.math.Vector2` or standard Python modules (`math`, `itertools`) already provide them. Utilize built-in, optimized methods.
3. **Zero Over-Importing:** Maintain an absolute minimum of dependencies. For 2D vector mathematics, rely on the standard library `math` or vectors from `pygame`. Strictly avoid importing heavy third-party packages (such as `numpy`, `pandas`, `scipy`, `matplotlib`) unless explicitly requested by the user.
4. **Memory Management:** Within the main Pygame rendering loop (inside `while running:`), avoid continuously instantiating new objects, allocating surfaces, or reloading fonts and assets. Initialize them once during `__init__` to prevent memory leaks and maintain steady FPS.