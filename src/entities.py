import random
import pygame


class Food:
    """Represents an ecological resource (food / apple) in the simulation world."""

    def __init__(self, x: float, y: float, radius: float = 4.0):
        self.pos = pygame.math.Vector2(x, y)
        self.radius = radius
        self.color = (46, 204, 113)  # Vibrant green color

    def respawn(self, width: int, height: int, margin: int = 30):
        """Relocates the existing food entity to a new random location, preventing repeated allocations."""
        self.pos.x = random.randint(margin, width - margin)
        self.pos.y = random.randint(margin, height - margin)

    def draw(self, screen: pygame.Surface):
        """Renders the food entity as an efficient circle primitive."""
        pygame.draw.circle(screen, self.color, (int(self.pos.x), int(self.pos.y)), int(self.radius))


class Poison:
    """Represents poison (environmental hazard) draining vital energy."""

    def __init__(self, x: float, y: float, size: float = 8.0):
        self.pos = pygame.math.Vector2(x, y)
        self.size = size
        self.radius = size / 2.0  # Effective radius for collision detection
        self.color = (155, 89, 182)  # Purple color

    def respawn(self, width: int, height: int, margin: int = 30):
        """Relocates the poison entity to a new random position."""
        self.pos.x = random.randint(margin, width - margin)
        self.pos.y = random.randint(margin, height - margin)

    def draw(self, screen: pygame.Surface):
        """Renders the poison entity as a purple rectangle."""
        top_left_x = int(self.pos.x - self.radius)
        top_left_y = int(self.pos.y - self.radius)
        pygame.draw.rect(screen, self.color, (top_left_x, top_left_y, int(self.size), int(self.size)))


class Hazard:
    """Represents a dynamic hazard / roving predator in the simulation world."""

    def __init__(self, x: float, y: float, radius: float = 12.0):
        self.pos = pygame.math.Vector2(x, y)
        self.radius = radius
        # Slow drift velocity across the arena
        self.vel = pygame.math.Vector2(
            random.uniform(-1.0, 1.0),
            random.uniform(-1.0, 1.0)
        )
        if self.vel.length_squared() > 0:
            self.vel = self.vel.normalize() * random.uniform(0.5, 1.5)
        self.color = (231, 76, 60)  # Warning red color

    def update(self, width: int, height: int, margin: int = 20):
        """Updates hazard position and bounces it off arena boundaries."""
        self.pos += self.vel

        # Horizontal boundary bounce
        if self.pos.x < margin:
            self.pos.x = margin
            self.vel.x *= -1
        elif self.pos.x > width - margin:
            self.pos.x = width - margin
            self.vel.x *= -1

        # Vertical boundary bounce
        if self.pos.y < margin:
            self.pos.y = margin
            self.vel.y *= -1
        elif self.pos.y > height - margin:
            self.pos.y = height - margin
            self.vel.y *= -1

    def draw(self, screen: pygame.Surface):
        """Renders the hazard entity as a red circle primitive."""
        pygame.draw.circle(screen, self.color, (int(self.pos.x), int(self.pos.y)), int(self.radius))
