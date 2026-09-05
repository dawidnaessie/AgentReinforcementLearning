from typing import List, Optional, Tuple
import random
import pygame


class Food:
    """Represents an ecological resource (food / apple) in the simulation world."""

    _active_hotspot: Optional[Tuple[float, float]] = None
    _cluster_remaining: int = 0

    def __init__(self, x: float, y: float, radius: float = 4.0):
        self.pos = pygame.math.Vector2(x, y)
        self.radius = radius
        self.color = (46, 204, 113)  # Vibrant green color

    @classmethod
    def reset_cluster_state(cls):
        """Resets dynamic cluster state for clean simulation boundaries and unit testing."""
        cls._active_hotspot = None
        cls._cluster_remaining = 0

    @classmethod
    def set_hotspot(cls, cx: float, cy: float, cluster_size: int = 5):
        """Manually sets an active hotspot and remaining apple budget for testing or scenarios."""
        cls._active_hotspot = (cx, cy)
        cls._cluster_remaining = cluster_size

    @classmethod
    def create_clustered(
        cls,
        count: int,
        width: int,
        height: int,
        margin: int = 60,
        sigma: float = 60.0
    ) -> List['Food']:
        """
        Spawns food entities distributed in dense spatial clusters (patch dispersion).
        Picks random hotspots (cx, cy) and places 4-6 apples using Gaussian offsets.
        """
        foods: List['Food'] = []
        remaining = count
        while remaining > 0:
            cluster_size = min(remaining, random.randint(4, 6))
            cx = random.uniform(margin + 40, max(margin + 40, width - margin - 40))
            cy = random.uniform(margin + 40, max(margin + 40, height - margin - 40))
            for _ in range(cluster_size):
                ox = random.gauss(0.0, sigma)
                oy = random.gauss(0.0, sigma)
                fx = max(float(margin), min(float(width - margin), cx + ox))
                fy = max(float(margin), min(float(height - margin), cy + oy))
                foods.append(cls(fx, fy))
            remaining -= cluster_size
        return foods

    @classmethod
    def respawn_clustered(
        cls,
        foods: List['Food'],
        width: int,
        height: int,
        margin: int = 60,
        sigma: float = 60.0
    ):
        """Relocates an existing list of food entities into dense spatial clusters/patches."""
        idx = 0
        total = len(foods)
        while idx < total:
            cluster_size = min(total - idx, random.randint(4, 6))
            cx = random.uniform(margin + 40, max(margin + 40, width - margin - 40))
            cy = random.uniform(margin + 40, max(margin + 40, height - margin - 40))
            for _ in range(cluster_size):
                ox = random.gauss(0.0, sigma)
                oy = random.gauss(0.0, sigma)
                foods[idx].pos.x = max(float(margin), min(float(width - margin), cx + ox))
                foods[idx].pos.y = max(float(margin), min(float(height - margin), cy + oy))
                idx += 1

    def respawn(
        self,
        width: int,
        height: int,
        margin: int = 30,
        hotspot: Optional[Tuple[float, float]] = None,
        sigma: float = 60.0
    ):
        """
        Relocates the existing food entity using patch dispersion (dynamic clustering).
        Periodically selects a new hotspot coordinate (cx, cy) and groups 4-6 apples
        around it using Gaussian distribution clamped to arena margins.
        """
        if hotspot is not None:
            cx, cy = hotspot
        else:
            if Food._active_hotspot is None or Food._cluster_remaining <= 0:
                cx = random.uniform(margin + 40, max(margin + 40, width - margin - 40))
                cy = random.uniform(margin + 40, max(margin + 40, height - margin - 40))
                Food._active_hotspot = (cx, cy)
                Food._cluster_remaining = random.randint(4, 6)
            cx, cy = Food._active_hotspot
            Food._cluster_remaining -= 1

        ox = random.gauss(0.0, sigma)
        oy = random.gauss(0.0, sigma)
        self.pos.x = max(float(margin), min(float(width - margin), cx + ox))
        self.pos.y = max(float(margin), min(float(height - margin), cy + oy))

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
