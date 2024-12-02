import numpy as np
from typing import Callable
import torch
import tqdm


class Swarm:
    """
    Manages the state of particles (positions, velocities, personal bests, global best)
    A particle is a torch.Tensor(dim).
    """

    def __init__(self, n_particles, dim, c1, c2, w, pos_constraint, vel_constraint):
        assert vel_constraint.shape == (2, dim)
        assert pos_constraint.shape == (2, dim)

        # Parameters
        self.n_particles = n_particles
        self.dim = dim
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.pos_constraint = pos_constraint
        self.vel_constraint = vel_constraint

        # Storage:
        # Positions and velocities are initialized randomly and clipped.
        pos_lower_bound = pos_constraint[0]
        pos_upper_bound = pos_constraint[1]
        vel_lower_bound = vel_constraint[0]
        vel_upper_bound = vel_constraint[1]
        self.positions = pos_lower_bound + (pos_upper_bound - pos_lower_bound) * torch.rand((n_particles, dim))
        self.velocities = vel_lower_bound + (vel_upper_bound - vel_lower_bound) * torch.rand((n_particles, dim))
        self.pbest = torch.ones((n_particles, dim)) * float("inf")
        self.gbest = torch.ones((1, dim)) * float("inf")
        self.pbest_costs = torch.ones(n_particles) * float("inf")
        self.gbest_cost = torch.ones(1) * float("inf")

    def reset(self):
        """
        Restart by initializing all particles' positions and velocities randomly within constraints, and emptying bests
        """
        pos_lower_bound = self.pos_constraint[0]
        pos_upper_bound = self.pos_constraint[1]
        vel_lower_bound = self.vel_constraint[0]
        vel_upper_bound = self.vel_constraint[1]

        self.positions = pos_lower_bound + (pos_upper_bound - pos_lower_bound) * torch.rand(
            (self.n_particles, self.dim)
        )
        self.velocities = vel_lower_bound + (vel_upper_bound - vel_lower_bound) * torch.rand(
            (self.n_particles, self.dim)
        )
        self.pbest = torch.ones((self.n_particles, self.dim)) * float("inf")
        self.gbest = torch.ones((1, self.dim)) * float("inf")
        self.pbest_costs = torch.ones(self.n_particles) * float("inf")
        self.gbest_cost = torch.ones(1) * float("inf")

    def update_positions(self):
        """Update all particles' positions"""
        self.positions = self.positions + self.velocities
        self.positions = self.positions.clip(self.pos_constraint[0], self.pos_constraint[1])

    def update_velocities(self):
        """Update all particles' velocities"""

        # Generate random weights in [0,1]
        rand_1 = torch.rand((self.n_particles, self.dim))
        rand_2 = torch.rand((self.n_particles, self.dim))

        # Update velocities
        self.velocities = (
            self.w * self.velocities
            + self.c1 * rand_1 * (self.pbest - self.positions)
            + self.c2 * rand_2 * (self.gbest - self.positions)
        )
        self.velocities = self.velocities.clip(self.vel_constraint[0], self.vel_constraint[1])


class PSO:
    def __init__(
        self,
        num_iters,
        n_particles,
        dim,
        c1,
        c2,
        w,
        pos_constraint,
        vel_constraint,
    ):
        self.swarm = Swarm(n_particles, dim, c1, c2, w, pos_constraint, vel_constraint)
        self.num_iters = num_iters

    def optimize(self, cost_fn: Callable):
        """
        Find best particle (minimizer) based on the specified cost function that evaluates a particle's cost.
        Terminates by number of iterations.
        """

        # Reinitialize swarm
        self.swarm.reset()

        for _ in tqdm.tqdm(range(self.num_iters)):
            # Evaluate all particles's costs at their current positions, and update their personal bests.
            for j in range(self.swarm.n_particles):
                # Compute its cost
                particle_cost = cost_fn(self.swarm.positions[j])
                # Update personal best
                if particle_cost < self.swarm.pbest_costs[j]:
                    self.swarm.pbest_costs[j] = particle_cost
                    self.swarm.pbest[j] = self.swarm.positions[j]
                # Update global best
                if self.swarm.pbest_costs[j] < self.swarm.gbest_cost:
                    self.swarm.gbest_cost = self.swarm.pbest_costs[j]
                    self.swarm.gbest = self.swarm.positions[j]

            # Update all particles's positions and velocities.
            self.swarm.update_velocities()
            self.swarm.update_positions()

        return (
            self.swarm.gbest_cost,
            self.swarm.gbest,
        )
