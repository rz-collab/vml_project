import numpy as np
from typing import Callable
import torch
import tqdm

# PSO implementation for batch of inputs of size N.


class Swarm:
    """
    Manages the state of particles (positions, velocities, personal bests, global best)
    Positions, velocities, personal best are shape (batch_size = N, n_particles, 6)
    Global best is shape (N, 6)
    """

    def __init__(self, batch_size, n_particles, dim, c1, c2, w, pos_constraint, vel_constraint):
        assert vel_constraint.shape == (2, dim)
        assert pos_constraint.shape == (2, dim)
        # Parameters
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.batch_size = batch_size
        self.n_particles = n_particles
        self.dim = dim
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.pos_constraint = pos_constraint.to(self.device)
        self.vel_constraint = vel_constraint.to(self.device)

        # Storage:
        # Positions and velocities are initialized randomly and clipped.
        pos_lower_bound = self.pos_constraint[0]
        pos_upper_bound = self.pos_constraint[1]
        vel_lower_bound = self.vel_constraint[0]
        vel_upper_bound = self.vel_constraint[1]
        self.positions = pos_lower_bound + (pos_upper_bound - pos_lower_bound) * torch.rand(
            (self.batch_size, self.n_particles, self.dim), device=self.device
        )
        self.velocities = vel_lower_bound + (vel_upper_bound - vel_lower_bound) * torch.rand(
            (self.batch_size, self.n_particles, self.dim), device=self.device
        )
        self.pbest = torch.ones((self.batch_size, self.n_particles, self.dim)).to(self.device) * float("inf")
        self.gbest = torch.ones((self.batch_size, self.dim)).to(self.device) * float("inf")
        self.pbest_costs = torch.ones((self.batch_size, self.n_particles)).to(self.device) * float("inf")
        self.gbest_cost = torch.ones(self.batch_size).to(self.device) * float("inf")

    def reset(self):
        """
        Restart by initializing all particles' positions and velocities randomly within constraints, and emptying bests
        """
        pos_lower_bound = self.pos_constraint[0]
        pos_upper_bound = self.pos_constraint[1]
        vel_lower_bound = self.vel_constraint[0]
        vel_upper_bound = self.vel_constraint[1]
        self.positions = pos_lower_bound + (pos_upper_bound - pos_lower_bound) * torch.rand(
            (self.batch_size, self.n_particles, self.dim), device=self.device
        )
        self.velocities = vel_lower_bound + (vel_upper_bound - vel_lower_bound) * torch.rand(
            (self.batch_size, self.n_particles, self.dim), device=self.device
        )
        self.pbest = torch.ones((self.batch_size, self.n_particles, self.dim)).to(self.device) * float("inf")
        self.gbest = torch.ones((self.batch_size, self.dim)).to(self.device) * float("inf")
        self.pbest_costs = torch.ones((self.batch_size, self.n_particles)).to(self.device) * float("inf")
        self.gbest_cost = torch.ones(self.batch_size).to(self.device) * float("inf")

    def update_positions(self):
        """Update all particles' positions"""
        self.positions = self.positions + self.velocities
        self.positions = self.positions.clip(self.pos_constraint[0], self.pos_constraint[1])

    def update_velocities(self):
        """Update all particles' velocities"""
        # Generate random weights in [0,1]
        rand_1 = torch.rand((self.batch_size, self.n_particles, self.dim), device=self.device)
        rand_2 = torch.rand((self.batch_size, self.n_particles, self.dim), device=self.device)

        # Update velocities
        self.velocities = (
            self.w * self.velocities
            + self.c1 * rand_1 * (self.pbest - self.positions)
            + self.c2 * rand_2 * (self.gbest.unsqueeze(1).expand(-1, self.n_particles, -1) - self.positions)
        )
        self.velocities = self.velocities.clip(self.vel_constraint[0], self.vel_constraint[1])


class PSO:
    def __init__(
        self,
        batch_size,
        num_iters,
        n_particles,
        dim,
        c1,
        c2,
        w,
        pos_constraint,
        vel_constraint,
    ):
        self.swarm = Swarm(batch_size, n_particles, dim, c1, c2, w, pos_constraint, vel_constraint)
        self.num_iters = num_iters

    def optimize(self, cost_fn: Callable):
        """
        Find best particle (minimizer) based on the specified particle cost function that accepts a batch of particles.
        Terminates by number of iterations.
        Args:
            cost_fn: Function that returns the cost for a batch of particles with shape (*, dim)
        Return:
            (self.swarm.gbest_cost, self.swarm.gbest) where gbest_cost of shape (N, ) and gbest of shape (N, dim)
        """
        # Reinitialize swarm
        self.swarm.reset()

        for _ in tqdm.tqdm(range(self.num_iters)):
            # Reshape particles positions from (N, n_particles, dim) to (N * n_particles, dim)
            particles_pos = self.swarm.positions.view(-1, self.swarm.dim)
            # Evaluate cost at their current positions
            particles_cost = cost_fn(particles_pos)
            particles_cost = particles_cost.view(self.swarm.batch_size, self.swarm.n_particles)
            # Update personal best (N, n_particles)
            update_mask = particles_cost < self.swarm.pbest_costs
            self.swarm.pbest_costs = torch.where(update_mask, particles_cost, self.swarm.pbest_costs)
            self.swarm.pbest = torch.where(update_mask.unsqueeze(-1), self.swarm.positions, self.swarm.pbest)
            # Update global best (N,)
            batch_best_cost, batch_best_idx = self.swarm.pbest_costs.min(dim=1)
            self.swarm.gbest_cost = torch.min(self.swarm.gbest_cost, batch_best_cost)
            self.swarm.gbest = self.swarm.pbest[torch.arange(self.swarm.batch_size), batch_best_idx]
            # Update all particles's positions and velocities.
            self.swarm.update_velocities()
            self.swarm.update_positions()

        return (
            self.swarm.gbest_cost,
            self.swarm.gbest,
        )
