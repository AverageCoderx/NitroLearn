//! Drives every group of a pool from Rust with caller-supplied actions: the Rust-only baseline of the
//! throughput benchmark and the driver of the pool tests.

use crate::config::PoolConfig;
use crate::pool::{EnvPool, OwnedBuffers, Work};

pub struct PoolDriver {
    // Declared first so it is dropped first: its workers stop before the buffers are freed.
    pub pool: EnvPool,
    pub buffers: Vec<OwnedBuffers>,
}

impl PoolDriver {
    pub fn new(config: &PoolConfig) -> Result<Self, String> {
        let pool = EnvPool::new(config)?;
        let mut buffers: Vec<_> = (0..pool.num_groups())
            .map(|group| OwnedBuffers::new(pool.shape(group)))
            .collect();
        for (group, owned) in buffers.iter_mut().enumerate() {
            pool.submit(group, Work::Observe, owned.view(pool.shape(group)))?;
            pool.wait(group)?;
        }
        Ok(Self { pool, buffers })
    }

    /// One step of every group, all running at once; returns when all are done. `fill` writes the
    /// actions of each group in turn.
    pub fn step(&mut self, mut fill: impl FnMut(&mut [i32])) -> Result<(), String> {
        for group in 0..self.buffers.len() {
            self.submit(group, &mut fill)?;
        }
        self.finish()
    }

    /// Starts the next step of each group as soon as that group's previous step is done, and returns
    /// with the groups still running: the trainer's pattern, where one group is prepared while the
    /// others step. Call [`PoolDriver::finish`] before reading the buffers.
    pub fn step_pipelined(&mut self, mut fill: impl FnMut(&mut [i32])) -> Result<(), String> {
        for group in 0..self.buffers.len() {
            self.pool.wait(group)?;
            self.submit(group, &mut fill)?;
        }
        Ok(())
    }

    pub fn finish(&self) -> Result<(), String> {
        (0..self.buffers.len()).try_for_each(|group| self.pool.wait(group))
    }

    fn submit(&mut self, group: usize, fill: &mut impl FnMut(&mut [i32])) -> Result<(), String> {
        let owned = &mut self.buffers[group];
        fill(&mut owned.actions);
        self.pool
            .submit(group, Work::Step, owned.view(self.pool.shape(group)))
    }
}
