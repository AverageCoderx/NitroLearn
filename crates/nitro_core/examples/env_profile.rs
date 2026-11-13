//! Steps a pool from a settings JSON (as `nitro.envs.core_settings` writes it) with seeded random allowed
//! actions, for profiling the games' processor time: `env_profile <settings.json> <steps>`.

use nitro_core::config::PoolConfig;
use nitro_core::driver::PoolDriver;

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let json = std::fs::read_to_string(&args[1]).map_err(|e| e.to_string())?;
    let steps: usize = args[2]
        .parse()
        .map_err(|_| "steps must be a number".to_owned())?;
    let config = PoolConfig::from_json(&json)?;
    let mut driver = PoolDriver::new(&config)?;
    let mut rng = fastrand::Rng::with_seed(1);
    let actions = driver.buffers[0].mask.len() / driver.buffers[0].actions.len();
    let start = std::time::Instant::now();
    for _ in 0..steps {
        let masks: Vec<Vec<u8>> = driver.buffers.iter().map(|b| b.mask.clone()).collect();
        let mut group = 0;
        driver.step(|chosen| {
            for (agent, action) in chosen.iter_mut().enumerate() {
                let allowed = &masks[group][agent * actions..(agent + 1) * actions];
                loop {
                    let pick = rng.usize(..actions);
                    if allowed[pick] != 0 {
                        *action = pick as i32;
                        break;
                    }
                }
            }
            group += 1;
        })?;
    }
    let stats = driver.pool.take_stats();
    eprintln!(
        "{} env steps in {:.2} s; step {:.1} us, physics {:.1} us each",
        stats.steps,
        start.elapsed().as_secs_f64(),
        stats.step_time.as_secs_f64() * 1e6 / stats.steps as f64,
        stats.physics_time.as_secs_f64() * 1e6 / stats.steps as f64
    );
    Ok(())
}
