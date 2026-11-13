//! `nitro._core.Policy`: the exported-policy reader, so Python can check an exported file against the
//! PyTorch model (the policy parity test).

use std::path::PathBuf;

use nitro_core::policy::Policy;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyclass(frozen, name = "Policy")]
pub struct PyPolicy {
    policy: Policy,
}

#[pymethods]
impl PyPolicy {
    #[new]
    fn new(path: PathBuf) -> PyResult<Self> {
        let policy = Policy::load(&path).map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(Self { policy })
    }

    #[getter]
    fn obs_size(&self) -> usize {
        self.policy.obs_size()
    }

    #[getter]
    fn num_actions(&self) -> usize {
        self.policy.num_actions()
    }

    /// The header's game settings and provenance as a dict of plain values.
    fn header(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyDict>> {
        let header = self.policy.header();
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("format", header.format)?;
        dict.set_item("run", &header.run)?;
        dict.set_item("agent_steps", header.agent_steps)?;
        dict.set_item("created", &header.created)?;
        dict.set_item("team_size", header.game.team_size)?;
        dict.set_item("opponents", header.game.opponents)?;
        dict.set_item("obs_team_size", header.game.obs_team_size)?;
        dict.set_item("tick_skip", header.game.tick_skip)?;
        dict.set_item("action_delay", header.game.action_delay)?;
        dict.set_item("layers", header.layers.len())?;
        Ok(dict.unbind())
    }

    /// Action logits for one observation, FP32 on the CPU.
    fn logits(&self, obs: Vec<f32>) -> PyResult<Vec<f32>> {
        if obs.len() != self.policy.obs_size() {
            return Err(PyValueError::new_err(format!(
                "the observation has {} values; the policy expects {}",
                obs.len(),
                self.policy.obs_size()
            )));
        }
        let mut out = vec![0.0; self.policy.num_actions()];
        self.policy.logits(&obs, &mut out);
        Ok(out)
    }

    /// Logits for many observations (a flat list, `rows * obs_size` long); returns `rows * num_actions`.
    fn logits_batch(&self, obs: Vec<f32>, rows: usize) -> PyResult<Vec<f32>> {
        let (inputs, outputs) = (self.policy.obs_size(), self.policy.num_actions());
        if obs.len() != rows * inputs {
            return Err(PyValueError::new_err(format!(
                "{} values do not make {rows} observations of {inputs}",
                obs.len()
            )));
        }
        let mut out = vec![0.0; rows * outputs];
        for (row, chunk) in obs.chunks_exact(inputs).enumerate() {
            self.policy
                .logits(chunk, &mut out[row * outputs..(row + 1) * outputs]);
        }
        Ok(out)
    }

    /// Whether the policy starts with an attention encoder, which reads any number of cars a side.
    #[getter]
    fn reads_any_cars(&self) -> bool {
        self.policy.reads_any_cars()
    }

    /// Logits for one observation holding exactly `teammates` and `opponents` car blocks, however many
    /// (the bot's observation in a game beyond the policy's room).
    fn logits_cars(&self, obs: Vec<f32>, teammates: usize, opponents: usize) -> PyResult<Vec<f32>> {
        let Some((own, block)) = self.policy.car_blocks() else {
            return Err(PyValueError::new_err(
                "only a policy with an attention encoder reads any number of cars",
            ));
        };
        if obs.len() != own + (teammates + opponents) * block {
            return Err(PyValueError::new_err(format!(
                "{} values are not an own part of {own} and {} car blocks of {block}",
                obs.len(),
                teammates + opponents
            )));
        }
        let mut out = vec![0.0; self.policy.num_actions()];
        self.policy
            .logits_cars(&obs, teammates, opponents, &mut out);
        Ok(out)
    }

    /// The allowed action with the highest logit (`mask`: one 0/1 per action).
    #[staticmethod]
    fn argmax(logits: Vec<f32>, mask: Vec<u8>) -> PyResult<usize> {
        if logits.len() != mask.len() {
            return Err(PyValueError::new_err("logits and mask differ in length"));
        }
        Ok(Policy::argmax(&logits, &mask))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPolicy>()
}
