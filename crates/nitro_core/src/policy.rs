//! A trained policy as a plain file (`policy.nwt`) and its CPU forward pass, for the RLBot bot.
//!
//! `nitro export` writes the file from a checkpoint (`python/nitro/export.py`); this module reads it and
//! computes action logits with no PyTorch anywhere. Format, all little-endian:
//!
//! ```text
//! b"NITROPOL"  u32 format version (1 to 8)  u32 header length  header (JSON, UTF-8)  f32 weights...
//! ```
//!
//! Format 2 adds the `entity_encoder` layer, an attention encoder in front of the policy;
//! format 3 its `skip` flag (the own part of the observation passed on beside the encoder's
//! output). Format 4 adds the `entity_talk` layer, the attention encoder whose tokens first talk to each
//! other (`model.encoder = "attention_communication"`). Format 6 adds the `transformer` layer
//! (`model.encoder = "transformer"`). Format 8 adds `residual_start` and `residual_add` (`model.residual`):
//! the vector at the start is kept and added back at the add, `x + f(x)` around the layers between.
//! The numbers 5 and 7 are not used. Plain networks are written as format 1, encoders without the flag as
//! format 2, and every format loads.
//!
//! An attention encoder's weights do not depend on how many cars it reads, so a policy that starts with
//! one plays any number of cars a side (`Policy::logits_cars`), beyond the room it was trained with.
//!
//! The header describes the game settings the policy was trained with (so the bot can build the same
//! observation) and the network layer by layer. The weights follow in that order: for a `linear` layer
//! its matrix (`outputs` rows of `inputs`, PyTorch's layout) then its bias; for a `layer_norm` its scale
//! then its shift; activations carry nothing. Only the policy (with any shared trunk) is exported; the
//! value network is training-only.

use std::fmt;
use std::fs;
use std::io;
use std::path::Path;

use serde::Deserialize;

use crate::team::Team;

pub const MAGIC: &[u8; 8] = b"NITROPOL";
/// The newest format this bot reads; every lower one loads too (but `UNUSED_FORMATS`).
pub const FORMAT_VERSION: u32 = 8;
/// The first format that may hold residual blocks (`residual_start`, `residual_add`).
pub const RESIDUAL_FORMAT: u32 = 8;
/// The format plain networks are written in (no attention encoder): the first one.
pub const PLAIN_FORMAT: u32 = 1;
const OLDEST_FORMAT: u32 = 1;
/// Format numbers no file is written in.
const UNUSED_FORMATS: [u32; 2] = [5, 7];
/// PyTorch's default LayerNorm epsilon.
const LAYER_NORM_EPS: f32 = 1e-5;
/// PyTorch's default LeakyReLU slope.
const LEAKY_SLOPE: f32 = 0.01;
const MAX_HEADER_BYTES: usize = 1 << 20;

/// One layer of the exported network, as written in the header.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LayerSpec {
    Linear {
        inputs: usize,
        outputs: usize,
    },
    LayerNorm {
        size: usize,
    },
    Relu,
    LeakyRelu,
    Tanh,
    /// Format 8: keeps the vector here (no weights); the next `residual_add` adds it back. Not nested.
    ResidualStart,
    /// Format 8: adds the vector kept at the last `residual_start` (it must be the same size).
    ResidualAdd,
    /// Format 2: the attention encoder of `nitro.model.EntityEncoder` over a padded observation: the own
    /// part, then `teammates` and `opponents` car blocks of `block` numbers (the last the 1/0 "filled"
    /// flag). Its weights, in order: the query's two linears, the car's two linears, the side embeddings
    /// (2 × width), the null car (width), then per step the query's, the cars' and the feed-forward's norms,
    /// q, k, v, out, and the feed-forward's two linears. Format 3: with `skip`, the output is the own part
    /// followed by the encoder's `width` numbers (no weights of its own).
    EntityEncoder {
        own: usize,
        block: usize,
        teammates: usize,
        opponents: usize,
        width: usize,
        steps: usize,
        heads: usize,
        #[serde(default)]
        skip: bool,
    },
    /// Format 4: `nitro.model.TalkingEncoder`. Tokens: a null token, the ball (the own part's first `ball`
    /// numbers), the own car (`car` numbers from `car_at`, through the cars' embedding with zeros after
    /// them) and the other cars. Each block the tokens talk (self-attention, 2x feed-forward), then the own
    /// part's query looks at them (an `entity_encoder` step). Its weights, in order: the query's, the
    /// car's and the ball's two linears, the kind embeddings (3 × width: teammate, opponent, own car), the
    /// null token, then per block the talk's norm, q, k, v, out, feed-forward norm and two linears, then
    /// the look's three norms, q, k, v, out and two linears.
    EntityTalk {
        own: usize,
        block: usize,
        teammates: usize,
        opponents: usize,
        ball: usize,
        car_at: usize,
        car: usize,
        width: usize,
        blocks: usize,
        heads: usize,
        skip: bool,
    },
    /// Format 6: `nitro.model.TransformerEncoder`. Tokens: with `summary` a learned summary token first;
    /// the own car (its `car` numbers from `car_at` through the cars' embedding with zeros after them, plus
    /// the embedding of the rest of the own part and the own-car kind); the ball (the own part's first `ball`
    /// numbers); every other car (plus the teammate or opponent kind); with `pads` above 0 one token per boost
    /// pad (its number from `pads_at` through the pads' embedding, plus its place's). The rest of the own part
    /// is every number that is not the ball's, the own car's or a pad token's. Then `layers` pre-norm blocks:
    /// masked multi-head self-attention over every present token (one q, k, v layer), then a ReLU
    /// feed-forward `ff` wide, each added back; the output is the first token (summary or own car) after a
    /// final norm. Its weights, in order: the car's, the ball's and the rest's two linears; with pads the
    /// pads' two linears and their places (pads × width); the kinds (3 × width: teammate, opponent, own car);
    /// with `summary` the summary token; per layer the norm, qkv, out, the feed-forward's norm and two
    /// linears; the final norm. With `skip`, the output is the own part followed by the encoder's `width`.
    Transformer {
        own: usize,
        block: usize,
        teammates: usize,
        opponents: usize,
        ball: usize,
        car_at: usize,
        car: usize,
        pads_at: usize,
        pads: usize,
        width: usize,
        layers: usize,
        heads: usize,
        ff: usize,
        summary: bool,
        skip: bool,
    },
}

/// The game settings an observation depends on, as the policy was trained with them.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct GameSettings {
    pub observation: crate::obs::ObsKind,
    pub action_masks: crate::actions::MaskRule,
    pub team_size: usize,
    pub opponents: bool,
    /// Players a side the observation has places for (0 = exactly `team_size`). Absent: 0.
    #[serde(default)]
    pub obs_team_size: usize,
    pub tick_skip: u32,
    pub action_delay: u32,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct PolicyHeader {
    pub format: u32,
    /// Where the policy came from, for people: run name, agent steps, date.
    #[serde(default)]
    pub run: String,
    #[serde(default)]
    pub agent_steps: u64,
    #[serde(default)]
    pub created: String,
    pub game: GameSettings,
    pub obs_size: usize,
    pub num_actions: usize,
    pub layers: Vec<LayerSpec>,
}

#[derive(Debug)]
pub enum PolicyError {
    Io(io::Error),
    NotAPolicyFile,
    UnsupportedVersion(u32),
    BadHeader(String),
    /// The layers do not chain from `obs_size` to `num_actions`.
    Shape(String),
    /// The file holds a different number of weights than its header describes.
    WeightCount {
        expected: usize,
        found: usize,
    },
}

impl fmt::Display for PolicyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "could not read the policy file: {err}"),
            Self::NotAPolicyFile => {
                write!(f, "not a NitroLearn policy file (wrong magic bytes)")
            }
            Self::UnsupportedVersion(v) => write!(
                f,
                "policy file format {v} is not supported by this bot (it knows formats \
                 {OLDEST_FORMAT} to {FORMAT_VERSION}); a newer bot reads it"
            ),
            Self::BadHeader(reason) => write!(f, "the policy file's header is invalid: {reason}"),
            Self::Shape(reason) => {
                write!(f, "the policy file's layers do not fit together: {reason}")
            }
            Self::WeightCount { expected, found } => write!(
                f,
                "the policy file holds {found} weights but its header describes {expected}"
            ),
        }
    }
}

impl std::error::Error for PolicyError {}

impl From<io::Error> for PolicyError {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}

enum Layer {
    Linear {
        inputs: usize,
        outputs: usize,
        weight: Vec<f32>,
        bias: Vec<f32>,
    },
    Norm {
        scale: Vec<f32>,
        shift: Vec<f32>,
    },
    Relu,
    LeakyRelu,
    Tanh,
    ResidualStart,
    ResidualAdd,
    Encoder(Box<Encoder>),
    Talker(Box<Talker>),
    Transformer(Box<Transformer>),
}

impl Layer {
    /// For an attention encoder (which reads any number of cars): the own part's size and a car block's.
    fn reads_cars(&self) -> Option<(usize, usize)> {
        match self {
            Self::Encoder(e) => Some((e.own, e.block)),
            Self::Talker(t) => Some((t.own, t.block)),
            Self::Transformer(t) => Some((t.own, t.block)),
            _ => None,
        }
    }
}

/// A linear layer: `outputs` rows of `inputs` weights (PyTorch's layout), then the biases.
struct Dense {
    inputs: usize,
    outputs: usize,
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl Dense {
    fn apply(&self, x: &[f32]) -> Vec<f32> {
        (0..self.outputs)
            .map(|row| {
                let w = &self.weight[row * self.inputs..(row + 1) * self.inputs];
                self.bias[row] + dot(w, x)
            })
            .collect()
    }

    /// `apply` for inputs side by side (`x.len()` a multiple of `inputs`), each the same bits, the weight
    /// rows shared out to `team`.
    fn apply_rows(&self, x: &[f32], team: &Team<'_>) -> Vec<f32> {
        let mut out = vec![0.0f32; x.len() / self.inputs * self.outputs];
        linear_rows(
            Weights::Full(&self.weight),
            &self.bias,
            self.inputs,
            x,
            &mut out,
            team,
        );
        out
    }
}

struct Norm {
    scale: Vec<f32>,
    shift: Vec<f32>,
}

impl Norm {
    fn applied(&self, x: &[f32]) -> Vec<f32> {
        let mut out = x.to_vec();
        layer_norm(&mut out, &self.scale, &self.shift);
        out
    }
}

struct AttentionStep {
    norm_query: Norm,
    norm_cars: Norm,
    norm_ff: Norm,
    q: Dense,
    k: Dense,
    v: Dense,
    out: Dense,
    ff_in: Dense,
    ff_out: Dense,
}

impl AttentionStep {
    /// One step of the query's attention over `tokens` (masked where not `present`), in place.
    fn attend(&self, query: &mut [f32], tokens: &[Vec<f32>], present: &[bool], heads: usize) {
        let w = query.len();
        let d = w / heads;
        let scale = 1.0 / (d as f32).sqrt();
        let q = self.q.apply(&self.norm_query.applied(query));
        let normed: Vec<Vec<f32>> = tokens.iter().map(|t| self.norm_cars.applied(t)).collect();
        let keys: Vec<Vec<f32>> = normed.iter().map(|t| self.k.apply(t)).collect();
        let values: Vec<Vec<f32>> = normed.iter().map(|t| self.v.apply(t)).collect();
        let mixed = mix(&q, &keys, &values, present, heads, scale);
        for (x, o) in query.iter_mut().zip(self.out.apply(&mixed)) {
            *x += o;
        }
        let ff = self
            .ff_out
            .apply(&relu(self.ff_in.apply(&self.norm_ff.applied(query))));
        for (x, f) in query.iter_mut().zip(ff) {
            *x += f;
        }
    }
}

/// Multi-head attention of one query over keys and values (masked where not `present`): per head, the
/// softmax of the scaled scores mixes the values. Every sum runs in token order, so masked tokens add
/// exact zeros: the answer is the same, bit for bit, with them or without them.
fn mix<K: AsRef<[f32]>>(
    q: &[f32],
    keys: &[K],
    values: &[K],
    present: &[bool],
    heads: usize,
    scale: f32,
) -> Vec<f32> {
    let d = q.len() / heads;
    let mut mixed = vec![0.0f32; q.len()];
    for head in 0..heads {
        let part = head * d..(head + 1) * d;
        let scores: Vec<f32> = keys
            .iter()
            .zip(present)
            .map(|(k, &on)| {
                if on {
                    dot(&q[part.clone()], &k.as_ref()[part.clone()]) * scale
                } else {
                    f32::NEG_INFINITY
                }
            })
            .collect();
        let top = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let weights: Vec<f32> = scores.iter().map(|s| (s - top).exp()).collect();
        let total: f32 = weights.iter().sum();
        for (weight, value) in weights.iter().zip(values) {
            for (m, v) in mixed[part.clone()]
                .iter_mut()
                .zip(&value.as_ref()[part.clone()])
            {
                *m += weight / total * v;
            }
        }
    }
    mixed
}

/// The attention encoder: the own car's query attends to the other cars, same maths as
/// `nitro.model.EntityEncoder` on the CPU in FP32.
struct Encoder {
    own: usize,
    block: usize,
    teammates: usize,
    opponents: usize,
    width: usize,
    heads: usize,
    skip: bool,
    query: (Dense, Dense),
    car: (Dense, Dense),
    sides: Vec<f32>,
    null: Vec<f32>,
    steps: Vec<AttentionStep>,
}

fn relu(mut x: Vec<f32>) -> Vec<f32> {
    x.iter_mut().for_each(|v| *v = v.max(0.0));
    x
}

impl Encoder {
    /// The encoding of `obs`, which holds `teammates` and `opponents` car blocks after the own part.
    fn forward(&self, obs: &[f32], (teammates, opponents): (usize, usize)) -> Vec<f32> {
        let w = self.width;
        let mut query = self
            .query
            .1
            .apply(&relu(self.query.0.apply(&obs[..self.own])));
        let mut tokens = vec![self.null.clone()];
        let mut present = vec![true];
        for place in 0..teammates + opponents {
            let start = self.own + place * self.block;
            let features = &obs[start..start + self.block - 1];
            let side = usize::from(place >= teammates);
            let mut token = self.car.1.apply(&relu(self.car.0.apply(features)));
            for (t, s) in token.iter_mut().zip(&self.sides[side * w..(side + 1) * w]) {
                *t += s;
            }
            tokens.push(token);
            present.push(obs[start + self.block - 1] > 0.5);
        }
        for step in &self.steps {
            step.attend(&mut query, &tokens, &present, self.heads);
        }
        if self.skip {
            let mut out = obs[..self.own].to_vec();
            out.extend(query);
            return out;
        }
        query
    }
}

/// One round of the tokens talking: self-attention over the present tokens, then a 2x
/// feed-forward, each added back (pre-norm).
struct TalkStep {
    norm: Norm,
    q: Dense,
    k: Dense,
    v: Dense,
    out: Dense,
    norm_ff: Norm,
    ff_in: Dense,
    ff_out: Dense,
}

impl TalkStep {
    fn apply(&self, tokens: &[Vec<f32>], present: &[bool], heads: usize) -> Vec<Vec<f32>> {
        let d = tokens[0].len() / heads;
        let scale = 1.0 / (d as f32).sqrt();
        let normed: Vec<Vec<f32>> = tokens.iter().map(|t| self.norm.applied(t)).collect();
        let queries: Vec<Vec<f32>> = normed.iter().map(|t| self.q.apply(t)).collect();
        let keys: Vec<Vec<f32>> = normed.iter().map(|t| self.k.apply(t)).collect();
        let values: Vec<Vec<f32>> = normed.iter().map(|t| self.v.apply(t)).collect();
        tokens
            .iter()
            .zip(&queries)
            .map(|(token, q)| {
                let mixed = mix(q, &keys, &values, present, heads, scale);
                let mut token: Vec<f32> = token
                    .iter()
                    .zip(self.out.apply(&mixed))
                    .map(|(x, o)| x + o)
                    .collect();
                let ff = self
                    .ff_out
                    .apply(&relu(self.ff_in.apply(&self.norm_ff.applied(&token))));
                for (x, f) in token.iter_mut().zip(ff) {
                    *x += f;
                }
                token
            })
            .collect()
    }
}

/// The talking encoder, same maths as `nitro.model.TalkingEncoder` on the CPU in FP32.
struct Talker {
    own: usize,
    block: usize,
    teammates: usize,
    opponents: usize,
    ball: usize,
    car_at: usize,
    car: usize,
    width: usize,
    heads: usize,
    skip: bool,
    query: (Dense, Dense),
    car_mlp: (Dense, Dense),
    ball_mlp: (Dense, Dense),
    kinds: Vec<f32>,
    null: Vec<f32>,
    talk: Vec<TalkStep>,
    listen: Vec<AttentionStep>,
}

impl Talker {
    /// The encoding of `obs`, which holds `teammates` and `opponents` car blocks after the own part.
    fn forward(&self, obs: &[f32], (teammates, opponents): (usize, usize)) -> Vec<f32> {
        let w = self.width;
        let embed = |mlp: &(Dense, Dense), x: &[f32]| mlp.1.apply(&relu(mlp.0.apply(x)));
        let with_kind = |mut token: Vec<f32>, kind: usize| {
            for (t, k) in token.iter_mut().zip(&self.kinds[kind * w..(kind + 1) * w]) {
                *t += k;
            }
            token
        };
        let mut query = embed(&self.query, &obs[..self.own]);
        let mut me = obs[self.car_at..self.car_at + self.car].to_vec();
        me.resize(self.block - 1, 0.0);
        let mut tokens = vec![
            self.null.clone(),
            embed(&self.ball_mlp, &obs[..self.ball]),
            with_kind(embed(&self.car_mlp, &me), 2),
        ];
        let mut present = vec![true; 3];
        for place in 0..teammates + opponents {
            let start = self.own + place * self.block;
            let kind = usize::from(place >= teammates);
            tokens.push(with_kind(
                embed(&self.car_mlp, &obs[start..start + self.block - 1]),
                kind,
            ));
            present.push(obs[start + self.block - 1] > 0.5);
        }
        for (talk, listen) in self.talk.iter().zip(&self.listen) {
            tokens = talk.apply(&tokens, &present, self.heads);
            listen.attend(&mut query, &tokens, &present, self.heads);
        }
        if self.skip {
            let mut out = obs[..self.own].to_vec();
            out.extend(query);
            return out;
        }
        query
    }
}

/// One pre-norm transformer block (`nitro.model._TransformerBlock`): masked multi-head self-attention
/// from one q, k, v layer, then a ReLU feed-forward, each added back.
struct TransformerBlock {
    norm: Norm,
    qkv: Dense,
    out: Dense,
    norm_ff: Norm,
    ff_in: Dense,
    ff_out: Dense,
}

impl TransformerBlock {
    /// In place. A token not `present` is never read (masked as a key, never the output), so it is not
    /// updated either. Each linear layer takes the tokens together (`Dense::apply_rows`), `team` sharing
    /// the work: the same bits as token by token, since every key and value comes from the tokens before
    /// the block and each token's feed-forward reads that token alone.
    fn apply(&self, tokens: &mut [Vec<f32>], present: &[bool], heads: usize, team: &Team<'_>) {
        let all = 0..tokens.len();
        self.apply_groups(tokens, present, std::slice::from_ref(&all), heads, team);
    }

    /// `apply` for several inputs' tokens side by side, `groups` (in order, contiguous) saying which are
    /// whose: a token attends to its own group's alone, so each group gets the bits it gets alone, and the
    /// linear layers read their weights once for all.
    fn apply_groups(
        &self,
        tokens: &mut [Vec<f32>],
        present: &[bool],
        groups: &[std::ops::Range<usize>],
        heads: usize,
        team: &Team<'_>,
    ) {
        let w = tokens[0].len();
        let scale = 1.0 / ((w / heads) as f32).sqrt();
        let normed: Vec<f32> = tokens.iter().flat_map(|t| self.norm.applied(t)).collect();
        let qkv = self.qkv.apply_rows(&normed, team);
        let qkv: Vec<&[f32]> = qkv.chunks_exact(3 * w).collect();
        let keys: Vec<&[f32]> = qkv.iter().map(|x| &x[w..2 * w]).collect();
        let values: Vec<&[f32]> = qkv.iter().map(|x| &x[2 * w..]).collect();
        let mut on = Vec::with_capacity(tokens.len());
        let mut mixed = Vec::with_capacity(tokens.len() * w);
        for group in groups {
            let (keys, values, here) = (
                &keys[group.clone()],
                &values[group.clone()],
                &present[group.clone()],
            );
            for i in group.clone().filter(|&i| present[i]) {
                on.push(i);
                mixed.extend(mix(&qkv[i][..w], keys, values, here, heads, scale));
            }
        }
        let attended = self.out.apply_rows(&mixed, team);
        for (&i, o) in on.iter().zip(attended.chunks_exact(w)) {
            for (t, o) in tokens[i].iter_mut().zip(o) {
                *t += o;
            }
        }
        let normed: Vec<f32> = on
            .iter()
            .flat_map(|&i| self.norm_ff.applied(&tokens[i]))
            .collect();
        let ff = self
            .ff_out
            .apply_rows(&relu(self.ff_in.apply_rows(&normed, team)), team);
        for (&i, f) in on.iter().zip(ff.chunks_exact(w)) {
            for (t, f) in tokens[i].iter_mut().zip(f) {
                *t += f;
            }
        }
    }
}

/// The boost pads as tokens: each pad's number through one embedding, plus its place's.
struct PadTokens {
    at: usize,
    mlp: (Dense, Dense),
    places: Vec<f32>,
}

/// The transformer encoder (`model.encoder = "transformer"`), same maths as `nitro.model.TransformerEncoder`
/// on the CPU in FP32.
struct Transformer {
    own: usize,
    block: usize,
    teammates: usize,
    opponents: usize,
    ball: usize,
    car_at: usize,
    car: usize,
    /// The own part's numbers that go into the own car's token beside its car numbers, in order.
    rest: Vec<usize>,
    width: usize,
    heads: usize,
    skip: bool,
    car_mlp: (Dense, Dense),
    ball_mlp: (Dense, Dense),
    rest_mlp: (Dense, Dense),
    pads: Option<PadTokens>,
    kinds: Vec<f32>,
    summary: Option<Vec<f32>>,
    blocks: Vec<TransformerBlock>,
    norm_out: Norm,
}

impl Transformer {
    /// The encoding of `obs`, which holds `teammates` and `opponents` car blocks after the own part.
    fn forward(&self, obs: &[f32], places: (usize, usize), team: &Team<'_>) -> Vec<f32> {
        let (mut tokens, present) = self.tokens(obs, places);
        for block in &self.blocks {
            block.apply(&mut tokens, &present, self.heads, team);
        }
        self.encoded(obs, &tokens[0])
    }

    /// `forward` for several observations of the trained places at once, each the bits it gets alone:
    /// every block takes all their tokens together (`TransformerBlock::apply_groups`).
    fn forward_batch(&self, inputs: &[&[f32]], team: &Team<'_>) -> Vec<Vec<f32>> {
        let mut tokens = Vec::new();
        let mut present = Vec::new();
        let mut groups = Vec::with_capacity(inputs.len());
        for obs in inputs {
            let (t, p) = self.tokens(obs, (self.teammates, self.opponents));
            groups.push(tokens.len()..tokens.len() + t.len());
            tokens.extend(t);
            present.extend(p);
        }
        for block in &self.blocks {
            block.apply_groups(&mut tokens, &present, &groups, self.heads, team);
        }
        inputs
            .iter()
            .zip(&groups)
            .map(|(obs, group)| self.encoded(obs, &tokens[group.start]))
            .collect()
    }

    /// The tokens going into the first block, and which of them are present.
    fn tokens(
        &self,
        obs: &[f32],
        (teammates, opponents): (usize, usize),
    ) -> (Vec<Vec<f32>>, Vec<bool>) {
        let w = self.width;
        let embed = |mlp: &(Dense, Dense), x: &[f32]| mlp.1.apply(&relu(mlp.0.apply(x)));
        let plus = |mut token: Vec<f32>, extra: &[f32]| {
            for (t, e) in token.iter_mut().zip(extra) {
                *t += e;
            }
            token
        };
        let kind = |k: usize| &self.kinds[k * w..(k + 1) * w];
        let mut tokens = Vec::with_capacity(3 + teammates + opponents);
        tokens.extend(self.summary.clone());
        let mut me = obs[self.car_at..self.car_at + self.car].to_vec();
        me.resize(self.block - 1, 0.0);
        let rest: Vec<f32> = self.rest.iter().map(|&i| obs[i]).collect();
        let me = plus(embed(&self.car_mlp, &me), &embed(&self.rest_mlp, &rest));
        tokens.push(plus(me, kind(2)));
        tokens.push(embed(&self.ball_mlp, &obs[..self.ball]));
        let mut present = vec![true; tokens.len()];
        for place in 0..teammates + opponents {
            let start = self.own + place * self.block;
            let car = embed(&self.car_mlp, &obs[start..start + self.block - 1]);
            tokens.push(plus(car, kind(usize::from(place >= teammates))));
            present.push(obs[start + self.block - 1] > 0.5);
        }
        if let Some(pads) = &self.pads {
            for (i, place) in pads.places.chunks_exact(w).enumerate() {
                tokens.push(plus(
                    embed(&pads.mlp, &obs[pads.at + i..=pads.at + i]),
                    place,
                ));
                present.push(true);
            }
        }
        (tokens, present)
    }

    /// The encoder's output, from the first token after the blocks.
    fn encoded(&self, obs: &[f32], first: &[f32]) -> Vec<f32> {
        let encoded = self.norm_out.applied(first);
        if self.skip {
            let mut out = obs[..self.own].to_vec();
            out.extend(encoded);
            return out;
        }
        encoded
    }
}

/// A loaded policy: the header and the network, ready to run on the CPU.
pub struct Policy {
    header: PolicyHeader,
    net: Net,
}

impl fmt::Debug for Policy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Policy")
            .field("header", &self.header)
            .finish_non_exhaustive()
    }
}

impl Policy {
    pub fn load(path: &Path) -> Result<Self, PolicyError> {
        Self::from_bytes(&fs::read(path)?)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, PolicyError> {
        if bytes.len() < 16 || &bytes[..8] != MAGIC {
            return Err(PolicyError::NotAPolicyFile);
        }
        let u32_at = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        let version = u32_at(8);
        if !(OLDEST_FORMAT..=FORMAT_VERSION).contains(&version) || UNUSED_FORMATS.contains(&version)
        {
            return Err(PolicyError::UnsupportedVersion(version));
        }
        let header_len = u32_at(12) as usize;
        if header_len > MAX_HEADER_BYTES || bytes.len() < 16 + header_len {
            return Err(PolicyError::BadHeader(
                "header length exceeds the file".into(),
            ));
        }
        let header: PolicyHeader = serde_json::from_slice(&bytes[16..16 + header_len])
            .map_err(|err| PolicyError::BadHeader(err.to_string()))?;
        if header.format != version {
            return Err(PolicyError::BadHeader(
                "the header's format number differs from the file's".into(),
            ));
        }
        let body = &bytes[16 + header_len..];
        if !body.len().is_multiple_of(4) {
            return Err(PolicyError::BadHeader(
                "the weights are not a whole number of 32-bit floats".into(),
            ));
        }
        let weights: Vec<f32> = body
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect();
        Self::assemble(header, &weights)
    }

    fn assemble(header: PolicyHeader, weights: &[f32]) -> Result<Self, PolicyError> {
        let game = &header.game;
        if !(1..=3).contains(&game.team_size)
            || game.tick_skip == 0
            || game.action_delay >= game.tick_skip
        {
            return Err(PolicyError::BadHeader(format!(
                "game settings team_size {}, tick_skip {}, action_delay {} cannot be played",
                game.team_size, game.tick_skip, game.action_delay
            )));
        }
        let mut at = 0;
        let mut take = |count: usize| -> Result<Vec<f32>, PolicyError> {
            let slice = weights
                .get(at..at + count)
                .ok_or(PolicyError::WeightCount {
                    expected: at + count,
                    found: weights.len(),
                })?;
            at += count;
            Ok(slice.to_vec())
        };
        let net = assemble_net(
            &header.layers,
            header.format,
            header.obs_size,
            header.num_actions,
            &mut take,
        )?;
        if at != weights.len() {
            return Err(PolicyError::WeightCount {
                expected: at,
                found: weights.len(),
            });
        }
        if weights.iter().any(|w| !w.is_finite()) {
            return Err(PolicyError::BadHeader("a weight is NaN or infinite".into()));
        }
        Ok(Self { header, net })
    }

    pub fn header(&self) -> &PolicyHeader {
        &self.header
    }

    pub fn obs_size(&self) -> usize {
        self.header.obs_size
    }

    pub fn num_actions(&self) -> usize {
        self.header.num_actions
    }

    /// Action logits for one observation (`obs.len() == obs_size`, `out.len() == num_actions`), in FP32,
    /// the same maths as the PyTorch model on the CPU.
    pub fn logits(&self, obs: &[f32], out: &mut [f32]) {
        assert_eq!(obs.len(), self.header.obs_size, "observation size");
        assert_eq!(out.len(), self.header.num_actions, "logit count");
        self.net.run(obs, out);
    }

    /// `logits` with `team` sharing each linear layer's work (the RLBot bot: one decision split across
    /// cores); the same bits.
    pub fn logits_team(&self, obs: &[f32], out: &mut [f32], team: &Team<'_>) {
        assert_eq!(obs.len(), self.header.obs_size, "observation size");
        assert_eq!(out.len(), self.header.num_actions, "logit count");
        self.net.run_cars(obs, None, out, team);
    }

    /// Action logits for `rows` observations side by side (`obs.len() == rows * obs_size`), each the same
    /// bits as `logits` gives it; a plain network reads each weight once per four of them, `team` sharing
    /// the work.
    pub fn logits_rows(&self, obs: &[f32], rows: usize, out: &mut [f32], team: &Team<'_>) {
        self.net.run_rows(obs, rows, out, team);
    }

    /// Has `logits_rows` read the weights in half precision: half the bytes to read, at a small cost in
    /// precision; `logits` and `decide` are unchanged. Returns whether it does (only on processors that
    /// convert half precision, x86 with F16C).
    pub fn read_half(&mut self, on: bool) -> bool {
        self.net.read_half(on)
    }

    /// Whether the policy starts with an attention encoder, whose weights do not depend on the number of
    /// cars, so it plays any number a side (`logits_cars`).
    pub fn reads_any_cars(&self) -> bool {
        self.car_blocks().is_some()
    }

    /// For a policy that `reads_any_cars`: its observation's own part and car block sizes.
    pub fn car_blocks(&self) -> Option<(usize, usize)> {
        self.net.layers.first().and_then(Layer::reads_cars)
    }

    /// `logits` for an observation with exactly `teammates` and `opponents` car blocks
    /// (`ObsKind::build_cars`), however many, for a policy that `reads_any_cars`. With the trained
    /// places, the same answer as `logits`.
    pub fn logits_cars(&self, obs: &[f32], teammates: usize, opponents: usize, out: &mut [f32]) {
        assert!(
            self.reads_any_cars(),
            "only an attention encoder reads any number of cars"
        );
        self.logits_cars_team(obs, teammates, opponents, out, &Team::ALONE);
    }

    /// `logits_cars` with `team` sharing the work, as `logits_team`.
    pub fn logits_cars_team(
        &self,
        obs: &[f32],
        teammates: usize,
        opponents: usize,
        out: &mut [f32],
        team: &Team<'_>,
    ) {
        assert!(
            self.reads_any_cars(),
            "only an attention encoder reads any number of cars"
        );
        assert_eq!(out.len(), self.header.num_actions, "logit count");
        self.net
            .run_cars(obs, Some((teammates, opponents)), out, team);
    }

    /// `logits_team` for `rows` observations of the trained size side by side (`obs.len() == rows *
    /// obs_size`, `out.len() == rows * num_actions`), each the bits `logits` gives it, the weights read
    /// once for all rows (the cars of a hivemind decide in one pass).
    pub fn logits_batch(&self, obs: &[f32], rows: usize, out: &mut [f32], team: &Team<'_>) {
        self.net.run_batch(obs, rows, out, team);
    }

    /// The action for one observation and its mask: `pick` chooses from the logits and the mask (argmax or
    /// sample).
    pub fn decide(
        &self,
        obs: &[f32],
        mask: &[u8],
        pick: &mut dyn FnMut(&[f32], &[u8]) -> usize,
    ) -> usize {
        assert_eq!(mask.len(), self.header.num_actions, "mask size");
        let mut logits = vec![0.0f32; self.header.num_actions];
        self.logits(obs, &mut logits);
        pick(&logits, mask)
    }

    /// The allowed action with the highest logit.
    pub fn argmax(logits: &[f32], mask: &[u8]) -> usize {
        logits
            .iter()
            .zip(mask)
            .enumerate()
            .filter(|(_, (_, allowed))| **allowed != 0)
            .max_by(|a, b| a.1.0.total_cmp(b.1.0))
            .map_or(0, |(index, _)| index)
    }

    /// An allowed action drawn with the policy's probabilities (Gumbel-max over the allowed logits,
    /// as the trainer samples), using `rng` for the noise.
    pub fn sample(logits: &[f32], mask: &[u8], rng: &mut fastrand::Rng) -> usize {
        let mut best = None;
        for (index, (&logit, &allowed)) in logits.iter().zip(mask).enumerate() {
            if allowed == 0 {
                continue;
            }
            let uniform = rng.f32().clamp(1e-20, 1.0 - 1e-7);
            let score = logit - (-uniform.ln()).ln();
            if best.is_none_or(|(_, top)| score > top) {
                best = Some((index, score));
            }
        }
        best.map_or(0, |(index, _)| index)
    }
}

/// One network of a policy file: its layers, ready to run.
struct Net {
    layers: Vec<Layer>,
    /// Widest layer, so a forward pass can reuse two scratch buffers.
    width: usize,
    inputs: usize,
    outputs: usize,
    /// The linear layers' weights in half precision, in order, when `run_rows` reads those.
    half: Option<Vec<Vec<u16>>>,
}

impl Net {
    /// The network's outputs for one input (`x.len() == inputs`, `out.len() == outputs`), in FP32.
    fn run(&self, x: &[f32], out: &mut [f32]) {
        self.run_cars(x, None, out, &Team::ALONE);
    }

    /// `run` for `rows` inputs side by side, each giving the same bits as alone. Linear layers take the
    /// inputs together (`linear_rows`), so a large network's weights are read once per four inputs instead
    /// of once per input; attention layers run input by input.
    fn run_rows(&self, x: &[f32], rows: usize, out: &mut [f32], team: &Team<'_>) {
        assert_eq!(x.len(), rows * self.inputs, "input size");
        assert_eq!(out.len(), rows * self.outputs, "output count");
        let attention = self.layers.iter().any(|layer| layer.reads_cars().is_some());
        if attention {
            for (input, output) in x
                .chunks_exact(self.inputs)
                .zip(out.chunks_exact_mut(self.outputs))
            {
                self.run(input, output);
            }
            return;
        }
        let mut current = x.to_vec();
        let mut width = self.inputs;
        let mut next = Vec::with_capacity(rows * self.width);
        let mut skip = Vec::new();
        let mut linear = 0;
        for layer in &self.layers {
            match layer {
                Layer::Linear {
                    inputs,
                    outputs,
                    weight,
                    bias,
                } => {
                    next.clear();
                    next.resize(rows * outputs, 0.0);
                    let weights = match &self.half {
                        Some(half) => Weights::Half(&half[linear]),
                        None => Weights::Full(weight),
                    };
                    linear_rows(weights, bias, *inputs, &current, &mut next, team);
                    linear += 1;
                    std::mem::swap(&mut current, &mut next);
                    width = *outputs;
                }
                Layer::Norm { scale, shift } => current
                    .chunks_exact_mut(width)
                    .for_each(|values| layer_norm(values, scale, shift)),
                Layer::Relu => current.iter_mut().for_each(|x| *x = x.max(0.0)),
                Layer::LeakyRelu => current
                    .iter_mut()
                    .for_each(|x| *x = if *x >= 0.0 { *x } else { LEAKY_SLOPE * *x }),
                Layer::Tanh => current.iter_mut().for_each(|x| *x = x.tanh()),
                Layer::ResidualStart => keep(&mut skip, &current),
                Layer::ResidualAdd => add_back(&mut current, &skip),
                Layer::Encoder(_) | Layer::Talker(_) | Layer::Transformer(_) => {
                    unreachable!("attention layers run input by input")
                }
            }
        }
        out.copy_from_slice(&current);
    }

    /// Has `run_rows` read the linear layers' weights in half precision (when `on` and the processor
    /// converts them, x86 with F16C), half the bytes from memory; returns whether it does.
    fn read_half(&mut self, on: bool) -> bool {
        self.half = None;
        if !on || !half_supported() {
            return false;
        }
        let halves = self
            .layers
            .iter()
            .filter_map(|layer| match layer {
                Layer::Linear { weight, .. } => Some(to_half(weight)),
                _ => None,
            })
            .collect();
        self.half = Some(halves);
        true
    }

    /// `run`, with `cars` (teammates, opponents) the number of car blocks the input holds when it is not
    /// the trained number (the first layer is then an attention encoder, which reads any number).
    fn run_cars(&self, x: &[f32], cars: Option<(usize, usize)>, out: &mut [f32], team: &Team<'_>) {
        match cars {
            None => assert_eq!(x.len(), self.inputs, "input size"),
            Some((teammates, opponents)) => {
                let (own, block) = self.layers[0]
                    .reads_cars()
                    .expect("an attention encoder comes first");
                assert_eq!(
                    x.len(),
                    own + (teammates + opponents) * block,
                    "observation size"
                );
            }
        }
        assert_eq!(out.len(), self.outputs, "output count");
        let mut current = Vec::with_capacity(self.width);
        let mut next = vec![0.0f32; self.width];
        let mut skip = Vec::new();
        current.extend_from_slice(x);
        for layer in &self.layers {
            match layer {
                Layer::Linear {
                    inputs,
                    outputs,
                    weight,
                    bias,
                } => {
                    next.clear();
                    if team.members() > 1 {
                        // The same bits as below (`linear_rows`), the weight rows shared out.
                        next.resize(*outputs, 0.0);
                        linear_rows(
                            Weights::Full(weight),
                            bias,
                            *inputs,
                            &current,
                            &mut next,
                            team,
                        );
                    } else {
                        next.extend((0..*outputs).map(|row| {
                            let w = &weight[row * inputs..(row + 1) * inputs];
                            bias[row] + dot(w, &current)
                        }));
                    }
                    std::mem::swap(&mut current, &mut next);
                }
                Layer::Norm { scale, shift } => layer_norm(&mut current, scale, shift),
                Layer::Relu => current.iter_mut().for_each(|x| *x = x.max(0.0)),
                Layer::LeakyRelu => current
                    .iter_mut()
                    .for_each(|x| *x = if *x >= 0.0 { *x } else { LEAKY_SLOPE * *x }),
                Layer::Tanh => current.iter_mut().for_each(|x| *x = x.tanh()),
                Layer::ResidualStart => keep(&mut skip, &current),
                Layer::ResidualAdd => add_back(&mut current, &skip),
                Layer::Encoder(encoder) => {
                    let places = cars.unwrap_or((encoder.teammates, encoder.opponents));
                    let encoded = encoder.forward(&current, places);
                    current.clear();
                    current.extend(encoded);
                }
                Layer::Talker(talker) => {
                    let places = cars.unwrap_or((talker.teammates, talker.opponents));
                    let encoded = talker.forward(&current, places);
                    current.clear();
                    current.extend(encoded);
                }
                Layer::Transformer(transformer) => {
                    let places = cars.unwrap_or((transformer.teammates, transformer.opponents));
                    let encoded = transformer.forward(&current, places, team);
                    current.clear();
                    current.extend(encoded);
                }
            }
        }
        out.copy_from_slice(&current);
    }

    /// `run` for `rows` inputs of the trained size side by side, each the bits it gets alone, `team`
    /// sharing the work: every linear layer (a transformer encoder's too) reads its weights once for all
    /// rows (both cars of a hivemind in one pass). Other attention encoders run row by row.
    fn run_batch(&self, x: &[f32], rows: usize, out: &mut [f32], team: &Team<'_>) {
        assert_eq!(x.len(), rows * self.inputs, "input size");
        assert_eq!(out.len(), rows * self.outputs, "output count");
        let mut current = x.to_vec();
        let mut width = self.inputs;
        let mut next = Vec::with_capacity(rows * self.width);
        let mut skip = Vec::new();
        for layer in &self.layers {
            match layer {
                Layer::Linear {
                    inputs,
                    outputs,
                    weight,
                    bias,
                } => {
                    next.clear();
                    next.resize(rows * outputs, 0.0);
                    linear_rows(
                        Weights::Full(weight),
                        bias,
                        *inputs,
                        &current,
                        &mut next,
                        team,
                    );
                    std::mem::swap(&mut current, &mut next);
                    width = *outputs;
                }
                Layer::Norm { scale, shift } => current
                    .chunks_exact_mut(width)
                    .for_each(|values| layer_norm(values, scale, shift)),
                Layer::Relu => current.iter_mut().for_each(|x| *x = x.max(0.0)),
                Layer::LeakyRelu => current
                    .iter_mut()
                    .for_each(|x| *x = if *x >= 0.0 { *x } else { LEAKY_SLOPE * *x }),
                Layer::Tanh => current.iter_mut().for_each(|x| *x = x.tanh()),
                Layer::ResidualStart => keep(&mut skip, &current),
                Layer::ResidualAdd => add_back(&mut current, &skip),
                Layer::Transformer(transformer) => {
                    let inputs: Vec<&[f32]> = current.chunks_exact(width).collect();
                    let encoded = transformer.forward_batch(&inputs, team);
                    width = encoded[0].len();
                    current = encoded.concat();
                }
                Layer::Encoder(_) | Layer::Talker(_) => {
                    let mut encoded = Vec::new();
                    for input in current.chunks_exact(width) {
                        let encoded_row = match layer {
                            Layer::Encoder(e) => e.forward(input, (e.teammates, e.opponents)),
                            Layer::Talker(t) => t.forward(input, (t.teammates, t.opponents)),
                            _ => unreachable!(),
                        };
                        width = encoded_row.len();
                        encoded.extend(encoded_row);
                    }
                    current = encoded;
                }
            }
        }
        out.copy_from_slice(&current);
    }
}

/// The layers `specs` describe, reading their weights with `take`, for `inputs` numbers in and `outputs`
/// out; `format` is the file's (some layers need a newer one).
fn assemble_net(
    specs: &[LayerSpec],
    format: u32,
    inputs: usize,
    outputs: usize,
    take: &mut impl FnMut(usize) -> Result<Vec<f32>, PolicyError>,
) -> Result<Net, PolicyError> {
    let mut layers = Vec::with_capacity(specs.len());
    let mut size = inputs;
    let mut width = size;
    // The size kept by an open `residual_start`.
    let mut kept: Option<usize> = None;
    for (index, spec) in specs.iter().enumerate() {
        layers.push(match *spec {
            LayerSpec::Linear { inputs, outputs } => {
                if inputs != size || outputs == 0 {
                    return Err(PolicyError::Shape(format!(
                        "layer {index} takes {inputs} inputs but receives {size}"
                    )));
                }
                let weight = take(outputs * inputs)?;
                let bias = take(outputs)?;
                size = outputs;
                Layer::Linear {
                    inputs,
                    outputs,
                    weight,
                    bias,
                }
            }
            LayerSpec::LayerNorm { size: norm_size } => {
                if norm_size != size {
                    return Err(PolicyError::Shape(format!(
                        "layer {index} normalises {norm_size} values but receives {size}"
                    )));
                }
                Layer::Norm {
                    scale: take(size)?,
                    shift: take(size)?,
                }
            }
            LayerSpec::Relu => Layer::Relu,
            LayerSpec::LeakyRelu => Layer::LeakyRelu,
            LayerSpec::Tanh => Layer::Tanh,
            LayerSpec::ResidualStart | LayerSpec::ResidualAdd if format < RESIDUAL_FORMAT => {
                return Err(PolicyError::BadHeader(format!(
                    "a residual block needs format {RESIDUAL_FORMAT}"
                )));
            }
            LayerSpec::ResidualStart => {
                if kept.replace(size).is_some() {
                    return Err(PolicyError::Shape(format!(
                        "layer {index} starts a residual block inside another"
                    )));
                }
                Layer::ResidualStart
            }
            LayerSpec::ResidualAdd => {
                if kept.take() != Some(size) {
                    return Err(PolicyError::Shape(format!(
                        "layer {index} adds back a residual block that was not started at {size} values"
                    )));
                }
                Layer::ResidualAdd
            }
            LayerSpec::EntityEncoder {
                own,
                block,
                teammates,
                opponents,
                width: encoded,
                steps,
                heads,
                skip,
            } => {
                if format < 2 {
                    return Err(PolicyError::BadHeader(
                        "an attention encoder needs format 2".into(),
                    ));
                }
                if skip && format < 3 {
                    return Err(PolicyError::BadHeader(
                        "an attention encoder with the own part passed beside it needs format 3".into(),
                    ));
                }
                if index != 0 || own + (teammates + opponents) * block != size || block < 2 {
                    return Err(PolicyError::Shape(format!(
                        "the attention encoder must come first and read all {size} observation numbers"
                    )));
                }
                if heads == 0 || encoded % heads != 0 || steps == 0 {
                    return Err(PolicyError::Shape(format!(
                        "an encoder {encoded} wide cannot have {heads} heads and {steps} steps"
                    )));
                }
                let mut dense = |inputs: usize, outputs: usize| -> Result<Dense, PolicyError> {
                    Ok(Dense {
                        inputs,
                        outputs,
                        weight: take(outputs * inputs)?,
                        bias: take(outputs)?,
                    })
                };
                let query = (dense(own, encoded)?, dense(encoded, encoded)?);
                let car = (dense(block - 1, encoded)?, dense(encoded, encoded)?);
                let sides = take(2 * encoded)?;
                let null = take(encoded)?;
                let mut attention = Vec::with_capacity(steps);
                for _ in 0..steps {
                    let mut norm = || -> Result<Norm, PolicyError> {
                        Ok(Norm {
                            scale: take(encoded)?,
                            shift: take(encoded)?,
                        })
                    };
                    let (norm_query, norm_cars, norm_ff) = (norm()?, norm()?, norm()?);
                    let mut dense = |inputs: usize, outputs: usize| -> Result<Dense, PolicyError> {
                        Ok(Dense {
                            inputs,
                            outputs,
                            weight: take(outputs * inputs)?,
                            bias: take(outputs)?,
                        })
                    };
                    attention.push(AttentionStep {
                        norm_query,
                        norm_cars,
                        norm_ff,
                        q: dense(encoded, encoded)?,
                        k: dense(encoded, encoded)?,
                        v: dense(encoded, encoded)?,
                        out: dense(encoded, encoded)?,
                        ff_in: dense(encoded, 4 * encoded)?,
                        ff_out: dense(4 * encoded, encoded)?,
                    });
                }
                size = encoded + if skip { own } else { 0 };
                Layer::Encoder(Box::new(Encoder {
                    own,
                    block,
                    teammates,
                    opponents,
                    width: encoded,
                    heads,
                    skip,
                    query,
                    car,
                    sides,
                    null,
                    steps: attention,
                }))
            }
            LayerSpec::EntityTalk {
                own,
                block,
                teammates,
                opponents,
                ball,
                car_at,
                car,
                width: encoded,
                blocks,
                heads,
                skip,
            } => {
                if format < 4 {
                    return Err(PolicyError::BadHeader(
                        "an attention_communication encoder needs format 4".into(),
                    ));
                }
                if index != 0 || own + (teammates + opponents) * block != size || block < 2 {
                    return Err(PolicyError::Shape(format!(
                        "the attention encoder must come first and read all {size} observation numbers"
                    )));
                }
                if ball == 0 || ball > own || car == 0 || car_at + car > own || car > block - 1 {
                    return Err(PolicyError::Shape(format!(
                        "the ball ({ball}) and own car ({car} from {car_at}) must lie in the own part \
                         ({own}) and the car fit a car block ({})",
                        block - 1
                    )));
                }
                if heads == 0 || encoded % heads != 0 || blocks == 0 {
                    return Err(PolicyError::Shape(format!(
                        "an encoder {encoded} wide cannot have {heads} heads and {blocks} blocks"
                    )));
                }
                let query = (
                    read_dense(take, own, encoded)?,
                    read_dense(take, encoded, encoded)?,
                );
                let car_mlp = (
                    read_dense(take, block - 1, encoded)?,
                    read_dense(take, encoded, encoded)?,
                );
                let ball_mlp = (
                    read_dense(take, ball, encoded)?,
                    read_dense(take, encoded, encoded)?,
                );
                let kinds = take(3 * encoded)?;
                let null = take(encoded)?;
                let mut talk = Vec::with_capacity(blocks);
                let mut listen = Vec::with_capacity(blocks);
                for _ in 0..blocks {
                    talk.push(TalkStep {
                        norm: read_norm(take, encoded)?,
                        q: read_dense(take, encoded, encoded)?,
                        k: read_dense(take, encoded, encoded)?,
                        v: read_dense(take, encoded, encoded)?,
                        out: read_dense(take, encoded, encoded)?,
                        norm_ff: read_norm(take, encoded)?,
                        ff_in: read_dense(take, encoded, 2 * encoded)?,
                        ff_out: read_dense(take, 2 * encoded, encoded)?,
                    });
                    listen.push(AttentionStep {
                        norm_query: read_norm(take, encoded)?,
                        norm_cars: read_norm(take, encoded)?,
                        norm_ff: read_norm(take, encoded)?,
                        q: read_dense(take, encoded, encoded)?,
                        k: read_dense(take, encoded, encoded)?,
                        v: read_dense(take, encoded, encoded)?,
                        out: read_dense(take, encoded, encoded)?,
                        ff_in: read_dense(take, encoded, 4 * encoded)?,
                        ff_out: read_dense(take, 4 * encoded, encoded)?,
                    });
                }
                size = encoded + if skip { own } else { 0 };
                Layer::Talker(Box::new(Talker {
                    own,
                    block,
                    teammates,
                    opponents,
                    ball,
                    car_at,
                    car,
                    width: encoded,
                    heads,
                    skip,
                    query,
                    car_mlp,
                    ball_mlp,
                    kinds,
                    null,
                    talk,
                    listen,
                }))
            }
            LayerSpec::Transformer {
                own,
                block,
                teammates,
                opponents,
                ball,
                car_at,
                car,
                pads_at,
                pads,
                width: encoded,
                layers,
                heads,
                ff,
                summary,
                skip,
            } => {
                if format < 6 {
                    return Err(PolicyError::BadHeader(
                        "a transformer encoder needs format 6".into(),
                    ));
                }
                if index != 0 || own + (teammates + opponents) * block != size || block < 2 {
                    return Err(PolicyError::Shape(format!(
                        "the attention encoder must come first and read all {size} observation numbers"
                    )));
                }
                let pads_fit = pads == 0 || (ball <= pads_at && pads_at + pads <= car_at);
                if ball == 0 || car == 0 || car_at + car > own || car > block - 1 || !pads_fit {
                    return Err(PolicyError::Shape(format!(
                        "the ball ({ball}), the pads ({pads} from {pads_at}) and the own car ({car} from \
                         {car_at}) must lie in order in the own part ({own}) and the car fit a car block \
                         ({})",
                        block - 1
                    )));
                }
                if heads == 0 || encoded % heads != 0 || layers == 0 || ff == 0 {
                    return Err(PolicyError::Shape(format!(
                        "a transformer {encoded} wide cannot have {heads} heads, {layers} layers and a \
                         feed-forward {ff} wide"
                    )));
                }
                let rest: Vec<usize> = (ball..own)
                    .filter(|i| !(car_at..car_at + car).contains(i))
                    .filter(|i| !(pads_at..pads_at + pads).contains(i))
                    .collect();
                let mut mlp = |inputs: usize| -> Result<(Dense, Dense), PolicyError> {
                    Ok((
                        read_dense(take, inputs, encoded)?,
                        read_dense(take, encoded, encoded)?,
                    ))
                };
                let car_mlp = mlp(block - 1)?;
                let ball_mlp = mlp(ball)?;
                let rest_mlp = mlp(rest.len())?;
                let pad_tokens = if pads > 0 {
                    Some(PadTokens {
                        at: pads_at,
                        mlp: mlp(1)?,
                        places: take(pads * encoded)?,
                    })
                } else {
                    None
                };
                let kinds = take(3 * encoded)?;
                let summary = if summary { Some(take(encoded)?) } else { None };
                let mut blocks = Vec::with_capacity(layers);
                for _ in 0..layers {
                    blocks.push(TransformerBlock {
                        norm: read_norm(take, encoded)?,
                        qkv: read_dense(take, encoded, 3 * encoded)?,
                        out: read_dense(take, encoded, encoded)?,
                        norm_ff: read_norm(take, encoded)?,
                        ff_in: read_dense(take, encoded, ff)?,
                        ff_out: read_dense(take, ff, encoded)?,
                    });
                }
                let norm_out = read_norm(take, encoded)?;
                size = encoded + if skip { own } else { 0 };
                Layer::Transformer(Box::new(Transformer {
                    own,
                    block,
                    teammates,
                    opponents,
                    ball,
                    car_at,
                    car,
                    rest,
                    width: encoded,
                    heads,
                    skip,
                    car_mlp,
                    ball_mlp,
                    rest_mlp,
                    pads: pad_tokens,
                    kinds,
                    summary,
                    blocks,
                    norm_out,
                }))
            }
        });
        width = width.max(size);
    }
    if kept.is_some() {
        return Err(PolicyError::Shape(
            "a residual block is never added back".into(),
        ));
    }
    if size != outputs {
        return Err(PolicyError::Shape(format!(
            "the last layer gives {size} values, not {outputs} (one per action)"
        )));
    }
    Ok(Net {
        half: None,
        layers,
        width,
        inputs,
        outputs,
    })
}

/// The next `outputs × inputs` weights and `outputs` biases of a file as a linear layer.
fn read_dense(
    take: &mut impl FnMut(usize) -> Result<Vec<f32>, PolicyError>,
    inputs: usize,
    outputs: usize,
) -> Result<Dense, PolicyError> {
    Ok(Dense {
        inputs,
        outputs,
        weight: take(outputs * inputs)?,
        bias: take(outputs)?,
    })
}

/// The next `size` scales and `size` shifts of a file as a layer norm.
fn read_norm(
    take: &mut impl FnMut(usize) -> Result<Vec<f32>, PolicyError>,
    size: usize,
) -> Result<Norm, PolicyError> {
    Ok(Norm {
        scale: take(size)?,
        shift: take(size)?,
    })
}

/// A dot product in eight running sums, which the compiler turns into vector instructions; one running sum
/// (`iter().sum()`) must add in order and runs one multiply at a time. Only the rounding differs.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut sums = [0.0f32; 8];
    let ((a8, a_tail), (b8, b_tail)) = (a.as_chunks::<8>(), b.as_chunks::<8>());
    for (x, y) in a8.iter().zip(b8) {
        for ((sum, x), y) in sums.iter_mut().zip(x).zip(y) {
            *sum += x * y;
        }
    }
    dot_total(&sums, a_tail, b_tail)
}

/// The end of `dot`: the eight running sums added in order, then the leftover products.
#[inline(always)]
fn dot_total(sums: &[f32; 8], a_tail: &[f32], b_tail: &[f32]) -> f32 {
    let tail: f32 = a_tail.iter().zip(b_tail).map(|(x, y)| x * y).sum();
    sums.iter().sum::<f32>() + tail
}

/// `M` weight rows against `R` inputs: each pair's `dot`, the same bits, with every weight read once for
/// all `R` inputs and `M × R` running sums on the go instead of one.
#[inline(always)]
fn tile<const M: usize, const R: usize>(w: [&[f32]; M], x: [&[f32]; R]) -> [[f32; R]; M] {
    let n = w[0].len();
    let chunks = n / 8;
    let mut sums = [[[0.0f32; 8]; R]; M];
    for c in 0..chunks {
        for m in 0..M {
            let wc: &[f32; 8] = w[m][c * 8..c * 8 + 8].try_into().unwrap();
            for r in 0..R {
                let xc: &[f32; 8] = x[r][c * 8..c * 8 + 8].try_into().unwrap();
                for j in 0..8 {
                    sums[m][r][j] += wc[j] * xc[j];
                }
            }
        }
    }
    std::array::from_fn(|m| {
        std::array::from_fn(|r| dot_total(&sums[m][r], &w[m][chunks * 8..], &x[r][chunks * 8..]))
    })
}

/// `tile` with 256-bit vectors: each register holds `dot`'s eight running sums, multiplied then added
/// (no fused multiply-add), so the bits are the same.
///
/// # Safety
/// The processor must have AVX, and every row of `w` and `x` the same length.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn tile_avx<const M: usize, const R: usize>(
    w: [&[f32]; M],
    x: [&[f32]; R],
) -> [[f32; R]; M] {
    use std::arch::x86_64::{
        __m256, _mm256_add_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_setzero_ps, _mm256_storeu_ps,
    };
    let n = w[0].len();
    let chunks = n / 8;
    let mut acc: [[__m256; R]; M] = [[_mm256_setzero_ps(); R]; M];
    let (wp, xp) = (w.map(<[f32]>::as_ptr), x.map(<[f32]>::as_ptr));
    for c in 0..chunks {
        // SAFETY: c * 8 + 8 <= n, the length of every row.
        let xs: [__m256; R] = std::array::from_fn(|r| unsafe { _mm256_loadu_ps(xp[r].add(c * 8)) });
        for m in 0..M {
            let wc = unsafe { _mm256_loadu_ps(wp[m].add(c * 8)) };
            for r in 0..R {
                acc[m][r] = _mm256_add_ps(acc[m][r], _mm256_mul_ps(wc, xs[r]));
            }
        }
    }
    std::array::from_fn(|m| {
        std::array::from_fn(|r| {
            let mut sums = [0.0f32; 8];
            // SAFETY: `sums` holds eight floats.
            unsafe { _mm256_storeu_ps(sums.as_mut_ptr(), acc[m][r]) };
            dot_total(&sums, &w[m][chunks * 8..], &x[r][chunks * 8..])
        })
    })
}

fn tile_any<const M: usize, const R: usize>(w: [&[f32]; M], x: [&[f32]; R]) -> [[f32; R]; M] {
    debug_assert!(w.iter().chain(&x).all(|row| row.len() == w[0].len()));
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx") {
        // SAFETY: the processor has AVX, and `linear_rows` passes whole weight rows and inputs.
        return unsafe { tile_avx::<M, R>(w, x) };
    }
    tile::<M, R>(w, x)
}

/// A linear layer's weights as `linear_rows` reads them.
#[derive(Clone, Copy)]
enum Weights<'a> {
    Full(&'a [f32]),
    /// IEEE half precision, `to_half`.
    Half(&'a [u16]),
}

/// Whether this processor converts half precision (x86 with AVX and F16C).
fn half_supported() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx") && std::arch::is_x86_feature_detected!("f16c")
    }
    #[cfg(not(target_arch = "x86_64"))]
    false
}

/// `values` in IEEE half precision, rounded to nearest even. Needs `half_supported`.
fn to_half(values: &[f32]) -> Vec<u16> {
    #[cfg(target_arch = "x86_64")]
    {
        assert!(half_supported(), "half precision needs AVX and F16C");
        // SAFETY: the processor has AVX and F16C.
        unsafe { to_half_f16c(values) }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = values;
        unreachable!("half precision needs x86 with F16C")
    }
}

/// # Safety
/// The processor must have AVX and F16C.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
unsafe fn to_half_f16c(values: &[f32]) -> Vec<u16> {
    use std::arch::x86_64::{_MM_FROUND_TO_NEAREST_INT, _mm256_cvtps_ph, _mm256_loadu_ps};
    let mut out = Vec::with_capacity(values.len());
    for chunk in values.chunks(8) {
        let mut eight = [0.0f32; 8];
        eight[..chunk.len()].copy_from_slice(chunk);
        // SAFETY: `eight` holds eight floats.
        let halves = _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(unsafe {
            _mm256_loadu_ps(eight.as_ptr())
        });
        let halves: [u16; 8] = unsafe { std::mem::transmute(halves) };
        out.extend_from_slice(&halves[..chunk.len()]);
    }
    out
}

/// One IEEE half-precision number as an f32 (exactly).
fn from_half(h: u16) -> f32 {
    let sign = u32::from(h >> 15) << 31;
    let exponent = u32::from((h >> 10) & 0x1f);
    let fraction = u32::from(h & 0x3ff);
    let bits = match exponent {
        0 if fraction == 0 => sign,
        0 => {
            // Subnormal: shift the fraction up until its leading one is the implicit bit.
            let shift = fraction.leading_zeros() - 21;
            sign | ((113 - shift) << 23) | (((fraction << shift) & 0x3ff) << 13)
        }
        31 => sign | 0x7f80_0000 | (fraction << 13),
        _ => sign | ((exponent + 112) << 23) | (fraction << 13),
    };
    f32::from_bits(bits)
}

/// `tile` with the weights in half precision: each converted exactly to f32, then the same sums.
///
/// # Safety
/// The processor must have AVX and F16C, and every row of `w` and `x` the same length.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
unsafe fn tile_half<const M: usize, const R: usize>(
    w: [&[u16]; M],
    x: [&[f32]; R],
) -> [[f32; R]; M] {
    use std::arch::x86_64::{
        __m128i, __m256, _mm_loadu_si128, _mm256_add_ps, _mm256_cvtph_ps, _mm256_loadu_ps,
        _mm256_mul_ps, _mm256_setzero_ps, _mm256_storeu_ps,
    };
    let n = x[0].len();
    let chunks = n / 8;
    let mut acc: [[__m256; R]; M] = [[_mm256_setzero_ps(); R]; M];
    let (wp, xp) = (w.map(<[u16]>::as_ptr), x.map(<[f32]>::as_ptr));
    for c in 0..chunks {
        // SAFETY: c * 8 + 8 <= n, the length of every row.
        let xs: [__m256; R] = std::array::from_fn(|r| unsafe { _mm256_loadu_ps(xp[r].add(c * 8)) });
        for m in 0..M {
            let wc =
                _mm256_cvtph_ps(unsafe { _mm_loadu_si128(wp[m].add(c * 8).cast::<__m128i>()) });
            for r in 0..R {
                acc[m][r] = _mm256_add_ps(acc[m][r], _mm256_mul_ps(wc, xs[r]));
            }
        }
    }
    std::array::from_fn(|m| {
        let tail: Vec<f32> = w[m][chunks * 8..].iter().map(|&h| from_half(h)).collect();
        std::array::from_fn(|r| {
            let mut sums = [0.0f32; 8];
            // SAFETY: `sums` holds eight floats.
            unsafe { _mm256_storeu_ps(sums.as_mut_ptr(), acc[m][r]) };
            dot_total(&sums, &tail, &x[r][chunks * 8..])
        })
    })
}

fn tile_half_any<const M: usize, const R: usize>(w: [&[u16]; M], x: [&[f32]; R]) -> [[f32; R]; M] {
    #[cfg(target_arch = "x86_64")]
    {
        debug_assert!(w.iter().all(|row| row.len() == x[0].len()));
        debug_assert!(x.iter().all(|row| row.len() == x[0].len()));
        assert!(half_supported(), "half precision needs AVX and F16C");
        // SAFETY: the processor has AVX and F16C, and `linear_rows` passes whole rows.
        unsafe { tile_half::<M, R>(w, x) }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (w, x);
        unreachable!("half precision needs x86 with F16C")
    }
}

/// Where the members of a team write their shares of a layer's outputs; each writes its own neurons only.
#[derive(Clone, Copy)]
struct Outputs(*mut f32);
// SAFETY: members write disjoint elements, and `Team::split` returns only when all are done.
unsafe impl Send for Outputs {}
unsafe impl Sync for Outputs {}

/// A linear layer for several inputs (`x`: inputs side by side): `out[r * outputs + o] = bias[o] +
/// dot(weight row o, input r)`, each the bits `Dense::apply` gives, the weight rows shared out to `team`
///.
fn linear_rows(
    weight: Weights<'_>,
    bias: &[f32],
    inputs: usize,
    x: &[f32],
    out: &mut [f32],
    team: &Team<'_>,
) {
    let (outputs, rows) = (bias.len(), x.len() / inputs);
    assert_eq!(out.len(), rows * outputs, "output count");
    let to = Outputs(out.as_mut_ptr());
    let input = |r: usize| &x[r * inputs..(r + 1) * inputs];
    let job = |member: usize| {
        // The whole wrapper, not its pointer field alone, so the closure is Sync.
        #[allow(clippy::redundant_locals)]
        let to = to;
        let mine = crate::team::share(outputs, 2, member, team.members());
        let mut o = mine.start;
        while o < mine.end {
            // One or two inputs: four weight rows at a time, more of memory on the way at once.
            let quad = rows <= 2 && o + 3 < mine.end;
            let pair = o + 1 < mine.end;
            let mut r = 0;
            while r < rows {
                let left = rows - r;
                let put = |m: usize, k: usize, value: f32| {
                    // SAFETY: neuron o + m of input r + k is this member's alone, inside `out`.
                    unsafe { *to.0.add((r + k) * outputs + o + m) = bias[o + m] + value };
                };
                macro_rules! run {
                    ($m:literal, $r:literal) => {{
                        let xs: [&[f32]; $r] = std::array::from_fn(|k| input(r + k));
                        let rows = |o: usize| o * inputs..(o + 1) * inputs;
                        let sums = match weight {
                            Weights::Full(all) => {
                                tile_any::<$m, $r>(std::array::from_fn(|m| &all[rows(o + m)]), xs)
                            }
                            Weights::Half(all) => tile_half_any::<$m, $r>(
                                std::array::from_fn(|m| &all[rows(o + m)]),
                                xs,
                            ),
                        };
                        for (m, sums) in sums.iter().enumerate() {
                            for (k, &value) in sums.iter().enumerate() {
                                put(m, k, value);
                            }
                        }
                    }};
                }
                match (pair, left.min(4)) {
                    _ if quad && left >= 2 => run!(4, 2),
                    _ if quad => run!(4, 1),
                    (true, 4) => run!(2, 4),
                    (true, 3) => run!(2, 3),
                    (true, 2) => run!(2, 2),
                    (true, _) => run!(2, 1),
                    (false, 4) => run!(1, 4),
                    (false, 3) => run!(1, 3),
                    (false, 2) => run!(1, 2),
                    (false, _) => run!(1, 1),
                }
                r += left.min(if quad { 2 } else { 4 });
            }
            o += if quad {
                4
            } else if pair {
                2
            } else {
                1
            };
        }
    };
    team.split(&job);
}

/// Keeps `current` in `skip` (a residual block's start).
fn keep(skip: &mut Vec<f32>, current: &[f32]) {
    skip.clear();
    skip.extend_from_slice(current);
}

/// Adds the kept vector back (a residual block's end): `skip + current`, as PyTorch's `x + f(x)`.
fn add_back(current: &mut [f32], skip: &[f32]) {
    debug_assert_eq!(current.len(), skip.len());
    for (c, s) in current.iter_mut().zip(skip) {
        *c += *s;
    }
}

/// In place: normalise to mean 0 and variance 1 over the whole vector (biased variance, PyTorch's), then
/// scale and shift.
fn layer_norm(values: &mut [f32], scale: &[f32], shift: &[f32]) {
    let n = values.len() as f32;
    let mean = values.iter().sum::<f32>() / n;
    let variance = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv = 1.0 / (variance + LAYER_NORM_EPS).sqrt();
    for ((v, s), b) in values.iter_mut().zip(scale).zip(shift) {
        *v = (*v - mean) * inv * s + b;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(layers: &str, obs_size: usize, num_actions: usize) -> String {
        format!(
            r#"{{"format": 1, "game": {{"observation": "native", "action_masks": "standard", "team_size": 1,
            "opponents": true, "tick_skip": 8, "action_delay": 7}}, "obs_size": {obs_size},
            "num_actions": {num_actions}, "layers": [{layers}]}}"#
        )
    }

    #[test]
    fn game_settings_ignore_keys_this_version_does_not_read() {
        // Game keys the bot does not read are left alone.
        let old = header("", 4, 90).replace(
            r#""action_delay": 7"#,
            r#""action_delay": 7, "an_old_key": {"a": [], "b": ["x"]}, "another_old_key": 0"#,
        );
        let old: PolicyHeader = serde_json::from_str(&old).unwrap();
        let now: PolicyHeader = serde_json::from_str(&header("", 4, 90)).unwrap();
        assert_eq!(old, now);
    }

    fn file(header: &str, weights: &[f32]) -> Vec<u8> {
        let mut bytes = MAGIC.to_vec();
        bytes.extend(PLAIN_FORMAT.to_le_bytes());
        bytes.extend((header.len() as u32).to_le_bytes());
        bytes.extend(header.as_bytes());
        for w in weights {
            bytes.extend(w.to_le_bytes());
        }
        bytes
    }

    fn random_policy(obs_size: usize, widths: &[usize], act: &str, seed: u64) -> Vec<u8> {
        crate::testing::random_policy(obs_size, 90, widths, act, seed)
    }

    #[test]
    fn many_observations_at_once_give_each_the_bits_it_gets_alone() {
        let mut rng = fastrand::Rng::with_seed(7);
        for (obs_size, widths, act) in [
            (13, &[5][..], "relu"),
            (70, &[33, 17][..], "leaky_relu"),
            (107, &[64, 48, 31][..], "tanh"),
        ] {
            let policy = Policy::from_bytes(&random_policy(obs_size, widths, act, 3)).unwrap();
            for rows in [1, 2, 3, 4, 5, 8, 9] {
                let obs: Vec<f32> = (0..rows * obs_size)
                    .map(|_| rng.f32() * 4.0 - 2.0)
                    .collect();
                let mut alone = vec![0.0f32; rows * 90];
                for (o, out) in obs.chunks(obs_size).zip(alone.chunks_mut(90)) {
                    policy.logits(o, out);
                }
                for members in 1..=3 {
                    let mut together = vec![0.0f32; rows * 90];
                    Team::run(members, |team| {
                        policy.logits_rows(&obs, rows, &mut together, team);
                    });
                    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                    assert_eq!(
                        bits(&together),
                        bits(&alone),
                        "{rows} rows, {members} threads"
                    );
                }
            }
        }
    }

    #[test]
    fn half_precision_converts_exactly_and_rounds_to_nearest() {
        if !half_supported() {
            eprintln!("SKIPPED: this processor has no F16C");
            return;
        }
        // Every finite half and the infinities come back as F16C converts them.
        let all: Vec<u16> = (0..=u16::MAX)
            .filter(|h| (h >> 10) & 0x1f != 0x1f || h & 0x3ff == 0)
            .collect();
        let floats: Vec<f32> = all.iter().map(|&h| from_half(h)).collect();
        assert_eq!(to_half(&floats), all);
        assert_eq!(from_half(0x3c00), 1.0);
        assert_eq!(from_half(0xc000), -2.0);
        assert_eq!(from_half(0x0001), 2.0f32.powi(-24));
        // Halfway between two halves goes to the even one.
        let halfway = f32::from_bits(1.0f32.to_bits() + (1 << 12));
        assert_eq!(to_half(&[halfway, 65504.0, 1e-9]), vec![0x3c00, 0x7bff, 0]);
    }

    #[test]
    fn half_weights_stay_close_and_give_each_row_the_same_bits() {
        let mut policy = Policy::from_bytes(&random_policy(70, &[64, 40], "relu", 4)).unwrap();
        let mut rng = fastrand::Rng::with_seed(9);
        let rows = 7;
        let obs: Vec<f32> = (0..rows * 70).map(|_| rng.f32() * 4.0 - 2.0).collect();
        let mut full = vec![0.0f32; rows * 90];
        policy.logits_rows(&obs, rows, &mut full, &Team::ALONE);
        if !policy.read_half(true) {
            eprintln!("SKIPPED: this processor has no F16C");
            return;
        }
        let mut half = vec![0.0f32; rows * 90];
        Team::run(2, |team| policy.logits_rows(&obs, rows, &mut half, team));
        let worst = full
            .iter()
            .zip(&half)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        let size = full.iter().map(|a| a.abs()).fold(0.0, f32::max);
        assert!(
            worst > 0.0 && worst < 2e-3 * size,
            "worst {worst} of {size}"
        );
        for (row, expected) in half.chunks(90).enumerate() {
            let mut alone = vec![0.0f32; 90];
            policy.logits_rows(&obs[row * 70..(row + 1) * 70], 1, &mut alone, &Team::ALONE);
            assert_eq!(alone, expected);
        }
        // `logits` (the bot's pass) keeps the full weights.
        let mut plain = vec![0.0f32; 90];
        policy.logits(&obs[..70], &mut plain);
        assert_eq!(plain, full[..90]);
        assert!(!policy.read_half(false));
        let mut exact = vec![0.0f32; rows * 90];
        policy.logits_rows(&obs, rows, &mut exact, &Team::ALONE);
        assert_eq!(exact, full);
    }

    #[test]
    fn the_unused_format_numbers_are_refused() {
        let bytes = random_policy(20, &[16, 8], "relu", 5);
        assert!(Policy::from_bytes(&bytes).is_ok());
        for unused in UNUSED_FORMATS {
            let start = 16 + u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
            let mut other = bytes.clone();
            let text = String::from_utf8(other[16..start].to_vec())
                .unwrap()
                .replace(r#""format": 1"#, &format!(r#""format": {unused}"#));
            other.splice(16..start, text.bytes());
            other[8..12].copy_from_slice(&unused.to_le_bytes());
            other[12..16].copy_from_slice(&(text.len() as u32).to_le_bytes());
            assert!(matches!(
                Policy::from_bytes(&other).unwrap_err(),
                PolicyError::UnsupportedVersion(v) if v == unused
            ));
        }
    }

    /// Not a check: how fast the batched pass runs here (`cargo test --release -- --ignored speed`).
    #[test]
    #[ignore = "a timing, not a check"]
    fn speed_of_many_rows() {
        let policy =
            Policy::from_bytes(&random_policy(110, &[2048, 2048, 1536], "relu", 1)).unwrap();
        for rows in [1usize, 4, 8] {
            let obs = vec![0.3f32; rows * 110];
            let mut out = vec![0.0f32; rows * 90];
            for members in [1, 2, 4] {
                let start = std::time::Instant::now();
                Team::run(members, |team| {
                    for _ in 0..20 {
                        policy.logits_rows(&obs, rows, &mut out, team);
                    }
                });
                let ms = start.elapsed().as_secs_f64() * 1000.0 / 20.0;
                eprintln!("{rows} rows, {members} threads: {ms:.2} ms a pass");
            }
        }
        let start = std::time::Instant::now();
        let mut out = vec![0.0f32; 90];
        for _ in 0..20 {
            policy.logits(&vec![0.3f32; 110], &mut out);
        }
        eprintln!(
            "logits alone: {:.2} ms",
            start.elapsed().as_secs_f64() * 1000.0 / 20.0
        );
    }

    #[test]
    fn a_two_layer_network_is_evaluated_by_hand() {
        // 2 inputs -> linear(3) -> layer_norm -> relu -> linear(2).
        let layers = r#"{"kind": "linear", "inputs": 2, "outputs": 3}, {"kind": "layer_norm", "size": 3},
            {"kind": "relu"}, {"kind": "linear", "inputs": 3, "outputs": 2}"#;
        #[rustfmt::skip]
        let weights = [
            1.0, 0.0,  0.0, 1.0,  1.0, 1.0, // W1 rows
            0.0, 0.0, 0.0, // b1
            1.0, 1.0, 1.0, 0.0, 0.0, 0.0, // norm scale, shift
            1.0, 0.0, 0.0,  0.0, 0.0, 1.0, // W2 rows
            0.5, -0.5, // b2
        ];
        let policy = Policy::from_bytes(&file(&header(layers, 2, 2), &weights)).unwrap();
        let mut out = [0.0; 2];
        policy.logits(&[1.0, 2.0], &mut out);
        // Hidden before norm: [1, 2, 3]; normalised: [-1.2247, 0, 1.2247] (var 2/3); relu keeps [0, 0, 1.2247].
        let z = (1.0f32 / (2.0 / 3.0 + LAYER_NORM_EPS).sqrt()) * 1.0;
        assert!((out[0] - 0.5).abs() < 1e-6, "{out:?}");
        assert!((out[1] - (z - 0.5)).abs() < 1e-5, "{out:?}");
    }

    /// A file of `format` with `layers` (its header and its first bytes both say `format`).
    fn file_of(
        format: u32,
        layers: &str,
        obs_size: usize,
        outputs: usize,
        weights: &[f32],
    ) -> Vec<u8> {
        let text = header(layers, obs_size, outputs).replacen(
            r#""format": 1"#,
            &format!(r#""format": {format}"#),
            1,
        );
        let mut bytes = file(&text, weights);
        bytes[8..12].copy_from_slice(&format.to_le_bytes());
        bytes
    }

    #[test]
    fn a_residual_block_is_evaluated_by_hand_and_needs_format_8() {
        // 2 inputs -> linear(2) -> [kept; linear(2) -> relu; added back] -> linear(1).
        let layers = r#"{"kind": "linear", "inputs": 2, "outputs": 2}, {"kind": "residual_start"},
            {"kind": "linear", "inputs": 2, "outputs": 2}, {"kind": "relu"}, {"kind": "residual_add"},
            {"kind": "linear", "inputs": 2, "outputs": 1}"#;
        #[rustfmt::skip]
        let weights = [
            1.0, 0.0,  0.0, 1.0,  0.0, 0.0, // identity, no bias
            1.0, 1.0,  -1.0, 0.0,  0.5, 0.0, // the block's linear
            1.0, 10.0,  0.0, // the output
        ];
        let policy = Policy::from_bytes(&file_of(8, layers, 2, 1, &weights)).unwrap();
        let mut out = [0.0; 1];
        policy.logits(&[1.0, 2.0], &mut out);
        // x = [1, 2]; f(x) = relu([3.5, -1]) = [3.5, 0]; x + f(x) = [4.5, 2]; output 4.5 + 20.
        assert_eq!(out[0], 24.5);
        // Batched and row by row, the same bits.
        let obs = [1.0, 2.0, -0.5, 0.25, 3.0, -2.0];
        let mut rows = [0.0; 3];
        policy.logits_rows(&obs, 3, &mut rows, &Team::ALONE);
        let mut batch = [0.0; 3];
        policy.logits_batch(&obs, 3, &mut batch, &Team::ALONE);
        for (row, input) in obs.as_chunks::<2>().0.iter().enumerate() {
            policy.logits(input, &mut out);
            assert_eq!((rows[row], batch[row]), (out[0], out[0]));
        }
        assert!(matches!(
            Policy::from_bytes(&file_of(6, layers, 2, 1, &weights)).unwrap_err(),
            PolicyError::BadHeader(_)
        ));
    }

    #[test]
    fn a_residual_block_must_add_back_what_it_kept() {
        let refused = |layers: &str, weights: usize| {
            let err =
                Policy::from_bytes(&file_of(8, layers, 2, 2, &vec![0.0; weights])).unwrap_err();
            assert!(matches!(err, PolicyError::Shape(_)), "{err}");
        };
        let start = r#"{"kind": "residual_start"}"#;
        let add = r#"{"kind": "residual_add"}"#;
        let widen = r#"{"kind": "linear", "inputs": 2, "outputs": 3}"#;
        let narrow = r#"{"kind": "linear", "inputs": 3, "outputs": 2}"#;
        refused(&format!("{start}, {widen}, {add}, {narrow}"), 9 + 8); // 3 values added to 2
        refused(&format!("{start}, {start}, {add}, {add}"), 0); // nested
        refused(add, 0); // never started
        refused(&format!("{start}, {widen}, {narrow}"), 9 + 8); // never added back
        let fine = format!("{start}, {widen}, {narrow}, {add}");
        assert!(Policy::from_bytes(&file_of(8, &fine, 2, 2, &[0.0; 17])).is_ok());
    }

    #[test]
    fn an_attention_encoder_needs_format_2_and_must_read_the_whole_observation() {
        // own 2, one car block of 3 (2 numbers and the flag): 5 observation numbers.
        let encoder = r#"{"kind": "entity_encoder", "own": 2, "block": 3, "teammates": 0,
            "opponents": 1, "width": 4, "steps": 1, "heads": 2}"#;
        let weights = (2 * 4 + 4)
            + (4 * 4 + 4)
            + (2 * 4 + 4)
            + (4 * 4 + 4)
            + 8
            + 4
            + 6 * 4
            + 4 * (4 * 4 + 4)
            + (4 * 16 + 16)
            + (16 * 4 + 4);
        let as_format_1 = header(encoder, 5, 4);
        assert!(matches!(
            Policy::from_bytes(&file(&as_format_1, &vec![0.0; weights])).unwrap_err(),
            PolicyError::BadHeader(_)
        ));
        let format_2 = |layers: &str, obs: usize| {
            let text = header(layers, obs, 4).replacen(r#""format": 1"#, r#""format": 2"#, 1);
            let mut bytes = file(&text, &vec![0.0; weights]);
            bytes[8] = 2;
            bytes
        };
        let policy = Policy::from_bytes(&format_2(encoder, 5)).unwrap();
        let mut out = [1.0f32; 4];
        policy.logits(&[0.5, -1.0, 2.0, 3.0, 1.0], &mut out);
        assert_eq!(
            out, [0.0; 4],
            "all-zero weights give zero logits, and nothing is NaN"
        );
        assert!(matches!(
            Policy::from_bytes(&format_2(encoder, 6)).unwrap_err(),
            PolicyError::Shape(_)
        ));
    }

    #[test]
    fn a_talking_encoder_needs_format_4_and_its_ball_and_car_inside_the_own_part() {
        // own 3 (ball 1, own car 1 from 2), one car block of 3 (2 numbers and the flag): 6 numbers.
        let talker = |ball: usize, car_at: usize| {
            format!(
                r#"{{"kind": "entity_talk", "own": 3, "block": 3, "teammates": 0, "opponents": 1,
                "ball": {ball}, "car_at": {car_at}, "car": 1, "width": 4, "blocks": 1, "heads": 2,
                "skip": false}}"#
            )
        };
        let linear = |i: usize, o: usize| i * o + o;
        let weights = linear(3, 4)
            + linear(4, 4)
            + linear(2, 4)
            + linear(4, 4)
            + linear(1, 4)
            + linear(4, 4)
            + 3 * 4
            + 4
            + (2 * 4 + 4 * linear(4, 4) + 2 * 4 + linear(4, 8) + linear(8, 4))
            + (6 * 4 + 4 * linear(4, 4) + linear(4, 16) + linear(16, 4));
        let with_format = |layers: &str, format: u32| {
            let text = header(layers, 6, 4).replacen(
                r#""format": 1"#,
                &format!(r#""format": {format}"#),
                1,
            );
            let mut bytes = file(&text, &vec![0.0; weights]);
            bytes[8] = format as u8;
            bytes
        };
        assert!(matches!(
            Policy::from_bytes(&with_format(&talker(1, 2), 3)).unwrap_err(),
            PolicyError::BadHeader(_)
        ));
        let policy = Policy::from_bytes(&with_format(&talker(1, 2), 4)).unwrap();
        let mut out = [1.0f32; 4];
        policy.logits(&[0.5, -1.0, 2.0, 3.0, 1.0, 0.0], &mut out);
        assert_eq!(
            out, [0.0; 4],
            "all-zero weights give zero logits, and nothing is NaN"
        );
        assert!(matches!(
            Policy::from_bytes(&with_format(&talker(1, 3), 4)).unwrap_err(),
            PolicyError::Shape(_)
        ));
    }

    /// The layers of a transformer over an own part of 6 (ball 2, one pad from 2, own car 2 from 3) and
    /// one car block of 3 (2 numbers and the flag).
    fn transformer(pads_at: usize, summary: bool) -> String {
        format!(
            r#"{{"kind": "transformer", "own": 6, "block": 3, "teammates": 0, "opponents": 1, "ball": 2,
            "car_at": 3, "car": 2, "pads_at": {pads_at}, "pads": 1, "width": 4, "layers": 1, "heads": 2,
            "ff": 8, "summary": {summary}, "skip": false}}"#
        )
    }

    #[test]
    fn a_transformer_needs_format_6_and_its_ball_pads_and_car_in_order() {
        let linear = |i: usize, o: usize| i * o + o;
        let mlp = |i: usize| linear(i, 4) + linear(4, 4);
        let block = 2 * 4 + linear(4, 12) + linear(4, 4) + 2 * 4 + linear(4, 8) + linear(8, 4);
        // car, ball, rest (the action-like number at 5), pads (one place), kinds, summary, a block, norm.
        let weights = mlp(2) + mlp(2) + mlp(1) + mlp(1) + 4 + 3 * 4 + 4 + block + 2 * 4;
        let with_format = |layers: &str, format: u32| {
            let text = header(layers, 9, 4).replacen(
                r#""format": 1"#,
                &format!(r#""format": {format}"#),
                1,
            );
            let mut bytes = file(&text, &vec![0.0; weights]);
            bytes[8] = format as u8;
            bytes
        };
        assert!(matches!(
            Policy::from_bytes(&with_format(&transformer(2, true), 4)).unwrap_err(),
            PolicyError::BadHeader(_)
        ));
        let policy = Policy::from_bytes(&with_format(&transformer(2, true), 6)).unwrap();
        assert!(policy.reads_any_cars());
        let mut out = [1.0f32; 4];
        policy.logits(&[0.5, -1.0, 0.2, 2.0, 3.0, 0.7, 1.0, 1.0, 1.0], &mut out);
        assert_eq!(
            out, [0.0; 4],
            "all-zero weights give zero logits, and nothing is NaN"
        );
        assert!(matches!(
            Policy::from_bytes(&with_format(&transformer(1, true), 6)).unwrap_err(),
            PolicyError::Shape(_)
        ));
        let without_summary = with_format(&transformer(2, false), 6);
        assert!(matches!(
            Policy::from_bytes(&without_summary).unwrap_err(),
            PolicyError::WeightCount { .. }
        ));
    }

    #[test]
    fn a_transformer_split_across_threads_gives_the_same_bits() {
        let linear = |i: usize, o: usize| i * o + o;
        let mlp = |i: usize| linear(i, 4) + linear(4, 4);
        let block = 2 * 4 + linear(4, 12) + linear(4, 4) + 2 * 4 + linear(4, 8) + linear(8, 4);
        let weights = mlp(2) + mlp(2) + mlp(1) + mlp(1) + 4 + 3 * 4 + 4 + 2 * block + 2 * 4;
        let layers = transformer(2, true).replace(r#""layers": 1"#, r#""layers": 2"#)
            + r#", {"kind": "relu"}, {"kind": "linear", "inputs": 4, "outputs": 5}"#;
        let text = header(&layers, 9, 5).replacen(r#""format": 1"#, r#""format": 6"#, 1);
        let mut rng = fastrand::Rng::with_seed(7);
        let random: Vec<f32> = (0..weights + linear(4, 5))
            .map(|_| rng.f32() - 0.5)
            .collect();
        let mut bytes = file(&text, &random);
        bytes[8] = 6;
        let policy = Policy::from_bytes(&bytes).unwrap();
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        for flag in [1.0, 0.0] {
            let obs = [0.5, -1.0, 0.2, 2.0, 3.0, 0.7, 1.0, -0.4, flag];
            let mut alone = [0.0f32; 5];
            policy.logits(&obs, &mut alone);
            for members in 1..=3 {
                let mut split = [0.0f32; 5];
                Team::run(members, |team| policy.logits_team(&obs, &mut split, team));
                assert_eq!(bits(&split), bits(&alone), "{members} threads, flag {flag}");
            }
        }
        let obs = [0.5, -1.0, 0.2, 2.0, 3.0, 0.7, 1.0, -0.4, 1.0, 0.3, 0.9, 1.0];
        let mut alone = [0.0f32; 5];
        policy.logits_cars(&obs, 0, 2, &mut alone);
        let mut split = [0.0f32; 5];
        Team::run(2, |team| {
            policy.logits_cars_team(&obs, 0, 2, &mut split, team)
        });
        assert_eq!(bits(&split), bits(&alone), "two opponents");
    }

    /// A policy of one attention encoder `layer` (giving `encoded` numbers) over an own part of 6 and 2
    /// teammate and 3 opponent places of 4 numbers (3 and the flag), then a linear layer to 3 actions; every
    /// weight random.
    fn encoder_policy(layer: &str, format: u32, encoded: usize) -> Policy {
        let layers = format!(r#"{layer}, {{"kind": "linear", "inputs": {encoded}, "outputs": 3}}"#);
        let text = header(&layers, 6 + 5 * 4, 3).replacen(
            r#""format": 1"#,
            &format!(r#""format": {format}"#),
            1,
        );
        let bytes = |weights: &[f32]| {
            let mut bytes = file(&text, weights);
            bytes[8] = format as u8;
            bytes
        };
        // Too many weights: the refusal says how many the layers take.
        let Err(PolicyError::WeightCount { expected, .. }) =
            Policy::from_bytes(&bytes(&vec![0.0; 1 << 20]))
        else {
            panic!("{layer}");
        };
        let mut rng = fastrand::Rng::with_seed(5);
        let weights: Vec<f32> = (0..expected).map(|_| rng.f32() - 0.5).collect();
        Policy::from_bytes(&bytes(&weights)).unwrap()
    }

    #[test]
    fn a_batch_of_observations_gives_each_the_bits_it_gets_alone() {
        let shape =
            r#""own": 6, "block": 4, "teammates": 2, "opponents": 3, "width": 8, "heads": 2"#;
        let layers = [
            (
                format!(r#"{{"kind": "entity_encoder", {shape}, "steps": 2}}"#),
                2,
            ),
            (
                format!(
                    r#"{{"kind": "entity_talk", {shape}, "ball": 2, "car_at": 4, "car": 2, "blocks": 2,
                    "skip": false}}"#
                ),
                4,
            ),
            (
                format!(
                    r#"{{"kind": "transformer", {shape}, "ball": 2, "car_at": 3, "car": 2, "pads_at": 2,
                    "pads": 1, "layers": 2, "ff": 16, "summary": true, "skip": false}}"#
                ),
                6,
            ),
            (
                format!(
                    r#"{{"kind": "transformer", {shape}, "ball": 2, "car_at": 3, "car": 2, "pads_at": 0,
                    "pads": 0, "layers": 2, "ff": 16, "summary": false, "skip": true}}"#
                )
                .replace(r#""width": 8"#, r#""width": 4"#),
                6,
            ),
        ];
        let mut rng = fastrand::Rng::with_seed(11);
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        let mut policies: Vec<Policy> = layers
            .iter()
            .map(|(layer, format)| {
                let skip = layer.contains(r#""skip": true"#);
                encoder_policy(layer, *format, if skip { 6 + 4 } else { 8 })
            })
            .collect();
        policies.push(Policy::from_bytes(&random_policy(26, &[33, 17], "relu", 4)).unwrap());
        for policy in &policies {
            let (size, actions) = (policy.obs_size(), policy.num_actions());
            for rows in [1, 2, 3, 5] {
                // Car places present or not at random (a flag of 1 or 0), so groups differ in what they read.
                let obs: Vec<f32> = (0..rows * size)
                    .map(|i| {
                        if i % size >= 6 && (i % size - 6) % 4 == 3 {
                            f32::from(u8::from(rng.bool()))
                        } else {
                            rng.f32() * 2.0 - 1.0
                        }
                    })
                    .collect();
                let mut alone = vec![0.0f32; rows * actions];
                for (o, out) in obs.chunks(size).zip(alone.chunks_mut(actions)) {
                    policy.logits(o, out);
                }
                for members in 1..=3 {
                    let mut together = vec![0.0f32; rows * actions];
                    Team::run(members, |team| {
                        policy.logits_batch(&obs, rows, &mut together, team)
                    });
                    assert_eq!(
                        bits(&together),
                        bits(&alone),
                        "{rows} rows, {members} threads"
                    );
                }
            }
        }
    }

    #[test]
    fn attention_encoders_read_any_number_of_cars_and_the_trained_places_bit_for_bit() {
        let shape =
            r#""own": 6, "block": 4, "teammates": 2, "opponents": 3, "width": 8, "heads": 2"#;
        let layers = [
            (
                format!(r#"{{"kind": "entity_encoder", {shape}, "steps": 2}}"#),
                2,
            ),
            (
                format!(
                    r#"{{"kind": "entity_talk", {shape}, "ball": 2, "car_at": 4, "car": 2, "blocks": 2,
                    "skip": false}}"#
                ),
                4,
            ),
            (
                format!(
                    r#"{{"kind": "transformer", {shape}, "ball": 2, "car_at": 3, "car": 2, "pads_at": 2,
                    "pads": 1, "layers": 2, "ff": 16, "summary": true, "skip": false}}"#
                ),
                6,
            ),
            (
                format!(
                    r#"{{"kind": "transformer", {shape}, "ball": 2, "car_at": 3, "car": 2, "pads_at": 0,
                    "pads": 0, "layers": 2, "ff": 16, "summary": false, "skip": true}}"#
                )
                .replace(r#""width": 8"#, r#""width": 4"#),
                6,
            ),
        ];
        let mut rng = fastrand::Rng::with_seed(9);
        let mut car = || -> Vec<f32> { vec![rng.f32() - 0.5, rng.f32(), -rng.f32(), 1.0] };
        for (layer, format) in &layers {
            let skip = layer.contains(r#""skip": true"#);
            let policy = encoder_policy(layer, *format, if skip { 6 + 4 } else { 8 });
            assert!(policy.reads_any_cars());
            let own = [0.3, -0.7, 0.1, 0.9, -0.2, 0.5];
            for (mates, opponents) in [(0, 0), (1, 1), (2, 3), (0, 2), (1, 0)] {
                // The padded observation (filled places first on each side) and one with only the cars
                // present: the same answer, bit for bit.
                let (mut padded, mut present) = (own.to_vec(), own.to_vec());
                for (filled, places) in [(mates, 2), (opponents, 3)] {
                    for place in 0..places {
                        if place < filled {
                            let numbers = car();
                            padded.extend(&numbers);
                            present.extend(&numbers);
                        } else {
                            padded.extend([0.0; 4]);
                        }
                    }
                }
                let (mut a, mut b) = ([0.0f32; 3], [0.0f32; 3]);
                policy.logits(&padded, &mut a);
                policy.logits_cars(&present, mates, opponents, &mut b);
                assert_eq!(
                    a.map(f32::to_bits),
                    b.map(f32::to_bits),
                    "{layer}: {mates}, {opponents}"
                );
            }
            // 12v12, and the order of the cars within a side changes nothing.
            let mut big = own.to_vec();
            for _ in 0..23 {
                big.extend(car());
            }
            let (mut a, mut b) = ([0.0f32; 3], [0.0f32; 3]);
            policy.logits_cars(&big, 11, 12, &mut a);
            let mut swapped = big.clone();
            let (first, second) = (6, 6 + 4 * 10); // teammates 1 and 11
            for i in 0..4 {
                swapped.swap(first + i, second + i);
            }
            let (third, last) = (6 + 4 * 11, 6 + 4 * 22); // opponents 1 and 12
            for i in 0..4 {
                swapped.swap(third + i, last + i);
            }
            policy.logits_cars(&swapped, 11, 12, &mut b);
            assert!(a.iter().all(|v| v.is_finite()));
            for (x, y) in a.iter().zip(&b) {
                assert!((x - y).abs() < 1e-5, "{layer}: {a:?} vs {b:?}");
            }
            // A teammate and an opponent trading places changes the answer (1v2 here).
            let small = &big[..6 + 3 * 4];
            let mut crossed = small.to_vec();
            for i in 0..4 {
                crossed.swap(6 + i, 6 + 4 + i);
            }
            policy.logits_cars(small, 1, 2, &mut a);
            policy.logits_cars(&crossed, 1, 2, &mut b);
            assert_ne!(a, b, "{layer}: the side matters");
        }
    }

    #[test]
    fn an_encoder_passing_the_own_part_beside_it_needs_format_3_and_puts_it_first() {
        let encoder = r#"{"kind": "entity_encoder", "own": 2, "block": 3, "teammates": 0,
            "opponents": 1, "width": 4, "steps": 1, "heads": 2, "skip": true}"#;
        let weights = (2 * 4 + 4)
            + (4 * 4 + 4)
            + (2 * 4 + 4)
            + (4 * 4 + 4)
            + 8
            + 4
            + 6 * 4
            + 4 * (4 * 4 + 4)
            + (4 * 16 + 16)
            + (16 * 4 + 4);
        // The encoder's 4 numbers and the own part's 2 are the 6 "actions" here.
        let in_format = |format: u8| {
            let text = header(encoder, 5, 6).replacen(
                r#""format": 1"#,
                &format!(r#""format": {format}"#),
                1,
            );
            let mut bytes = file(&text, &vec![0.0; weights]);
            bytes[8] = format;
            bytes
        };
        assert!(matches!(
            Policy::from_bytes(&in_format(2)).unwrap_err(),
            PolicyError::BadHeader(_)
        ));
        let policy = Policy::from_bytes(&in_format(3)).unwrap();
        let mut out = [1.0f32; 6];
        policy.logits(&[0.5, -1.0, 2.0, 3.0, 1.0], &mut out);
        assert_eq!(out, [0.5, -1.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn malformed_files_are_refused_with_a_reason() {
        let good = header(r#"{"kind": "linear", "inputs": 2, "outputs": 2}"#, 2, 2);
        assert!(matches!(
            Policy::from_bytes(b"NOTAPOLICY").unwrap_err(),
            PolicyError::NotAPolicyFile
        ));
        let mut wrong_version = file(&good, &[0.0; 6]);
        wrong_version[8] = 9;
        assert!(matches!(
            Policy::from_bytes(&wrong_version).unwrap_err(),
            PolicyError::UnsupportedVersion(9)
        ));
        assert!(matches!(
            Policy::from_bytes(&file(&good, &[0.0; 5])).unwrap_err(),
            PolicyError::WeightCount {
                expected: 6,
                found: 5
            }
        ));
        assert!(matches!(
            Policy::from_bytes(&file(&good, &[0.0; 7])).unwrap_err(),
            PolicyError::WeightCount {
                expected: 6,
                found: 7
            }
        ));
        let mismatch = header(r#"{"kind": "linear", "inputs": 3, "outputs": 2}"#, 2, 2);
        assert!(matches!(
            Policy::from_bytes(&file(&mismatch, &[0.0; 8])).unwrap_err(),
            PolicyError::Shape(_)
        ));
        let nan = file(&good, &[f32::NAN, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert!(matches!(
            Policy::from_bytes(&nan).unwrap_err(),
            PolicyError::BadHeader(_)
        ));
        // Game settings the bot could not play are refused here, not by a panic in the bot.
        for (from, to) in [
            (r#""tick_skip": 8"#, r#""tick_skip": 0"#),
            (r#""action_delay": 7"#, r#""action_delay": 8"#),
            (r#""team_size": 1"#, r#""team_size": 0"#),
            (r#""team_size": 1"#, r#""team_size": 4"#),
        ] {
            let unplayable = good.replace(from, to);
            assert_ne!(unplayable, good);
            assert!(matches!(
                Policy::from_bytes(&file(&unplayable, &[0.0; 6])).unwrap_err(),
                PolicyError::BadHeader(_)
            ));
        }
    }

    #[test]
    fn argmax_and_sampling_respect_the_mask() {
        let logits = [5.0, 1.0, 0.0, 4.0];
        let mask = [0u8, 1, 1, 1];
        assert_eq!(Policy::argmax(&logits, &mask), 3);
        let mut rng = fastrand::Rng::with_seed(1);
        let mut counts = [0usize; 4];
        for _ in 0..4000 {
            counts[Policy::sample(&logits, &mask, &mut rng)] += 1;
        }
        assert_eq!(counts[0], 0, "a forbidden action was sampled");
        // Probabilities over the allowed actions: e^1 : e^0 : e^4 = 0.046 : 0.017 : 0.937.
        assert!(
            (counts[3] as f32 / 4000.0 - 0.937).abs() < 0.03,
            "{counts:?}"
        );
    }
}
