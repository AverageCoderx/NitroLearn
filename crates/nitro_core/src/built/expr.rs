//! Expressions of built rewards: a small formula language read once from the settings, checked
//! (names, types, limits) and then interpreted per car and step. No loops, no calls out: the work per step is
//! bounded by the expression's size times the number of cars.
//!
//! Values are numbers or 3-vectors; true/false are 1/0. Entities (`me`, `ball`, `mate`, `opp`, `it` inside
//! a sum over cars) have fields (`me.pos`, `ball.vel`, `me.on_wall` ...). Functions that remember (`since`,
//! `held`, `count`, `once`) keep numbers per car between steps; `prev(...)` reads the step before.

use rocketsim::Vec3A;

use crate::custom::kit::{
    attack_goal, attack_net_back, attack_sign, closest_to_ball, own_goal, own_net_back,
};
use crate::snapshot::{GameSnapshot, NUM_BOOST_PADS, StepEvents};

/// Most nodes one expression may have.
pub const MAX_NODES: usize = 400;
/// What `since` gives before its condition was ever true (seconds): "never".
pub const NEVER: f32 = 1.0e9;
const CEILING_Z: f32 = 2044.0;
const SIDE_WALL_X: f32 = 4096.0;
const BACK_WALL_Y: f32 = 5120.0;
/// A car on its wheels with its roof's up.z at least this is on the floor; at most −this, the ceiling.
const FLOOR_UP_Z: f32 = 0.7;

/// The boost pads (x, y, big), in the snapshot's order (by y, then x).
const PADS: [(f32, f32, bool); NUM_BOOST_PADS] = [
    (0.0, -4240.0, false),
    (-1792.0, -4184.0, false),
    (1792.0, -4184.0, false),
    (-3072.0, -4096.0, true),
    (3072.0, -4096.0, true),
    (-940.0, -3308.0, false),
    (940.0, -3308.0, false),
    (0.0, -2816.0, false),
    (-3584.0, -2484.0, false),
    (3584.0, -2484.0, false),
    (-1788.0, -2300.0, false),
    (1788.0, -2300.0, false),
    (-2048.0, -1036.0, false),
    (0.0, -1024.0, false),
    (2048.0, -1036.0, false),
    (-3584.0, 0.0, true),
    (-1024.0, 0.0, false),
    (1024.0, 0.0, false),
    (3584.0, 0.0, true),
    (-2048.0, 1036.0, false),
    (0.0, 1024.0, false),
    (2048.0, 1036.0, false),
    (-1788.0, 2300.0, false),
    (1788.0, 2300.0, false),
    (-3584.0, 2484.0, false),
    (3584.0, 2484.0, false),
    (0.0, 2816.0, false),
    (-940.0, 3310.0, false),
    (940.0, 3308.0, false),
    (-3072.0, 4096.0, true),
    (3072.0, 4096.0, true),
    (-1792.0, 4184.0, false),
    (1792.0, 4184.0, false),
    (0.0, 4240.0, false),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ty {
    Num,
    Vec,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ent {
    Me,
    Mate,
    Opp,
    It,
    Ball,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Field {
    Pos,
    Vel,
    AngVel,
    Forward,
    Up,
    Right,
    X,
    Y,
    Z,
    Speed,
    Boost,
    OnGround,
    OnFloor,
    OnWall,
    OnCeiling,
    InAir,
    HasFlip,
    Jumped,
    DoubleJumped,
    Flipped,
    Flipping,
    Boosting,
    Supersonic,
    Demoed,
    AirTime,
    Team,
    WallDist,
    CeilingDist,
    Closest,
    Touch,
    TouchAir,
    TouchStrength,
    Shot,
    Bump,
    Demo,
    GotDemoed,
    Jump,
    Flip,
    RegainedFlip,
    FlipReset,
    LastTouch,
    Touched,
}

/// How a field may be read: of the state, of the step's details (`prev` reads the step before's), or of
/// what happened during the step (not inside `prev`).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    State,
    Detail,
    Event,
}

/// (name, field, type, kind, on cars, on the ball)
const FIELDS: &[(&str, Field, Ty, Kind, bool, bool)] = &[
    ("pos", Field::Pos, Ty::Vec, Kind::State, true, true),
    ("vel", Field::Vel, Ty::Vec, Kind::State, true, true),
    ("angvel", Field::AngVel, Ty::Vec, Kind::State, true, true),
    ("forward", Field::Forward, Ty::Vec, Kind::State, true, false),
    ("up", Field::Up, Ty::Vec, Kind::State, true, false),
    ("right", Field::Right, Ty::Vec, Kind::State, true, false),
    ("x", Field::X, Ty::Num, Kind::State, true, true),
    ("y", Field::Y, Ty::Num, Kind::State, true, true),
    ("z", Field::Z, Ty::Num, Kind::State, true, true),
    ("speed", Field::Speed, Ty::Num, Kind::State, true, true),
    ("boost", Field::Boost, Ty::Num, Kind::State, true, false),
    (
        "on_ground",
        Field::OnGround,
        Ty::Num,
        Kind::State,
        true,
        false,
    ),
    (
        "on_floor",
        Field::OnFloor,
        Ty::Num,
        Kind::State,
        true,
        false,
    ),
    ("on_wall", Field::OnWall, Ty::Num, Kind::State, true, false),
    (
        "on_ceiling",
        Field::OnCeiling,
        Ty::Num,
        Kind::State,
        true,
        false,
    ),
    ("in_air", Field::InAir, Ty::Num, Kind::State, true, false),
    (
        "has_flip",
        Field::HasFlip,
        Ty::Num,
        Kind::State,
        true,
        false,
    ),
    ("jumped", Field::Jumped, Ty::Num, Kind::State, true, false),
    (
        "double_jumped",
        Field::DoubleJumped,
        Ty::Num,
        Kind::State,
        true,
        false,
    ),
    ("flipped", Field::Flipped, Ty::Num, Kind::State, true, false),
    (
        "flipping",
        Field::Flipping,
        Ty::Num,
        Kind::Detail,
        true,
        false,
    ),
    (
        "boosting",
        Field::Boosting,
        Ty::Num,
        Kind::Detail,
        true,
        false,
    ),
    (
        "supersonic",
        Field::Supersonic,
        Ty::Num,
        Kind::State,
        true,
        false,
    ),
    ("demoed", Field::Demoed, Ty::Num, Kind::State, true, false),
    (
        "air_time",
        Field::AirTime,
        Ty::Num,
        Kind::Detail,
        true,
        false,
    ),
    ("team", Field::Team, Ty::Num, Kind::State, true, false),
    (
        "wall_dist",
        Field::WallDist,
        Ty::Num,
        Kind::State,
        true,
        true,
    ),
    (
        "ceiling_dist",
        Field::CeilingDist,
        Ty::Num,
        Kind::State,
        true,
        true,
    ),
    ("closest", Field::Closest, Ty::Num, Kind::State, true, false),
    ("touch", Field::Touch, Ty::Num, Kind::Event, true, false),
    (
        "touch_air",
        Field::TouchAir,
        Ty::Num,
        Kind::Event,
        true,
        false,
    ),
    (
        "touch_strength",
        Field::TouchStrength,
        Ty::Num,
        Kind::Event,
        true,
        false,
    ),
    ("shot", Field::Shot, Ty::Num, Kind::Event, true, false),
    ("bump", Field::Bump, Ty::Num, Kind::Event, true, false),
    ("demo", Field::Demo, Ty::Num, Kind::Event, true, false),
    (
        "got_demoed",
        Field::GotDemoed,
        Ty::Num,
        Kind::Event,
        true,
        false,
    ),
    ("jump", Field::Jump, Ty::Num, Kind::Event, true, false),
    ("flip", Field::Flip, Ty::Num, Kind::Event, true, false),
    (
        "regained_flip",
        Field::RegainedFlip,
        Ty::Num,
        Kind::Event,
        true,
        false,
    ),
    (
        "flip_reset",
        Field::FlipReset,
        Ty::Num,
        Kind::Event,
        true,
        false,
    ),
    (
        "last_touch",
        Field::LastTouch,
        Ty::Num,
        Kind::Event,
        true,
        false,
    ),
    ("touched", Field::Touched, Ty::Num, Kind::Event, false, true),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Konst {
    Dt,
    Goal,
    Concede,
    Kickoff,
    HasMate,
    HasOpp,
    OwnGoal,
    OppGoal,
    OwnNet,
    OppNet,
}

/// (name, constant, type, depends on the step's events)
const CONSTANTS: &[(&str, Konst, Ty, bool)] = &[
    ("dt", Konst::Dt, Ty::Num, false),
    ("goal", Konst::Goal, Ty::Num, false),
    ("concede", Konst::Concede, Ty::Num, false),
    ("kickoff", Konst::Kickoff, Ty::Num, false),
    ("has_mate", Konst::HasMate, Ty::Num, false),
    ("has_opp", Konst::HasOpp, Ty::Num, false),
    ("own_goal", Konst::OwnGoal, Ty::Vec, false),
    ("opp_goal", Konst::OppGoal, Ty::Vec, false),
    ("own_net", Konst::OwnNet, Ty::Vec, false),
    ("opp_net", Konst::OppNet, Ty::Vec, false),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Div,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
    And,
    Or,
    Min,
    Max,
    Pow,
    Decay,
    Dot,
    Dist,
    Angle,
    VAdd,
    VSub,
    VScale,
    VDiv,
    Cross,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Un {
    Neg,
    Not,
    Abs,
    Sign,
    Sqrt,
    Exp,
    Log,
    Length,
    VNeg,
    Norm,
    Flat,
    Local,
    CompX,
    CompY,
    CompZ,
    NearestPad,
    NearestBigPad,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Agg {
    Sum,
    Min,
    Max,
    Count,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Remember {
    /// Seconds since the condition was last true (`NEVER` before).
    Since,
    /// Seconds the condition has held without a break (0 while false).
    Held,
    /// Steps on which the condition was true this episode.
    Count,
    /// 1 on the first step the condition is true this episode, else 0.
    Once,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Node {
    Num(f32),
    Const(Konst),
    Var(usize),
    Field(Ent, Field),
    Un(Un, Box<Node>),
    Bin(Op, Box<Node>, Box<Node>),
    Clamp(Box<Node>, Box<Node>, Box<Node>),
    If(Box<Node>, Box<Node>, Box<Node>),
    MkVec(Box<Node>, Box<Node>, Box<Node>),
    Prev(Box<Node>),
    /// Over the teammates (`true`) or the opponents, `it` being each in turn.
    Agg(Agg, bool, Box<Node>),
    /// The output of remembering function number n.
    State(usize),
}

/// A remembering function: its condition and where its numbers are.
#[derive(Clone, Debug, PartialEq)]
pub struct Stateful {
    pub kind: Remember,
    pub cond: Node,
}

/// What an expression reads, for deciding how it may be used.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Uses {
    /// Something that happened during the step (touches, jumps, goals...).
    pub events: bool,
    /// The step before (`prev`, `delta`).
    pub prev: bool,
    /// A variable or a remembering function.
    pub memory: bool,
    /// The step's length (`dt`).
    pub dt: bool,
}

impl Uses {
    pub fn with(self, other: Self) -> Self {
        Self {
            events: self.events || other.events,
            prev: self.prev || other.prev,
            memory: self.memory || other.memory,
            dt: self.dt || other.dt,
        }
    }

    /// Of the state alone: the same at any tick skip when paid every step.
    pub fn state_only(self) -> bool {
        !(self.events || self.prev || self.memory || self.dt)
    }
}

/// The functions a formula can call.
pub const FUNCTIONS: &[&str] = &[
    "if",
    "min",
    "max",
    "clamp",
    "abs",
    "sign",
    "sqrt",
    "pow",
    "exp",
    "log",
    "decay",
    "dot",
    "cross",
    "dist",
    "angle",
    "length",
    "norm",
    "flat",
    "local",
    "vec",
    "nearest_pad",
    "nearest_big_pad",
    "prev",
    "delta",
    "since",
    "held",
    "count",
    "once",
    "sum_mates",
    "sum_opps",
    "min_mates",
    "min_opps",
    "max_mates",
    "max_opps",
    "count_mates",
    "count_opps",
];

/// Names a variable cannot take: the language's own.
pub fn reserved(name: &str) -> bool {
    const WORDS: &[&str] = &["and", "or", "not", "me", "mate", "opp", "it", "ball", "pi"];
    WORDS.contains(&name) || FUNCTIONS.contains(&name) || CONSTANTS.iter().any(|(n, ..)| *n == name)
}

/// The words formulas know: car fields, ball fields, named values, functions.
pub fn words() -> (
    Vec<&'static str>,
    Vec<&'static str>,
    Vec<&'static str>,
    Vec<&'static str>,
) {
    let fields = |cars: bool| -> Vec<&'static str> {
        FIELDS
            .iter()
            .filter(|f| if cars { f.4 } else { f.5 })
            .map(|f| f.0)
            .collect()
    };
    (
        fields(true),
        fields(false),
        CONSTANTS.iter().map(|c| c.0).collect(),
        FUNCTIONS.to_vec(),
    )
}

// ---------------------------------------------------------------------------------------------------- parsing

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Num(f32),
    Ident(String),
    Op(&'static str),
    LParen,
    RParen,
    Comma,
    Dot,
}

fn lex(text: &str) -> Result<Vec<(Tok, usize)>, String> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let at = i + 1;
        if c.is_whitespace() {
            i += 1;
        } else if c.is_ascii_digit()
            || (c == '.' && chars.get(i + 1).is_some_and(char::is_ascii_digit))
        {
            let start = i;
            while i < chars.len()
                && (chars[i].is_ascii_digit() || chars[i] == '.' || chars[i] == '_')
            {
                i += 1;
            }
            if i < chars.len() && (chars[i] == 'e' || chars[i] == 'E') {
                i += 1;
                if i < chars.len() && (chars[i] == '-' || chars[i] == '+') {
                    i += 1;
                }
                while i < chars.len() && chars[i].is_ascii_digit() {
                    i += 1;
                }
            }
            let written: String = chars[start..i].iter().filter(|&&c| c != '_').collect();
            let n = written
                .parse::<f32>()
                .ok()
                .filter(|n| n.is_finite())
                .ok_or_else(|| format!("'{written}' at {at} is not a number (or too big)"))?;
            out.push((Tok::Num(n), at));
        } else if c.is_ascii_alphabetic() || c == '_' {
            let start = i;
            while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            out.push((Tok::Ident(chars[start..i].iter().collect()), at));
        } else {
            let two: String = chars[i..(i + 2).min(chars.len())].iter().collect();
            let op = ["<=", ">=", "==", "!="].into_iter().find(|op| *op == two);
            if let Some(op) = op {
                out.push((Tok::Op(op), at));
                i += 2;
                continue;
            }
            let tok = match c {
                '+' => Tok::Op("+"),
                '-' => Tok::Op("-"),
                '*' => Tok::Op("*"),
                '/' => Tok::Op("/"),
                '<' => Tok::Op("<"),
                '>' => Tok::Op(">"),
                '(' => Tok::LParen,
                ')' => Tok::RParen,
                ',' => Tok::Comma,
                '.' => Tok::Dot,
                _ => return Err(format!("'{c}' at {at} is not something a formula can hold")),
            };
            out.push((tok, at));
            i += 1;
        }
    }
    Ok(out)
}

/// The names an expression can see beyond the built-in ones: the reward's variables, in order.
pub struct Scope<'a> {
    pub vars: &'a [String],
}

/// An expression checked and ready, with the remembering functions it holds.
pub struct Compiled {
    pub node: Node,
    pub ty: Ty,
    pub uses: Uses,
}

struct Parser<'a> {
    toks: Vec<(Tok, usize)>,
    pos: usize,
    scope: &'a Scope<'a>,
    stateful: &'a mut Vec<Stateful>,
    nodes: usize,
    /// Inside a sum over cars (where `it` means each car) / inside `prev` / inside a remembering function.
    in_agg: bool,
    in_prev: bool,
    in_memory: bool,
}

type Out = (Node, Ty, Uses);

/// Parses and checks `text`; remembering functions it holds are added to `stateful` (their numbers
/// `State(n)` count on from the ones already there).
pub fn compile(
    text: &str,
    scope: &Scope,
    stateful: &mut Vec<Stateful>,
) -> Result<Compiled, String> {
    let toks = lex(text)?;
    if toks.is_empty() {
        return Err("the formula is empty".into());
    }
    let mut parser = Parser {
        toks,
        pos: 0,
        scope,
        stateful,
        nodes: 0,
        in_agg: false,
        in_prev: false,
        in_memory: false,
    };
    let (node, ty, uses) = parser.or()?;
    if let Some((tok, at)) = parser.toks.get(parser.pos) {
        return Err(format!("unexpected {} at {at}", describe(tok)));
    }
    Ok(Compiled { node, ty, uses })
}

fn describe(tok: &Tok) -> String {
    match tok {
        Tok::Num(n) => format!("number {n}"),
        Tok::Ident(name) => format!("'{name}'"),
        Tok::Op(op) => format!("'{op}'"),
        Tok::LParen => "'('".into(),
        Tok::RParen => "')'".into(),
        Tok::Comma => "','".into(),
        Tok::Dot => "'.'".into(),
    }
}

fn num(out: Out, what: &str) -> Result<(Node, Uses), String> {
    match out.1 {
        Ty::Num => Ok((out.0, out.2)),
        Ty::Vec => Err(format!(
            "{what} needs a number, not a direction or position (use length(...) or .x)"
        )),
    }
}

fn vect(out: Out, what: &str) -> Result<(Node, Uses), String> {
    match out.1 {
        Ty::Vec => Ok((out.0, out.2)),
        Ty::Num => Err(format!(
            "{what} needs a direction or position, not a number"
        )),
    }
}

impl Parser<'_> {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos).map(|(t, _)| t)
    }

    fn at(&self) -> usize {
        self.toks.get(self.pos).map_or_else(
            || self.toks.last().map_or(1, |(_, at)| at + 1),
            |(_, at)| *at,
        )
    }

    fn word(&self, word: &str) -> bool {
        matches!(self.peek(), Some(Tok::Ident(w)) if w == word)
    }

    fn op(&self, op: &str) -> bool {
        matches!(self.peek(), Some(Tok::Op(o)) if *o == op)
    }

    fn grow(&mut self) -> Result<(), String> {
        self.nodes += 1;
        if self.nodes > MAX_NODES {
            return Err(format!(
                "the formula is too long (at most {MAX_NODES} parts)"
            ));
        }
        Ok(())
    }

    fn expect(&mut self, tok: Tok, what: &str) -> Result<(), String> {
        if self.peek() == Some(&tok) {
            self.pos += 1;
            Ok(())
        } else {
            let found = self.peek().map_or_else(|| "the end".to_string(), describe);
            Err(format!("expected {what} at {}, found {found}", self.at()))
        }
    }

    fn bin(&mut self, op: Op, a: Out, b: Out) -> Result<Out, String> {
        self.grow()?;
        let uses = a.2.with(b.2);
        let (ta, tb) = (a.1, b.1);
        let (op, ty, a, b) = match (op, ta, tb) {
            (Op::Add, Ty::Vec, Ty::Vec) => (Op::VAdd, Ty::Vec, a.0, b.0),
            (Op::Sub, Ty::Vec, Ty::Vec) => (Op::VSub, Ty::Vec, a.0, b.0),
            (Op::Mul, Ty::Vec, Ty::Num) => (Op::VScale, Ty::Vec, a.0, b.0),
            (Op::Mul, Ty::Num, Ty::Vec) => (Op::VScale, Ty::Vec, b.0, a.0),
            (Op::Div, Ty::Vec, Ty::Num) => (Op::VDiv, Ty::Vec, a.0, b.0),
            (_, Ty::Num, Ty::Num) => (op, Ty::Num, a.0, b.0),
            _ => {
                return Err(format!(
                    "this {} mixes numbers and directions in a way that has no meaning (at {})",
                    op_word(op),
                    self.at()
                ));
            }
        };
        Ok((Node::Bin(op, Box::new(a), Box::new(b)), ty, uses))
    }

    fn or(&mut self) -> Result<Out, String> {
        let mut left = self.and()?;
        while self.word("or") {
            self.pos += 1;
            let right = self.and()?;
            left = self.logic(Op::Or, left, right)?;
        }
        Ok(left)
    }

    fn and(&mut self) -> Result<Out, String> {
        let mut left = self.not()?;
        while self.word("and") {
            self.pos += 1;
            let right = self.not()?;
            left = self.logic(Op::And, left, right)?;
        }
        Ok(left)
    }

    fn logic(&mut self, op: Op, a: Out, b: Out) -> Result<Out, String> {
        let word = if op == Op::And { "and" } else { "or" };
        let a = num(a, word)?;
        let b = num(b, word)?;
        self.grow()?;
        Ok((
            Node::Bin(op, Box::new(a.0), Box::new(b.0)),
            Ty::Num,
            a.1.with(b.1),
        ))
    }

    fn not(&mut self) -> Result<Out, String> {
        if self.word("not") {
            self.pos += 1;
            // Counted before going deeper, so a long run of signs stops at the limit, not the stack's end.
            self.grow()?;
            let inner = num(self.not()?, "not")?;
            return Ok((Node::Un(Un::Not, Box::new(inner.0)), Ty::Num, inner.1));
        }
        self.cmp()
    }

    fn cmp(&mut self) -> Result<Out, String> {
        let left = self.add()?;
        let op = match self.peek() {
            Some(Tok::Op("<")) => Op::Lt,
            Some(Tok::Op("<=")) => Op::Le,
            Some(Tok::Op(">")) => Op::Gt,
            Some(Tok::Op(">=")) => Op::Ge,
            Some(Tok::Op("==")) => Op::Eq,
            Some(Tok::Op("!=")) => Op::Ne,
            _ => return Ok(left),
        };
        self.pos += 1;
        let right = self.add()?;
        let a = num(left, "a comparison")?;
        let b = num(right, "a comparison")?;
        self.grow()?;
        Ok((
            Node::Bin(op, Box::new(a.0), Box::new(b.0)),
            Ty::Num,
            a.1.with(b.1),
        ))
    }

    fn add(&mut self) -> Result<Out, String> {
        let mut left = self.mul()?;
        loop {
            let op = if self.op("+") {
                Op::Add
            } else if self.op("-") {
                Op::Sub
            } else {
                return Ok(left);
            };
            self.pos += 1;
            let right = self.mul()?;
            left = self.bin(op, left, right)?;
        }
    }

    fn mul(&mut self) -> Result<Out, String> {
        let mut left = self.unary()?;
        loop {
            let op = if self.op("*") {
                Op::Mul
            } else if self.op("/") {
                Op::Div
            } else {
                return Ok(left);
            };
            self.pos += 1;
            let right = self.unary()?;
            left = self.bin(op, left, right)?;
        }
    }

    fn unary(&mut self) -> Result<Out, String> {
        if self.op("-") {
            self.pos += 1;
            self.grow()?;
            let (node, ty, uses) = self.unary()?;
            let un = if ty == Ty::Vec { Un::VNeg } else { Un::Neg };
            return Ok((Node::Un(un, Box::new(node)), ty, uses));
        }
        self.postfix()
    }

    fn postfix(&mut self) -> Result<Out, String> {
        let mut out = self.primary()?;
        while self.peek() == Some(&Tok::Dot) {
            self.pos += 1;
            let at = self.at();
            let Some(Tok::Ident(name)) = self.peek().cloned() else {
                return Err(format!("expected x, y or z after '.' at {at}"));
            };
            self.pos += 1;
            let un = match name.as_str() {
                "x" => Un::CompX,
                "y" => Un::CompY,
                "z" => Un::CompZ,
                _ => {
                    return Err(format!(
                        "'.{name}' at {at}: a direction or position has only .x, .y and .z"
                    ));
                }
            };
            let (node, uses) = vect(out, &format!("'.{name}'"))?;
            self.grow()?;
            out = (Node::Un(un, Box::new(node)), Ty::Num, uses);
        }
        Ok(out)
    }

    fn args(&mut self, name: &str) -> Result<Vec<Out>, String> {
        self.expect(Tok::LParen, &format!("'(' after {name}"))?;
        let mut args = Vec::new();
        if self.peek() == Some(&Tok::RParen) {
            self.pos += 1;
            return Ok(args);
        }
        loop {
            args.push(self.or()?);
            if self.peek() == Some(&Tok::Comma) {
                self.pos += 1;
                continue;
            }
            self.expect(Tok::RParen, &format!("',' or ')' in {name}(...)"))?;
            return Ok(args);
        }
    }

    fn primary(&mut self) -> Result<Out, String> {
        let at = self.at();
        let Some(tok) = self.peek().cloned() else {
            return Err(format!("the formula ends too soon (at {at})"));
        };
        self.pos += 1;
        self.grow()?;
        match tok {
            Tok::Num(n) => Ok((Node::Num(n), Ty::Num, Uses::default())),
            Tok::LParen => {
                let inner = self.or()?;
                self.expect(Tok::RParen, "')'")?;
                Ok(inner)
            }
            Tok::Ident(name) if self.peek() == Some(&Tok::LParen) => self.call(&name, at),
            Tok::Ident(name) => self.name(&name, at),
            other => Err(format!("unexpected {} at {at}", describe(&other))),
        }
    }

    fn name(&mut self, name: &str, at: usize) -> Result<Out, String> {
        if let Some(i) = self.scope.vars.iter().position(|v| v == name) {
            let uses = Uses {
                memory: true,
                ..Uses::default()
            };
            return Ok((Node::Var(i), Ty::Num, uses));
        }
        let ent = match name {
            "me" => Some(Ent::Me),
            "mate" => Some(Ent::Mate),
            "opp" => Some(Ent::Opp),
            "it" => Some(Ent::It),
            "ball" => Some(Ent::Ball),
            _ => None,
        };
        if let Some(ent) = ent {
            if ent == Ent::It && !self.in_agg {
                return Err(format!(
                    "'it' at {at} means each car inside sum_mates(...), max_opps(...) and the like"
                ));
            }
            self.expect(
                Tok::Dot,
                &format!("'.' and a field after '{name}' (e.g. {name}.pos)"),
            )?;
            let fat = self.at();
            let Some(Tok::Ident(field)) = self.peek().cloned() else {
                return Err(format!("expected a field of '{name}' at {fat}"));
            };
            self.pos += 1;
            let car = ent != Ent::Ball;
            let Some(&(_, f, ty, kind, on_car, on_ball)) =
                FIELDS.iter().find(|(n, ..)| *n == field)
            else {
                return Err(format!("'{name}' has no field '{field}' (at {fat})"));
            };
            if (car && !on_car) || (!car && !on_ball) {
                return Err(format!("'{name}' has no field '{field}' (at {fat})"));
            }
            if kind == Kind::Event && self.in_prev {
                return Err(format!(
                    "{name}.{field} (at {fat}) is something that happens during a step: prev(...) cannot read it"
                ));
            }
            let uses = Uses {
                events: kind == Kind::Event,
                ..Uses::default()
            };
            return Ok((Node::Field(ent, f), ty, uses));
        }
        if name == "pi" {
            return Ok((Node::Num(std::f32::consts::PI), Ty::Num, Uses::default()));
        }
        if let Some(&(_, k, ty, _)) = CONSTANTS.iter().find(|(n, ..)| *n == name) {
            let uses = Uses {
                events: matches!(k, Konst::Goal | Konst::Concede),
                dt: k == Konst::Dt,
                ..Uses::default()
            };
            if self.in_prev && matches!(k, Konst::Goal | Konst::Concede | Konst::Dt) {
                return Err(format!("prev(...) cannot read '{name}' (at {at})"));
            }
            return Ok((Node::Const(k), ty, uses));
        }
        Err(format!("there is nothing called '{name}' (at {at})"))
    }

    fn call(&mut self, name: &str, at: usize) -> Result<Out, String> {
        let wrong = |n: usize| format!("{name}(...) at {at} takes {n}");
        match name {
            "prev" | "delta" => {
                if self.in_prev {
                    return Err(format!("{name}(...) at {at} cannot be inside prev(...)"));
                }
                let was = self.in_prev;
                self.in_prev = true;
                let args = self.args(name);
                self.in_prev = was;
                let mut args = args?;
                if args.len() != 1 {
                    return Err(wrong(1));
                }
                let (node, ty, uses) = args.remove(0);
                let uses = uses.with(Uses {
                    prev: true,
                    ..Uses::default()
                });
                let past = Node::Prev(Box::new(node.clone()));
                if name == "prev" {
                    return Ok((past, ty, uses));
                }
                let op = if ty == Ty::Vec { Op::VSub } else { Op::Sub };
                Ok((Node::Bin(op, Box::new(node), Box::new(past)), ty, uses))
            }
            "since" | "held" | "count" | "once" => {
                if self.in_prev || self.in_agg || self.in_memory {
                    return Err(format!(
                        "{name}(...) at {at} cannot be inside prev(...), a sum over cars or another {name}-like function"
                    ));
                }
                self.in_memory = true;
                let args = self.args(name);
                self.in_memory = false;
                let mut args = args?;
                if args.len() != 1 {
                    return Err(wrong(1));
                }
                let (cond, uses) = num(args.remove(0), name)?;
                let kind = match name {
                    "since" => Remember::Since,
                    "held" => Remember::Held,
                    "count" => Remember::Count,
                    _ => Remember::Once,
                };
                self.stateful.push(Stateful { kind, cond });
                let uses = uses.with(Uses {
                    memory: true,
                    ..Uses::default()
                });
                Ok((Node::State(self.stateful.len() - 1), Ty::Num, uses))
            }
            "sum_mates" | "sum_opps" | "min_mates" | "min_opps" | "max_mates" | "max_opps"
            | "count_mates" | "count_opps" => {
                if self.in_agg {
                    return Err(format!(
                        "{name}(...) at {at} cannot be inside another sum over cars"
                    ));
                }
                self.in_agg = true;
                let args = self.args(name);
                self.in_agg = false;
                let mut args = args?;
                if args.len() != 1 {
                    return Err(wrong(1));
                }
                let (inner, uses) = num(args.remove(0), name)?;
                let agg = match &name[..3] {
                    "sum" => Agg::Sum,
                    "min" => Agg::Min,
                    "max" => Agg::Max,
                    _ => Agg::Count,
                };
                Ok((
                    Node::Agg(agg, name.ends_with("mates"), Box::new(inner)),
                    Ty::Num,
                    uses,
                ))
            }
            _ => {
                let args = self.args(name)?;
                self.function(name, at, args)
            }
        }
    }

    fn function(&mut self, name: &str, at: usize, mut args: Vec<Out>) -> Result<Out, String> {
        let count = args.len();
        let need = |n: usize| -> Result<(), String> {
            if count == n {
                Ok(())
            } else {
                Err(format!("{name}(...) at {at} takes {n}, not {count}"))
            }
        };
        let un_num = |un: Un, args: &mut Vec<Out>| -> Result<Out, String> {
            let (a, uses) = num(args.remove(0), name)?;
            Ok((Node::Un(un, Box::new(a)), Ty::Num, uses))
        };
        match name {
            "abs" | "sign" | "sqrt" | "exp" | "log" => {
                need(1)?;
                let un = match name {
                    "abs" => Un::Abs,
                    "sign" => Un::Sign,
                    "sqrt" => Un::Sqrt,
                    "exp" => Un::Exp,
                    _ => Un::Log,
                };
                un_num(un, &mut args)
            }
            "length" | "norm" | "flat" | "local" | "nearest_pad" | "nearest_big_pad" => {
                need(1)?;
                let (a, uses) = vect(args.remove(0), name)?;
                let (un, ty) = match name {
                    "length" => (Un::Length, Ty::Num),
                    "norm" => (Un::Norm, Ty::Vec),
                    "flat" => (Un::Flat, Ty::Vec),
                    "local" => (Un::Local, Ty::Vec),
                    "nearest_pad" => (Un::NearestPad, Ty::Num),
                    _ => (Un::NearestBigPad, Ty::Num),
                };
                Ok((Node::Un(un, Box::new(a)), ty, uses))
            }
            "min" | "max" => {
                if count < 2 {
                    return Err(format!("{name}(...) at {at} takes 2 or more"));
                }
                let op = if name == "min" { Op::Min } else { Op::Max };
                let mut iter = args.into_iter();
                let (mut node, mut uses) = num(iter.next().expect("2+"), name)?;
                for arg in iter {
                    let (b, u) = num(arg, name)?;
                    self.grow()?;
                    node = Node::Bin(op, Box::new(node), Box::new(b));
                    uses = uses.with(u);
                }
                Ok((node, Ty::Num, uses))
            }
            "pow" | "decay" => {
                need(2)?;
                let (a, ua) = num(args.remove(0), name)?;
                let (b, ub) = num(args.remove(0), name)?;
                let op = if name == "pow" { Op::Pow } else { Op::Decay };
                Ok((
                    Node::Bin(op, Box::new(a), Box::new(b)),
                    Ty::Num,
                    ua.with(ub),
                ))
            }
            "dot" | "dist" | "angle" | "cross" => {
                need(2)?;
                let (a, ua) = vect(args.remove(0), name)?;
                let (b, ub) = vect(args.remove(0), name)?;
                let (op, ty) = match name {
                    "dot" => (Op::Dot, Ty::Num),
                    "dist" => (Op::Dist, Ty::Num),
                    "angle" => (Op::Angle, Ty::Num),
                    _ => (Op::Cross, Ty::Vec),
                };
                Ok((Node::Bin(op, Box::new(a), Box::new(b)), ty, ua.with(ub)))
            }
            "clamp" | "if" | "vec" => {
                need(3)?;
                let c = args.pop().expect("3");
                let b = args.pop().expect("3");
                let a = args.pop().expect("3");
                if name == "if" {
                    let (cond, uc) = num(a, "if's condition")?;
                    if b.1 != c.1 {
                        return Err(format!(
                            "if(...) at {at}: both answers must be numbers, or both directions"
                        ));
                    }
                    let ty = b.1;
                    let uses = uc.with(b.2).with(c.2);
                    return Ok((
                        Node::If(Box::new(cond), Box::new(b.0), Box::new(c.0)),
                        ty,
                        uses,
                    ));
                }
                let (a, ua) = num(a, name)?;
                let (b, ub) = num(b, name)?;
                let (c, uc) = num(c, name)?;
                let uses = ua.with(ub).with(uc);
                if name == "clamp" {
                    Ok((
                        Node::Clamp(Box::new(a), Box::new(b), Box::new(c)),
                        Ty::Num,
                        uses,
                    ))
                } else {
                    Ok((
                        Node::MkVec(Box::new(a), Box::new(b), Box::new(c)),
                        Ty::Vec,
                        uses,
                    ))
                }
            }
            _ => Err(format!("there is no function called '{name}' (at {at})")),
        }
    }
}

fn op_word(op: Op) -> &'static str {
    match op {
        Op::Add => "'+'",
        Op::Sub => "'-'",
        Op::Mul => "'*'",
        _ => "'/'",
    }
}

// ---------------------------------------------------------------------------------------------------- evaluation

/// What an expression is evaluated on: one car's view of one step, and its memory.
pub struct Ctx<'a> {
    pub prev: &'a GameSnapshot,
    pub cur: &'a GameSnapshot,
    pub events: &'a StepEvents,
    pub car: usize,
    /// The car `it` means inside a sum over cars.
    pub it: usize,
    /// Read the step before (inside `prev`).
    pub past: bool,
    /// The reward's variables now and at the step before.
    pub vars: &'a [f32],
    pub prev_vars: &'a [f32],
    /// Each remembering function's output.
    pub states: &'a [f32],
}

fn bit(b: bool) -> f32 {
    f32::from(u8::from(b))
}

fn truth(x: f32) -> bool {
    x != 0.0
}

impl Ctx<'_> {
    fn game(&self) -> &GameSnapshot {
        if self.past { self.prev } else { self.cur }
    }

    fn nearest(&self, mates: bool) -> Option<usize> {
        let game = self.game();
        let me = &game.cars[self.car];
        (0..game.cars.len())
            .filter(|&o| o != self.car && (game.cars[o].team == me.team) == mates)
            .min_by(|&a, &b| {
                let d = |c: usize| game.cars[c].body.pos.distance(me.body.pos);
                d(a).total_cmp(&d(b))
            })
    }

    fn car_of(&self, ent: Ent) -> Option<usize> {
        match ent {
            Ent::Me => Some(self.car),
            Ent::It => Some(self.it),
            Ent::Mate => self.nearest(true),
            Ent::Opp => self.nearest(false),
            Ent::Ball => None,
        }
    }

    fn field_v(&self, ent: Ent, field: Field) -> Vec3A {
        let game = self.game();
        let body = if ent == Ent::Ball {
            &game.ball
        } else {
            match self.car_of(ent) {
                Some(car) => &game.cars[car].body,
                None => return Vec3A::ZERO,
            }
        };
        match field {
            Field::Pos => body.pos,
            Field::Vel => body.vel,
            Field::AngVel => body.ang_vel,
            Field::Forward => body.rot.x_axis,
            Field::Right => body.rot.y_axis,
            _ => body.rot.z_axis,
        }
    }

    fn field_n(&self, ent: Ent, field: Field) -> f32 {
        let game = self.game();
        let wall = |p: Vec3A| (SIDE_WALL_X - p.x.abs()).min(BACK_WALL_Y - p.y.abs());
        if ent == Ent::Ball {
            let ball = &game.ball;
            return match field {
                Field::X => ball.pos.x,
                Field::Y => ball.pos.y,
                Field::Z => ball.pos.z,
                Field::Speed => ball.vel.length(),
                Field::WallDist => wall(ball.pos),
                Field::CeilingDist => CEILING_Z - ball.pos.z,
                _ => bit(self.events.cars.iter().any(|e| e.touched_ball)),
            };
        }
        let Some(car) = self.car_of(ent) else {
            return 0.0;
        };
        let me = &game.cars[car];
        let before = &self.prev.cars[car];
        let detail = || {
            if self.past {
                self.events.prev_detail(car)
            } else {
                self.events.detail(car)
            }
        };
        let happened = || self.events.cars.get(car).copied().unwrap_or_default();
        let up = me.body.up().z;
        match field {
            Field::X => me.body.pos.x,
            Field::Y => me.body.pos.y,
            Field::Z => me.body.pos.z,
            Field::Speed => me.body.vel.length(),
            Field::Boost => me.boost,
            Field::OnGround => bit(me.on_ground),
            Field::OnFloor => bit(me.on_ground && up >= FLOOR_UP_Z),
            Field::OnWall => bit(me.on_ground && up < FLOOR_UP_Z && up > -FLOOR_UP_Z),
            Field::OnCeiling => bit(me.on_ground && up <= -FLOOR_UP_Z),
            Field::InAir => bit(!me.on_ground),
            Field::HasFlip => bit(me.can_flip),
            Field::Jumped => bit(me.has_jumped),
            Field::DoubleJumped => bit(me.has_double_jumped),
            Field::Flipped => bit(me.has_flipped),
            Field::Flipping => bit(detail().is_flipping),
            Field::Boosting => bit(detail().is_boosting),
            Field::Supersonic => bit(me.is_supersonic),
            Field::Demoed => bit(me.is_demoed),
            Field::AirTime => detail().air_time,
            Field::Team => attack_sign(me.team),
            Field::WallDist => wall(me.body.pos),
            Field::CeilingDist => CEILING_Z - me.body.pos.z,
            Field::Closest => bit(closest_to_ball(game, me.team) == Some(car)),
            Field::Touch => bit(happened().touched_ball),
            Field::TouchAir => bit(happened().touched_ball_in_air),
            Field::TouchStrength => {
                if happened().touched_ball {
                    (game.ball.vel - self.prev.ball.vel).length()
                } else {
                    0.0
                }
            }
            Field::Shot => bit(happened().touched_ball
                && crate::mechanics::heading_into_goal(
                    &game.ball,
                    attack_sign(me.team) * BACK_WALL_Y,
                )),
            Field::Bump => bit(happened().bumped_opponent),
            Field::Demo => bit(happened().demoed_opponent),
            Field::GotDemoed => bit(me.is_demoed && !before.is_demoed),
            Field::Jump => bit(me.has_jumped && !before.has_jumped),
            Field::Flip => bit(me.has_flipped && !before.has_flipped),
            Field::RegainedFlip => bit(me.can_flip && !before.can_flip),
            Field::FlipReset => bit(me.can_flip && !before.can_flip && !me.on_ground),
            Field::LastTouch => bit(self.events.last_touch == Some(car)),
            _ => 0.0,
        }
    }

    fn konst_n(&self, k: Konst) -> f32 {
        let game = self.game();
        let team = game.cars[self.car].team;
        match k {
            Konst::Dt => self.events.step_seconds,
            Konst::Goal => bit(game.goal_scored == Some(team)),
            Konst::Concede => bit(game.goal_scored == Some(team.opposite())),
            Konst::Kickoff => bit(crate::custom::kit::is_kickoff(game)),
            Konst::HasMate => bit(self.nearest(true).is_some()),
            Konst::HasOpp => bit(self.nearest(false).is_some()),
            _ => 0.0,
        }
    }

    fn konst_v(&self, k: Konst) -> Vec3A {
        let team = self.game().cars[self.car].team;
        match k {
            Konst::OwnGoal => own_goal(team),
            Konst::OppGoal => attack_goal(team),
            Konst::OwnNet => own_net_back(team),
            _ => attack_net_back(team),
        }
    }

    pub fn num(&self, node: &Node) -> f32 {
        match node {
            Node::Num(n) => *n,
            Node::Const(k) => self.konst_n(*k),
            Node::Var(i) => {
                if self.past {
                    self.prev_vars[*i]
                } else {
                    self.vars[*i]
                }
            }
            Node::State(i) => self.states[*i],
            Node::Field(ent, field) => self.field_n(*ent, *field),
            Node::Un(un, a) => match un {
                Un::Neg => -self.num(a),
                Un::Not => bit(!truth(self.num(a))),
                Un::Abs => self.num(a).abs(),
                Un::Sign => {
                    let x = self.num(a);
                    if x > 0.0 {
                        1.0
                    } else if x < 0.0 {
                        -1.0
                    } else {
                        0.0
                    }
                }
                Un::Sqrt => self.num(a).sqrt(),
                Un::Exp => self.num(a).exp(),
                Un::Log => self.num(a).ln(),
                Un::Length => self.vec(a).length(),
                Un::CompX => self.vec(a).x,
                Un::CompY => self.vec(a).y,
                Un::CompZ => self.vec(a).z,
                Un::NearestPad | Un::NearestBigPad => {
                    let p = self.vec(a);
                    let big_only = *un == Un::NearestBigPad;
                    let game = self.game();
                    PADS.iter()
                        .zip(&game.pads)
                        .filter(|((_, _, big), pad)| pad.active && (*big || !big_only))
                        .map(|(&(x, y, _), _)| Vec3A::new(x - p.x, y - p.y, 0.0).length())
                        .fold(f32::INFINITY, f32::min)
                }
                _ => unreachable!("checked when compiled"),
            },
            Node::Bin(op, a, b) => match op {
                Op::Add => self.num(a) + self.num(b),
                Op::Sub => self.num(a) - self.num(b),
                Op::Mul => self.num(a) * self.num(b),
                Op::Div => self.num(a) / self.num(b),
                Op::Lt => bit(self.num(a) < self.num(b)),
                Op::Le => bit(self.num(a) <= self.num(b)),
                Op::Gt => bit(self.num(a) > self.num(b)),
                Op::Ge => bit(self.num(a) >= self.num(b)),
                Op::Eq => bit(self.num(a) == self.num(b)),
                Op::Ne => bit(self.num(a) != self.num(b)),
                Op::And => bit(truth(self.num(a)) && truth(self.num(b))),
                Op::Or => bit(truth(self.num(a)) || truth(self.num(b))),
                Op::Min => self.num(a).min(self.num(b)),
                Op::Max => self.num(a).max(self.num(b)),
                Op::Pow => self.num(a).powf(self.num(b)),
                Op::Decay => 0.5f32.powf(self.num(a) / self.num(b)),
                Op::Dot => self.vec(a).dot(self.vec(b)),
                Op::Dist => self.vec(a).distance(self.vec(b)),
                Op::Angle => {
                    let cos = self
                        .vec(a)
                        .normalize_or_zero()
                        .dot(self.vec(b).normalize_or_zero());
                    cos.clamp(-1.0, 1.0).acos().to_degrees()
                }
                _ => unreachable!("checked when compiled"),
            },
            Node::Clamp(x, lo, hi) => self.num(x).max(self.num(lo)).min(self.num(hi)),
            Node::If(c, a, b) => {
                if truth(self.num(c)) {
                    self.num(a)
                } else {
                    self.num(b)
                }
            }
            Node::Prev(a) => self.past().num(a),
            Node::Agg(agg, mates, inner) => {
                let game = self.game();
                let team = game.cars[self.car].team;
                let cars = (0..game.cars.len())
                    .filter(|&o| o != self.car && (game.cars[o].team == team) == *mates);
                let mut each = cars.map(|o| self.with_it(o).num(inner));
                match agg {
                    Agg::Sum => each.sum(),
                    Agg::Count => each.filter(|&v| truth(v)).count() as f32,
                    Agg::Min => each.fold(f32::INFINITY, f32::min),
                    Agg::Max => {
                        let first = each.next();
                        first.map_or(f32::NEG_INFINITY, |f| each.fold(f, f32::max))
                    }
                }
            }
            Node::MkVec(..) => unreachable!("checked when compiled"),
        }
    }

    pub fn vec(&self, node: &Node) -> Vec3A {
        match node {
            Node::Const(k) => self.konst_v(*k),
            Node::Field(ent, field) => self.field_v(*ent, *field),
            Node::Un(un, a) => match un {
                Un::VNeg => -self.vec(a),
                Un::Norm => self.vec(a).normalize_or_zero(),
                Un::Flat => {
                    let v = self.vec(a);
                    Vec3A::new(v.x, v.y, 0.0)
                }
                Un::Local => {
                    let v = self.vec(a);
                    let body = &self.game().cars[self.car].body;
                    Vec3A::new(
                        v.dot(body.rot.x_axis),
                        v.dot(body.rot.y_axis),
                        v.dot(body.rot.z_axis),
                    )
                }
                _ => unreachable!("checked when compiled"),
            },
            Node::Bin(op, a, b) => match op {
                Op::VAdd => self.vec(a) + self.vec(b),
                Op::VSub => self.vec(a) - self.vec(b),
                Op::VScale => self.vec(a) * self.num(b),
                Op::VDiv => self.vec(a) / self.num(b),
                Op::Cross => self.vec(a).cross(self.vec(b)),
                _ => unreachable!("checked when compiled"),
            },
            Node::If(c, a, b) => {
                if truth(self.num(c)) {
                    self.vec(a)
                } else {
                    self.vec(b)
                }
            }
            Node::MkVec(x, y, z) => Vec3A::new(self.num(x), self.num(y), self.num(z)),
            Node::Prev(a) => self.past().vec(a),
            _ => unreachable!("checked when compiled"),
        }
    }

    fn past(&self) -> Self {
        Ctx {
            past: true,
            ..*self
        }
    }

    fn with_it(&self, it: usize) -> Self {
        Ctx { it, ..*self }
    }
}

impl Clone for Ctx<'_> {
    fn clone(&self) -> Self {
        *self
    }
}

impl Copy for Ctx<'_> {}
