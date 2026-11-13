//! Finding the rewards in Rust files, shared by the build script and its tests: the
//! structs that implement `Reward`, their `///` descriptions, their settings (one field a line) and whether
//! they are paid every step. A plain reading of the text: a file that does not compile is the compiler's to
//! report, not this.

use std::fmt::Write as _;

/// The built-in rewards' names, which a reward of one's own cannot take.
pub const BUILT_IN: [&str; 14] = [
    "air",
    "face_ball",
    "velocity_player_to_ball",
    "velocity_ball_to_goal",
    "touch_ball",
    "strong_touch",
    "pickup_boost",
    "save_boost",
    "bump",
    "demo",
    "goal",
    "alignment",
    "all_of",
    "goal_view",
];

#[derive(Debug, PartialEq, Eq)]
pub struct Found {
    /// The struct, as written (`FlipReset`).
    pub type_name: String,
    /// Its name in a settings file (`flip_reset`).
    pub name: String,
    pub doc: String,
    pub every_step: bool,
    pub params: Vec<Param>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Param {
    pub name: String,
    /// Its Rust type as written (`f32`, `bool`, `u32` …).
    pub kind: String,
    /// Marked `#[serde(default)]`: it may be left out of the settings.
    pub optional: bool,
    pub doc: String,
    /// The value it takes when left out, as JSON (`1.5`, `true`, `"text"`), when the file says it plainly
    /// (a literal in its `default = "…"` function, or the type's own default); empty otherwise.
    pub default: String,
}

/// `FlipReset` -> `flip_reset`.
pub fn snake(name: &str) -> String {
    let mut out = String::new();
    for (i, c) in name.chars().enumerate() {
        if c.is_uppercase() {
            if i > 0 {
                out.push('_');
            }
            out.extend(c.to_lowercase());
        } else {
            out.push(c);
        }
    }
    out
}

/// The text without comments and with string contents blanked, so `impl Reward for` in either is not
/// mistaken for code. Line breaks are kept.
fn code_only(source: &str) -> String {
    let mut out = String::with_capacity(source.len());
    let mut chars = source.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '/' if chars.peek() == Some(&'/') => {
                for next in chars.by_ref() {
                    if next == '\n' {
                        out.push('\n');
                        break;
                    }
                }
            }
            '/' if chars.peek() == Some(&'*') => {
                chars.next();
                let mut last = ' ';
                for next in chars.by_ref() {
                    if next == '\n' {
                        out.push('\n');
                    }
                    if last == '*' && next == '/' {
                        break;
                    }
                    last = next;
                }
                out.push(' ');
            }
            '"' => {
                out.push('"');
                let mut escaped = false;
                for next in chars.by_ref() {
                    if next == '\n' {
                        out.push('\n');
                    }
                    if !escaped && next == '"' {
                        break;
                    }
                    escaped = !escaped && next == '\\';
                }
                out.push('"');
            }
            _ => out.push(c),
        }
    }
    out
}

fn tokens(code: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut chars = code.chars().peekable();
    while let Some(&c) = chars.peek() {
        if c.is_whitespace() {
            chars.next();
        } else if c.is_alphanumeric() || c == '_' {
            let mut word = String::new();
            while let Some(&d) = chars.peek() {
                if d.is_alphanumeric() || d == '_' {
                    word.push(d);
                    chars.next();
                } else {
                    break;
                }
            }
            out.push(word);
        } else if c == ':' {
            chars.next();
            if chars.peek() == Some(&':') {
                chars.next();
                out.push("::".into());
            } else {
                out.push(":".into());
            }
        } else {
            out.push(c.to_string());
            chars.next();
        }
    }
    out
}

fn is_ident(token: &str) -> bool {
    token
        .chars()
        .next()
        .is_some_and(|c| c.is_alphabetic() || c == '_')
}

/// The tokens of the `{ … }` block starting at `open`, braces included, and the index after it.
fn block(tokens: &[String], open: usize) -> (&[String], usize) {
    let mut depth = 0;
    for (i, token) in tokens.iter().enumerate().skip(open) {
        match token.as_str() {
            "{" => depth += 1,
            "}" => {
                depth -= 1;
                if depth == 0 {
                    return (&tokens[open..=i], i + 1);
                }
            }
            _ => {}
        }
    }
    (&tokens[open..], tokens.len())
}

/// Every `impl … Reward for Name { … }` in `source`: (Name, whether its `every_step` says `true`).
fn reward_impls(tokens: &[String]) -> Vec<(String, bool)> {
    let mut found = Vec::new();
    let mut i = 0;
    while i < tokens.len() {
        if tokens[i] != "impl" {
            i += 1;
            continue;
        }
        let mut j = i + 1;
        let mut last = "";
        while j < tokens.len() && (is_ident(&tokens[j]) || tokens[j] == "::") && tokens[j] != "for"
        {
            if tokens[j] != "::" {
                last = &tokens[j];
            }
            j += 1;
        }
        let named = j + 1 < tokens.len() && tokens[j] == "for" && is_ident(&tokens[j + 1]);
        if last == "Reward" && named {
            let name = tokens[j + 1].clone();
            let open = (j + 2..tokens.len()).find(|&k| tokens[k] == "{");
            let every_step = open.is_some_and(|open| {
                let (body, _) = block(tokens, open);
                body.windows(2)
                    .position(|w| w[0] == "fn" && w[1] == "every_step")
                    .is_some_and(|at| {
                        let start = (at..body.len()).find(|&k| body[k] == "{");
                        start.is_some_and(|start| {
                            let (inner, _) = block(body, start);
                            inner.len() == 3 && inner[1] == "true"
                        })
                    })
            });
            found.push((name, every_step));
        }
        i = j.max(i + 1);
    }
    found
}

/// The `///` lines just above line `at` (attribute lines skipped), joined.
fn doc_above(lines: &[&str], at: usize) -> String {
    let mut docs = Vec::new();
    for line in lines[..at].iter().rev() {
        let line = line.trim();
        if let Some(doc) = line.strip_prefix("///") {
            docs.push(doc.trim().to_string());
        } else if line.starts_with("#[") || line.starts_with("#![") {
            continue;
        } else {
            break;
        }
    }
    docs.reverse();
    docs.join(" ")
}

fn struct_details(source: &str, type_name: &str) -> (String, Vec<Param>) {
    let lines: Vec<&str> = source.lines().collect();
    let code = code_only(source);
    let code_lines: Vec<&str> = code.lines().collect();
    let Some(start) = code_lines.iter().position(|line| {
        let words = tokens(line);
        words
            .windows(2)
            .any(|w| w[0] == "struct" && w[1] == type_name)
    }) else {
        return (String::new(), Vec::new());
    };
    let doc = doc_above(&lines, start);
    let mut params = Vec::new();
    let header = code_lines[start];
    let Some(open) = header.find('{') else {
        return (doc, params); // a unit or tuple struct: no settings
    };
    if header[open..].contains('}') {
        return (doc, params); // `{}` on its own line: no settings
    }
    for (offset, line) in code_lines[start + 1..].iter().enumerate() {
        let at = start + 1 + offset;
        if line.trim_start().starts_with('}') {
            break;
        }
        let words = tokens(line);
        let words: Vec<&str> = words
            .iter()
            .map(String::as_str)
            .filter(|w| *w != "pub")
            .collect();
        if words.len() >= 3 && is_ident(words[0]) && words[1] == ":" {
            let kind = words[2..]
                .iter()
                .take_while(|w| **w != ",")
                .copied()
                .collect::<String>();
            let attributes: Vec<String> = lines[..at]
                .iter()
                .rev()
                .take_while(|l| l.trim().starts_with("#[") || l.trim().starts_with("///"))
                .map(|l| l.replace(' ', ""))
                .collect();
            let marked = attributes.iter().find(|l| l.contains("serde(default"));
            params.push(Param {
                name: words[0].to_string(),
                optional: marked.is_some(),
                default: marked.map_or(String::new(), |l| default_of(source, l, &kind)),
                kind,
                doc: doc_above(&lines, at),
            });
        }
    }
    (doc, params)
}

/// The JSON of the value a setting takes when left out: from `#[serde(default = "f")]` (`attribute`, spaces
/// removed), the literal `f`'s body returns, or from `#[serde(default)]`, its type's own default. Empty when
/// the function computes it some other way.
fn default_of(source: &str, attribute: &str, kind: &str) -> String {
    let Some(start) = attribute.find("default=\"") else {
        return match kind {
            "f32" | "f64" => "0.0".into(),
            "bool" => "false".into(),
            "String" => "\"\"".into(),
            _ if kind.starts_with(['i', 'u']) => "0".into(),
            _ => String::new(),
        };
    };
    let path = attribute[start + 9..].split('"').next().unwrap_or_default();
    let function = path.rsplit("::").next().unwrap_or(path); // `md::scale`: a module in the same file
    let body =
        if let Some(at) = source.find(&format!("fn {function}()")) {
            let rest = &source[at..];
            let (Some(open), Some(close)) = (rest.find('{'), rest.find('}')) else {
                return String::new();
            };
            rest[open + 1..close.max(open + 1)].trim()
        } else {
            // A `defaults!` list (the library's aerial_kit): a line `name: type = value,`.
            let prefix = format!("{function}:");
            let Some(line) = source.lines().map(str::trim).find(|line| {
                line.starts_with(&prefix) && line.contains(" = ") && line.ends_with(',')
            }) else {
                return String::new();
            };
            let value = line.split_once(" = ").map_or("", |(_, value)| value);
            value.strip_suffix(',').unwrap_or(value).trim()
        };
    literal_json(body)
}

/// A Rust literal (as a default function returns it) as JSON, or empty when it is not a plain literal.
fn literal_json(body: &str) -> String {
    if body == "true" || body == "false" {
        return body.into();
    }
    let text = body
        .strip_suffix(".into()")
        .or_else(|| body.strip_suffix(".to_string()"))
        .or_else(|| body.strip_suffix(".to_owned()"))
        .or_else(|| {
            body.strip_prefix("String::from(")
                .and_then(|b| b.strip_suffix(')'))
        })
        .unwrap_or(body);
    if let Some(inner) = text.strip_prefix('"').and_then(|t| t.strip_suffix('"')) {
        let plain = !inner.contains(['"', '\\']);
        return if plain {
            format!("\"{inner}\"")
        } else {
            String::new()
        };
    }
    let number = body.replace('_', "");
    let number = ["f32", "f64", "i32", "i64", "u32", "u64", "usize", "u8"]
        .iter()
        .find_map(|suffix| number.strip_suffix(suffix))
        .unwrap_or(&number);
    match number.parse::<f64>() {
        Ok(value) if value.is_finite() && !number.starts_with('.') && !number.ends_with('.') => {
            number.trim_start_matches('+').to_string()
        }
        _ => String::new(),
    }
}

/// Every reward that `source` (one Rust file) defines.
pub fn rewards(source: &str) -> Vec<Found> {
    let words = tokens(&code_only(source));
    reward_impls(&words)
        .into_iter()
        .map(|(type_name, every_step)| {
            let (doc, params) = struct_details(source, &type_name);
            Found {
                name: snake(&type_name),
                type_name,
                doc,
                every_step,
                params,
            }
        })
        .collect()
}

/// The code the core includes: each file as a module, every reward found listed with its details, and the
/// function that makes one from its settings. `files` are (path, source); an error names the problem.
pub fn generate(files: &[(String, String)]) -> Result<String, String> {
    let mut code = String::from(
        "// Generated by crates/nitro_core/build.rs from the reward files it was given.\n",
    );
    let mut makers = String::new();
    let mut infos = String::new();
    let mut library = String::new();
    let mut seen: Vec<String> = Vec::new();
    for (index, (path, source)) in files.iter().enumerate() {
        writeln!(code, "#[path = {path:?}]\npub mod file_{index};").expect("writing to a String");
        let file = std::path::Path::new(path);
        let in_library = file
            .parent()
            .and_then(|folder| folder.file_name())
            .is_some_and(|name| name == "library");
        let stem = file
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or_default();
        if in_library && is_ident(stem) && !stem.starts_with(|c: char| c.is_ascii_digit()) {
            writeln!(library, "    pub use super::file_{index} as {stem};")
                .expect("writing to a String");
        }
        for found in rewards(source) {
            if BUILT_IN.contains(&found.name.as_str()) {
                return Err(format!(
                    "{path}: '{}' is the name of a built-in reward; give {} another name",
                    found.name, found.type_name
                ));
            }
            if seen.contains(&found.name) {
                return Err(format!(
                    "{path}: a second reward is called '{}'",
                    found.name
                ));
            }
            seen.push(found.name.clone());
            let params = found
                .params
                .iter()
                .map(|p| {
                    format!(
                        "Param {{ name: {:?}, kind: {:?}, optional: {}, doc: {:?}, default: {:?} }}",
                        p.name, p.kind, p.optional, p.doc, p.default
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            writeln!(
                infos,
                "    Info {{ name: {:?}, doc: {:?}, every_step: {}, file: {:?}, params: &[{params}] }},",
                found.name, found.doc, found.every_step, path
            )
            .expect("writing to a String");
            writeln!(
                makers,
                "        {:?} => Ok(std::sync::Arc::new(serde_json::from_value::<file_{index}::{}>(settings)\n            \
                 .map_err(|err| format!(\"{}: {{err}}\"))?)),",
                found.name, found.type_name, found.name
            )
            .expect("writing to a String");
        }
    }
    writeln!(
        code,
        "\n/// The files that come with NitroLearn (custom/library) by name, so one can use another's \
         helpers.\npub mod library {{\n{library}}}"
    )
    .expect("writing to a String");
    writeln!(
        code,
        "\n/// Every reward of one's own in this build.\npub const REWARDS: &[Info] = &[\n{infos}];"
    )
    .expect("writing to a String");
    writeln!(
        code,
        "\n/// Makes the reward called `name` from its settings.\npub fn make(name: &str, settings: serde_json::Value) \
         -> Result<std::sync::Arc<dyn Reward>, String> {{\n    match name {{\n{makers}        _ => Err(format!(\
         \"there is no reward called '{{name}}' in this build of NitroLearn\")),\n    }}\n}}"
    )
    .expect("writing to a String");
    Ok(code)
}
