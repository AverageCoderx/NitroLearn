//! Where each row of a CSV file starts and ends, for the dashboard's metrics reader: one streamed
//! pass over the bytes, so a long run's metrics.csv (hundreds of MB) is never held in memory and no cell is
//! parsed. A row counts when it has `width` cells by its commas; a row holding a quote is left for the caller
//! to decide (a quoted cell can hold a comma).

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

const CHUNK: usize = 1 << 20;

/// The rows found from one byte on.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Rows {
    /// Each row's first byte, in file order.
    pub starts: Vec<u64>,
    /// The byte after each row's text (a `\r` before its `\n` left out).
    pub ends: Vec<u64>,
    /// Indices into `starts` of rows holding a quote: their width is the caller's to check.
    pub quoted: Vec<u64>,
    /// The byte after the last whole line (`\n`): where the next pass starts.
    pub end: u64,
}

/// The non-empty lines of `path` from byte `from` up to its last `\n`, keeping those with `width` cells by
/// their commas and every line that holds a quote (marked in `quoted`).
pub fn rows(path: &Path, from: u64, width: usize) -> io::Result<Rows> {
    let mut file = File::open(path)?;
    file.seek(SeekFrom::Start(from))?;
    let commas = width.saturating_sub(1);
    let mut out = Rows {
        end: from,
        ..Rows::default()
    };
    let mut buffer = vec![0u8; CHUNK];
    // Bytes of a line that runs past the end of a chunk, kept until its `\n` comes; `start`: its first byte.
    let mut carry: Vec<u8> = Vec::new();
    let mut start = from;
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        let mut rest = &buffer[..read];
        while let Some(at) = rest.iter().position(|&b| b == b'\n') {
            let line = if carry.is_empty() {
                &rest[..at]
            } else {
                carry.extend_from_slice(&rest[..at]);
                &carry[..]
            };
            let text = line.strip_suffix(b"\r").unwrap_or(line);
            if !text.is_empty() {
                let quote = text.contains(&b'"');
                if quote || text.iter().filter(|&&b| b == b',').count() == commas {
                    if quote {
                        out.quoted.push(out.starts.len() as u64);
                    }
                    out.starts.push(start);
                    out.ends.push(start + text.len() as u64);
                }
            }
            start += line.len() as u64 + 1;
            out.end = start;
            carry.clear();
            rest = &rest[at + 1..];
        }
        carry.extend_from_slice(rest);
    }
    Ok(out)
}

/// The cells of rows (each `starts[i]..ends[i]` of `path`, in file order) as numbers: `width` per row, NaN for
/// an empty cell. A row the caller must read itself is listed in the second value, its numbers left NaN: one
/// holding a quote, one not of `width` cells or past the file's end (it was rewritten), one with a cell that
/// is not plain number text (a reader of the caller's own may still take it).
pub fn numbers(
    path: &Path,
    starts: &[u64],
    ends: &[u64],
    width: usize,
) -> io::Result<(Vec<f64>, Vec<u64>)> {
    let mut file = io::BufReader::with_capacity(CHUNK, File::open(path)?);
    let mut out = vec![f64::NAN; starts.len() * width];
    let mut unsure = Vec::new();
    let mut line = Vec::new();
    let mut at: Option<u64> = None;
    for (i, (&start, &end)) in starts.iter().zip(ends).enumerate() {
        match at {
            Some(here) if start >= here => file.seek_relative((start - here) as i64)?,
            _ => {
                file.seek(SeekFrom::Start(start))?;
            }
        }
        line.resize(end.saturating_sub(start) as usize, 0);
        let read = file.read_exact(&mut line);
        at = read.is_ok().then_some(end);
        if let Err(err) = read
            && err.kind() != io::ErrorKind::UnexpectedEof
        {
            return Err(err);
        }
        let row = &mut out[i * width..(i + 1) * width];
        if at.is_none() || !fill(&line, row) {
            row.fill(f64::NAN);
            unsure.push(i as u64);
        }
    }
    Ok((out, unsure))
}

fn fill(line: &[u8], row: &mut [f64]) -> bool {
    if line.contains(&b'"') {
        return false;
    }
    let mut cells = line.split(|&b| b == b',');
    for slot in row.iter_mut() {
        let Some(cell) = cells.next() else {
            return false;
        };
        let cell = cell.trim_ascii();
        if cell.is_empty() {
            continue;
        }
        match std::str::from_utf8(cell)
            .ok()
            .and_then(|t| t.parse::<f64>().ok())
        {
            Some(value) => *slot = value,
            None => return false,
        }
    }
    cells.next().is_none()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn found(text: &[u8], from: u64, width: usize) -> Rows {
        let dir =
            std::env::temp_dir().join(format!("csv_rows_{}_{}", std::process::id(), text.len()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("m.csv");
        std::fs::write(&file, text).unwrap();
        let out = rows(&file, from, width).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        out
    }

    #[test]
    fn rows_of_the_headers_width_with_either_line_end() {
        let text = b"a,b,c\r\n1,2,3\r\n4,5\n\n6,7,8\n9,\"1,0\",2\n10,11,1";
        let out = found(text, 7, 3);
        let at = |s: &str| {
            text.windows(s.len())
                .position(|w| w == s.as_bytes())
                .unwrap() as u64
        };
        assert_eq!(out.starts, vec![7, at("6,7"), at("9,")]);
        assert_eq!(out.ends, vec![12, at("6,7") + 5, at("9,") + 9]);
        assert_eq!(
            out.quoted,
            vec![2],
            "the quoted row is the caller's to check"
        );
        assert_eq!(
            out.end,
            at("10,"),
            "the half-written last line waits for the next pass"
        );
    }

    #[test]
    fn lines_across_chunks_and_a_pass_from_the_middle() {
        let mut text = b"x,y\n".to_vec();
        let mut expected = Vec::new();
        for i in 0..200_000u64 {
            expected.push(text.len() as u64);
            text.extend_from_slice(format!("{i},{}\r\n", i * 7).as_bytes());
        }
        let out = found(&text, 4, 2);
        assert_eq!(out.starts, expected);
        assert_eq!(out.end, text.len() as u64);
        assert!(
            out.starts
                .iter()
                .zip(&out.ends)
                .all(|(s, e)| text[*e as usize] == b'\r' && s < e)
        );
        let later = found(&text, expected[150_000], 2);
        assert_eq!(later.starts, expected[150_000..]);
    }

    #[test]
    fn numbers_of_rows_and_the_rows_left_to_the_caller() {
        let text = b"a,b,c\n1,,3.5\n4,\"5\",6\n7,8\n-1e3, 2 ,inf\nx,y,z\n";
        let dir = std::env::temp_dir().join(format!("csv_numbers_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("m.csv");
        std::fs::write(&file, text).unwrap();
        let lines: Vec<(u64, u64)> = [6u64, 13, 21, 25, 38]
            .iter()
            .map(|&s| {
                (
                    s,
                    s + text[s as usize..].iter().position(|&b| b == b'\n').unwrap() as u64,
                )
            })
            .collect();
        let mut starts: Vec<u64> = lines.iter().map(|l| l.0).collect();
        let mut ends: Vec<u64> = lines.iter().map(|l| l.1).collect();
        starts.push(100); // a row the file no longer reaches (rewritten shorter)
        ends.push(110);
        let (got, unsure) = numbers(&file, &starts, &ends, 3).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(
            unsure,
            vec![1, 2, 4, 5],
            "a quote, two cells, words, past the end"
        );
        assert_eq!(got[0], 1.0);
        assert!(got[1].is_nan());
        assert_eq!(got[2], 3.5);
        assert_eq!(&got[9..12], &[-1000.0, 2.0, f64::INFINITY]);
        assert!(got[3..9].iter().chain(&got[12..]).all(|v| v.is_nan()));
    }
}
