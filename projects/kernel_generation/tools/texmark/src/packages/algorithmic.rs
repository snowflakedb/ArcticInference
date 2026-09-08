//! Native rendering for the `algorithmic` / `algorithmicx` packages.
//!
//! Converts the raw source of an `algorithmic` environment into a readable
//! pseudocode fenced block. The engine reads the body and delegates here; this
//! module is entirely self-contained and has no knowledge of the token stream.

use super::Package;
use crate::engine::{Engine, Event};
use crate::node::{Element, Node};

/// The algorithmic package handler.
pub struct Algorithmic;

impl Package for Algorithmic {
    fn package_names(&self) -> &[&str] {
        &["algorithmic", "algorithmicx", "algpseudocode"]
    }

    fn environments(&self) -> &[&str] {
        &["algorithmic", "algorithmic*"]
    }

    fn environment(&self, name: &str, e: &mut Engine) -> Event {
        // The body uses paper-defined macros (\vQ, \softmax, ...); expand them
        // so they reach the pseudocode as standard TeX, then reconstruct to text.
        let _ = e.grab_optional(); // e.g. line-numbering frequency
        let body = e.read_environment_body_raw(name);
        let expanded = e.expand_tokens(body);
        let raw = crate::engine::reconstruct(&expanded);

        let (text, labels) = clean(&raw);
        let mut node = Element::new("verbatim");
        node.attributes
            .push(("language".into(), "pseudocode".into()));
        if !labels.is_empty() {
            node.attributes.push(("labels".into(), labels.join(",")));
        }
        node.push(Node::text(text));
        Event::Blocks(vec![Node::Element(node)])
    }
}

/// Clean the raw source of an `algorithmic` environment into readable pseudocode.
///
/// When the environment body comes from an `\input`-ed file it is reconstructed
/// from tokens, where newlines collapse to spaces. [`normalize_lines`] inserts
/// a newline before each all-caps keyword so that [`strip_prefix`] can operate
/// line-by-line regardless of whether the source was raw-read or reconstructed.
fn clean(src: &str) -> (String, Vec<String>) {
    let normalized = normalize_lines(src);
    let mut out = String::new();
    let mut labels = Vec::new();
    for line in normalized.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let (trimmed, line_labels) = strip_labels(trimmed);
        labels.extend(line_labels);
        let cleaned = strip_prefix(trimmed.trim());
        let cleaned = cleaned.trim();
        if !cleaned.is_empty() {
            out.push_str(cleaned);
            out.push('\n');
        }
    }
    (out.trim_end().to_string(), labels)
}

/// Whether `name` is an algorithmic control/structure keyword, in any case —
/// old `algorithmic` uses ALL-CAPS (`\STATE`, `\ENDFOR`), `algorithmicx`/
/// `algpseudocode` use mixed case (`\State`, `\EndFor`). Only the structural
/// keywords are listed, so ordinary commands (`\to`, `\gets`) are left alone.
fn is_algo_keyword(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "state"
            | "statex"
            | "for"
            | "forall"
            | "endfor"
            | "if"
            | "elsif"
            | "elseif"
            | "else"
            | "endif"
            | "while"
            | "endwhile"
            | "repeat"
            | "until"
            | "loop"
            | "endloop"
            | "function"
            | "endfunction"
            | "procedure"
            | "endprocedure"
            | "require"
            | "ensure"
            | "input"
            | "output"
            | "return"
            | "comment"
            | "call"
            | "print"
            | "globals"
    )
}

/// Insert a newline before each algorithmic keyword command so they each start
/// their own line. Idempotent when the source already has newlines there.
fn normalize_lines(src: &str) -> String {
    let chars: Vec<char> = src.chars().collect();
    let mut out = String::with_capacity(src.len() + 32);
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '\\' {
            let name_start = i + 1;
            let name_end = chars[name_start..]
                .iter()
                .position(|c| !c.is_ascii_alphabetic())
                .map(|p| name_start + p)
                .unwrap_or(chars.len());
            if name_end > name_start
                && is_algo_keyword(&chars[name_start..name_end].iter().collect::<String>())
                && !out.is_empty()
                && !out.ends_with('\n')
            {
                out.push('\n');
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

/// Render a leading algorithmic keyword command as readable pseudocode.
fn strip_prefix(s: &str) -> String {
    let Some(rest) = s.strip_prefix('\\') else {
        return s.to_string();
    };
    let end = rest
        .find(|c: char| !c.is_ascii_alphabetic())
        .unwrap_or(rest.len());
    if end == 0 {
        return s.to_string();
    }
    let cmd = &rest[..end];
    if !is_algo_keyword(cmd) {
        return s.to_string();
    }
    // Strip outer braces from the argument, e.g. {$cond$} → $cond$.
    let content = rest[end..].trim();
    let content = strip_outer_braces(content);
    let command = cmd.to_ascii_lowercase();
    match command.as_str() {
        "state" | "statex" => content.to_string(),
        "endfor" => "End for".into(),
        "endif" => "End if".into(),
        "endwhile" => "End while".into(),
        "endloop" => "End loop".into(),
        "endfunction" => "End function".into(),
        "endprocedure" => "End procedure".into(),
        "else" => "Else".into(),
        _ if content.is_empty() => capitalize(&command),
        _ => format!("{} {content}", capitalize(&command)),
    }
}

/// Remove all `\label{...}` occurrences from `s` and retain their targets.
fn strip_labels(s: &str) -> (String, Vec<String>) {
    let mut out = String::with_capacity(s.len());
    let mut labels = Vec::new();
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' {
            // Peek at what follows.
            let word: String = chars
                .clone()
                .take_while(|ch| ch.is_ascii_alphabetic())
                .collect();
            if word == "label" {
                // Consume "label".
                for _ in 0..word.len() {
                    chars.next();
                }
                // Skip optional whitespace.
                while chars.peek() == Some(&' ') {
                    chars.next();
                }
                // Consume {balanced braces}.
                if chars.peek() == Some(&'{') {
                    chars.next();
                    let mut depth = 1u32;
                    let mut label = String::new();
                    for ch in chars.by_ref() {
                        if ch == '{' {
                            depth += 1;
                            label.push(ch);
                        } else if ch == '}' {
                            depth -= 1;
                            if depth == 0 {
                                break;
                            }
                            label.push(ch);
                        } else {
                            label.push(ch);
                        }
                    }
                    if !label.trim().is_empty() {
                        labels.push(label.trim().to_string());
                    }
                }
                continue;
            }
        }
        out.push(c);
    }
    (out, labels)
}

fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

/// Strip a single layer of outer `{...}` braces if present.
fn strip_outer_braces(s: &str) -> &str {
    if s.starts_with('{') && s.ends_with('}') {
        &s[1..s.len() - 1]
    } else {
        s
    }
}
