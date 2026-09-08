//! The tokenizer.
//!
//! Turns a character stream into [`Token`]s according to the current category
//! codes, implementing the line-state machine described in *The TeXbook*
//! (chapter 8). It operates over an in-memory string so it is WASM-friendly —
//! all file access lives in the CLI layer.
//!
//! The catcode table is supplied on each call rather than held by the
//! tokenizer, because TeX lets constructs like `\makeatletter`, `\catcode`, and
//! verbatim change catcodes *while* tokenizing. The tokenizer is thus a pure
//! function of its position and the table it is handed.

use crate::token::{CatCode, CatCodeTable, Token};

pub(crate) fn control_word(source: &str, slash: usize) -> Option<(&str, usize)> {
    let bytes = source.as_bytes();
    if bytes.get(slash) != Some(&b'\\') {
        return None;
    }
    let start = slash + 1;
    let mut end = start;
    while bytes.get(end).is_some_and(u8::is_ascii_alphabetic) {
        end += 1;
    }
    Some((&source[start..end], end))
}

pub(crate) fn skip_whitespace(source: &str, mut from: usize) -> usize {
    while source
        .as_bytes()
        .get(from)
        .is_some_and(u8::is_ascii_whitespace)
    {
        from += 1;
    }
    from
}

pub(crate) fn balanced_group(
    source: &str,
    from: usize,
    open: u8,
    close: u8,
) -> Option<(&str, usize)> {
    let start = skip_whitespace(source, from);
    let bytes = source.as_bytes();
    if bytes.get(start) != Some(&open) {
        return None;
    }
    let mut depth = 0usize;
    for index in start..bytes.len() {
        if bytes[index] == open {
            depth += 1;
        } else if bytes[index] == close {
            depth -= 1;
            if depth == 0 {
                return Some((&source[start + 1..index], index + 1));
            }
        }
    }
    None
}

/// The reader's line state, controlling how spaces and line ends are handled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// At the start of a line: leading spaces are ignored; a blank line yields
    /// `\par`.
    NewLine,
    /// In the middle of a line: spaces produce a single space token.
    MidLine,
    /// Skipping blanks (e.g. right after a control word): spaces and line ends
    /// are ignored.
    SkipBlanks,
}

/// A tokenizer over an in-memory source string.
pub struct Tokenizer {
    chars: Vec<char>,
    pos: usize,
    state: State,
}

impl Tokenizer {
    /// Create a tokenizer reading `source`.
    pub fn new(source: &str) -> Self {
        Tokenizer {
            chars: source.chars().collect(),
            pos: 0,
            state: State::NewLine,
        }
    }

    /// Tokenize the entire input using a fixed catcode table.
    pub fn tokenize(mut self, catcodes: &CatCodeTable) -> Vec<Token> {
        let mut out = Vec::new();
        while let Some(t) = self.next_token(catcodes) {
            out.push(t);
        }
        out
    }

    /// Read raw source, ignoring category codes, up to (and consuming) the
    /// literal `marker`. Returns the text before it. This is how verbatim
    /// content is read faithfully — no tokenization, no space collapsing, no
    /// comment stripping. If the marker is absent, the rest of the input is
    /// returned.
    pub fn read_until_literal(&mut self, marker: &str) -> String {
        let marker: Vec<char> = marker.chars().collect();
        let mut out = String::new();
        while self.pos < self.chars.len() {
            if self.matches_at(self.pos, &marker) {
                self.pos += marker.len();
                return out;
            }
            out.push(self.chars[self.pos]);
            self.pos += 1;
        }
        out
    }

    /// Read a single raw character, ignoring category codes and state (used to
    /// pick up a `\verb` delimiter exactly as it appears — a space or `%` is a
    /// literal delimiter, not a skipped blank or a comment). `None` at end.
    pub fn read_raw_char(&mut self) -> Option<char> {
        let c = self.chars.get(self.pos).copied()?;
        self.pos += 1;
        Some(c)
    }

    fn matches_at(&self, at: usize, marker: &[char]) -> bool {
        marker
            .iter()
            .enumerate()
            .all(|(k, &c)| self.chars.get(at + k) == Some(&c))
    }

    /// Read the character at `i`, resolving TeX `^^` notation. Returns the
    /// decoded character and the index of the following character.
    fn resolved_char(&self, i: usize, catcodes: &CatCodeTable) -> Option<(char, usize)> {
        let c = *self.chars.get(i)?;
        // `^^` notation only triggers when the char is a superscript and is
        // immediately followed by an identical character.
        if catcodes.get(c) == CatCode::Superscript
            && let Some(&c2) = self.chars.get(i + 1)
            && c2 == c
        {
            // Two lowercase hex digits form a byte value.
            if let (Some(&h1), Some(&h2)) = (self.chars.get(i + 2), self.chars.get(i + 3))
                && is_lower_hex(h1)
                && is_lower_hex(h2)
            {
                let v = (hex_val(h1) << 4) | hex_val(h2);
                if let Some(ch) = char::from_u32(v) {
                    return Some((ch, i + 4));
                }
            }
            // Otherwise, a single following character is XOR'd with 64.
            if let Some(&n) = self.chars.get(i + 2) {
                let code = n as u32;
                if code < 128 {
                    let decoded = if code < 64 { code + 64 } else { code - 64 };
                    if let Some(ch) = char::from_u32(decoded) {
                        return Some((ch, i + 3));
                    }
                }
            }
        }
        Some((c, i + 1))
    }

    /// Peek the resolved character, its catcode, and the following index.
    fn peek(&self, catcodes: &CatCodeTable) -> Option<(char, CatCode, usize)> {
        let (c, next) = self.resolved_char(self.pos, catcodes)?;
        Some((c, catcodes.get(c), next))
    }

    /// Consume any remaining characters on the current physical line, including
    /// the terminating line end.
    fn skip_to_line_end(&mut self, catcodes: &CatCodeTable) {
        while let Some((_, cc, next)) = self.peek(catcodes) {
            if cc == CatCode::EndLine {
                self.consume_line_end(next);
                return;
            }
            self.pos = next;
        }
    }

    /// Advance past a line ending, collapsing a `\r\n` pair into one.
    fn consume_line_end(&mut self, after_first: usize) {
        let first = self.chars[self.pos];
        self.pos = after_first;
        if first == '\r'
            && let Some(&'\n') = self.chars.get(self.pos)
        {
            self.pos += 1;
        }
    }

    /// Produce the next token, or `None` at end of input.
    pub fn next_token(&mut self, catcodes: &CatCodeTable) -> Option<Token> {
        loop {
            let (c, cc, next) = self.peek(catcodes)?;
            match cc {
                CatCode::Escape => return Some(self.read_control_sequence(next, catcodes)),
                CatCode::Comment => {
                    self.skip_to_line_end(catcodes);
                    self.state = State::NewLine;
                }
                CatCode::EndLine => {
                    self.consume_line_end(next);
                    let produced = match self.state {
                        State::NewLine => Some(Token::cs("par")),
                        State::MidLine => Some(Token::Char(' ', CatCode::Space)),
                        State::SkipBlanks => None,
                    };
                    self.state = State::NewLine;
                    if let Some(t) = produced {
                        return Some(t);
                    }
                }
                CatCode::Space => {
                    self.pos = next;
                    if self.state == State::MidLine {
                        self.state = State::SkipBlanks;
                        return Some(Token::Char(' ', CatCode::Space));
                    }
                    // NewLine / SkipBlanks: ignore the space.
                }
                CatCode::Ignored | CatCode::Invalid => {
                    self.pos = next;
                }
                CatCode::Active => {
                    self.pos = next;
                    self.state = State::MidLine;
                    // The default active character is `~` = non-breaking space.
                    // Emit it as the nbsp character directly, rather than as the
                    // control sequence named "~", so it stays distinct from the
                    // control symbol `\~` (the tilde accent), which the engine
                    // reads as `\cs("~")`. Any other active character keeps the
                    // control-sequence form so it can be `\def`-defined.
                    if c == '~' {
                        return Some(Token::Char('\u{00A0}', CatCode::Other));
                    }
                    return Some(Token::cs(c.to_string()));
                }
                _ => {
                    // Ordinary character token.
                    self.pos = next;
                    self.state = State::MidLine;
                    return Some(Token::Char(c, cc));
                }
            }
        }
    }

    /// Read a control sequence, given the index just past the escape char.
    fn read_control_sequence(&mut self, after_escape: usize, catcodes: &CatCodeTable) -> Token {
        self.pos = after_escape;
        let Some((c, cc, next)) = self.peek(catcodes) else {
            // Escape char at end of input: empty control sequence.
            self.state = State::MidLine;
            return Token::cs("");
        };
        match cc {
            CatCode::Letter => {
                // A control word: read all following letters.
                let mut name = String::new();
                name.push(c);
                self.pos = next;
                while let Some((c2, CatCode::Letter, next2)) = self.peek(catcodes) {
                    name.push(c2);
                    self.pos = next2;
                }
                self.state = State::SkipBlanks;
                Token::cs(name)
            }
            CatCode::Space => {
                // Control space `\ `.
                self.pos = next;
                self.state = State::SkipBlanks;
                Token::cs(" ")
            }
            _ => {
                // Single-character control symbol.
                self.pos = next;
                self.state = State::MidLine;
                Token::cs(c.to_string())
            }
        }
    }
}

fn is_lower_hex(c: char) -> bool {
    c.is_ascii_digit() || ('a'..='f').contains(&c)
}

fn hex_val(c: char) -> u32 {
    if c.is_ascii_digit() {
        c as u32 - '0' as u32
    } else {
        c as u32 - 'a' as u32 + 10
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(s: &str) -> Vec<Token> {
        let cc = CatCodeTable::default();
        Tokenizer::new(s).tokenize(&cc)
    }

    #[test]
    fn simple_word() {
        assert_eq!(
            toks("abc"),
            vec![
                Token::Char('a', CatCode::Letter),
                Token::Char('b', CatCode::Letter),
                Token::Char('c', CatCode::Letter),
            ]
        );
    }

    #[test]
    fn control_word_eats_trailing_space() {
        // `\foo bar` -> \foo, b, a, r  (space after control word is swallowed)
        assert_eq!(
            toks("\\foo bar"),
            vec![
                Token::cs("foo"),
                Token::Char('b', CatCode::Letter),
                Token::Char('a', CatCode::Letter),
                Token::Char('r', CatCode::Letter),
            ]
        );
    }

    #[test]
    fn control_symbol() {
        // `\$x` -> \$, x  (single non-letter control symbol keeps mid-line state)
        assert_eq!(
            toks("\\$x"),
            vec![Token::cs("$"), Token::Char('x', CatCode::Letter)]
        );
    }

    #[test]
    fn collapse_spaces_midline() {
        // multiple spaces collapse to one; leading spaces on a line are dropped
        assert_eq!(
            toks("a   b"),
            vec![
                Token::Char('a', CatCode::Letter),
                Token::Char(' ', CatCode::Space),
                Token::Char('b', CatCode::Letter),
            ]
        );
    }

    #[test]
    fn newline_is_space_midline() {
        assert_eq!(
            toks("a\nb"),
            vec![
                Token::Char('a', CatCode::Letter),
                Token::Char(' ', CatCode::Space),
                Token::Char('b', CatCode::Letter),
            ]
        );
    }

    #[test]
    fn blank_line_is_par() {
        // `a\n\nb` -> a, <space from first eol>, \par, b
        assert_eq!(
            toks("a\n\nb"),
            vec![
                Token::Char('a', CatCode::Letter),
                Token::Char(' ', CatCode::Space),
                Token::cs("par"),
                Token::Char('b', CatCode::Letter),
            ]
        );
    }

    #[test]
    fn comment_removes_rest_of_line() {
        assert_eq!(
            toks("a% comment\nb"),
            vec![
                Token::Char('a', CatCode::Letter),
                // comment swallows the newline too, so no space is produced
                Token::Char('b', CatCode::Letter),
            ]
        );
    }

    #[test]
    fn groups_and_math() {
        assert_eq!(
            toks("{$x$}"),
            vec![
                Token::Char('{', CatCode::BeginGroup),
                Token::Char('$', CatCode::MathShift),
                Token::Char('x', CatCode::Letter),
                Token::Char('$', CatCode::MathShift),
                Token::Char('}', CatCode::EndGroup),
            ]
        );
    }

    #[test]
    fn makeatletter_changes_control_word_boundary() {
        // With @ as a letter, `\foo@bar` is one control sequence.
        let mut cc = CatCodeTable::default();
        cc.set('@', CatCode::Letter);
        let got = Tokenizer::new("\\foo@bar").tokenize(&cc);
        assert_eq!(got, vec![Token::cs("foo@bar")]);
    }
}
