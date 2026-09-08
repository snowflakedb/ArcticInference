//! TeX tokens and category codes.
//!
//! This models TeX's tokenization primitives. A [`CatCode`] classifies each
//! input character; the tokenizer ([`crate::tokenizer`]) uses the active category
//! code table to turn characters into [`Token`]s.

use std::fmt;

/// TeX category codes (the "catcodes"), numbered as in *The TeXbook*.
///
/// These drive how the tokenizer interprets each character. The default
/// assignment is set up in [`CatCodeTable::default`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum CatCode {
    /// `\` — starts a control sequence.
    Escape = 0,
    /// `{` — begin group.
    BeginGroup = 1,
    /// `}` — end group.
    EndGroup = 2,
    /// `$` — math shift.
    MathShift = 3,
    /// `&` — alignment tab.
    AlignTab = 4,
    /// end of line (carriage return).
    EndLine = 5,
    /// `#` — macro parameter.
    Param = 6,
    /// `^` — superscript.
    Superscript = 7,
    /// `_` — subscript.
    Subscript = 8,
    /// ignored character.
    Ignored = 9,
    /// space.
    Space = 10,
    /// letter (`a`–`z`, `A`–`Z`).
    Letter = 11,
    /// any other character.
    Other = 12,
    /// active character (e.g. `~`), behaves like a control sequence.
    Active = 13,
    /// `%` — comment; consumes the rest of the line.
    Comment = 14,
    /// invalid character.
    Invalid = 15,
}

/// The category-code table mapping characters to their [`CatCode`].
///
/// TeX allows catcodes to be reassigned at runtime (`\catcode`); this table is
/// therefore part of the mutable engine state. It is initialized with TeX's
/// standard plain-format assignments.
#[derive(Debug, Clone)]
pub struct CatCodeTable {
    /// Dense table for the ASCII range, which covers the overwhelming majority
    /// of lookups. Non-ASCII characters fall back to [`CatCode::Other`] unless
    /// overridden in `overrides`.
    ascii: [CatCode; 128],
    /// Overrides for non-ASCII characters (rare; e.g. active Unicode chars).
    overrides: std::collections::HashMap<char, CatCode>,
}

impl CatCodeTable {
    /// Look up the category code for `c`.
    pub fn get(&self, c: char) -> CatCode {
        if (c as u32) < 128 {
            self.ascii[c as usize]
        } else {
            self.overrides.get(&c).copied().unwrap_or(CatCode::Other)
        }
    }

    /// Set the category code for `c`.
    pub fn set(&mut self, c: char, cc: CatCode) {
        if (c as u32) < 128 {
            self.ascii[c as usize] = cc;
        } else {
            self.overrides.insert(c, cc);
        }
    }
}

impl Default for CatCodeTable {
    /// TeX's standard (plain/LaTeX) category-code assignments.
    fn default() -> Self {
        let mut ascii = [CatCode::Other; 128];
        ascii[b'\\' as usize] = CatCode::Escape;
        ascii[b'{' as usize] = CatCode::BeginGroup;
        ascii[b'}' as usize] = CatCode::EndGroup;
        ascii[b'$' as usize] = CatCode::MathShift;
        ascii[b'&' as usize] = CatCode::AlignTab;
        ascii[b'\r' as usize] = CatCode::EndLine;
        ascii[b'\n' as usize] = CatCode::EndLine;
        ascii[b'#' as usize] = CatCode::Param;
        ascii[b'^' as usize] = CatCode::Superscript;
        ascii[b'_' as usize] = CatCode::Subscript;
        ascii[0_usize] = CatCode::Ignored; // null
        ascii[b' ' as usize] = CatCode::Space;
        ascii[b'\t' as usize] = CatCode::Space;
        ascii[b'%' as usize] = CatCode::Comment;
        ascii[127_usize] = CatCode::Invalid; // delete
        ascii[b'~' as usize] = CatCode::Active;
        for c in b'a'..=b'z' {
            ascii[c as usize] = CatCode::Letter;
        }
        for c in b'A'..=b'Z' {
            ascii[c as usize] = CatCode::Letter;
        }
        Self {
            ascii,
            overrides: std::collections::HashMap::new(),
        }
    }
}

/// A single TeX token: either a control sequence or a character with an
/// attached category code.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Token {
    /// A control sequence, e.g. `\section`. Stored without the leading escape
    /// character. Active characters are also represented here (the name is the
    /// single character).
    ControlSequence(String),
    /// A character token carrying the catcode it was read with.
    Char(char, CatCode),
}

impl Token {
    /// Construct a control-sequence token.
    pub fn cs(name: impl Into<String>) -> Token {
        Token::ControlSequence(name.into())
    }

    /// Is this a character with the given catcode?
    pub fn is_cat(&self, cc: CatCode) -> bool {
        matches!(self, Token::Char(_, c) if *c == cc)
    }

    /// The control-sequence name, if this is one.
    pub fn cs_name(&self) -> Option<&str> {
        match self {
            Token::ControlSequence(n) => Some(n),
            _ => None,
        }
    }

    /// The character, if this is a character token.
    pub fn char(&self) -> Option<char> {
        match self {
            Token::Char(c, _) => Some(*c),
            _ => None,
        }
    }
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Token::ControlSequence(name) => {
                // Single-character non-letter control sequences print without a
                // trailing space; multi-letter ones get one, matching TeX's
                // \meaning-ish rendering closely enough for diagnostics.
                if name.chars().count() == 1 && !name.chars().next().unwrap().is_alphabetic() {
                    write!(f, "\\{name}")
                } else {
                    write!(f, "\\{name} ")
                }
            }
            Token::Char(c, _) => write!(f, "{c}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_catcodes() {
        let t = CatCodeTable::default();
        assert_eq!(t.get('\\'), CatCode::Escape);
        assert_eq!(t.get('{'), CatCode::BeginGroup);
        assert_eq!(t.get('a'), CatCode::Letter);
        assert_eq!(t.get('Z'), CatCode::Letter);
        assert_eq!(t.get('1'), CatCode::Other);
        assert_eq!(t.get(' '), CatCode::Space);
        assert_eq!(t.get('%'), CatCode::Comment);
        assert_eq!(t.get('~'), CatCode::Active);
    }

    #[test]
    fn catcode_override() {
        let mut t = CatCodeTable::default();
        t.set('@', CatCode::Letter);
        assert_eq!(t.get('@'), CatCode::Letter);
    }
}
