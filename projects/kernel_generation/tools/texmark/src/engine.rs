//! The engine: expands macros and executes LaTeX constructs into a document
//! tree ([`Node`]).
//!
//! It reads tokens through a small pushback stack layered over the [`Tokenizer`],
//! so macro expansions and `\input`ed files are spliced in simply by pushing
//! their tokens back to be read next.
//!
//! Structure is produced in two easy steps rather than one clever one:
//! `Engine::next_event` emits a flat stream of inline and block pieces, with
//! sections and list items left as *markers*; a small folding pass
//! (`fold_sections`, `fold_items`) then nests them. Keeping the main loop
//! free of "close the previously-open thing" bookkeeping is what keeps it
//! readable.

use crate::node::{Element, Node};
use crate::packages;
use crate::state::{Environment, Macro, Resolver, State};
use crate::token::{CatCode, Token};
use crate::tokenizer::Tokenizer;
use std::collections::{HashMap, HashSet};

/// Upper bound on macro expansions per parse. A self-referential macro (or a
/// package's layout code that texmark can't faithfully run) would otherwise loop
/// forever. Real papers use only a few hundred expansions, so this leaves ~100×
/// headroom while bounding any runaway to a small, terminating amount of work.
const MAX_EXPANSIONS: usize = 100_000;

/// Upper bound on macro expansions performed since the last live *source*
/// character was consumed. A cyclic definition (`\def\x{\x}`, or the classic
/// `\let\a=\b`\`\def\b{...\a...}` alias loop) expands forever while consuming no
/// source, so this depth catches a runaway almost immediately — after which the
/// engine abandons only the offending construct's pending expansion and keeps
/// parsing the rest of the document, instead of the single global
/// [`MAX_EXPANSIONS`] budget tripping once and blanking everything after it.
/// Expansion nests only a handful deep between source reads in real documents
/// (even a math body's macro shorthands expand a bounded amount), so this leaves
/// ample headroom.
const MAX_EXPANSION_DEPTH: usize = 4_000;

/// Problems encountered during a parse that the caller must not ignore.
///
/// texmark never fails silently: when it cannot faithfully convert its input it
/// says so here. A default (all-false) value means a clean conversion.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Diagnostics {
    /// The macro-expansion budget (`MAX_EXPANSIONS`) was exhausted: the input
    /// contains unbounded recursion or package code texmark cannot evaluate, so
    /// the output is truncated and must not be trusted as complete.
    pub expansion_limit_exceeded: bool,
    /// Control sequences texmark did not recognize and dropped (deduplicated,
    /// without the leading `\`). Deliberate no-ops (spacing, layout) are not
    /// listed — only commands whose meaning was genuinely unknown, so a caller
    /// can see what was lost rather than trust a silently-degraded result.
    pub dropped_commands: std::collections::BTreeSet<String>,
    /// `\input`/`\include`/`\subfile` targets the resolver could not find, so
    /// their content is missing from the output. (Unresolved `\usepackage`/
    /// `\RequirePackage` are NOT listed — texmark bundles few packages and
    /// handles most natively, so a missing `.sty` is expected, not lost content.)
    pub unresolved_inputs: std::collections::BTreeSet<String>,
}

impl Diagnostics {
    /// Whether the conversion completed with no loss or degradation.
    pub fn is_clean(&self) -> bool {
        !self.expansion_limit_exceeded
            && self.dropped_commands.is_empty()
            && self.unresolved_inputs.is_empty()
    }

    /// Whether output is INCOMPLETE — content was lost, not merely rendered at
    /// lower fidelity. Drives the `texmark_truncated` flag. (Dropped commands are
    /// a fidelity loss but the surrounding text survives, so they don't count.)
    pub fn is_incomplete(&self) -> bool {
        self.expansion_limit_exceeded || !self.unresolved_inputs.is_empty()
    }
}

/// The engine's pending input: a stack of source frames, read most-recent-first.
/// A frame is either live source text (a [`Tokenizer`], which can also be read
/// raw for verbatim) or a token list produced by macro expansion or `unread`.
///
/// Sharing one stack is what keeps a single, correctly-ordered stream while
/// still retaining the raw text verbatim needs: an `\input` encountered
/// mid-stream pushes a text frame that is read to exhaustion before the
/// surrounding file resumes, and a macro expanding to `\input{f}rest`
/// interleaves `f` before `rest` for free.
enum Frame {
    /// A character source and its cursor.
    Text(Tokenizer),
    /// Tokens to read before anything below, stored reversed so `pop` yields
    /// them front-to-back.
    Tokens(Vec<Token>),
    /// Enter or leave a package source while traversing the stack.
    Package(bool),
    /// A read boundary: [`Input::next`] returns `None` here instead of reading
    /// the frames below, until [`Input::pop_barrier`] removes it. Lets a
    /// self-contained token list be expanded without a trailing macro's
    /// argument grab spilling into the surrounding document.
    Barrier,
}

struct Input {
    /// The last frame is the top of the stack: read first, popped when empty.
    stack: Vec<Frame>,
    /// Set by [`next`](Self::next): whether the token it just returned came from
    /// a live character source ([`Frame::Text`]) rather than spliced expansion
    /// output. The engine resets its expansion-depth counter on a source read, so
    /// a runaway macro *cycle* — which re-expands forever while consuming no
    /// source — is distinguished from a document that merely uses many macros.
    last_from_source: bool,
    package_depth: usize,
}

impl Input {
    fn from_source(source: &str) -> Self {
        Input {
            stack: vec![Frame::Text(Tokenizer::new(source))],
            last_from_source: false,
            package_depth: 0,
        }
    }

    fn next(&mut self, state: &State) -> Option<Token> {
        while let Some(frame) = self.stack.last_mut() {
            let from_source = matches!(frame, Frame::Text(_));
            let token = match frame {
                Frame::Text(tokenizer) => tokenizer.next_token(&state.catcodes),
                Frame::Tokens(tokens) => tokens.pop(),
                Frame::Package(entering) => {
                    if *entering {
                        self.package_depth += 1;
                    } else {
                        self.package_depth = self.package_depth.saturating_sub(1);
                    }
                    self.stack.pop();
                    continue;
                }
                Frame::Barrier => return None,
            };
            if token.is_some() {
                self.last_from_source = from_source;
                return token;
            }
            self.stack.pop();
        }
        None
    }

    /// Discard pending spliced-token frames at the top of the stack (down to the
    /// nearest live source [`Frame::Text`] or [`Barrier`](Frame::Barrier)). Used
    /// to contain a runaway macro cycle: the frames it spliced are abandoned so
    /// the rest of the document below reads cleanly, instead of the cycle's
    /// half-expanded garbage leaking into later output.
    fn drop_pending_tokens(&mut self) {
        while matches!(self.stack.last(), Some(Frame::Tokens(_))) {
            self.stack.pop();
        }
    }

    fn unread(&mut self, token: Token) {
        // Extend the top token frame if there is one, so a peek/unread cycle
        // doesn't litter the stack with single-token frames.
        match self.stack.last_mut() {
            Some(Frame::Tokens(tokens)) => tokens.push(token),
            _ => self.stack.push(Frame::Tokens(vec![token])),
        }
    }

    /// Splice `tokens` in to be read next, preserving their order.
    fn splice(&mut self, tokens: &[Token]) {
        if !tokens.is_empty() {
            self.stack
                .push(Frame::Tokens(tokens.iter().rev().cloned().collect()));
        }
    }

    /// Push a character source to be read next (as `\input` does), so its
    /// verbatim content stays readable as raw text and its catcode changes
    /// (`\makeatletter`, `\catcode`) take effect as it is read.
    fn push_source(&mut self, source: &str) {
        self.stack.push(Frame::Text(Tokenizer::new(source)));
    }

    fn mark_package(&mut self, entering: bool) {
        self.stack.push(Frame::Package(entering));
    }

    fn in_package(&self) -> bool {
        self.package_depth > 0
    }

    /// Seal off the current stack: [`next`](Self::next) returns `None` at this
    /// barrier instead of reading the frames below it, until [`pop_barrier`]
    /// removes it. Splice tokens *after* pushing the barrier to expand them in
    /// isolation.
    fn push_barrier(&mut self) {
        self.stack.push(Frame::Barrier);
    }

    /// Remove the topmost [`Barrier`](Frame::Barrier) and any residual frames
    /// above it, resuming reads of the frames below.
    fn pop_barrier(&mut self) {
        while let Some(frame) = self.stack.pop() {
            if matches!(frame, Frame::Barrier) {
                break;
            }
        }
    }

    /// Read raw source up to `marker` from the current character source — the
    /// normal case at `\begin{verbatim}`, including inside an `\input`ed file.
    /// Returns `None` only when the top frame is spliced tokens (e.g. verbatim
    /// produced by macro expansion), so the caller can fall back to token
    /// reconstruction.
    fn read_raw(&mut self, marker: &str) -> Option<String> {
        match self.stack.last_mut() {
            Some(Frame::Text(tokenizer)) => Some(tokenizer.read_until_literal(marker)),
            _ => None,
        }
    }

    /// Read one raw character from the live source, or `None` if the top frame
    /// is spliced tokens (e.g. content from macro expansion).
    fn read_raw_char(&mut self) -> Option<char> {
        match self.stack.last_mut() {
            Some(Frame::Text(tokenizer)) => tokenizer.read_raw_char(),
            _ => None,
        }
    }
}

/// Where the current build should stop.
#[derive(Debug, Clone, PartialEq)]
enum Stop {
    /// End of input (top level).
    Eof,
    /// A closing `}`.
    Group,
    /// A matching `\end{name}`.
    Environment(String),
}

/// The engine, borrowing the persistent [`State`] and a [`Resolver`] so nested
/// builds (arguments, environments, included files) share the same definitions.
pub struct Engine<'a> {
    state: &'a mut State,
    resolver: &'a dyn Resolver,
    input: Input,
    /// Whether the source has a `\begin{document}`; if so, everything before it
    /// is preamble and only its definitions and metadata are kept.
    has_document: bool,
    /// Document metadata collected from the preamble.
    title: Option<Vec<Node>>,
    authors: Vec<Vec<Node>>,
    date: Option<Vec<Node>>,
    footnote_number: usize,
    pending_footnote: Option<String>,
    footnote_texts: HashMap<String, Vec<Node>>,
    preamble_footnotes: Vec<Node>,
    /// Running count of macro expansions, to bound pathological recursion
    /// (e.g. a self-referential `\def\x{\x}`) so parsing always terminates.
    expansions: usize,
    /// Macro expansions performed since the last live *source* token was read
    /// (see [`MAX_EXPANSION_DEPTH`]). Reset to zero on every source read, so it
    /// measures how far the current expansion has run without making forward
    /// progress through the document — the signal of a cyclic definition, as
    /// opposed to a document that merely uses many macros.
    expansion_depth: usize,
    /// Degradation reported to the caller — texmark never fails silently.
    diagnostics: Diagnostics,
}

impl<'a> Engine<'a> {
    /// Create an engine over `source`.
    pub fn new(source: &str, state: &'a mut State, resolver: &'a dyn Resolver) -> Self {
        Engine {
            state,
            resolver,
            has_document: source.contains("\\begin{document}"),
            input: Input::from_source(source),
            title: None,
            authors: Vec::new(),
            date: None,
            footnote_number: 0,
            pending_footnote: None,
            footnote_texts: HashMap::new(),
            preamble_footnotes: Vec::new(),
            expansions: 0,
            expansion_depth: 0,
            diagnostics: Diagnostics::default(),
        }
    }

    // --- low-level token reading ------------------------------------------

    fn next_raw(&mut self) -> Option<Token> {
        let token = self.input.next(self.state);
        // A token read from live source is genuine forward progress through the
        // document, so the current expansion is not a stuck cycle: reset the
        // expansion-depth counter. (A cycle re-expands spliced tokens only and
        // never reaches source, so its depth climbs unchecked to the cap.)
        if token.is_some() && self.input.last_from_source {
            self.expansion_depth = 0;
        }
        token
    }

    /// Read one raw character from the live source (for `\verb` delimiters).
    fn read_raw_char(&mut self) -> Option<char> {
        self.input.read_raw_char()
    }

    fn unread(&mut self, token: Token) {
        self.input.unread(token);
    }

    /// Read the next token that is not a space.
    fn next_nonspace(&mut self) -> Option<Token> {
        loop {
            let t = self.next_raw()?;
            if !t.is_cat(CatCode::Space) {
                return Some(t);
            }
        }
    }

    /// Grab one macro-style argument: a `{...}` group (returned without the
    /// braces) or, failing that, a single token. Leading spaces are skipped.
    pub(crate) fn grab_argument(&mut self) -> Vec<Token> {
        match self.next_nonspace() {
            None => Vec::new(),
            Some(t) if t.is_cat(CatCode::BeginGroup) => self.read_until_group_end(),
            Some(t) => vec![t],
        }
    }

    /// Read tokens until the `}` that closes the group already opened, honoring
    /// nesting. The closing brace is consumed but not returned.
    fn read_until_group_end(&mut self) -> Vec<Token> {
        let mut depth = 1usize;
        let mut out = Vec::new();
        while let Some(t) = self.next_raw() {
            if t.is_cat(CatCode::BeginGroup) {
                depth += 1;
            } else if t.is_cat(CatCode::EndGroup) {
                depth -= 1;
                if depth == 0 {
                    break;
                }
            }
            out.push(t);
        }
        out
    }

    /// Grab a bracketed optional argument `[...]` if one is present next
    /// (skipping spaces). Returns the tokens inside the brackets.
    pub(crate) fn grab_optional(&mut self) -> Option<Vec<Token>> {
        let t = self.next_nonspace()?;
        if t == Token::Char('[', CatCode::Other) {
            Some(self.read_until_bracket_end())
        } else {
            self.unread(t);
            None
        }
    }

    /// Read tokens up to a top-level `]`, honoring brace nesting so that `]`
    /// inside `{...}` does not close the optional argument.
    fn read_until_bracket_end(&mut self) -> Vec<Token> {
        let mut depth = 0usize;
        let mut out = Vec::new();
        while let Some(t) = self.next_raw() {
            if t.is_cat(CatCode::BeginGroup) {
                depth += 1;
            } else if t.is_cat(CatCode::EndGroup) {
                depth = depth.saturating_sub(1);
            } else if depth == 0 && t == Token::Char(']', CatCode::Other) {
                break;
            }
            out.push(t);
        }
        out
    }

    /// Grab an argument and render it back to source-like text (used for names,
    /// keys, and URLs where the literal characters matter).
    pub(crate) fn grab_argument_text(&mut self) -> String {
        reconstruct(&self.grab_argument())
    }

    pub(crate) fn grab_expanded_argument_text(&mut self) -> String {
        let tokens = self.grab_argument();
        reconstruct(&self.expand_tokens(tokens))
    }

    pub(crate) fn splice_tokens(&mut self, tokens: &[Token]) {
        self.input.splice(tokens);
    }

    fn grab_file_name(&mut self) -> String {
        let Some(first) = self.next_nonspace() else {
            return String::new();
        };
        if first.is_cat(CatCode::BeginGroup) {
            return reconstruct(&self.read_until_group_end());
        }
        let mut tokens = vec![first];
        while let Some(token) = self.next_raw() {
            if token.is_cat(CatCode::Space) {
                break;
            }
            if matches!(token, Token::ControlSequence(_)) || token.is_cat(CatCode::EndGroup) {
                self.unread(token);
                break;
            }
            tokens.push(token);
        }
        reconstruct(&tokens)
    }
}

/// Render tokens back to approximate LaTeX source. Used for math bodies and for
/// arguments whose literal text matters (labels, citation keys, URLs).
pub(crate) fn reconstruct(tokens: &[Token]) -> String {
    let mut out = String::new();
    for t in tokens {
        match t {
            Token::ControlSequence(name) => {
                out.push('\\');
                out.push_str(name);
                // A control word (letters) is followed by a space in source; a
                // control symbol is not.
                let is_word = name.chars().count() > 1
                    || name.chars().next().is_some_and(|c| c.is_alphabetic());
                if is_word {
                    out.push(' ');
                }
            }
            Token::Char(c, _) => out.push(*c),
        }
    }
    out
}

/// Whether a control-word name opens a TeX `\if…\fi` conditional (for nesting
/// bookkeeping when skipping branches). This is `if`-prefixed — covering the
/// primitives and any `\newif`-defined `\if<name>` — but excludes look-alikes
/// that are NOT conditionals and have no `\fi`: `\iff` (the ⟺ relation) and
/// `\ifthenelse` (the LaTeX macro).
pub(crate) fn is_conditional_cs(name: &str) -> bool {
    name.starts_with("if") && !matches!(name, "iff" | "ifthenelse")
}

impl Engine<'_> {
    /// Whether `name` should expand as a user macro. A control sequence texmark
    /// handles natively (sectioning, font sizes) is *not* expandable even if a
    /// package `\def`s it — our semantic handling wins over layout plumbing we
    /// cannot execute (and which is often self-referential, e.g. `\@setsize`).
    fn is_user_macro(&self, name: &str) -> bool {
        self.state.macros.contains_key(name) && !is_protected(name)
    }

    /// Expand a user macro named `name`: grab its arguments and splice the
    /// substituted body back onto the input.
    fn expand_macro(&mut self, name: &str) {
        // Bound runaway recursion two ways. The global count is a hard ceiling on
        // total work (a document with genuinely enormous but non-cyclic expansion
        // still terminates). The depth catches a *cycle* — a definition that
        // re-expands forever without consuming source (e.g. `\def\x{\x}`, or a
        // `\let`/`\def` alias loop) — almost immediately, and crucially contains
        // it: only this construct's pending expansion is abandoned, so the rest
        // of the document keeps parsing rather than every later macro silently
        // vanishing once a single global budget trips.
        self.expansions += 1;
        self.expansion_depth += 1;
        if self.expansion_depth > MAX_EXPANSION_DEPTH {
            self.diagnostics.expansion_limit_exceeded = true;
            // Drop the frames this runaway spliced so its half-expanded garbage
            // does not leak into later output, and reset the depth so reads that
            // resume below (the rest of the document) start fresh.
            self.input.drop_pending_tokens();
            self.expansion_depth = 0;
            return;
        }
        if self.expansions > MAX_EXPANSIONS {
            self.diagnostics.expansion_limit_exceeded = true;
            return;
        }
        let mac = self.state.macros[name].clone();
        let mut args: Vec<Vec<Token>> = Vec::with_capacity(mac.params);
        if let Some(default) = &mac.optional_default {
            args.push(self.grab_optional().unwrap_or_else(|| default.clone()));
            for _ in 1..mac.params {
                args.push(self.grab_argument());
            }
        } else {
            for _ in 0..mac.params {
                args.push(self.grab_argument());
            }
        }
        let body = substitute(&mac.body, &args);
        self.input.splice(&body);
    }

    /// Run a `\@for\var:=<list>\do{<body>}` loop (LaTeX's `\@for`/`\@tfor`):
    /// bind `\var` to each list element and splice the body per iteration, so
    /// the generated content reaches the output. `per_token` iterates over
    /// individual tokens (`\@tfor`) rather than comma-separated items (`\@for`).
    fn run_for_loop(&mut self, per_token: bool) {
        let Some(Token::ControlSequence(var)) = self.next_nonspace() else {
            return;
        };
        // The `:=` separator.
        match self.next_nonspace() {
            Some(Token::Char(':', _)) => match self.next_raw() {
                Some(Token::Char('=', _)) => {}
                Some(t) => self.unread(t),
                None => {}
            },
            Some(t) => self.unread(t),
            None => return,
        }
        // The list runs up to `\do`; the body is its argument.
        let mut list = Vec::new();
        loop {
            match self.next_raw() {
                Some(Token::ControlSequence(n)) if n == "do" => break,
                Some(t) => list.push(t),
                None => return,
            }
        }
        let body = self.grab_argument();

        let items = if per_token {
            list.into_iter().map(|t| vec![t]).collect()
        } else {
            split_top_commas(&list)
        };
        let mut expanded = Vec::new();
        for item in items {
            expanded.extend(substitute_cs(&body, &var, &item));
        }
        self.input.splice(&expanded);
    }

    /// Run a `\forloop{counter}{start}{condition}{body}`: set the counter, then
    /// while the condition holds, render the body (which reads the counter's
    /// current value) and step the counter by one. Iterations are bounded so a
    /// malformed condition terminates.
    fn run_counter_loop(
        &mut self,
        counter: &str,
        start: &[Token],
        cond: &[Token],
        body: &[Token],
    ) -> Vec<Node> {
        let start_val = self.eval_number(start);
        self.state.counters.insert(counter.to_string(), start_val);
        let mut out = Vec::new();
        let mut iterations = 0;
        while self.eval_condition(cond) {
            iterations += 1;
            if iterations > MAX_EXPANSIONS {
                self.diagnostics.expansion_limit_exceeded = true;
                break;
            }
            out.extend(self.render_inline(body.to_vec()));
            *self.state.counters.entry(counter.to_string()).or_insert(0) += 1;
        }
        out
    }

    /// Evaluate a `\forloop` condition of the form `<num> <rel> <num>` where a
    /// number is a literal or `\value{counter}` and `<rel>` is `<`, `=`, or `>`.
    fn eval_condition(&self, tokens: &[Token]) -> bool {
        let mut depth = 0i32;
        for (i, t) in tokens.iter().enumerate() {
            match t {
                _ if t.is_cat(CatCode::BeginGroup) => depth += 1,
                _ if t.is_cat(CatCode::EndGroup) => depth -= 1,
                Token::Char(c, _) if depth == 0 && matches!(c, '<' | '=' | '>') => {
                    let lhs = self.eval_number(&tokens[..i]);
                    let rhs = self.eval_number(&tokens[i + 1..]);
                    return match c {
                        '<' => lhs < rhs,
                        '>' => lhs > rhs,
                        _ => lhs == rhs,
                    };
                }
                _ => {}
            }
        }
        false
    }

    /// Handle `\def`/`\gdef` (`expand_body` false) and `\edef`/`\xdef`
    /// (`expand_body` true). For the `e`/`x` variants TeX expands the body at
    /// definition time; texmark approximates that by fully expanding user
    /// macros in the body before storing it. This makes the common
    /// self-referential option-processing idiom safe: after `\let\x\@empty`,
    /// `\edef\x{...\x...\@ptionlist...}` expands the already-defined `\x` to its
    /// current (empty) value rather than freezing a literal `\x` into its own
    /// body — which would otherwise recurse forever when `\x` is later used.
    /// (A self-referential `\edef` to a name that was never defined is a TeX
    /// error; here its `\x` stays literal and is bounded only by the expansion
    /// budget, so it terminates and reports rather than hanging.)
    fn define_with_def(&mut self, expand_body: bool) {
        let Some(Token::ControlSequence(name)) = self.next_raw() else {
            return;
        };
        // Parameter text: everything up to the opening brace. Count parameters.
        let mut params = 0usize;
        loop {
            match self.next_raw() {
                Some(t) if t.is_cat(CatCode::BeginGroup) => break,
                Some(Token::Char(_, CatCode::Param)) => {
                    // Consume the following digit.
                    if let Some(d) = self.next_raw()
                        && let Some(n) = d.char().and_then(|c| c.to_digit(10))
                    {
                        params = params.max(n as usize);
                    }
                }
                Some(_) => {} // literal delimiter tokens are ignored (unsupported)
                None => return,
            }
        }
        let body = self.read_until_group_end();
        let body = if expand_body {
            self.expand_tokens(body)
        } else {
            body
        };
        self.state.macros.insert(
            name,
            Macro {
                params,
                optional_default: None,
                body,
            },
        );
    }

    /// Handle `\newcommand`/`\renewcommand`/`\providecommand`, including the
    /// optional `*` and `[n][default]` forms.
    fn define_with_newcommand(&mut self, only_if_missing: bool) {
        self.skip_star();
        let name = match self.next_nonspace() {
            Some(t) if t.is_cat(CatCode::BeginGroup) => {
                // `{\name}` form.
                match self.read_until_group_end().into_iter().next() {
                    Some(Token::ControlSequence(n)) => n,
                    _ => return,
                }
            }
            Some(Token::ControlSequence(n)) => n,
            _ => return,
        };
        let (params, optional_default) = self.read_command_signature();
        let body = self.grab_argument();
        if only_if_missing
            && (self.state.macros.contains_key(&name) || packages::for_command(&name).is_some())
        {
            return;
        }
        self.state.macros.insert(
            name,
            Macro {
                params,
                optional_default,
                body,
            },
        );
    }

    /// Handle `\newenvironment{name}[n][default]{begin}{end}`.
    fn define_with_newenvironment(&mut self) {
        self.skip_star();
        let name = self.grab_argument_text();
        let (params, optional_default) = self.read_command_signature();
        let begin = self.grab_argument();
        let end = self.grab_argument();
        self.state.environments.insert(
            name,
            Environment {
                params,
                optional_default,
                begin,
                end,
            },
        );
    }

    /// Read the `[n]` parameter count and optional `[default]` that follow a
    /// command or environment name.
    fn read_command_signature(&mut self) -> (usize, Option<Vec<Token>>) {
        let params = self
            .grab_optional()
            .and_then(|toks| reconstruct(&toks).trim().parse::<usize>().ok())
            .unwrap_or(0);
        let optional_default = self.grab_optional();
        (params, optional_default)
    }

    /// Consume a `*` immediately following a command name, if present.
    fn skip_star(&mut self) {
        if let Some(t) = self.next_nonspace()
            && t != Token::Char('*', CatCode::Other)
        {
            self.unread(t);
        }
    }

    /// Like [`skip_star`](Self::skip_star), but reports whether a `*` was
    /// consumed — used where the star changes semantics (e.g. `\newtheorem*`,
    /// or a package's `\captionof*`).
    pub(crate) fn consume_star(&mut self) -> bool {
        match self.next_nonspace() {
            Some(Token::Char('*', CatCode::Other)) => true,
            Some(t) => {
                self.unread(t);
                false
            }
            None => false,
        }
    }

    /// Read the base of a text accent: a braced group (`\'{e}`, `\c{c}`), a
    /// single following character (`\'e`), or a dotless-i/j control word
    /// (`\'\i`). Returns the base as a string (empty if the accent stands alone).
    fn grab_accent_base(&mut self) -> String {
        match self.next_nonspace() {
            Some(t) if t.is_cat(CatCode::BeginGroup) => {
                self.unread(t);
                reconstruct(&self.grab_argument())
            }
            Some(Token::ControlSequence(n)) => match n.as_str() {
                "i" => "i".to_string(),
                "j" => "j".to_string(),
                _ => String::new(),
            },
            Some(Token::Char(c, _)) => c.to_string(),
            None => String::new(),
        }
    }
}

/// Substitute macro arguments into a body: `#1`..`#9` become the corresponding
/// argument; `##` becomes one parameter token for a nested definition.
fn substitute(body: &[Token], args: &[Vec<Token>]) -> Vec<Token> {
    let mut out = Vec::with_capacity(body.len());
    let mut i = 0;
    while i < body.len() {
        match &body[i] {
            Token::Char(_, CatCode::Param) if i + 1 < body.len() => match &body[i + 1] {
                Token::Char('#', _) | Token::Char(_, CatCode::Param) => {
                    out.push(Token::Char('#', CatCode::Param));
                    i += 2;
                }
                Token::Char(c, _) if c.is_ascii_digit() => {
                    let n = c.to_digit(10).unwrap() as usize;
                    if n >= 1
                        && let Some(arg) = args.get(n - 1)
                    {
                        out.extend(arg.iter().cloned());
                    }
                    i += 2;
                }
                _ => {
                    out.push(body[i].clone());
                    i += 1;
                }
            },
            other => {
                out.push(other.clone());
                i += 1;
            }
        }
    }
    out
}

/// Replace each occurrence of the control sequence named `var` in `body` with
/// the `value` tokens — used to bind a `\@for` loop variable per iteration.
fn substitute_cs(body: &[Token], var: &str, value: &[Token]) -> Vec<Token> {
    let mut out = Vec::with_capacity(body.len());
    for t in body {
        match t {
            Token::ControlSequence(n) if n == var => out.extend(value.iter().cloned()),
            _ => out.push(t.clone()),
        }
    }
    out
}

/// Split a token list on top-level commas (ignoring commas inside `{...}`),
/// trimming whitespace tokens around each element and dropping empty ones.
fn split_top_commas(tokens: &[Token]) -> Vec<Vec<Token>> {
    let mut items = Vec::new();
    let mut current = Vec::new();
    let mut depth = 0i32;
    for t in tokens {
        match t {
            _ if t.is_cat(CatCode::BeginGroup) => {
                depth += 1;
                current.push(t.clone());
            }
            _ if t.is_cat(CatCode::EndGroup) => {
                depth -= 1;
                current.push(t.clone());
            }
            Token::Char(',', _) if depth == 0 => {
                items.push(std::mem::take(&mut current));
            }
            _ => current.push(t.clone()),
        }
    }
    items.push(current);
    items
        .into_iter()
        .map(trim_space_tokens)
        .filter(|item| !item.is_empty())
        .collect()
}

/// Drop leading/trailing whitespace character tokens from a token list.
fn trim_space_tokens(mut tokens: Vec<Token>) -> Vec<Token> {
    while tokens
        .first()
        .is_some_and(|t| matches!(t, Token::Char(c, _) if c.is_whitespace()))
    {
        tokens.remove(0);
    }
    while tokens
        .last()
        .is_some_and(|t| matches!(t, Token::Char(c, _) if c.is_whitespace()))
    {
        tokens.pop();
    }
    tokens
}

/// One step of output from the token stream.
pub(crate) enum Event {
    /// Nothing more to read for the current [`Stop`].
    End,
    /// A paragraph break (`\par` / blank line).
    Par,
    /// Inline content (text, emphasis, math, ...).
    Inline(Vec<Node>),
    /// Block content (paragraphs' peers: sections, lists, figures, ...).
    Blocks(Vec<Node>),
    /// A font *declaration* (`\bf`, `\itshape`, …): wrap the rest of the current
    /// scope in an element with this tag, the way the declaration applies until
    /// the end of its group.
    Wrap(&'static str),
}

impl Engine<'_> {
    // --- the build loops ---------------------------------------------------

    /// Build inline content until `stop`, merging adjacent text.
    fn build_inline(&mut self, stop: Stop) -> Vec<Node> {
        let mut out = Vec::new();
        loop {
            match self.next_event(&stop) {
                Event::End => break,
                Event::Par => out.push(Node::text(" ")),
                Event::Inline(ns) | Event::Blocks(ns) => out.extend(ns),
                Event::Wrap(tag) => {
                    // A declaration wraps the remainder of this scope; the inner
                    // build consumes the scope terminator, so we stop after.
                    let rest = self.build_inline(stop.clone());
                    out.push(Node::Element(wrap_el(tag, rest)));
                    break;
                }
            }
        }
        merge_text(out)
    }

    /// Build block content until `stop`, gathering inline runs into `<p>`
    /// paragraphs and passing block pieces through.
    fn build_block(&mut self, stop: Stop) -> Vec<Node> {
        let mut blocks = Vec::new();
        let mut paragraph: Vec<Node> = Vec::new();
        loop {
            match self.next_event(&stop) {
                Event::End => break,
                Event::Par => flush_paragraph(&mut paragraph, &mut blocks),
                Event::Inline(ns) => paragraph.extend(ns),
                Event::Blocks(ns) => {
                    flush_paragraph(&mut paragraph, &mut blocks);
                    blocks.extend(ns);
                }
                Event::Wrap(tag) => {
                    // Bare declaration at block level: wrap the rest of the
                    // scope, keeping block structure intact.
                    flush_paragraph(&mut paragraph, &mut blocks);
                    let rest = self.build_block(stop.clone());
                    blocks.push(Node::Element(wrap_el(tag, rest)));
                    break;
                }
            }
        }
        flush_paragraph(&mut paragraph, &mut blocks);
        blocks
    }

    /// Render a token list as inline content, sharing the current state. Used
    /// for command arguments such as `\textbf{...}`.
    pub(crate) fn render_inline(&mut self, tokens: Vec<Token>) -> Vec<Node> {
        // Splice the tokens followed by a closing brace, then build to it.
        self.input.splice(&[Token::Char('}', CatCode::EndGroup)]);
        self.input.splice(&tokens);
        self.build_inline(Stop::Group)
    }

    /// Read and classify the next token into an [`Event`].
    fn next_event(&mut self, stop: &Stop) -> Event {
        loop {
            let Some(t) = self.next_raw() else {
                return Event::End;
            };
            match t {
                Token::Char(_, CatCode::EndGroup) => {
                    if *stop == Stop::Group {
                        return Event::End;
                    }
                    // Stray close brace: ignore.
                }
                Token::Char(_, CatCode::BeginGroup) => {
                    return Event::Inline(self.build_inline(Stop::Group));
                }
                Token::Char(_, CatCode::MathShift) => return self.read_math(),
                Token::Char(_, CatCode::Space) => return Event::Inline(vec![Node::text(" ")]),
                Token::Char(_, CatCode::Param)
                | Token::Char(_, CatCode::AlignTab)
                | Token::Char(_, CatCode::Superscript)
                | Token::Char(_, CatCode::Subscript) => {
                    // Only meaningful in math or tables, handled there; ignore
                    // when they appear in ordinary text.
                }
                Token::Char(c, _) => return Event::Inline(vec![Node::text(c.to_string())]),
                Token::ControlSequence(name) => {
                    if self.is_user_macro(&name) {
                        self.expand_macro(&name);
                        continue;
                    }
                    return self.run_command(&name, stop);
                }
            }
        }
    }
}

/// Merge consecutive text nodes into single runs.
fn merge_text(nodes: Vec<Node>) -> Vec<Node> {
    let mut out: Vec<Node> = Vec::with_capacity(nodes.len());
    for node in nodes {
        match (out.last_mut(), node) {
            (Some(Node::Text(prev)), Node::Text(next)) => prev.push_str(&next),
            (_, node) => out.push(node),
        }
    }
    out
}

/// Move accumulated inline content into a `<p>` block, unless it is only
/// whitespace. Leading and trailing whitespace is trimmed.
fn flush_paragraph(paragraph: &mut Vec<Node>, blocks: &mut Vec<Node>) {
    let merged = merge_text(std::mem::take(paragraph));
    let trimmed = trim_edges(merged);
    if trimmed.is_empty() {
        return;
    }

    // If the paragraph content contains block-level nodes (e.g. a tabular inside
    // a {group}), hoist them out rather than wrapping everything in <p>.
    let has_blocks = trimmed
        .iter()
        .any(|n| matches!(n, Node::Element(e) if is_block_element(e)));
    if !has_blocks {
        let mut p = Element::new("p");
        p.children = trimmed;
        blocks.push(Node::Element(p));
        return;
    }

    let mut inline_acc: Vec<Node> = Vec::new();
    for node in trimmed {
        let is_block = matches!(&node, Node::Element(e) if is_block_element(e));
        if is_block {
            let acc = trim_edges(merge_text(std::mem::take(&mut inline_acc)));
            if !acc.is_empty() {
                let mut p = Element::new("p");
                p.children = acc;
                blocks.push(Node::Element(p));
            }
            blocks.push(node);
        } else {
            inline_acc.push(node);
        }
    }
    let acc = trim_edges(merge_text(inline_acc));
    if !acc.is_empty() {
        let mut p = Element::new("p");
        p.children = acc;
        blocks.push(Node::Element(p));
    }
}

/// Control sequences texmark handles natively and a document/package may not
/// override. Their semantics (document structure, font sizing) are fixed by
/// texmark; class files routinely `\def` them into TeX-level layout code that
/// texmark cannot evaluate (and which is often self-referential).
fn is_protected(name: &str) -> bool {
    matches!(
        name,
        // Sectioning — extracted as <section> markers; a class's \@startsection
        // redefinition would destroy the document structure.
        "part"
            | "chapter"
            | "section"
            | "subsection"
            | "subsubsection"
            | "paragraph"
            | "subparagraph"
            // Font sizes — no-ops for Markdown; class files redefine them via
            // \@setsize (unrunnable here), producing self-referential loops.
            | "tiny"
            | "scriptsize"
            | "footnotesize"
            | "small"
            | "normalsize"
            | "large"
            | "Large"
            | "LARGE"
            | "huge"
            | "Huge"
    )
}

/// Element names that are block-level and must not be wrapped in `<p>`.
fn is_block_node(name: &str) -> bool {
    matches!(
        name,
        "tabular"
            | "verbatim"
            | "float"
            | "align"
            | "itemize"
            | "enumerate"
            | "description"
            | "blockquote"
            | "bibliography"
            | "pending-bibliography"
            | "environment"
            | "section"
            | "appendix"
            | "abstract"
    )
}

fn is_block_element(element: &Element) -> bool {
    is_block_node(&element.name)
        || (element.name == "math"
            && element
                .attributes
                .iter()
                .any(|(key, value)| key == "mode" && value == "display"))
}

/// Drop leading/trailing whitespace-only text and trim the outer text nodes.
fn trim_edges(mut nodes: Vec<Node>) -> Vec<Node> {
    while let Some(Node::Text(t)) = nodes.first() {
        if t.trim().is_empty() {
            nodes.remove(0);
        } else {
            break;
        }
    }
    while let Some(Node::Text(t)) = nodes.last() {
        if t.trim().is_empty() {
            nodes.pop();
        } else {
            break;
        }
    }
    if let Some(Node::Text(t)) = nodes.first_mut() {
        *t = t.trim_start().to_string();
    }
    if let Some(Node::Text(t)) = nodes.last_mut() {
        *t = t.trim_end().to_string();
    }
    nodes
}

impl Engine<'_> {
    // --- command dispatch --------------------------------------------------

    /// Follow a `\let` alias chain to its final non-alias target, bounded so a
    /// self- or mutually-referential alias (`\let\a=\b`\`\let\b=\a`, or
    /// `\let\zz=\zz`) terminates with `None` instead of looping. Chains are short
    /// in practice; the visited set makes any cycle a definite stop rather than
    /// relying on a hop count alone.
    fn resolve_let_alias(&self, name: &str) -> Option<String> {
        let mut current = name;
        let mut seen = std::collections::HashSet::new();
        while let Some(next) = self.state.let_aliases.get(current) {
            if !seen.insert(current.to_string()) {
                return None; // cycle
            }
            current = next;
        }
        (current != name).then(|| current.to_string())
    }

    /// Execute a control sequence that is not a user macro.
    fn run_command(&mut self, name: &str, stop: &Stop) -> Event {
        // Resolve a `\let` alias to its non-macro target (see the `"let"`
        // handler): `\foo` behaves exactly as its target's meaning. Chains are
        // followed iteratively with a bound, so a cyclic alias (`\let\a=\b`\
        // `\let\b=\a`) terminates instead of recursing natively into a stack
        // overflow. The target is never a user macro, so no expansion cycle can
        // form once we dispatch it.
        if self.state.let_aliases.contains_key(name) {
            match self.resolve_let_alias(name) {
                Some(target) => return self.run_command(&target, stop),
                // A cyclic alias chase: report the runaway and emit nothing,
                // matching how the engine bounds other non-terminating input.
                None => {
                    self.diagnostics.expansion_limit_exceeded = true;
                    return Event::Inline(vec![]);
                }
            }
        }
        // Dispatch to native packages before built-in arms.
        if let Some(pkg) = packages::for_command(name) {
            return pkg.command(name, self);
        }

        match name {
            "par" => Event::Par,
            // Line break.
            "\\" | "newline" | "cr" | "tabularnewline" => {
                Event::Inline(vec![Node::Element(Element::new("linebreak"))])
            }

            // Definitions.
            "def" | "gdef" => {
                self.define_with_def(false);
                Event::Inline(vec![])
            }
            "edef" | "xdef" => {
                self.define_with_def(true);
                Event::Inline(vec![])
            }
            // `\DeclareRobustCommand`/`\CheckCommand` share `\newcommand`'s
            // `[*]{\name}[n][default]{body}` syntax; the "robust" distinction is a
            // runtime-protection concern texmark does not model, so treat them as
            // plain definitions. Unit macros are frequently declared this way
            // (`\DeclareRobustCommand{\GeV}{\ifmmode…\fi}`), and without this the
            // control sequence stays undefined and leaks raw into math.
            "newcommand" | "renewcommand" | "DeclareRobustCommand" | "CheckCommand" => {
                self.define_with_newcommand(false);
                Event::Inline(vec![])
            }
            "providecommand" => {
                self.define_with_newcommand(true);
                Event::Inline(vec![])
            }
            "newenvironment" | "renewenvironment" => {
                self.define_with_newenvironment();
                Event::Inline(vec![])
            }
            "newcolumntype" => {
                // `array`-package custom column type:
                // `\newcolumntype{<char>}[<nargs>]{<spec>}`. texmark drops the
                // column specification when building tables (GFM has no column
                // formatting), so a custom type has no rendering effect — but we
                // must still consume its full definition here. Otherwise the
                // `{char}[n]{spec}` arguments fall through to normal text and
                // leak into the output (e.g. `x[1]>p1pt`), and any tabular using
                // the type would also see raw spec fragments in a data cell.
                let _ = self.grab_argument(); // column-type character
                let _ = self.grab_optional(); // [nargs]
                let _ = self.grab_argument(); // replacement column spec
                Event::Inline(vec![])
            }
            "newtheorem" => {
                // \newtheorem{name}{Heading}, \newtheorem{name}[shared]{Heading}
                // (shares another counter), \newtheorem{name}{Heading}[within]
                // (numbers within a section/chapter, resetting each time it
                // steps), or \newtheorem*{name}{Heading} (unnumbered). Capture
                // the definition so the resolve pass can number and label it.
                let starred = self.consume_star();
                let name = self.grab_argument_text();
                let shared = self.grab_optional().map(|t| reconstruct(&t).trim().to_string());
                let printed = self.grab_argument_text().trim().to_string();
                let within = self
                    .grab_optional()
                    .map(|t| reconstruct(&t).trim().to_string())
                    .filter(|s| !s.is_empty());
                let counter = if starred {
                    None
                } else {
                    Some(shared.unwrap_or_else(|| name.clone()))
                };
                self.state.theorems.insert(
                    name,
                    crate::state::TheoremDef {
                        printed,
                        counter,
                        within,
                    },
                );
                Event::Inline(vec![])
            }
            "DeclareMathOperator" => {
                self.skip_star();
                // \DeclareMathOperator{\name}{text} → \name expands to
                // \operatorname{text}, which math renderers understand.
                let name = match self.grab_argument().into_iter().next() {
                    Some(Token::ControlSequence(n)) => n,
                    _ => return Event::Inline(vec![]),
                };
                let text = self.grab_argument();
                let mut body = vec![
                    Token::cs("operatorname"),
                    Token::Char('{', CatCode::BeginGroup),
                ];
                body.extend(text);
                body.push(Token::Char('}', CatCode::EndGroup));
                self.state.macros.insert(
                    name,
                    Macro {
                        params: 0,
                        optional_default: None,
                        body,
                    },
                );
                Event::Inline(vec![])
            }

            // Diagnostics that write to the log/terminal: drop the message.
            "message" | "typeout" | "wlog" | "typein" => {
                let _ = self.grab_argument();
                Event::Inline(vec![])
            }
            "PackageWarning" | "PackageError" | "PackageInfo" | "ClassWarning" | "ClassError"
            | "GenericWarning" | "GenericError" => {
                let _ = self.grab_argument();
                let _ = self.grab_argument();
                let _ = self.grab_optional();
                Event::Inline(vec![])
            }

            // Preamble noise: consume arguments, emit nothing.
            "documentclass" => {
                let _ = self.grab_optional();
                let name = self.grab_argument_text();
                // Select the class's numbering scheme (article vs book/report);
                // unknown classes keep the article default.
                if let Some(class) = crate::classes::for_class(name.trim()) {
                    self.state.numbering = class.numbering();
                }
                Event::Inline(vec![])
            }
            // Native packages fully replace their `.sty` implementation. For an
            // unknown package, load a bundled file so its document-level macro
            // definitions remain available.
            "usepackage" | "RequirePackage" => {
                let options = self
                    .grab_optional()
                    .map(|t| reconstruct(&t))
                    .unwrap_or_default();
                let name = self.grab_argument_text();
                for pkg in name.split(',') {
                    let pkg = pkg.trim();
                    if pkg.is_empty() {
                        continue;
                    }
                    if let Some(handler) = packages::for_package(pkg) {
                        handler.configure(&options, self.state);
                    } else {
                        self.include_file(&format!("{pkg}.sty"), false);
                    }
                }
                Event::Inline(vec![])
            }
            "PassOptionsToPackage" | "PassOptionsToClass" => {
                let options = self.grab_argument_text();
                let target = self.grab_argument_text();
                if let Some(handler) = packages::for_package(target.trim()) {
                    handler.configure(&options, self.state);
                }
                Event::Inline(vec![])
            }
            // .sty file primitives: no-ops so bundled style files can be loaded
            // without crashing. \newcommand/\def etc. still work normally.
            "makeatletter" => {
                self.state.catcodes.set('@', CatCode::Letter);
                Event::Inline(vec![])
            }
            "makeatother" => {
                self.state.catcodes.set('@', CatCode::Other);
                Event::Inline(vec![])
            }
            "DeclareOption" => {
                self.skip_star();
                let _ = self.grab_argument(); // option name
                let _ = self.grab_argument(); // code
                Event::Inline(vec![])
            }
            "ProcessOptions" => {
                self.skip_star();
                Event::Inline(vec![])
            }
            "ExecuteOptions" => {
                let _ = self.grab_argument();
                Event::Inline(vec![])
            }
            "ProvidesPackage" | "ProvidesClass" | "ProvidesFile" | "NeedsTeXFormat" => {
                let _ = self.grab_argument();
                let _ = self.grab_optional();
                Event::Inline(vec![])
            }

            // Layout/length assignments: two-argument no-ops.
            "setlength" | "addtolength" | "settowidth" | "settodepth" | "settoheight"
            | "setstretch" | "fontsize" => {
                let _ = self.grab_argument();
                let _ = self.grab_argument();
                Event::Inline(vec![])
            }
            // Counters.
            "newcounter" => {
                let name = self.grab_argument_text();
                let _ = self.grab_optional(); // [within]: reset relationship, ignored
                self.state.counters.insert(name, 0);
                Event::Inline(vec![])
            }
            "setcounter" => {
                let name = self.grab_argument_text();
                let value = self.grab_argument();
                let n = self.eval_number(&value);
                self.state.counters.insert(name.trim().to_string(), n);
                Event::Inline(vec![])
            }
            "addtocounter" => {
                let name = self.grab_argument_text();
                let value = self.grab_argument();
                let n = self.eval_number(&value);
                *self.state.counters.entry(name.trim().to_string()).or_insert(0) += n;
                Event::Inline(vec![])
            }
            "stepcounter" | "refstepcounter" => {
                let name = self.grab_argument_text();
                *self.state.counters.entry(name.trim().to_string()).or_insert(0) += 1;
                Event::Inline(vec![])
            }
            "value" | "arabic" | "roman" | "Roman" | "alph" | "Alph" | "fnsymbol" => {
                let style = name;
                let counter = self.grab_argument_text();
                let n = self.state.counters.get(counter.trim()).copied().unwrap_or(0);
                Event::Inline(vec![Node::text(format_counter(style, n))])
            }
            // No-argument layout switches.
            "sloppy" | "fussy" | "frenchspacing" | "nonfrenchspacing" | "begingroup"
            | "endgroup" => Event::Inline(vec![]),
            // LaTeX layout commands taking a braced argument.
            "usecounter" | "pagenumbering" | "pagestyle"
            | "thispagestyle" | "hyphenation" | "captionsetup"
            // Index/glossary entries produce no visible text; their argument must
            // be consumed (else `Word\index{Word}` renders as "WordWord").
            | "index" | "glossary"
            // Register/length/style declarations that produce no content.
            | "newlength" | "newdimen" | "newcount" | "newskip" | "newmuskip"
            | "urlstyle" => {
                let _ = self.grab_optional();
                let _ = self.grab_argument();
                Event::Inline(vec![])
            }
            // TeX length/glue registers set by bare assignment (`\parindent 0pt`,
            // `\baselineskip=18pt`): consume the dimension so it doesn't leak.
            "columnsep" | "columnwidth" | "linewidth" | "textwidth" | "textheight"
            | "topmargin" | "oddsidemargin" | "evensidemargin" | "headheight" | "headsep"
            | "footskip" | "marginparsep" | "marginparwidth" | "parskip" | "parindent"
            | "baselineskip" | "lineskip" | "topskip" | "floatsep" | "textfloatsep"
            | "intextsep" | "dblfloatsep" | "dbltextfloatsep" | "abovecaptionskip"
            | "belowcaptionskip" | "arraycolsep" | "tabcolsep" | "arrayrulewidth"
            | "doublerulesep" | "itemsep" | "parsep" | "topsep" | "partopsep"
            | "listparindent" | "labelwidth" | "labelsep" | "leftmargin" | "rightmargin"
            | "itemindent" | "vskip" | "hskip" | "kern" | "abovedisplayskip"
            | "belowdisplayskip" | "parfillskip" => {
                self.skip_dimen();
                Event::Inline(vec![])
            }
            // Box registers and rules produce no Markdown; consume their operands.
            // (`\global`/`\long` prefixes are dropped as unknown, so `\global\setbox`
            // reaches this arm as `\setbox`.)
            "setbox" => {
                let _ = self.next_nonspace(); // the box register
                self.skip_optional_equals();
                self.skip_box();
                Event::Inline(vec![])
            }
            "hrule" | "vrule" => {
                self.skip_rule_spec();
                Event::Inline(vec![])
            }
            // TeX conditionals texmark cannot evaluate (numeric/mode tests): drop
            // the whole construct — the condition tokens (`>`, `0`, dimensions)
            // are not content, and the guarded branches are layout plumbing.
            "ifdim" | "ifnum" | "ifodd" | "ifvmode" | "ifhmode" | "ifinner"
            | "ifvoid" | "ifhbox" | "ifvbox" | "ifeof" | "ifcase" | "ifdefined" => {
                self.skip_to_fi();
                Event::Inline(vec![])
            }
            // In text-mode dispatch `\ifmmode` is always FALSE, so take the else
            // branch: skip the true branch through `\else` (its content then
            // processes normally); the trailing `\fi` is absorbed by the inert
            // `\fi` arm below. Common in unit macros like
            // `\def\GeV{\ifmmode ... \else \textrm{GeV}\fi}`.
            "ifmmode" => {
                self.skip_to_else_or_fi();
                Event::Inline(vec![])
            }
            "ifx" => {
                let equal = match (self.next_raw(), self.next_raw()) {
                    (Some(left), Some(right)) => self.ifx_equal(&left, &right),
                    _ => false,
                };
                if !equal {
                    self.skip_to_else_or_fi();
                }
                Event::Inline(vec![])
            }
            // `\ifcsname name\endcsname`: evaluate whether the (computed) control
            // sequence is defined, and take the corresponding branch.
            "ifcsname" => {
                let name = self.build_csname();
                if !self.is_defined(&name) {
                    self.skip_to_else_or_fi();
                }
                Event::Inline(vec![])
            }
            // \twocolumn[spanning material]: drop the column switch, render the
            // bracketed content (title block, etc.) as blocks.
            "twocolumn" | "onecolumn" => match self.grab_optional() {
                Some(toks) => Event::Blocks(self.render_block(toks)),
                None => Event::Inline(vec![]),
            },
            // Box constructors used standalone: render their content.
            "vbox" | "vtop" | "vcenter" => {
                let toks = self.grab_argument();
                Event::Inline(self.render_inline(toks))
            }
            // `\csname name\endcsname` builds a control sequence from the enclosed
            // tokens (expanding `\the<counter>` etc.) and re-dispatches it, so a
            // defined name expands and an undefined one vanishes.
            "csname" => {
                let name = self.build_csname();
                self.unread(Token::cs(name));
                Event::Inline(vec![])
            }
            "endcsname" => Event::Inline(vec![]),
            // `\expandafter\A\B…`: expand `\B` once before processing `\A`. Only
            // the `\B = \csname…\endcsname` case (computed macro names) is
            // meaningfully expanded; other tokens are restored in order.
            "expandafter" => {
                let first = self.next_raw();
                match self.next_raw() {
                    Some(Token::ControlSequence(n)) if n == "csname" => {
                        let built = self.build_csname();
                        let mut toks = Vec::new();
                        toks.extend(first);
                        toks.push(Token::cs(built));
                        self.input.splice(&toks);
                    }
                    second => {
                        let mut toks = Vec::new();
                        toks.extend(first);
                        toks.extend(second);
                        self.input.splice(&toks);
                    }
                }
                Event::Inline(vec![])
            }
            // `\@for\var:=list\do{body}` / `\@tfor` — content-generating loops.
            "@for" => {
                self.run_for_loop(false);
                Event::Inline(vec![])
            }
            "@tfor" => {
                self.run_for_loop(true);
                Event::Inline(vec![])
            }
            // `\forloop[step]{counter}{start}{condition}{body}` (forloop package):
            // a counter loop whose body generates content each iteration.
            "forloop" => {
                let _ = self.grab_optional(); // [step], default 1
                let counter = self.grab_argument_text();
                let start = self.grab_argument();
                let cond = self.grab_argument();
                let body = self.grab_argument();
                let nodes = self.run_counter_loop(counter.trim(), &start, &cond, &body);
                Event::Inline(nodes)
            }
            // TeX penalty/glue primitives carry a numeric argument (e.g. the
            // `\penalty0` breakpoints BibTeX sprinkles through `.bbl` entries);
            // consume the number so its digits don't leak into the text.
            "penalty" => {
                self.skip_number();
                Event::Inline(vec![])
            }
            // \resizebox{width}{height}{content} → render the content.
            "resizebox" => {
                let _ = self.grab_argument(); // width
                let _ = self.grab_argument(); // height
                let toks = self.grab_argument();
                Event::Inline(self.render_inline(toks))
            }
            // \scalebox{ratio}[vratio]{content} → render the content.
            "scalebox" => {
                let _ = self.grab_argument(); // scale
                let _ = self.grab_optional(); // optional vertical scale
                let toks = self.grab_argument();
                Event::Inline(self.render_inline(toks))
            }
            // \texorpdfstring{TeX}{PDF-bookmark text}: hyperref uses the first
            // (rich) form for the document and the second (ASCII) only for PDF
            // bookmarks. Render the first; drop the second (else both leak, e.g.
            // a section title comes out `$H\to\gamma$H to gamma`).
            "texorpdfstring" => {
                let toks = self.grab_argument(); // TeX form
                let _ = self.grab_argument(); // PDF-bookmark form
                Event::Inline(self.render_inline(toks))
            }
            // Wrappers whose leading braced arg is a *presentation* value (a
            // colour, a length, a link target), not content: drop it and render
            // only the content arg. Without this the value leaks as text
            // (e.g. `\textcolor{blue}{hi}` → "bluehi"). Leading `[..]` (colour
            // model, box position) is skipped.
            "textcolor" | "colorbox" | "parbox" | "hyperlink" | "hypertarget" => {
                while self.grab_optional().is_some() {}
                let _ = self.grab_argument(); // colour / width / target
                let toks = self.grab_argument(); // content
                Event::Inline(self.render_inline(toks))
            }
            // \raisebox{lift}[ht][dp]{content}: the optionals sit *after* the
            // mandatory lift, so skip them between the two.
            "raisebox" => {
                let _ = self.grab_argument(); // lift
                while self.grab_optional().is_some() {}
                let toks = self.grab_argument(); // content
                Event::Inline(self.render_inline(toks))
            }
            // \fcolorbox{frame}{bg}{content} — two presentation args, then content.
            "fcolorbox" => {
                let _ = self.grab_optional();
                let _ = self.grab_argument();
                let _ = self.grab_argument();
                let toks = self.grab_argument();
                Event::Inline(self.render_inline(toks))
            }
            // \makebox[w][pos]{content} / \framebox[...] — optional sizing, content.
            "makebox" | "framebox" => {
                while self.grab_optional().is_some() {}
                let toks = self.grab_argument();
                Event::Inline(self.render_inline(toks))
            }
            // \rule[raise]{width}{height} — a printed rule, no textual content.
            "rule" => {
                let _ = self.grab_optional();
                let _ = self.grab_argument();
                let _ = self.grab_argument();
                Event::Inline(vec![])
            }
            // \adjustbox{opts}{content} → render the content.
            "adjustbox" => {
                let _ = self.grab_argument(); // options
                let toks = self.grab_argument();
                Event::Inline(self.render_inline(toks))
            }
            "newif" => {
                // \newif\ifXXX — consume the \ifXXX token. The conditional itself
                // is not evaluated; if its branches later desync a skip, the
                // `\begin{document}` boundary guard in skip_to_fi contains it.
                let _ = self.next_nonspace();
                Event::Inline(vec![])
            }
            "let" => {
                // \let\foo=\bar or \let\foo\bar — bind \foo to \bar's meaning.
                let lhs = self.next_nonspace();
                let mut rhs = self.next_nonspace();
                // Skip an optional `=` between the two (catcode-robust).
                if rhs.as_ref().and_then(Token::char) == Some('=') {
                    rhs = self.next_nonspace();
                }
                if let (Some(Token::ControlSequence(name)), Some(rhs)) = (lhs, rhs) {
                    // A `\let` fully replaces whatever `\foo` meant before, so clear
                    // any stale binding of either kind first.
                    self.state.macros.remove(&name);
                    self.state.let_aliases.remove(&name);
                    match rhs {
                        // \let\a=\b. If \b is a user macro, clone its binding — a
                        // faithful snapshot matching TeX. Otherwise \b is a native
                        // or unknown control sequence: record \a as an *alias* to
                        // \b's target (following any chain), NOT as a macro whose
                        // body is `\b`. A macro body of `\b` would re-dispatch \b
                        // through normal expansion, so a later `\def\b{...\a...}`
                        // (the classic `\let\etaa=\eta`\`\def\eta{...\etaa...}`
                        // idiom) would loop \a→\b→\a forever; an alias dispatches
                        // to \b's *original* meaning and cannot form that cycle.
                        Token::ControlSequence(b) => {
                            if let Some(existing) = self.state.macros.get(&b) {
                                let existing = existing.clone();
                                self.state.macros.insert(name, existing);
                            } else {
                                // Follow \b's own alias chain to its ultimate
                                // target so the snapshot is a direct binding. Skip
                                // a self-alias (`\let\a=\a`, or a chain that
                                // resolves back to \a): storing `name→name` would
                                // be a one-node cycle. Leaving it unbound makes \a
                                // inert, exactly its meaning here (\b/\a is not a
                                // macro), and keeps the resolver cycle-free.
                                let target = self.resolve_let_alias(&b).unwrap_or(b);
                                if target != name {
                                    self.state.let_aliases.insert(name, target);
                                }
                            }
                        }
                        // \let\a=<char> (e.g. \let\bgroup={): re-emit that char.
                        other => {
                            self.state.macros.insert(
                                name,
                                Macro {
                                    params: 0,
                                    optional_default: None,
                                    body: vec![other],
                                },
                            );
                        }
                    }
                }
                Event::Inline(vec![])
            }

            // File inclusion via the resolver.
            "input" | "include" | "subfile" => {
                let file = self.grab_file_name();
                self.include_file(&file, !self.input.in_package());
                Event::Inline(vec![])
            }

            // Document metadata.
            "title" => {
                let _ = self.grab_optional(); // running (short) title
                let toks = self.grab_argument();
                self.title = Some(self.render_inline(toks));
                Event::Inline(vec![])
            }
            "author" => {
                let _ = self.grab_optional(); // affiliation marks, e.g. \author[1,2]{...}
                let group = self.grab_argument();
                // A single \author may list several people separated by \and.
                for chunk in split_on_and(&group) {
                    let nodes = self.render_inline(chunk);
                    self.authors.push(nodes);
                }
                Event::Inline(vec![])
            }
            // Affiliation/contact commands inside \author: keep no visible text.
            "inst" | "affil" | "affiliation" | "institute" | "email" | "orcid"
            | "IEEEauthorblockA" => {
                let _ = self.grab_optional();
                let _ = self.grab_argument();
                Event::Inline(vec![])
            }
            "and" | "And" | "AND" => Event::Inline(vec![]),
            "date" => {
                let toks = self.grab_argument();
                self.date = Some(self.render_inline(toks));
                Event::Inline(vec![])
            }
            "maketitle" => Event::Inline(vec![]),

            // Environments.
            "begin" => self.begin_environment(),
            "end" => {
                let _ = self.grab_argument_text();
                let _ = stop;
                Event::End
            }

            // etoolbox boolean toggles.
            "newtoggle" | "providetoggle" => {
                let name = self.grab_argument_text();
                self.state.toggles.entry(name).or_insert(false);
                Event::Inline(vec![])
            }
            "toggletrue" => {
                let name = self.grab_argument_text();
                self.state.toggles.insert(name, true);
                Event::Inline(vec![])
            }
            "togglefalse" => {
                let name = self.grab_argument_text();
                self.state.toggles.insert(name, false);
                Event::Inline(vec![])
            }
            "settoggle" => {
                let name = self.grab_argument_text();
                let value = self.grab_argument_text();
                self.state.toggles.insert(name, value.trim() == "true");
                Event::Inline(vec![])
            }
            "iftoggle" | "nottoggle" => {
                let negate = name == "nottoggle";
                let toggle = self.grab_argument_text();
                let then_branch = self.grab_argument();
                let else_branch = self.grab_argument();
                let on = self.state.toggles.get(&toggle).copied().unwrap_or(false);
                let taken = if on ^ negate {
                    then_branch
                } else {
                    else_branch
                };
                self.input.splice(&taken);
                Event::Inline(vec![])
            }

            // Classic TeX conditionals (\newif-style and primitives): we cannot
            // always evaluate them, so keep the material up to \else (the "true"
            // branch) and drop the \else..\fi part. \else and \fi seen on their
            // own are handled by the conditional reader, so here they are inert.
            "else" => {
                self.skip_to_fi();
                Event::Inline(vec![])
            }
            "fi" => Event::Inline(vec![]),

            // Sectioning.
            "part" | "chapter" | "section" | "subsection" | "subsubsection" | "paragraph"
            | "subparagraph" => self.sectioning(name),
            // `\appendix` is a numbering-mode switch, not a visible heading.
            "appendix" => Event::Blocks(vec![Node::Element(Element::new("appendix"))]),

            // List / bibliography items.
            "item" => {
                let mut marker = Element::new("item").attr("marker", "");
                if let Some(label) = self.grab_optional() {
                    let term_nodes = self.render_inline(label);
                    let mut term = Element::new("term");
                    term.children = term_nodes;
                    marker.push(Node::Element(term));
                }
                Event::Blocks(vec![Node::Element(marker)])
            }
            "bibitem" => {
                // The optional `[label]` is natbib's citation label, e.g.
                // `[Aggarwal and Vitter(1988)]` — the author-year the reference
                // list and \citet/\citep render in author-year mode. Keep it.
                let label = self.grab_optional().map(|t| reconstruct(&t));
                let key = self.grab_argument_text();
                let mut marker = Element::new("bibitem").attr("marker", "").attr("key", key);
                if let Some(label) = label {
                    marker = marker.attr("label", label.trim());
                }
                Event::Blocks(vec![Node::Element(marker)])
            }

            // Captions and cross-references.
            "caption" => {
                let _ = self.grab_optional();
                let toks = self.grab_argument();
                let nodes = self.render_inline(toks);
                let mut caption = Element::new("caption");
                caption.children = nodes;
                Event::Blocks(vec![Node::Element(caption)])
            }
            "label" => {
                let id = self.grab_argument_text();
                Event::Inline(vec![Node::Element(Element::new("label").attr("id", id))])
            }
            "ref" | "eqref" | "cref" | "Cref" | "autoref" | "pageref" | "vref" => {
                let target = self.grab_argument_text();
                // Record which command produced the reference: the resolve pass
                // needs it to decide whether to prepend a type name ("Figure 2"
                // for \cref) and how to format ("(2)" for \eqref).
                Event::Inline(vec![Node::Element(
                    Element::new("ref")
                        .attr("target", target)
                        .attr("cmd", name),
                )])
            }

            // Hyperlinks.
            "href" => {
                let target = self.grab_argument_text();
                let toks = self.grab_argument();
                let text = self.render_inline(toks);
                let mut link = Element::new("link").attr("href", target);
                link.children = text;
                Event::Inline(vec![Node::Element(link)])
            }

            // Inline text styling — command forms take a braced argument.
            "textbf" | "textmd" => self.styled("bold"),
            "textit" | "textsl" => self.styled("italic"),
            "emph" => self.styled("emph"),
            "texttt" => self.styled("code"),
            "verb" => self.verb_command(),
            "textsc" => self.styled("smallcaps"),
            // Font *declarations* apply to the rest of the current group/scope.
            "bf" | "bfseries" => Event::Wrap("bold"),
            "it" | "itshape" | "sl" | "slshape" | "em" => Event::Wrap("italic"),
            "tt" | "ttfamily" => Event::Wrap("code"),
            "sc" | "scshape" => Event::Wrap("smallcaps"),
            "underline" | "uline" => self.styled("underline"),
            "textsuperscript" => self.styled("superscript"),
            "textsubscript" => self.styled("subscript"),
            "footnote" | "thanks" => {
                let key = self.next_footnote_key();
                let tokens = self.grab_argument();
                let text = self.render_inline(tokens);
                self.footnote_texts.insert(key.clone(), text);
                Event::Inline(vec![Node::Element(
                    Element::new("footnote-mark").attr("key", key),
                )])
            }
            "footnotemark" => {
                let key = self.next_footnote_key();
                self.pending_footnote = Some(key.clone());
                Event::Inline(vec![Node::Element(
                    Element::new("footnote-mark").attr("key", key),
                )])
            }
            "footnotetext" => {
                let key = self
                    .optional_footnote_key()
                    .or_else(|| self.pending_footnote.take())
                    .unwrap_or_else(|| {
                        self.footnote_number += 1;
                        format!("auto:{}", self.footnote_number)
                    });
                let tokens = self.grab_argument();
                let text = self.render_inline(tokens);
                self.footnote_texts.insert(key.clone(), text);
                Event::Inline(vec![Node::Element(
                    Element::new("footnote-text").attr("key", key),
                )])
            }
            // Font switches with no semantic mapping: render the argument as-is.
            "textrm" | "textsf" | "textnormal" | "mbox" | "hbox" | "text"
            | "mathrm" | "mathnormal" => {
                let toks = self.grab_argument();
                Event::Inline(self.render_inline(toks))
            }
            // \ensuremath{X} forces math mode. When it reaches here we are in text
            // (inside `$…$` the body is captured raw, never dispatched), so emit an
            // inline math node — otherwise math commands in X (`\to`, `\gamma`) are
            // rendered as prose and dropped (`\ensuremath{H\to\gamma}` → "H"). The
            // body is reconstructed as raw TeX; the KaTeX backend cleans it.
            "ensuremath" => {
                let toks = self.grab_argument();
                let tex = reconstruct(&self.expand_tokens(toks));
                let mut math = Element::new("math").attr("mode", "inline");
                math.push(Node::text(tex.trim().to_string()));
                Event::Inline(vec![Node::Element(math)])
            }

            // Table commands: render only the content, drop column/span args.
            "multicolumn" => {
                let _ = self.grab_argument(); // column count
                let _ = self.grab_argument(); // column spec
                let toks = self.grab_argument();
                Event::Inline(self.render_inline(toks))
            }
            "multirow" => {
                let _ = self.grab_argument(); // row count
                let _ = self.grab_optional(); // optional vpos
                let _ = self.grab_argument(); // width
                let toks = self.grab_argument();
                Event::Inline(self.render_inline(toks))
            }
            "makecell" | "thead" | "rotatebox" => {
                let _ = self.grab_optional(); // optional spec
                let toks = self.grab_argument();
                Event::Inline(self.render_inline(toks))
            }

            // `\xspace` adds a space unless the next character is punctuation
            // that a space should not precede.
            "xspace" => {
                let space = match self.next_raw() {
                    Some(t) => {
                        let suppress = matches!(
                            &t,
                            Token::Char(c, _) if ".,;:!?')-/".contains(*c)
                        );
                        self.unread(t);
                        !suppress
                    }
                    None => false,
                };
                Event::Inline(if space { vec![Node::text(" ")] } else { vec![] })
            }

            // \ding{N}: a Zapf Dingbats glyph (pifont), commonly a check/cross.
            "ding" => {
                let code = self.grab_argument_text();
                Event::Inline(vec![Node::text(ding_glyph(code.trim()).to_string())])
            }

            // Inline and display math shorthands.
            "(" => self.read_delimited_math(")", false),
            "[" => self.read_delimited_math("]", true),

            // Text-mode accents: `\'e`→é, `\"o`→ö, `\~n`→ñ, `\c{c}`→ç, ... The
            // accent command takes one argument (a bare char or a braced group)
            // and combines with it. User macros of the same name are resolved
            // earlier, so this only fires for the built-in accents. `\~` reaches
            // here as `\cs("~")`; the active non-breaking-space character `~` is
            // tokenized separately (as the nbsp char), so it does not collide.
            "'" | "`" | "^" | "~" | "\"" | "=" | "." | "u" | "v" | "H" | "r" | "c"
            | "k" | "b" | "d" | "t"
                if accent_mark(name).is_some() =>
            {
                let base = self.grab_accent_base();
                Event::Inline(vec![Node::text(apply_accent(&base, name))])
            }

            _ => {
                // `\the<counter>` (e.g. `\the@affilnum`) prints the counter value.
                if let Some(ctr) = name.strip_prefix("the")
                    && let Some(&n) = self.state.counters.get(ctr)
                {
                    return Event::Inline(vec![Node::text(n.to_string())]);
                }
                if let Some(text) = symbol(name) {
                    Event::Inline(vec![Node::text(text)])
                } else if let Some(space) = spacing(name) {
                    // Some spacing commands take a length argument.
                    if matches!(name, "hspace" | "vspace" | "hspace*" | "vspace*") {
                        let _ = self.grab_argument();
                    }
                    Event::Inline(if space.is_empty() {
                        vec![]
                    } else {
                        vec![Node::text(space)]
                    })
                } else {
                    // A control sequence texmark does not understand. Drop it,
                    // but record it so the loss is reported, never silent. (An
                    // empty name is a stray `\` at end of input — nothing to report.)
                    if !name.is_empty() {
                        self.diagnostics.dropped_commands.insert(name.to_string());
                    }
                    Event::Inline(vec![])
                }
            }
        }
    }

    /// `\verb<d>...<d>` (and `\verb*`): read the character `<d>` as a delimiter,
    /// then the raw source up to its next occurrence, as a code span. Reading is
    /// verbatim (no tokenizing), so `_`, `\`, `{`, `#` etc. are literal. `\verb*`
    /// makes spaces visible. Falls back to empty if `\verb` reached us via macro
    /// expansion (no raw frame) or the delimiter is malformed.
    fn verb_command(&mut self) -> Event {
        // Read the delimiter as a RAW character from the live source, so a space
        // or `%` is a literal delimiter (matching LaTeX), not a skipped blank or
        // a comment. `*` selects the visible-space variant.
        let body = match self.read_raw_char() {
            Some('*') => match self.read_raw_char() {
                Some(delim) => Some((
                    true,
                    self.input.read_raw(&delim.to_string()).unwrap_or_default(),
                )),
                None => return Event::Inline(vec![]),
            },
            Some(delim) => Some((
                false,
                self.input.read_raw(&delim.to_string()).unwrap_or_default(),
            )),
            // `\verb` arrived via macro expansion: no live source to read raw.
            // Fall back to reading tokens up to the delimiter so the content is
            // captured (approximately) rather than lost or leaked.
            None => self.verb_from_tokens(),
        };
        let Some((starred, raw)) = body else {
            return Event::Inline(vec![]);
        };
        let raw = if starred {
            raw.replace(' ', "\u{2423}") // ␣ open box, LaTeX's visible space
        } else {
            raw
        };
        let mut el = Element::new("code");
        el.push(Node::text(raw));
        Event::Inline(vec![Node::Element(el)])
    }

    /// Fallback for `\verb` reached through macro expansion: read tokens up to
    /// the delimiter and reconstruct them (catcodes already applied, so not
    /// perfectly verbatim, but content is preserved rather than leaked).
    fn verb_from_tokens(&mut self) -> Option<(bool, String)> {
        let first = match self.next_raw()? {
            Token::Char(c, _) => c,
            t => {
                self.unread(t);
                return None;
            }
        };
        let (starred, delim) = if first == '*' {
            match self.next_raw()? {
                Token::Char(c, _) => (true, c),
                _ => return None,
            }
        } else {
            (false, first)
        };
        let mut toks = Vec::new();
        while let Some(t) = self.next_raw() {
            if matches!(&t, Token::Char(c, _) if *c == delim) {
                break;
            }
            toks.push(t);
        }
        Some((starred, reconstruct(&toks)))
    }

    /// Wrap the next argument in an inline element with the given tag.
    fn styled(&mut self, tag: &str) -> Event {
        let toks = self.grab_argument();
        let nodes = self.render_inline(toks);
        let mut el = Element::new(tag);
        el.children = nodes;
        Event::Inline(vec![Node::Element(el)])
    }

    /// Assemble a control-sequence name from tokens up to `\endcsname`, expanding
    /// control sequences inside it to their character form (`\the<counter>` → its
    /// value, nested `\csname` → its built name's value).
    fn build_csname(&mut self) -> String {
        let mut name = String::new();
        loop {
            match self.next_raw() {
                None => break,
                Some(Token::ControlSequence(n)) if n == "endcsname" => break,
                Some(Token::ControlSequence(n)) if n == "csname" => {
                    let inner = self.build_csname();
                    name.push_str(&self.cs_value_string(&inner));
                }
                Some(Token::ControlSequence(n)) => name.push_str(&self.cs_value_string(&n)),
                Some(Token::Char(c, _)) => name.push(c),
            }
        }
        name
    }

    /// The string a control sequence contributes inside a `\csname`: a
    /// `\the<counter>` yields the counter's value; anything else contributes its
    /// own name (best effort).
    fn cs_value_string(&self, cs: &str) -> String {
        if let Some(ctr) = cs.strip_prefix("the")
            && let Some(&n) = self.state.counters.get(ctr)
        {
            return n.to_string();
        }
        cs.to_string()
    }

    /// Whether the control sequence `\name` is defined — a user macro, or a
    /// counter's `\the<counter>` representation.
    fn is_defined(&self, name: &str) -> bool {
        self.state.macros.contains_key(name)
            || name
                .strip_prefix("the")
                .is_some_and(|c| self.state.counters.contains_key(c))
    }

    fn ifx_equal(&self, left: &Token, right: &Token) -> bool {
        match (left, right) {
            (Token::Char(_, _), Token::Char(_, _)) => left == right,
            (Token::ControlSequence(left), Token::ControlSequence(right)) => {
                let left = self.resolve_let_alias(left).unwrap_or_else(|| left.clone());
                let right = self
                    .resolve_let_alias(right)
                    .unwrap_or_else(|| right.clone());
                left == right
                    || matches!(
                        (self.state.macros.get(&left), self.state.macros.get(&right)),
                        (Some(left), Some(right)) if left == right
                    )
                    || (!self.state.macros.contains_key(&left)
                        && !self.state.macros.contains_key(&right)
                        && !self.state.let_aliases.contains_key(&left)
                        && !self.state.let_aliases.contains_key(&right)
                        && packages::for_command(&left).is_none()
                        && packages::for_command(&right).is_none()
                        && !is_protected(&left)
                        && !is_protected(&right))
            }
            _ => false,
        }
    }

    /// Skip a conditional's true branch to the matching `\else` (consumed, so the
    /// else branch runs) or `\fi` (consumed, no else), honoring nesting.
    fn skip_to_else_or_fi(&mut self) {
        let mut depth = 1usize;
        while let Some(t) = self.next_raw() {
            if let Token::ControlSequence(n) = &t {
                if is_conditional_cs(n) {
                    depth += 1;
                } else if n == "fi" {
                    depth -= 1;
                    if depth == 0 {
                        return;
                    }
                } else if (n == "else" && depth == 1) || (n == "begin" && self.at_document_start())
                {
                    return;
                }
            }
        }
    }

    /// Fully expand user macros in a token list, leaving primitives and
    /// unknown control sequences untouched. Used for math bodies (so paper-defined
    /// shorthands like `\vQ`/`\softmax` reach the output as standard TeX) and for
    /// `\edef`/`\xdef` bodies (definition-time expansion).
    pub(crate) fn expand_tokens(&mut self, tokens: Vec<Token>) -> Vec<Token> {
        // Seal the surrounding input behind a barrier, then splice the tokens on
        // top. Reads — including a trailing macro's argument grab — stop at the
        // barrier (as end-of-input) instead of spilling into the document, which
        // would silently swallow it. The barrier is removed once the tokens drain.
        self.input.push_barrier();
        self.input.splice(&tokens);
        let mut out = Vec::with_capacity(tokens.len());
        while let Some(t) = self.next_raw() {
            if let Token::ControlSequence(name) = &t {
                // `\noexpand` protects the next token from expansion: emit that
                // token verbatim (and drop the `\noexpand` itself).
                if name == "noexpand" {
                    if let Some(next) = self.next_raw() {
                        out.push(next);
                    }
                    continue;
                }
                if matches!(name.as_str(), "iftoggle" | "nottoggle") {
                    let negate = name == "nottoggle";
                    let toggle = self.grab_argument_text();
                    let then_branch = self.grab_argument();
                    let else_branch = self.grab_argument();
                    let on = self.state.toggles.get(&toggle).copied().unwrap_or(false);
                    self.input.splice(if on ^ negate {
                        &then_branch
                    } else {
                        &else_branch
                    });
                    continue;
                }
                if name == "ifx" {
                    let equal = match (self.next_raw(), self.next_raw()) {
                        (Some(left), Some(right)) => self.ifx_equal(&left, &right),
                        _ => false,
                    };
                    if !equal {
                        self.skip_to_else_or_fi();
                    }
                    continue;
                }
                if name == "else" {
                    self.skip_to_fi();
                    continue;
                }
                if name == "fi" {
                    continue;
                }
                if self.is_user_macro(name) {
                    let name = name.clone();
                    self.expand_macro(&name);
                    continue;
                }
                // A `\let` alias to a non-macro (see the `"let"` handler): emit its
                // *target* control sequence, so a math body reconstructs to the
                // target's TeX (`\etaa`→`\eta`) — the meaning snapshotted at `\let`
                // time — rather than the undefined alias name. Resolution is
                // bounded (`resolve_let_alias`); a cyclic alias just emits the name
                // inertly. Either way this pushes one token and cannot loop.
                if self.state.let_aliases.contains_key(name) {
                    let target = self.resolve_let_alias(name).unwrap_or_else(|| name.clone());
                    out.push(Token::cs(target));
                    continue;
                }
            }
            out.push(t);
        }
        self.input.pop_barrier();
        out
    }

    fn include_file(&mut self, name: &str, report_missing: bool) {
        // A file that (transitively) `\input`s itself would splice source
        // forever; bound it by the same budget that limits macro expansion, and
        // report the truncation rather than looping.
        self.expansions += 1;
        if self.expansions > MAX_EXPANSIONS {
            self.diagnostics.expansion_limit_exceeded = true;
            return;
        }
        let Some(src) = self.resolver.resolve(name) else {
            // A content include (`\input`/`\include`) the resolver can't find is
            // silent content loss — record it so the caller isn't told the output
            // is complete. (Package includes pass report_missing=false: a missing
            // `.sty` is expected, not lost content.)
            if report_missing {
                self.diagnostics.unresolved_inputs.insert(name.to_string());
            }
            return;
        };
        // Package/class files are read with `@` as a letter (LaTeX sets it
        // before loading them), so internal `\@foo` control words tokenize
        // as one control sequence. Bracket the source with an implicit
        // `\makeatletter … <restore>` and let the normal command handlers
        // flip the catcode as the frame is read. The trailing token restores
        // `@` to *whatever it was before this include* rather than blindly
        // resetting it to `other`: a package that `\input`s another `.sty`
        // (e.g. eso-pic → cvpr_eso.sty) must keep `@` a letter for its own
        // code that follows the nested input — a blind `\makeatother` there
        // would mis-tokenize every subsequent `\@foo`. A plain `.tex` needs
        // no bracket: any `\makeatletter` it contains now takes effect
        // incrementally through the same live-catcode read.
        let at_letter = name.ends_with(".sty") || name.ends_with(".cls");
        if !report_missing {
            self.input.mark_package(false);
        }
        if at_letter {
            let restore = if self.state.catcodes.get('@') == CatCode::Letter {
                "makeatletter"
            } else {
                "makeatother"
            };
            self.input.splice(&[Token::cs(restore)]);
            self.input.push_source(&src);
            self.input.splice(&[Token::cs("makeatletter")]);
        } else {
            self.input.push_source(&src);
        }
        if !report_missing {
            self.input.mark_package(true);
        }
    }

    fn next_footnote_key(&mut self) -> String {
        self.optional_footnote_key().unwrap_or_else(|| {
            self.footnote_number += 1;
            format!("auto:{}", self.footnote_number)
        })
    }

    fn optional_footnote_key(&mut self) -> Option<String> {
        let key = reconstruct(&self.grab_optional()?).trim().to_string();
        if key.is_empty() {
            return None;
        }
        if key
            .chars()
            .filter(|c| !c.is_whitespace())
            .collect::<String>()
            == r"\value{footnote}"
        {
            return Some(format!("auto:{}", self.footnote_number));
        }
        if let Ok(number) = key.parse::<usize>() {
            let automatic = format!("auto:{number}");
            if self.footnote_texts.contains_key(&automatic)
                || self.pending_footnote.as_deref() == Some(automatic.as_str())
            {
                return Some(automatic);
            }
        }
        Some(format!("explicit:{key}"))
    }

    // --- package-facing engine surface ------------------------------------

    /// The resolver, so a package can pull in external files (e.g. natbib
    /// reading a `.bbl`/`.bib`).
    pub(crate) fn resolver(&self) -> &dyn Resolver {
        self.resolver
    }

    /// The engine's mutable state — lets a package read the options it stashed
    /// in [`State::package_state`] during its `finalize` hook.
    pub(crate) fn state(&self) -> &State {
        self.state
    }

    /// Mutable state, so a package command can stash options in
    /// [`State::package_state`] (e.g. `\addbibresource`).
    pub(crate) fn state_mut(&mut self) -> &mut State {
        self.state
    }

    pub(crate) fn report_dropped_command(&mut self, name: &str) {
        self.diagnostics.dropped_commands.insert(name.to_string());
    }

    /// Push `src` as a character source to be read next (as `\input` does), so
    /// generated fragments stay lazily tokenized and readable raw.
    pub(crate) fn splice_source(&mut self, src: &str) {
        self.input.push_source(src);
    }

    /// Render a generated LaTeX fragment to block nodes, reusing the full
    /// engine. Used by `finalize` hooks that build content post-parse.
    pub(crate) fn render_fragment(&mut self, src: &str) -> Vec<Node> {
        let tokens = Tokenizer::new(src).tokenize(&self.state.catcodes);
        self.render_block(tokens)
    }

    /// Read an environment body as raw, unexpanded text (verbatim-style), for
    /// packages like listings whose body is not LaTeX. The text is returned
    /// exactly as written — including any leading optional argument
    /// (`[key=val,…]`) on the `\begin` line — so the caller parses it in the raw
    /// domain rather than tokenizing past the body. Falls back to token
    /// reconstruction when the source is not directly readable (e.g. a body
    /// produced by macro expansion rather than a live character source).
    pub(crate) fn read_environment_raw_text(&mut self, name: &str) -> String {
        let marker = format!("\\end{{{name}}}");
        match self.input.read_raw(&marker) {
            Some(r) => r,
            None => reconstruct(&self.read_environment_body_raw(name)),
        }
    }

    /// Evaluate a small integer expression from argument tokens: a decimal
    /// literal or `\value{counter}`. Anything else evaluates to 0.
    fn eval_number(&self, tokens: &[Token]) -> i32 {
        let s = reconstruct(tokens);
        let s = s.trim();
        if let Some(rest) = s.strip_prefix("\\value") {
            let name = rest
                .trim()
                .trim_start_matches('{')
                .trim_end_matches('}')
                .trim();
            return self.state.counters.get(name).copied().unwrap_or(0);
        }
        s.split_whitespace()
            .next()
            .and_then(|w| w.parse::<i32>().ok())
            .unwrap_or(0)
    }

    /// Consume a TeX integer argument (optional leading spaces, optional sign,
    /// then decimal digits) from the input, discarding it. Used for primitives
    /// like `\penalty` whose numeric operand must not reach the output.
    fn skip_number(&mut self) {
        while let Some(t) = self.next_raw() {
            if t.is_cat(CatCode::Space) {
                continue;
            }
            self.unread(t);
            break;
        }
        // Optional sign: consume a leading '+'/'-', else put the token back.
        match self.next_raw() {
            Some(Token::Char(c, _)) if c == '+' || c == '-' => {}
            Some(t) => self.unread(t),
            None => {}
        }
        while let Some(t) = self.next_raw() {
            if matches!(&t, Token::Char(c, _) if c.is_ascii_digit()) {
                continue;
            }
            self.unread(t);
            break;
        }
    }

    /// Consume an optional `=` (and surrounding spaces), as in a TeX assignment.
    fn skip_optional_equals(&mut self) {
        while let Some(t) = self.next_raw() {
            if t.is_cat(CatCode::Space) {
                continue;
            }
            if !matches!(&t, Token::Char('=', _)) {
                self.unread(t);
            }
            return;
        }
    }

    /// Consume a TeX dimension/glue operand — `=10pt`, `.25in`, `\textwidth`,
    /// `\wd\box`, `18pt` — discarding it. Used for length-register assignments
    /// and glue primitives (`\vskip`, `\kern`, …) whose operands are not content.
    fn skip_dimen(&mut self) {
        self.skip_optional_equals();
        // Optional sign.
        match self.next_nonspace() {
            Some(Token::Char(c, _)) if c == '+' || c == '-' => {}
            Some(t) => self.unread(t),
            None => {}
        }
        match self.next_raw() {
            // A length register/command (`\textwidth`); `\wd`/`\ht`/`\dp` take a
            // following box register.
            Some(Token::ControlSequence(n)) => {
                if matches!(n.as_str(), "wd" | "ht" | "dp") {
                    let _ = self.next_nonspace();
                }
            }
            // A numeric dimension: digits/point, then the unit letters that
            // immediately follow it (`pt`, `in`, `fil`, …).
            Some(t @ Token::Char(c, _)) if c.is_ascii_digit() || c == '.' => {
                self.unread(t);
                while let Some(t) = self.next_raw() {
                    if matches!(&t, Token::Char(c, _) if c.is_ascii_digit() || *c == '.') {
                        continue;
                    }
                    self.unread(t);
                    break;
                }
                while let Some(t) = self.next_raw() {
                    if matches!(&t, Token::Char(c, _) if c.is_ascii_alphabetic()) {
                        continue;
                    }
                    self.unread(t);
                    break;
                }
            }
            Some(t) => self.unread(t),
            None => {}
        }
    }

    /// Consume a `\hrule`/`\vrule` specification: any run of `height`/`width`/
    /// `depth <dimen>` keywords, discarding it.
    fn skip_rule_spec(&mut self) {
        loop {
            let mut word = String::new();
            let mut buf = Vec::new();
            // Skip leading spaces, then read a run of letters (a keyword).
            while let Some(t) = self.next_raw() {
                match &t {
                    Token::Char(c, _) if c.is_whitespace() && word.is_empty() => {}
                    Token::Char(c, _) if c.is_ascii_alphabetic() => {
                        word.push(*c);
                        buf.push(t);
                        continue;
                    }
                    _ => {
                        self.unread(t);
                        break;
                    }
                }
            }
            if matches!(word.as_str(), "height" | "width" | "depth") {
                self.skip_dimen();
            } else {
                for t in buf.into_iter().rev() {
                    self.unread(t);
                }
                break;
            }
        }
    }

    /// Consume a box after `\setbox<reg>=`: a box constructor and its group,
    /// discarding it (the box is stored, never typeset inline).
    fn skip_box(&mut self) {
        match self.next_nonspace() {
            Some(Token::ControlSequence(_)) => {
                // The box maker (\hbox/\vbox/…) was just consumed; drop its group.
                if let Some(t) = self.next_nonspace() {
                    if t.is_cat(CatCode::BeginGroup) {
                        self.unread(t);
                        let _ = self.grab_argument();
                    } else {
                        self.unread(t);
                    }
                }
            }
            Some(t) => self.unread(t),
            None => {}
        }
    }

    /// Skip tokens from a stray `\else` to its matching `\fi`, honoring nested
    /// `\if...\fi`. Approximates TeX by treating any `\if...` as an opener and
    /// `\fi` as a closer.
    fn skip_to_fi(&mut self) {
        let mut depth = 1usize;
        while let Some(t) = self.next_raw() {
            if let Token::ControlSequence(n) = &t {
                if is_conditional_cs(n) {
                    depth += 1;
                } else if n == "fi" {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                } else if n == "begin" && self.at_document_start() {
                    break;
                }
            }
        }
    }

    /// While skipping a conditional branch, a `\begin` was seen: peek whether it
    /// starts the document body. If so, restore `\begin{document}` and report it
    /// — a conditional must never run away past the preamble/body boundary and
    /// swallow the whole document (e.g. a `\newif`/`\ifx`/`\loop` desync in a
    /// bundled `.sty`). Otherwise consume the environment name and keep skipping.
    fn at_document_start(&mut self) -> bool {
        let name = self.grab_argument_text();
        if name == "document" {
            self.splice_source("\\begin{document}");
            true
        } else {
            false
        }
    }

    // --- math --------------------------------------------------------------

    /// Read a `$...$` (inline) or `$$...$$` (display) math run. Called just
    /// after the opening `$` was consumed.
    fn read_math(&mut self) -> Event {
        let display = match self.next_raw() {
            Some(t) if t.is_cat(CatCode::MathShift) => true,
            Some(t) => {
                self.unread(t);
                false
            }
            None => false,
        };
        let mut tokens = Vec::new();
        // Track brace grouping: the closing `$`/`$$` is always at the same group
        // level as the opener. A `$` nested inside a group — e.g. `\text{... $x$
        // ...}` inside display math — opens a sub-formula and must NOT be taken
        // as the closer, or the scanner desyncs and swallows the rest of the doc.
        let mut depth: i32 = 0;
        while let Some(t) = self.next_raw() {
            if t.is_cat(CatCode::BeginGroup) {
                depth += 1;
            } else if t.is_cat(CatCode::EndGroup) {
                depth -= 1;
            } else if t.is_cat(CatCode::MathShift) && depth <= 0 {
                if display {
                    // Consume the second `$` of the closing `$$`.
                    if let Some(t2) = self.next_raw()
                        && !t2.is_cat(CatCode::MathShift)
                    {
                        self.unread(t2);
                    }
                }
                break;
            }
            tokens.push(t);
        }
        // Store raw (macro-expanded) TeX; KaTeX-specific cleaning is the backend's
        // job. `\label` survives here so the resolve pass can number equations.
        let tokens = self.expand_tokens(tokens);
        self.math_event(&tokens, display)
    }

    /// Read math delimited by a closing control sequence (`\)` or `\]`).
    fn read_delimited_math(&mut self, end: &str, display: bool) -> Event {
        let mut tokens = Vec::new();
        while let Some(t) = self.next_raw() {
            if matches!(&t, Token::ControlSequence(n) if n == end) {
                break;
            }
            tokens.push(t);
        }
        let tokens = self.expand_tokens(tokens);
        self.math_event(&tokens, display)
    }

    /// Build a `<math>` element from math tokens.
    fn math_event(&self, tokens: &[Token], display: bool) -> Event {
        let tex = reconstruct(tokens);
        let mode = if display { "display" } else { "inline" };
        let mut math = Element::new("math").attr("mode", mode);
        math.push(Node::text(tex.trim().to_string()));
        let node = Node::Element(math);
        if display {
            Event::Blocks(vec![node])
        } else {
            Event::Inline(vec![node])
        }
    }
}

impl Engine<'_> {
    // --- environments ------------------------------------------------------

    /// Handle `\begin{name}`, dispatching to the matching construct.
    fn begin_environment(&mut self) -> Event {
        let name = self.grab_argument_text();

        // Native packages are checked before user-defined environments so that
        // loading a package's .sty (which may also define the same environment)
        // does not override our handler.
        if let Some(pkg) = packages::for_environment(&name) {
            return pkg.environment(&name, self);
        }

        // Standard float environments carry native semantics. A document class
        // commonly `\renewenvironment{table}`/`{figure}` to tweak spacing,
        // delegating to `\@float` internals we can't execute; that redefinition
        // would otherwise shadow the native handler here and leak the body as
        // text (e.g. `table[t]`) while never producing a numbered `<float>`.
        // Our semantic handler wins over such a re-plumbing redefinition.
        match name.as_str() {
            "figure" | "figure*" => return self.float_environment(&name, "figure"),
            "table" | "table*" => return self.float_environment(&name, "table"),
            "algorithm" | "algorithm*" => return self.float_environment(&name, "algorithm"),
            "thebibliography" => {
                let _ = self.grab_argument();
                let body = self.build_block(Stop::Environment(name));
                return Event::Blocks(vec![wrap("bibliography", fold_items(body))]);
            }
            _ => {}
        }

        // User-defined environments expand their begin/end token lists around
        // the (raw) body.
        if let Some(env) = self.state.environments.get(&name).cloned() {
            return self.expand_user_environment(&name, &env);
        }

        match name.as_str() {
            "document" => Event::Blocks(self.build_block(Stop::Environment("document".into()))),
            "abstract" => {
                let body = self.build_block(Stop::Environment(name));
                Event::Blocks(vec![wrap("abstract", body)])
            }
            "itemize" => self.list_environment(&name, "itemize"),
            "enumerate" => self.list_environment(&name, "enumerate"),
            "description" => self.list_environment(&name, "description"),
            "quote" | "quotation" | "displayquote" | "verse" => {
                let body = self.build_block(Stop::Environment(name));
                Event::Blocks(vec![wrap("blockquote", body)])
            }
            // `\begin{proof}[optional lead]` — capture the optional argument (which
            // may contain a `\cref`, resolved later) as the block's lead-in so it
            // does not leak as literal `[...]`. Marked so md renders "**...**".
            "proof" => {
                let head = self.grab_optional().map(|toks| {
                    let mut term = Element::new("term");
                    term.children = self.render_inline(toks);
                    Node::Element(term)
                });
                let mut body = self.build_block(Stop::Environment(name));
                if let Some(head) = head {
                    body.insert(0, head);
                }
                Event::Blocks(vec![Node::Element({
                    let mut e = Element::new("environment").attr("name", "proof");
                    e.children = body;
                    e
                })])
            }
            "center" | "flushleft" | "flushright" => {
                let align = match name.as_str() {
                    "flushleft" => "left",
                    "flushright" => "right",
                    _ => "center",
                };
                let body = self.build_block(Stop::Environment(name));
                Event::Blocks(vec![Node::Element({
                    let mut e = Element::new("align").attr("to", align);
                    e.children = body;
                    e
                })])
            }
            "figure" | "figure*" | "table" | "table*" | "algorithm" | "algorithm*" => {
                // Handled natively above (before the user-env lookup) so a class's
                // `\renewenvironment` cannot shadow it; unreachable here.
                self.float_environment(&name, name.trim_end_matches('*'))
            }
            // subfigure: like figure but has a required {width} argument.
            "subfigure" | "subfloat" => {
                let _ = self.grab_optional(); // optional caption position
                let _ = self.grab_argument(); // required width
                self.float_environment(&name, "figure")
            }
            // minipage: transparent wrapper; consume required {width} arg.
            "minipage" => {
                let _ = self.grab_optional(); // optional vertical alignment
                let _ = self.grab_argument(); // required width
                let body = self.build_block(Stop::Environment(name));
                Event::Blocks(body)
            }
            // TikZ / PGF picture environments cannot be represented in GFM.
            "tikzpicture" | "pgfpicture" | "tikz" => {
                // Consume and discard the entire body.
                self.read_environment_body_raw(&name);
                Event::Inline(vec![])
            }
            "verbatim" | "verbatim*" => self.verbatim_environment(&name),
            "equation" | "equation*" | "align" | "align*" | "aligned" | "eqnarray"
            | "eqnarray*" | "gather" | "gather*" | "multline" | "multline*" | "displaymath"
            | "split" | "flalign" | "flalign*" => {
                let body = self.read_environment_body_raw(&name);
                // Store raw (macro-expanded) TeX and the environment name. The
                // backend decides KaTeX alignment wrapping / command cleaning; the
                // resolve pass reads `env` + the raw `\label`s to number equations.
                let body = self.expand_tokens(body);
                let tex = reconstruct(&body);
                let math = Element::new("math")
                    .attr("mode", "display")
                    .attr("env", name.as_str());
                let mut math = math;
                math.push(Node::text(tex.trim().to_string()));
                Event::Blocks(vec![Node::Element(math)])
            }
            "tabular" | "tabular*" | "array" | "tabularx" | "longtable" => {
                self.tabular_environment(&name)
            }
            // A registered `\newtheorem` environment. Capture its optional note
            // (`\begin{thm}[Pythagoras]`) as a `<term>` so it does not leak as
            // literal `[...]`; the resolve pass stamps the "Theorem 1" prefix.
            n if self.state.theorems.contains_key(n) => {
                let head = self.grab_optional().map(|toks| {
                    let mut term = Element::new("term");
                    term.children = self.render_inline(toks);
                    Node::Element(term)
                });
                let mut body = self.build_block(Stop::Environment(name.clone()));
                if let Some(head) = head {
                    body.insert(0, head);
                }
                Event::Blocks(vec![Node::Element({
                    let mut e =
                        Element::new("environment").attr("name", name.trim_end_matches('*'));
                    e.children = body;
                    e
                })])
            }
            _ => {
                // Unknown environment: keep the content, wrap it so structure
                // is not lost.
                let body = self.build_block(Stop::Environment(name.clone()));
                let clean = name.trim_end_matches('*').to_string();
                Event::Blocks(vec![Node::Element({
                    let mut e = Element::new("environment").attr("name", clean);
                    e.children = body;
                    e
                })])
            }
        }
    }

    /// Build a list environment (`itemize`/`enumerate`/`description`), folding
    /// `\item` markers into `<item>` children.
    fn list_environment(&mut self, name: &str, tag: &str) -> Event {
        let body = self.build_block(Stop::Environment(name.to_string()));
        Event::Blocks(vec![wrap(tag, fold_items(body))])
    }

    /// Read a verbatim environment's body as raw source, so backslashes,
    /// braces, spacing, and comments survive untouched.
    fn verbatim_environment(&mut self, name: &str) -> Event {
        let marker = format!("\\end{{{name}}}");
        let body = match self.input.read_raw(&marker) {
            Some(raw) => strip_leading_newline(&raw),
            None => reconstruct(&self.read_environment_body_raw(name)),
        };
        let mut e = Element::new("verbatim");
        e.push(Node::text(body));
        Event::Blocks(vec![Node::Element(e)])
    }

    /// Build a float environment (`figure`/`table`), keeping captions and
    /// labels inside.
    fn float_environment(&mut self, name: &str, kind: &str) -> Event {
        let _ = self.grab_optional(); // placement specifier
        let body = self.build_block(Stop::Environment(name.to_string()));
        let mut e = Element::new("float").attr("kind", kind);
        e.children = body;
        // The tree stays neutral: a `<float>` with `<caption>`/`<image>` children.
        // Deriving image alt text from the caption (and suppressing the now-
        // redundant visible caption) is a Markdown concern, done in the backend.
        Event::Blocks(vec![Node::Element(e)])
    }

    /// Expand a user-defined environment around its raw body.
    fn expand_user_environment(&mut self, name: &str, env: &Environment) -> Event {
        let mut args: Vec<Vec<Token>> = Vec::with_capacity(env.params);
        if let Some(default) = &env.optional_default {
            args.push(self.grab_optional().unwrap_or_else(|| default.clone()));
            for _ in 1..env.params {
                args.push(self.grab_argument());
            }
        } else {
            for _ in 0..env.params {
                args.push(self.grab_argument());
            }
        }
        let body = self.read_environment_body_raw(name);
        let mut tokens = substitute(&env.begin, &args);
        tokens.extend(body);
        tokens.extend(substitute(&env.end, &args));
        Event::Blocks(self.render_block(tokens))
    }

    /// Read an environment's body as raw tokens up to its matching
    /// `\end{name}`, honoring nested environments of the same name.
    pub(crate) fn read_environment_body_raw(&mut self, name: &str) -> Vec<Token> {
        let mut out = Vec::new();
        let mut depth = 1usize;
        while let Some(t) = self.next_raw() {
            if let Token::ControlSequence(cs) = &t {
                if cs == "end" {
                    let inner = self.grab_argument_text();
                    if inner == name {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                    out.extend(end_tokens(&inner));
                    continue;
                }
                if cs == "begin" {
                    let inner = self.grab_argument_text();
                    if inner == name {
                        depth += 1;
                    }
                    out.push(Token::cs("begin"));
                    out.extend(group_tokens(&inner));
                    continue;
                }
                // `\input` is textual inclusion: splice the file so its content
                // becomes part of the raw body (e.g. a table whose rows live in
                // an `\input`ed file). The spliced source is read by the loop.
                if matches!(cs.as_str(), "input" | "include" | "subfile") {
                    let file = self.grab_argument_text();
                    self.include_file(&file, true);
                    continue;
                }
            }
            out.push(t);
        }
        out
    }

    /// Render a raw token list as block content, sharing state.
    pub(crate) fn render_block(&mut self, tokens: Vec<Token>) -> Vec<Node> {
        self.input.splice(&end_tokens(""));
        self.input.splice(&tokens);
        self.build_block(Stop::Environment(String::new()))
    }

    /// Parse a `tabular`-like environment into rows and cells.
    fn tabular_environment(&mut self, name: &str) -> Event {
        let _ = self.grab_optional(); // vertical position
        let _ = self.grab_argument(); // column specification
        let body = reduce_longtable_body(self.read_environment_body_raw(name));

        let mut table = Element::new("tabular");
        for row_tokens in split_rows(&body) {
            let cells = split_cells(&row_tokens);
            // Skip rows that are entirely rule commands, empty, or a `\caption`
            // (longtable puts its caption in a row of its own).
            if cells.iter().all(|c| is_rule_only(c)) || is_caption_row(&row_tokens) {
                continue;
            }
            let mut row = Element::new("row");
            for cell_tokens in cells {
                // cmidrule/cline may appear without a \\ before data cells; strip them.
                let cell_tokens = strip_rule_prefix(cell_tokens);
                // `\multicolumn{n}{spec}{content}` spans n columns. GFM has no
                // colspan, so render the content in the first column and pad with
                // n-1 empty cells, keeping every later column aligned (otherwise a
                // grouped header row is short and its labels drift left).
                let (content_tokens, span) = match parse_leading_multicolumn(&cell_tokens) {
                    Some((n, content)) => (content, n.max(1)),
                    None => (cell_tokens, 1),
                };
                let nodes = self.render_inline(content_tokens);
                let mut cell = Element::new("cell");
                cell.children = trim_edges(nodes);
                row.push(Node::Element(cell));
                for _ in 1..span {
                    row.push(Node::Element(Element::new("cell")));
                }
            }
            table.push(Node::Element(row));
        }
        Event::Blocks(vec![Node::Element(table)])
    }
}

/// A longtable head/foot marker delimiting the repeated header and footer
/// blocks from the table's data rows.
fn is_longtable_marker(t: &Token) -> bool {
    matches!(t, Token::ControlSequence(n)
        if matches!(n.as_str(), "endfirsthead" | "endhead" | "endfoot" | "endlastfoot"))
}

/// Reduce a `longtable` body to just its first head plus its data rows, dropping
/// the repeated-header and footer blocks that `\endfirsthead`/`\endhead`/
/// `\endfoot`/`\endlastfoot` delimit. A body with none of those markers (an
/// ordinary `tabular`) is returned unchanged.
fn reduce_longtable_body(body: Vec<Token>) -> Vec<Token> {
    if !body.iter().any(is_longtable_marker) {
        return body;
    }
    // Split on the markers: the first segment is the head shown once, the last
    // is the data; everything between is the repeated head and the footer.
    let mut segments = vec![Vec::new()];
    for t in body {
        if is_longtable_marker(&t) {
            segments.push(Vec::new());
        } else {
            segments.last_mut().unwrap().push(t);
        }
    }
    let mut out = segments.first().cloned().unwrap_or_default();
    if let Some(data) = segments.last()
        && segments.len() > 1
    {
        out.extend(data.clone());
    }
    out
}

/// Whether a table row is a `\caption{…}` (with an optional `\label`), which
/// longtable emits as a row of its own rather than table data.
fn is_caption_row(row: &[Token]) -> bool {
    row.iter()
        .any(|t| matches!(t, Token::ControlSequence(n) if n == "caption"))
}

// --- token-list helpers -------------------------------------------------

/// The tokens for `\end{name}`.
fn end_tokens(name: &str) -> Vec<Token> {
    let mut v = vec![Token::cs("end")];
    v.extend(group_tokens(name));
    v
}

/// The tokens for `{name}` (a braced group of letters).
fn group_tokens(name: &str) -> Vec<Token> {
    let mut v = vec![Token::Char('{', CatCode::BeginGroup)];
    v.extend(name.chars().map(|c| Token::Char(c, CatCode::Letter)));
    v.push(Token::Char('}', CatCode::EndGroup));
    v
}

/// Split author tokens on `\and`/`\And`/`\AND` into one chunk per person.
fn split_on_and(tokens: &[Token]) -> Vec<Vec<Token>> {
    let mut chunks = vec![Vec::new()];
    for t in tokens {
        if matches!(t, Token::ControlSequence(n) if matches!(n.as_str(), "and" | "And" | "AND")) {
            chunks.push(Vec::new());
        } else {
            chunks.last_mut().unwrap().push(t.clone());
        }
    }
    chunks
}

/// Drop a single leading newline (verbatim content begins on the line after
/// `\begin{...}`).
fn strip_leading_newline(s: &str) -> String {
    s.strip_prefix("\r\n")
        .or_else(|| s.strip_prefix('\n'))
        .unwrap_or(s)
        .to_string()
}

/// Split table body tokens on a top-level separator, honoring brace groups and
/// nested environments (a `\\` or `&` inside `\makecell{a\\b}` or a nested
/// `tabular` is not a separator). `is_sep` selects the separator token;
/// `keep_trailing_empty` controls whether an empty final chunk is emitted (cells
/// keep it — a trailing `&` is a real empty cell — rows drop it).
fn split_on(
    tokens: &[Token],
    keep_trailing_empty: bool,
    is_sep: impl Fn(&Token) -> bool,
) -> Vec<Vec<Token>> {
    let mut out = Vec::new();
    let mut current = Vec::new();
    let mut depth = 0i32;
    let mut env_depth = 0i32;
    for t in tokens {
        if t.is_cat(CatCode::BeginGroup) {
            depth += 1;
            current.push(t.clone());
        } else if t.is_cat(CatCode::EndGroup) {
            depth -= 1;
            current.push(t.clone());
        } else if matches!(t, Token::ControlSequence(n) if n == "begin") {
            env_depth += 1;
            current.push(t.clone());
        } else if matches!(t, Token::ControlSequence(n) if n == "end") {
            // Clamp at zero: a stray `\end` (e.g. macro residue landing in a
            // cell) must not drive the depth negative and suppress every later
            // separator.
            env_depth = (env_depth - 1).max(0);
            current.push(t.clone());
        } else if depth == 0 && env_depth == 0 && is_sep(t) {
            out.push(std::mem::take(&mut current));
        } else {
            current.push(t.clone());
        }
    }
    if keep_trailing_empty || !current.is_empty() {
        out.push(current);
    }
    out
}

/// Split table body tokens into rows on a top-level row terminator: `\\`, its
/// robust alias `\tabularnewline` (used when a column redefines `\\`, e.g. a
/// `\raggedright` `p{}` column), or the TeX primitive `\cr`. (`\newline` is an
/// intra-cell break, not a row terminator, so it is deliberately excluded.)
fn split_rows(tokens: &[Token]) -> Vec<Vec<Token>> {
    split_on(
        tokens,
        false,
        |t| matches!(t, Token::ControlSequence(n) if matches!(n.as_str(), "\\" | "tabularnewline" | "cr")),
    )
}

/// Split a row's tokens into cells on top-level `&`.
fn split_cells(tokens: &[Token]) -> Vec<Vec<Token>> {
    split_on(tokens, true, |t| t.is_cat(CatCode::AlignTab))
}

/// From index `start` (skipping leading spaces), if a `{...}` group opens, return
/// its inner tokens (braces removed, nesting honored) and the index just past the
/// closing `}`. Used to peel `\multicolumn`'s brace arguments from a cell.
fn peel_group(tokens: &[Token], start: usize) -> Option<(Vec<Token>, usize)> {
    let mut i = start;
    while tokens.get(i).is_some_and(|t| t.is_cat(CatCode::Space)) {
        i += 1;
    }
    if !tokens.get(i)?.is_cat(CatCode::BeginGroup) {
        return None;
    }
    let mut depth = 0i32;
    let mut inner = Vec::new();
    while i < tokens.len() {
        let t = &tokens[i];
        if t.is_cat(CatCode::BeginGroup) {
            depth += 1;
            if depth > 1 {
                inner.push(t.clone());
            }
        } else if t.is_cat(CatCode::EndGroup) {
            depth -= 1;
            if depth == 0 {
                return Some((inner, i + 1));
            }
            inner.push(t.clone());
        } else {
            inner.push(t.clone());
        }
        i += 1;
    }
    None
}

/// If `tokens` is a cell whose leading command is `\multicolumn{n}{spec}{content}`,
/// return the span `n` and the content tokens (any tokens after the group are kept
/// in the content so nothing is dropped). Returns `None` if the cell isn't a
/// multicolumn or the span isn't a non-negative integer; a `0` span is passed
/// through and clamped to a single column by the caller.
fn parse_leading_multicolumn(tokens: &[Token]) -> Option<(usize, Vec<Token>)> {
    let mut i = 0;
    while tokens.get(i).is_some_and(|t| t.is_cat(CatCode::Space)) {
        i += 1;
    }
    match tokens.get(i) {
        Some(Token::ControlSequence(n)) if n == "multicolumn" => {}
        _ => return None,
    }
    let (count_toks, i) = peel_group(tokens, i + 1)?;
    let (_spec, i) = peel_group(tokens, i)?;
    let (mut content, i) = peel_group(tokens, i)?;
    let count = reconstruct(&count_toks).trim().parse::<usize>().ok()?;
    content.extend_from_slice(&tokens[i..]);
    Some((count, content))
}

/// Horizontal-rule commands that take no argument.
const PLAIN_RULES: &[&str] = &["hline", "toprule", "midrule", "bottomrule"];
/// Rule commands that carry trailing argument groups — `\cmidrule(lr){N-M}`,
/// `\specialrule{h}{a}{b}`, `\addlinespace[len]` — which must be consumed with
/// them so their dimensions don't leak into a cell.
const ARG_RULES: &[&str] = &[
    "cline",
    "cmidrule",
    "specialrule",
    "addlinespace",
    "morecmidrules",
];

/// Skip from `i` (positioned at an opening delimiter's char) to just past the
/// matching close: `(`/`[` scan flat to `)`/`]`; `{` balances nested braces.
fn skip_delimited(tokens: &[Token], mut i: usize, open: char, close: char) -> usize {
    let mut depth = 0i32;
    while i < tokens.len() {
        match &tokens[i] {
            Token::Char(c, _) if *c == open => depth += 1,
            Token::Char(c, _) if *c == close => {
                depth -= 1;
                if depth == 0 {
                    return i + 1;
                }
            }
            _ => {}
        }
        i += 1;
    }
    i
}

/// From `i` (just past a rule command), skip its trailing `(...)`, `[...]`, and
/// `{...}` argument groups (and interleaving whitespace); return the next index.
fn skip_rule_args(tokens: &[Token], mut i: usize) -> usize {
    loop {
        match tokens.get(i) {
            Some(Token::Char(c, _)) if c.is_whitespace() => i += 1,
            Some(Token::Char('(', _)) => i = skip_delimited(tokens, i, '(', ')'),
            Some(Token::Char('[', _)) => i = skip_delimited(tokens, i, '[', ']'),
            Some(Token::Char('{', CatCode::BeginGroup)) => i = skip_delimited(tokens, i, '{', '}'),
            _ => return i,
        }
    }
}

/// Is this cell only horizontal rules / whitespace (e.g. `\hline`)?
fn is_rule_only(tokens: &[Token]) -> bool {
    let mut i = 0;
    while i < tokens.len() {
        match &tokens[i] {
            Token::Char(c, _) if c.is_whitespace() => i += 1,
            Token::ControlSequence(n) if PLAIN_RULES.contains(&n.as_str()) => i += 1,
            Token::ControlSequence(n) if ARG_RULES.contains(&n.as_str()) => {
                i = skip_rule_args(tokens, i + 1);
            }
            _ => return false,
        }
    }
    true
}

/// Strip leading horizontal-rule commands (`\hline`/`\cmidrule`/`\cline`/…,
/// including any arguments) from a cell's token list. In some tables these rules
/// appear on the same `\\`-delimited row-chunk as the content row that follows
/// them — notably a leading `\hline` before the first cell, which would
/// otherwise hide a `\multicolumn` from [`parse_leading_multicolumn`] and defeat
/// column-span padding. Mirrors the rule set [`is_rule_only`] recognizes.
fn strip_rule_prefix(mut tokens: Vec<Token>) -> Vec<Token> {
    let mut i = 0;
    loop {
        match tokens.get(i) {
            Some(Token::Char(c, _)) if c.is_whitespace() => i += 1,
            Some(Token::ControlSequence(n)) if PLAIN_RULES.contains(&n.as_str()) => i += 1,
            Some(Token::ControlSequence(n)) if ARG_RULES.contains(&n.as_str()) => {
                i = skip_rule_args(&tokens, i + 1);
            }
            _ => break,
        }
    }
    tokens.drain(..i);
    tokens
}

/// Wrap block content in an element with the given tag.
fn wrap(tag: &str, children: Vec<Node>) -> Node {
    let mut e = Element::new(tag);
    e.children = children;
    Node::Element(e)
}

/// Build an inline element with the given tag around `children`.
fn wrap_el(tag: &str, children: Vec<Node>) -> Element {
    let mut e = Element::new(tag);
    e.children = children;
    e
}

/// Format a counter value per a LaTeX counter style command (`\arabic`,
/// `\roman`, `\alph`, `\fnsymbol`, …). `\value` behaves as `\arabic` in text.
fn format_counter(style: &str, n: i32) -> String {
    match style {
        "roman" => to_roman(n).to_lowercase(),
        "Roman" => to_roman(n),
        "alph" => to_alpha(n, 'a'),
        "Alph" => to_alpha(n, 'A'),
        "fnsymbol" => {
            const SYMS: [&str; 9] = ["*", "†", "‡", "§", "¶", "‖", "**", "††", "‡‡"];
            SYMS.get((n - 1) as usize)
                .copied()
                .unwrap_or("")
                .to_string()
        }
        // "arabic", "value", and anything else: decimal.
        _ => n.to_string(),
    }
}

/// Roman numerals (uppercase); `n <= 0` yields an empty string.
fn to_roman(mut n: i32) -> String {
    const TABLE: [(i32, &str); 13] = [
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ];
    let mut out = String::new();
    for (v, s) in TABLE {
        while n >= v {
            out.push_str(s);
            n -= v;
        }
    }
    out
}

/// Alphabetic counter: 1→a/A, 2→b/B, … (wraps past 26 the way LaTeX doesn't,
/// but this is only reached for small counters).
fn to_alpha(n: i32, base: char) -> String {
    if n < 1 {
        return String::new();
    }
    let idx = ((n - 1) % 26) as u8;
    ((base as u8 + idx) as char).to_string()
}

// --- structure folding --------------------------------------------------

/// Fold a flat block list with `<section>` markers into nested sections.
fn fold_sections(blocks: Vec<Node>) -> Vec<Node> {
    let mut root: Vec<Node> = Vec::new();
    // Each open section: (absolute level for nesting, display level for headings).
    let mut stack: Vec<(i32, i32, Element)> = Vec::new();

    for block in blocks {
        if let Node::Element(e) = &block
            && e.name == "section"
        {
            let level = section_attr_level(e);
            while stack.last().is_some_and(|(l, _, _)| *l >= level) {
                let (_, _, sec) = stack.pop().unwrap();
                attach_section(sec, &mut stack, &mut root);
            }
            // The heading level to render is one deeper than the enclosing
            // section, regardless of the `level` gap between them: a `\paragraph`
            // (level 4) placed directly under a `\subsection` (level 2) renders
            // one step below it, not two — never skipping a heading level. A
            // top-level section keeps its absolute level (so the class anchor
            // holds: article `\section`→h2, book `\chapter`→h1), floored at 0 so a
            // shallow `\part` (level -1) still leaves its children room rather than
            // colliding with them at the h1 clamp. `level` is retained for the
            // resolve pass's numbering; backends read `depth` for the heading.
            let depth = match stack.last() {
                Some((_, parent_depth, _)) => parent_depth + 1,
                None => level.max(0),
            };
            let mut e = e.clone();
            e.attributes.push(("depth".into(), depth.to_string()));
            stack.push((level, depth, e));
            continue;
        }
        match stack.last_mut() {
            Some((_, _, sec)) => sec.push(block),
            None => root.push(block),
        }
    }
    while let Some((_, _, sec)) = stack.pop() {
        attach_section(sec, &mut stack, &mut root);
    }
    root
}

fn attach_section(sec: Element, stack: &mut [(i32, i32, Element)], root: &mut Vec<Node>) {
    match stack.last_mut() {
        Some((_, _, parent)) => parent.push(Node::Element(sec)),
        None => root.push(Node::Element(sec)),
    }
}

fn section_attr_level(e: &Element) -> i32 {
    e.attributes
        .iter()
        .find(|(k, _)| k == "level")
        .and_then(|(_, v)| v.parse().ok())
        .unwrap_or(1)
}

/// Fold a block list with marked item markers into item elements, each holding
/// the blocks that follow it.
fn fold_items(blocks: Vec<Node>) -> Vec<Node> {
    let mut items: Vec<Node> = Vec::new();
    let mut current: Option<Element> = None;
    for block in blocks {
        if let Node::Element(e) = &block
            && e.attributes.iter().any(|(k, _)| k == "marker")
        {
            if let Some(item) = current.take() {
                items.push(Node::Element(item));
            }
            let mut item = Element::new(&e.name);
            for (k, v) in &e.attributes {
                if k != "marker" {
                    item.attributes.push((k.clone(), v.clone()));
                }
            }
            item.children.extend(e.children.clone());
            current = Some(item);
            continue;
        }
        if let Some(item) = current.as_mut() {
            item.push(block);
        }
    }
    if let Some(item) = current.take() {
        items.push(Node::Element(item));
    }
    items
}

impl Engine<'_> {
    /// Emit a sectioning command as an empty `<section>` marker carrying its
    /// level and title; [`fold_sections`] later gives it a body.
    fn sectioning(&mut self, name: &str) -> Event {
        let starred = self.consume_star();
        let _ = self.grab_optional(); // short (running-head) title
        let title_tokens = self.grab_argument();
        let title_nodes = self.render_inline(title_tokens);
        let mut section = Element::new("section").attr("level", section_level(name).to_string());
        // Starred forms (`\section*`) are unnumbered; record it so the resolve
        // pass does not step the section counter for them.
        if starred {
            section = section.attr("starred", "1");
        }
        let mut title = Element::new("title");
        title.children = title_nodes;
        section.push(Node::Element(title));
        Event::Blocks(vec![Node::Element(section)])
    }

    /// Run the engine to completion, returning the `<document>` element and the
    /// [`Diagnostics`] describing any degradation (never silent).
    pub fn parse(mut self) -> (Element, Diagnostics) {
        let blocks = if self.has_document {
            self.consume_preamble();
            self.build_block(Stop::Environment("document".into()))
        } else {
            self.build_block(Stop::Eof)
        };
        let body = fold_sections(blocks);

        let mut root = Element::new("document");
        if let Some(title) = self.title.take() {
            root.push(wrap("title", title));
        }
        for author in std::mem::take(&mut self.authors) {
            root.push(wrap("author", author));
        }
        if let Some(date) = self.date.take() {
            root.push(wrap("date", date));
        }
        root.children.append(&mut self.preamble_footnotes);
        root.children.extend(body);
        attach_footnote_texts(&mut root, &self.footnote_texts);

        // End-of-document hooks (e.g. natbib assembling the bibliography).
        packages::finalize_all(&mut root, &mut self);

        // Second pass: now that the whole document exists, assign numbers to
        // sections/floats/theorems/equations and to cited references, then
        // resolve every `\ref`/`\cref`/`\cite` against those tables. This is the
        // `.aux`-equivalent phase — cross-references cannot be resolved during
        // the single forward build because a reference may precede its target.
        crate::resolve::resolve(&mut root, self.state);

        (root, std::mem::take(&mut self.diagnostics))
    }

    /// Process the preamble for side effects only — definitions and metadata —
    /// discarding all typeset output, until `\begin{document}`.
    fn consume_preamble(&mut self) {
        while let Some(t) = self.next_raw() {
            let Token::ControlSequence(name) = &t else {
                continue; // discard preamble characters and groups
            };
            if name == "begin" {
                let env = self.grab_argument_text();
                if env == "document" {
                    return;
                }
                // A preamble environment (e.g. a package's setup): skip it.
                let _ = self.read_environment_body_raw(&env);
                continue;
            }
            let name = name.clone();
            if self.is_user_macro(&name) {
                self.expand_macro(&name);
                continue;
            }
            // Reuse the command handler for definitions and metadata. Preamble
            // footnote text is visible content, even when its mark is a literal
            // symbol in an author list, so retain that one output form.
            if let Event::Inline(nodes) = self.run_command(&name, &Stop::Eof) {
                self.preamble_footnotes.extend(nodes.into_iter().filter(|node| {
                    matches!(node, Node::Element(element) if element.name == "footnote-text")
                }));
            }
        }
    }
}

fn attach_footnote_texts(element: &mut Element, texts: &HashMap<String, Vec<Node>>) {
    let mut marks = HashSet::new();
    collect_footnote_marks(&element.children, &mut marks);
    resolve_footnotes(&mut element.children, texts, &marks);
}

fn collect_footnote_marks(nodes: &[Node], marks: &mut HashSet<String>) {
    for node in nodes {
        let Node::Element(element) = node else {
            continue;
        };
        if element.name == "footnote-mark" {
            if let Some(key) = element
                .attributes
                .iter()
                .find(|(name, _)| name == "key")
                .map(|(_, value)| value.clone())
            {
                marks.insert(key);
            }
        } else {
            collect_footnote_marks(&element.children, marks);
        }
    }
}

fn resolve_footnotes(
    nodes: &mut Vec<Node>,
    texts: &HashMap<String, Vec<Node>>,
    marks: &HashSet<String>,
) {
    let mut resolved = Vec::with_capacity(nodes.len());
    for node in nodes.drain(..) {
        let Node::Element(mut element) = node else {
            resolved.push(node);
            continue;
        };
        let key = element
            .attributes
            .iter()
            .find(|(name, _)| name == "key")
            .map(|(_, value)| value.clone())
            .unwrap_or_default();
        match element.name.as_str() {
            "footnote-mark" => {
                if let Some(text) = texts.get(&key) {
                    element.name = "footnote".into();
                    element.children = text.clone();
                } else {
                    element.name = "superscript".into();
                    element.attributes.clear();
                    element.children = vec![Node::text(footnote_label(&key))];
                }
                resolved.push(Node::Element(element));
            }
            "footnote-text" if marks.contains(&key) => {}
            "footnote-text" => {
                element.name = "footnote".into();
                element.children = texts.get(&key).cloned().unwrap_or_default();
                resolved.push(Node::Element(element));
            }
            _ => {
                resolve_footnotes(&mut element.children, texts, marks);
                resolved.push(Node::Element(element));
            }
        }
    }
    *nodes = resolved;
}

fn footnote_label(key: &str) -> &str {
    key.split_once(':').map(|(_, label)| label).unwrap_or(key)
}

/// The nesting depth of a sectioning command (smaller is higher-level).
pub(crate) fn section_level(name: &str) -> i32 {
    match name {
        "part" => -1,
        "chapter" => 0,
        "section" => 1,
        "subsection" => 2,
        "subsubsection" => 3,
        "paragraph" => 4,
        "subparagraph" => 5,
        _ => 2,
    }
}

/// The Unicode combining mark for a text-accent command, or `None` if the
/// command is not an accent.
fn accent_mark(cmd: &str) -> Option<char> {
    Some(match cmd {
        "'" => '\u{0301}',  // acute
        "`" => '\u{0300}',  // grave
        "^" => '\u{0302}',  // circumflex
        "~" => '\u{0303}',  // tilde
        "\"" => '\u{0308}', // diaeresis / umlaut
        "=" => '\u{0304}',  // macron
        "." => '\u{0307}',  // dot above
        "u" => '\u{0306}',  // breve
        "v" => '\u{030C}',  // caron / háček
        "H" => '\u{030B}',  // double acute
        "r" => '\u{030A}',  // ring above
        "c" => '\u{0327}',  // cedilla
        "k" => '\u{0328}',  // ogonek
        "b" => '\u{0331}',  // macron below
        "d" => '\u{0323}',  // dot below
        "t" => '\u{0361}',  // double inverted breve (tie)
        _ => return None,
    })
}

/// Apply a text accent to its base. Precomposed Latin letters are returned where
/// one exists (so `\'e` yields "é", not "e" + a separate combining mark);
/// otherwise the base is emitted followed by the combining mark, which renders
/// correctly even without a precomposed form.
fn apply_accent(base: &str, cmd: &str) -> String {
    let Some(mark) = accent_mark(cmd) else {
        return base.to_string();
    };
    let mut chars = base.chars();
    match (chars.next(), chars.next()) {
        (Some(c), None) => precompose(c, cmd)
            .map(|p| p.to_string())
            .unwrap_or_else(|| format!("{c}{mark}")),
        // Empty base (accent used alone) or multi-char: fall back to combining.
        (None, _) => mark.to_string(),
        _ => format!("{base}{mark}"),
    }
}

/// Precomposed Latin letter for a (base, accent) pair, covering the accented
/// letters that occur in names and prose. Uncovered pairs fall back to a
/// combining mark in [`apply_accent`].
fn precompose(base: char, cmd: &str) -> Option<char> {
    Some(match (cmd, base) {
        ("'", 'a') => 'á',
        ("'", 'e') => 'é',
        ("'", 'i') => 'í',
        ("'", 'o') => 'ó',
        ("'", 'u') => 'ú',
        ("'", 'y') => 'ý',
        ("'", 'n') => 'ń',
        ("'", 'c') => 'ć',
        ("'", 's') => 'ś',
        ("'", 'z') => 'ź',
        ("'", 'r') => 'ŕ',
        ("'", 'l') => 'ĺ',
        ("'", 'A') => 'Á',
        ("'", 'E') => 'É',
        ("'", 'I') => 'Í',
        ("'", 'O') => 'Ó',
        ("'", 'U') => 'Ú',
        ("'", 'Y') => 'Ý',
        ("'", 'N') => 'Ń',
        ("'", 'C') => 'Ć',
        ("'", 'S') => 'Ś',
        ("'", 'Z') => 'Ź',
        ("`", 'a') => 'à',
        ("`", 'e') => 'è',
        ("`", 'i') => 'ì',
        ("`", 'o') => 'ò',
        ("`", 'u') => 'ù',
        ("`", 'A') => 'À',
        ("`", 'E') => 'È',
        ("`", 'I') => 'Ì',
        ("`", 'O') => 'Ò',
        ("`", 'U') => 'Ù',
        ("^", 'a') => 'â',
        ("^", 'e') => 'ê',
        ("^", 'i') => 'î',
        ("^", 'o') => 'ô',
        ("^", 'u') => 'û',
        ("^", 'A') => 'Â',
        ("^", 'E') => 'Ê',
        ("^", 'I') => 'Î',
        ("^", 'O') => 'Ô',
        ("^", 'U') => 'Û',
        ("\"", 'a') => 'ä',
        ("\"", 'e') => 'ë',
        ("\"", 'i') => 'ï',
        ("\"", 'o') => 'ö',
        ("\"", 'u') => 'ü',
        ("\"", 'y') => 'ÿ',
        ("\"", 'A') => 'Ä',
        ("\"", 'E') => 'Ë',
        ("\"", 'I') => 'Ï',
        ("\"", 'O') => 'Ö',
        ("\"", 'U') => 'Ü',
        ("~", 'a') => 'ã',
        ("~", 'o') => 'õ',
        ("~", 'n') => 'ñ',
        ("~", 'e') => 'ẽ',
        ("~", 'i') => 'ĩ',
        ("~", 'u') => 'ũ',
        ("~", 'A') => 'Ã',
        ("~", 'O') => 'Õ',
        ("~", 'N') => 'Ñ',
        ("c", 'c') => 'ç',
        ("c", 'C') => 'Ç',
        ("c", 's') => 'ş',
        ("c", 'S') => 'Ş',
        ("=", 'a') => 'ā',
        ("=", 'e') => 'ē',
        ("=", 'i') => 'ī',
        ("=", 'o') => 'ō',
        ("=", 'u') => 'ū',
        ("=", 'A') => 'Ā',
        ("=", 'O') => 'Ō',
        ("v", 'c') => 'č',
        ("v", 's') => 'š',
        ("v", 'z') => 'ž',
        ("v", 'r') => 'ř',
        ("v", 'C') => 'Č',
        ("v", 'S') => 'Š',
        ("v", 'Z') => 'Ž',
        ("v", 'n') => 'ň',
        ("r", 'a') => 'å',
        ("r", 'A') => 'Å',
        ("r", 'u') => 'ů',
        ("u", 'g') => 'ğ',
        ("u", 'a') => 'ă',
        ("u", 'o') => 'ŏ',
        ("H", 'o') => 'ő',
        ("H", 'u') => 'ű',
        _ => return None,
    })
}

/// Map a symbol-producing control sequence to its text, if known.
fn symbol(name: &str) -> Option<&'static str> {
    Some(match name {
        "LaTeX" => "LaTeX",
        "TeX" => "TeX",
        "LaTeXe" => "LaTeX2e",
        "ldots" | "dots" | "textellipsis" => "\u{2026}",
        "textbackslash" => "\\",
        "textasciitilde" => "~",
        "textasciicircum" => "^",
        "textbar" => "|",
        "textless" => "<",
        "textgreater" => ">",
        "copyright" | "textcopyright" => "\u{00A9}",
        "textregistered" => "\u{00AE}",
        "texttrademark" => "\u{2122}",
        "dag" | "dagger" => "\u{2020}",
        "ddag" | "ddagger" => "\u{2021}",
        "S" | "sectionmark" => "\u{00A7}",
        "P" => "\u{00B6}",
        "pounds" | "textsterling" => "\u{00A3}",
        "texteuro" | "euro" => "\u{20AC}",
        "textdegree" => "\u{00B0}",
        "%" => "%",
        "&" => "&",
        "#" => "#",
        "_" => "_",
        "$" => "$",
        "{" => "{",
        "}" => "}",
        "ae" => "\u{00E6}",
        "AE" => "\u{00C6}",
        "oe" => "\u{0153}",
        "OE" => "\u{0152}",
        "aa" => "\u{00E5}",
        "AA" => "\u{00C5}",
        "ss" => "\u{00DF}",
        "o" => "\u{00F8}",
        "O" => "\u{00D8}",
        "l" => "\u{0142}",
        "L" => "\u{0141}",
        "i" => "\u{0131}",                  // dotless i (ı)
        "j" => "\u{0237}",                  // dotless j (ȷ)
        "textexclamdown" => "\u{00A1}",     // ¡
        "textquestiondown" => "\u{00BF}",   // ¿
        "textbullet" => "\u{2022}",         // •
        "textperiodcentered" => "\u{00B7}", // ·
        "guillemotleft" => "\u{00AB}",      // «
        "guillemotright" => "\u{00BB}",     // »
        "textcent" => "\u{00A2}",           // ¢
        "-" => "",                          // discretionary hyphen
        _ => return None,
    })
}

/// Map a spacing control sequence to the text it contributes (often a single
/// space, sometimes nothing).
fn spacing(name: &str) -> Option<&'static str> {
    // A control sequence whose name is whitespace is a control space / control
    // newline (`\ ` at end of line): a space, not an unknown command.
    if !name.is_empty() && name.chars().all(char::is_whitespace) {
        return Some(" ");
    }
    Some(match name {
        " " | "quad" | "qquad" | "," | ";" | ":" | "!" | "thinspace" | "enspace" | "hfill"
        | "hspace" | "hspace*" | "space" => " ",
        "vspace" | "vspace*" | "noindent" | "indent" | "smallskip" | "medskip" | "bigskip"
        | "newpage" | "clearpage" | "cleardoublepage" | "centering" | "raggedright"
        | "raggedleft" | "normalfont" | "normalsize" | "small" | "footnotesize" | "large"
        | "Large" | "LARGE" | "huge" | "Huge" | "tiny" | "scriptsize" | "protect" | "relax"
        | "ignorespaces" | "allowbreak" | "linebreak" | "nolinebreak"
        // Table rules (consumed inside tables; produce no content elsewhere) and
        // bibliography spacers.
        | "hline" | "toprule" | "midrule" | "bottomrule" | "newblock" => "",
        _ => return None,
    })
}

/// Map a Zapf Dingbats code (pifont `\ding{N}`) to its Unicode glyph. Only the
/// marks that show up in real papers are mapped — checks and crosses, used for
/// yes/no cells (`\cmark`/`\xmark`); anything else contributes nothing rather
/// than leaking the raw number.
fn ding_glyph(code: &str) -> &'static str {
    match code {
        "51" => "✓", // check mark
        "52" => "✔", // heavy check mark
        "55" => "✗", // ballot X
        "56" => "✘", // heavy ballot X
        _ => "",
    }
}
