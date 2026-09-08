//! Native boolean conditionals for the `ifthen` package.

use super::Package;
use crate::engine::{Engine, Event, reconstruct};
use crate::state::State;
use crate::tokenizer::{balanced_group, control_word};

pub struct Ifthen;

impl Package for Ifthen {
    fn package_names(&self) -> &[&str] {
        &["ifthen"]
    }

    fn commands(&self) -> &[&str] {
        &["boolean", "equal", "ifthenelse", "newboolean", "setboolean"]
    }

    fn command(&self, name: &str, engine: &mut Engine) -> Event {
        match name {
            "newboolean" => {
                let name = engine.grab_argument_text();
                set_boolean(engine.state_mut(), &name, false);
            }
            "setboolean" => {
                let name = engine.grab_argument_text();
                let value = engine.grab_argument_text();
                set_boolean(engine.state_mut(), &name, value.trim() == "true");
            }
            "ifthenelse" => {
                let condition = engine.grab_argument();
                let condition = reconstruct(&engine.expand_tokens(condition));
                let then_branch = engine.grab_argument();
                let else_branch = engine.grab_argument();
                match evaluate(&condition, engine.state()) {
                    Some(true) => engine.splice_tokens(&then_branch),
                    Some(false) => engine.splice_tokens(&else_branch),
                    None => {
                        engine.report_dropped_command("ifthenelse");
                        engine.splice_tokens(&then_branch);
                    }
                }
            }
            "boolean" | "equal" => {
                let _ = engine.grab_argument();
                if name == "equal" {
                    let _ = engine.grab_argument();
                }
            }
            _ => {}
        }
        Event::Inline(vec![])
    }
}

fn boolean_key(name: &str) -> String {
    format!("ifthen:boolean:{}", name.trim())
}

fn set_boolean(state: &mut State, name: &str, value: bool) {
    state
        .package_state
        .insert(boolean_key(name), value.to_string());
}

fn evaluate(condition: &str, state: &State) -> Option<bool> {
    let condition = condition.trim();
    let Some((command, end)) = control_word(condition, 0) else {
        return match condition {
            "true" => Some(true),
            "false" => Some(false),
            _ => None,
        };
    };
    match command {
        "boolean" => {
            let (name, end) = balanced_group(condition, end, b'{', b'}')?;
            if !condition[end..].trim().is_empty() {
                return None;
            }
            state
                .package_state
                .get(&boolean_key(name))
                .and_then(|value| value.parse().ok())
        }
        "equal" => {
            let (left, end) = balanced_group(condition, end, b'{', b'}')?;
            let (right, end) = balanced_group(condition, end, b'{', b'}')?;
            condition[end..].trim().is_empty().then_some(left == right)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn equal_expands_macros_and_booleans_are_package_scoped() {
        let markdown = crate::latex_to_markdown(
            r"\usepackage{ifthen}\def\x{a}\newboolean{ready}\setboolean{ready}{true}
              \begin{document}
              \ifthenelse{\equal{\x}{a}}{equal}{wrong}
              \ifthenelse{\boolean{ready}}{ ready}{wrong}
              \end{document}",
        );
        assert!(
            markdown.contains("equal") && markdown.contains("ready"),
            "{markdown}"
        );
        assert!(!markdown.contains("wrong"), "{markdown}");
    }

    #[test]
    fn compound_conditions_are_not_partially_evaluated() {
        let state = crate::state::State::new();
        assert_eq!(
            super::evaluate(r"\equal{a}{b}\or\equal{c}{c}", &state),
            None
        );
    }
}
