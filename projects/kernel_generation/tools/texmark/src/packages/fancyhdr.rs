//! Native ownership of the presentation-only `fancyhdr` package.

use super::Package;
use crate::engine::{Engine, Event};

pub struct Fancyhdr;

impl Package for Fancyhdr {
    fn package_names(&self) -> &[&str] {
        &["fancyhdr"]
    }

    fn commands(&self) -> &[&str] {
        &[
            "fancyhf",
            "fancyhead",
            "fancyfoot",
            "lhead",
            "chead",
            "rhead",
            "lfoot",
            "cfoot",
            "rfoot",
            "fancypagestyle",
        ]
    }

    fn command(&self, name: &str, engine: &mut Engine) -> Event {
        if name == "fancypagestyle" {
            let _ = engine.grab_argument();
            let _ = engine.grab_argument();
        } else {
            let _ = engine.grab_optional();
            let _ = engine.grab_argument();
        }
        Event::Inline(vec![])
    }
}
