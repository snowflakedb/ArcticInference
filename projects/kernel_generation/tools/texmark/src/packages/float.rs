//! Native configuration commands for the `float` package.

use super::Package;
use crate::engine::{Engine, Event};

pub struct Float;

impl Package for Float {
    fn package_names(&self) -> &[&str] {
        &["float"]
    }

    fn commands(&self) -> &[&str] {
        &["floatname", "floatstyle", "newfloat"]
    }

    fn command(&self, name: &str, engine: &mut Engine) -> Event {
        let arguments = match name {
            "floatstyle" => 1,
            "floatname" => 2,
            _ => 3,
        };
        for _ in 0..arguments {
            let _ = engine.grab_argument();
        }
        if name == "newfloat" {
            let _ = engine.grab_optional();
        }
        Event::Inline(vec![])
    }
}
