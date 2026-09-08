//! Native configuration commands for `cleveref`.

use super::Package;
use crate::engine::{Engine, Event};

pub struct Cleveref;

impl Package for Cleveref {
    fn package_names(&self) -> &[&str] {
        &["cleveref"]
    }

    fn commands(&self) -> &[&str] {
        &["crefname", "Crefname"]
    }

    fn command(&self, _name: &str, engine: &mut Engine) -> Event {
        let _ = engine.grab_argument();
        let _ = engine.grab_argument();
        let _ = engine.grab_argument();
        Event::Inline(vec![])
    }
}
