//! Native configuration commands for `enumitem`.

use super::Package;
use crate::engine::{Engine, Event};

pub struct Enumitem;

impl Package for Enumitem {
    fn package_names(&self) -> &[&str] {
        &["enumitem"]
    }

    fn commands(&self) -> &[&str] {
        &["setlist"]
    }

    fn command(&self, _name: &str, engine: &mut Engine) -> Event {
        let _ = engine.grab_optional();
        let _ = engine.grab_optional();
        let _ = engine.grab_argument();
        Event::Inline(vec![])
    }
}
