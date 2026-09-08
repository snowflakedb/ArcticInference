//! ANSI color/style escape codes for the terminal UI, shared by the streaming
//! renderer and the workspace tools that echo output as it arrives.

pub const RESET: &str = "\x1b[0m";
pub const BOLD: &str = "\x1b[1m";
pub const DIM: &str = "\x1b[2m";
pub const RED: &str = "\x1b[31m";
pub const GREEN: &str = "\x1b[32m";
pub const YELLOW: &str = "\x1b[33m";
pub const MAGENTA: &str = "\x1b[35m";
pub const WHITE: &str = "\x1b[37m";
