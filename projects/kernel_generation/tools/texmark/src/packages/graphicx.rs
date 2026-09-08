//! Native rendering for the `graphicx` package.
//!
//! Handles `\includegraphics[opts]{src}` → `<image src="...">` and records
//! `\graphicspath{{dir/}...}` so the image resolver can find figures that live
//! in a search directory rather than next to the main file.

use super::Package;
use crate::engine::{Engine, Event};
use crate::node::{Element, Node};

pub struct Graphicx;

impl Package for Graphicx {
    fn package_names(&self) -> &[&str] {
        &["graphics", "graphicx"]
    }

    fn commands(&self) -> &[&str] {
        &["includegraphics", "graphicspath"]
    }

    fn command(&self, name: &str, e: &mut Engine) -> Event {
        match name {
            // \graphicspath{{dir1/}{dir2/}} — the directories LaTeX prepends to a
            // graphics filename when searching. Record them (relative to the main
            // file) for the image resolver; produces no output.
            "graphicspath" => {
                let arg = e.grab_argument_text();
                let dirs = parse_dirs(&arg);
                if !dirs.is_empty() {
                    e.state_mut()
                        .package_state
                        .insert("graphicx:path".into(), dirs.join(","));
                }
                Event::Inline(vec![])
            }
            // \includegraphics[opts]{src}
            _ => {
                let _ = e.grab_optional(); // sizing/placement options
                let src = e.grab_expanded_argument_text();
                Event::Inline(vec![Node::Element(Element::new("image").attr("src", src))])
            }
        }
    }
}

/// Extract the directories from a `\graphicspath` argument, which is a braced
/// list of braced paths — the argument text looks like `{a/}{b/}`. Returns each
/// inner path (`a/`, `b/`).
fn parse_dirs(arg: &str) -> Vec<String> {
    let mut dirs = Vec::new();
    let mut cur = String::new();
    let mut depth = 0u32;
    for c in arg.chars() {
        match c {
            '{' => {
                if depth > 0 {
                    cur.push(c);
                }
                depth += 1;
            }
            '}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    if !cur.is_empty() {
                        dirs.push(std::mem::take(&mut cur));
                    }
                } else {
                    cur.push(c);
                }
            }
            _ if depth >= 1 => cur.push(c),
            _ => {}
        }
    }
    dirs
}

#[cfg(test)]
mod tests {
    use super::parse_dirs;
    use crate::latex_to_xml;

    #[test]
    fn parses_braced_dir_list() {
        assert_eq!(parse_dirs("{Figures/}"), ["Figures/"]);
        assert_eq!(parse_dirs("{a/}{b/}"), ["a/", "b/"]);
        assert_eq!(parse_dirs(""), Vec::<String>::new());
    }

    #[test]
    fn expands_figure_path_macros_and_toggles() {
        let xml = latex_to_xml(
            r"\newcommand{\impath}[1]{figures/#1}\newtoggle{hq}\togglefalse{hq}
              \begin{document}
              \includegraphics{\impath{sample}}
              \includegraphics{\iftoggle{hq}{large.pdf}{small.jpg}}
              \end{document}",
        );
        assert!(xml.contains("src=\"figures/sample\""), "{xml}");
        assert!(xml.contains("src=\"small.jpg\""), "{xml}");
    }
}
