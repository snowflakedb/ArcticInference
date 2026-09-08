# texmark

Convert LaTeX research papers into clean Markdown for agents.

## Install

```sh
# Install the agent skill.
gh skill install snowflakedb/ArcticInference texmark

# Install the CLI from this repository.
cargo install --git https://github.com/snowflakedb/ArcticInference texmark
```

or from the source:
```bash
git clone https://github.com/snowflakedb/ArcticInference.git
cd ArcticInference/projects/kernel_generation/tools/texmark
cargo install --path .
```

## CLI

```sh
# Local bundle.
texmark paper.tex
# writes paper/main.md and paper/figures/
# preserves PDF figures and converts EPS figures to PNG

# Override the bundle directory.
texmark paper.tex -o converted
# writes converted/main.md and converted/figures/

# arXiv bundle named after the paper.
texmark --arxiv 1706.03762
# writes attention-is-all-you-need/main.md and
# attention-is-all-you-need/figures/

# Standalone Markdown with inlined images.
texmark paper.tex --standalone
# writes paper.md

# Override the standalone filename.
texmark --arxiv 1706.03762 --standalone -o attention.md
# writes attention.md

# Override the default 120-second timeout.
texmark paper.tex --timeout 30
```

## Acknowledgements

Thank you to arXiv for use of its open access interoperability.

texmark's design is informed by [LaTeXML](https://github.com/brucemiller/LaTeXML).
texmark is an independent Rust implementation and contains no code ported from
LaTeXML.
