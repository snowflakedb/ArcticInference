---
name: texmark
description: Convert LaTeX projects and arXiv papers into clean Markdown with the texmark CLI. Use when the user provides a .tex file, LaTeX project, arXiv ID, or arXiv URL and wants the paper converted, read, summarized, searched, or analyzed. Prefer bundle output for agent work and use standalone output only when one portable Markdown file is required.
---

# texmark

Use the `texmark` CLI directly.

## Convert an arXiv paper

Prefer a bundle so `main.md` stays small and figures remain separate:

```sh
texmark --arxiv 1706.03762 -o paper
# paper/main.md
# paper/figures/
```

The input may be a versioned ID or an arXiv URL.

## Convert a local LaTeX project

Pass the project's main TeX file:

```sh
texmark path/to/main.tex -o paper
# paper/main.md
# paper/figures/
```

texmark resolves `\input`, `\include`, bibliographies, and figures relative to
the main file.

## Read the result

Read `paper/main.md`. Inspect `paper/figures/` only when the task requires
understanding a figure.

Check the command's exit status and stderr. Treat `warning:` messages and
`texmark_truncated: true` as evidence that content may be missing or degraded.
Say so instead of presenting the conversion as complete.

## Standalone output

Use standalone output only when the user needs one portable file:

```sh
texmark --arxiv 1706.03762 --standalone -o paper.md
```

Standalone files inline figures as base64 data URIs and may be very large. Do
not load the whole file into context before replacing those payloads:

```sh
sed -E 's#data:image/[^)]*#DATA_URI#g' paper.md > paper-lite.md
```

Read `paper-lite.md`, while retaining `paper.md` as the portable artifact.

## If texmark is not installed

From the texmark repository, use the release binary through Cargo:

```sh
cargo run --release -- --arxiv 1706.03762 -o paper
```
