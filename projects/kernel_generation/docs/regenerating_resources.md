# Regenerating the agent's docs and skills

Reference for `agent_ressources/.generate/generate.sh`, which rebuilds the
agent-facing tree under `agent_ressources/`. The README's *Generate docs/skills*
section has the mounts and the prerequisites; this covers what the script writes and
how to change it.

```sh
./agent_ressources/.generate/generate.sh
```

The script is one line per document: to refresh a single doc run just its line, and
to add one copy a line. Its tool guards are not decoration — the papers step clears
its output directories before writing, so a missing `texmark` discovered halfway
through would leave them empty.

## What it writes

Paths are relative to `agent_ressources/`.

| Regenerated | From |
|---|---|
| `docs/nvidia/*.md` | the CUDA and Nsight Compute doc sites, recovered to Markdown by `.generate/nvidia` |
| `docs/nvidia/cuda_binary_utilities/` | the same page, split per tool |
| `docs/nvidia/blackwell_ultra_blog.md` | a developer-blog post, via `.generate/nvidia_blog` |
| `docs/nvidia/nvidia-blackwell-*.pdf` | the architecture brief and Blackwell Ultra datasheet, fetched as PDFs |
| `docs/papers/<Name>/paper.md` | arXiv LaTeX source, via texmark |
| `skills/*/references/*.md` | copies of the `docs/` files above |

Hand-written, and so untouched by a run: every `SKILL.md`, plus
`docs/nvidia/torch_nvrtc.md` and `docs/nvidia/how_to_read_ptx_docs.md`. Don't
hand-edit anything in the table above — the next run overwrites it.

## Adding or retargeting a document

Copy the line that matches its kind and give it a new URL and output path: `doc` for
a CUDA or Nsight docs site, `blog` for a developer-blog post, `pdf` for a published
PDF, `texmark --arxiv` for a paper. To point one at a newer upstream version, edit
that line's URL in place.

Keep PDF links **unsigned**. A signed URL carries an expiry and silently rots, which
is how the previous Blackwell architecture brief link died.

Skill references need no edit here. They are *copies* rather than links, because
each skill is mounted on its own and has to be self-contained; the script refreshes
every copy that already exists, so adding one to a skill is enough.

## Reviewing a run

Everything the script writes is committed, so `git diff --stat` after a run is the
changelog and `git checkout` is the undo. Review the diff rather than committing it
blind.

Be suspicious of any doc that got much **smaller**. Upstream occasionally degrades a
page's Markdown mirror, and the parser then falls back to rendering its HTML —
usually equivalent, but worth a look. Compare word counts rather than bytes: table
formatting alone can swing the byte count by 10% with no change in content.

A papers diff with no upstream change means texmark itself changed — its output
depends on which binary is installed, and it is deliberately unpinned. Check
`texmark --version` before assuming a paper was revised.

Failure modes are loud by design: a renamed doc slug 404s, a renumbered
binary-utilities section raises `RuntimeError`, and an undetectable page theme raises
`StyleDetectionError` — whose message tells you to pass `--style`, which is the one
time that is the right answer.
