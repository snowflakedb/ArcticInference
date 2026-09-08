---
name: nvidia-binary-utilities
description: Use local NVIDIA CUDA Binary Utilities documentation for CUDA binary inspection, disassembly, demangling, pruning, and SASS instruction reference tasks. Trigger when Codex needs to choose or explain cuobjdump, nvdisasm, cu++filt, nvprune commands/options, inspect cubin/fatbin/ELF/PTX contents, decode or compare SASS instructions, use nvdisasm JSON output, or reason about CUDA binary formats and command option notation. Trigger when you need to verify that a hardware feature is actually used or what is actually running on the GPU after compiling a kernel. 
---

# NVIDIA Binary Utilities

Use this skill to ground CUDA binary tooling advice in the local CUDA Binary Utilities docs bundled under `references/`.

## Workflow

1. Identify the target task: host-binary extraction, cubin disassembly, SASS instruction lookup, C++ symbol demangling, architecture pruning, or machine-readable disassembly.
2. Read or search the relevant reference before giving exact flags, option names, architecture values, output formats, or instruction details.
3. Prefer minimal commands that make the input type explicit: host executable/object/archive/fatbin for `cuobjdump`, standalone cubin for `nvdisasm`, mangled symbols for `cu++filt`, and object/library pruning for `nvprune`.
4. When both disassemblers could apply, choose `cuobjdump` for extracting PTX/cubin from host binaries and choose `nvdisasm` for richer cubin-only disassembly, control-flow output, advanced displays, or JSON output.
5. Call out command-option behavior when it matters: boolean options take no argument, single-value options use the rightmost repeated value, and list options append values while ignoring duplicates.

## References

- `references/cuobjdump.md`: Read for extracting and displaying PTX, ELF sections, symbols, relocations, and SASS from host binaries, objects, archives, fatbins, and cubins.
- `references/nvdisasm.md`: Read for cubin disassembly, control-flow output, advanced display options, JSON schema/output, and nvdisasm-specific command-line options.
- `references/instruction_set_reference.md`: Read for Turing, Ampere/Ada, Hopper, and Blackwell SASS instruction mnemonics and instruction descriptions.
- `references/cuplusplusfilt.md`: Read for demangling CUDA/C++ symbols with `cu++filt`, including library availability and command options.
- `references/nvprune.md`: Read for pruning CUDA object files or static libraries to retain selected GPU architectures.

## Search Cues

Use `rg` inside `references/` instead of loading whole files when possible:

```bash
rg -n "cuobjdump|nvdisasm|cu\+\+filt|nvprune|cubin|fatbin|ELF|PTX|SASS|json|control flow|SM[0-9]+|Blackwell|Hopper" references/
```

For instruction lookup, search the instruction reference first by exact mnemonic, then by architecture or category:

```bash
rg -n "\bLDG\b|\bSTG\b|\bIMAD\b|\bMMA\b|\bBAR\b|\bEXIT\b" references/instruction_set_reference.md
```

## Command Patterns

Use these as starting points and verify exact flags in the references for the requested tool/version:

```bash
cuobjdump --dump-ptx app_or_object
cuobjdump --dump-sass app_or_object
cuobjdump --dump-elf app_or_object
nvdisasm --print-code standalone.cubin
nvdisasm --print-line-info standalone.cubin
nvdisasm --format json standalone.cubin
cu++filt _Z...
nvprune --generate-code arch=compute_90,code=sm_90 input.a -o pruned.a
```

Do not imply `nvdisasm` can consume host executables directly; use `cuobjdump` to extract cubins first when the input is not a standalone cubin.
