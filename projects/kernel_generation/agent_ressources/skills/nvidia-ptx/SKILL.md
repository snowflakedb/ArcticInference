---
name: nvidia-ptx
description: Use local NVIDIA PTX ISA and Inline PTX Assembly documentation for writing, reading, debugging, or explaining PTX code, PTX instruction syntax and qualifiers, state spaces, register/type rules, memory consistency, atomics, barriers, tensor-core PTX instructions, asynchronous copy/TMA, special registers, directives, target ISA versions, and CUDA inline asm constraints or pitfalls. Trigger when Codex needs exact PTX semantics, instruction variants, operands, qualifiers, or inline PTX integration details.
---

# NVIDIA PTX ISA

Use this skill to ground PTX ISA and CUDA inline assembly work in local NVIDIA documentation bundled under `references/`.

## Workflow

1. Identify whether the task is PTX ISA semantics, an individual instruction, PTX syntax/directives, memory/state spaces, target/version support, or CUDA inline `asm()` integration.
2. Read `references/how_to_read_ptx_docs.md` for PTX notation reminders when interpreting instruction forms, optional qualifiers, or grouped variants.
3. Search `references/ptx.md` before giving exact instruction syntax, operand order, type constraints, qualifiers, PTX ISA version, target architecture requirements, or memory consistency semantics.
4. Search `references/inline_ptx_assembly.md` before writing CUDA C++ `asm()` blocks, constraints, output/input operands, clobber assumptions, or guidance about pitfalls.
5. Distinguish PTX from SASS: use this skill for virtual PTX ISA and inline PTX source; use `nvidia-binary-utilities` for generated SASS/cubin disassembly and hardware instruction reference checks.
6. When performance depends on lowering, recommend verifying generated SASS with CUDA binary utilities rather than assuming PTX maps one-to-one to machine instructions.

## References

- `references/ptx.md`: Local NVIDIA Parallel Thread Execution ISA reference. Read for PTX syntax, directives, state spaces, types, instructions, memory model, special registers, and architecture/version notes.
- `references/inline_ptx_assembly.md`: Read for CUDA C++ inline `asm()` syntax, operand constraints, escaping, volatile usage, memory clobbers, namespace issues, and error checking.
- `references/how_to_read_ptx_docs.md`: Read for local notes on interpreting PTX instruction notation and variant blocks.

## Search Cues

Use these from inside this skill directory, or adjust the path if working from the repository root:

```bash
rg -n "^#{1,4} |\.version|\.target|\.address_size|\.entry|\.func|\.reg|\.param|\.shared|\.global|\.local|\.const" references/ptx.md
rg -n "state space|generic address|\.global|\.shared|\.local|\.const|\.param|cvta|isspacep|mapa|getctarank" references/ptx.md
rg -n "memory consistency|weak|strong|relaxed|acquire|release|\.sem|\.scope|fence|membar|atom|red" references/ptx.md
rg -n "barrier|bar\.sync|mbarrier|elect|redux|shfl|vote|match|activemask|lanemask|warpsize" references/ptx.md
rg -n "mma|wgmma|wmma|tcgen05|ldmatrix|stmatrix|tensor|satfinite|kind|block_scale" references/ptx.md
rg -n "cp\.async|cp\.reduce\.async|bulk|TMA|tensormap|prefetch|createpolicy|discard|applypriority" references/ptx.md
rg -n "special register|%tid|%ntid|%ctaid|%nctaid|%laneid|%clock|%globaltimer|%smid" references/ptx.md
```

For inline CUDA assembly:

```bash
rg -n "constraint|asm\(|volatile|memory clobber|namespace|memory space|optimization|incorrect PTX|Error Checking" references/inline_ptx_assembly.md
rg -n '"=r"|"\+r"|"f"|"d"|"l"|"h"|%%|%0|%1' references/inline_ptx_assembly.md
```

For a specific PTX instruction, search by exact mnemonic and nearby heading:

```bash
rg -n "^## .*\bcp\.async\b|\bcp\.async\b" references/ptx.md
rg -n "^## .*\bwgmma\.mma_async\b|\bwgmma\.mma_async\b" references/ptx.md
```

## Command Patterns

Use PTX generation and inspection as a source-level sanity check; verify exact flags in `nvidia-cuda` or local CUDA docs when needed:

```bash
nvcc -ptx -arch=sm_90 kernel.cu -o kernel.ptx
nvcc -keep -lineinfo -O3 -arch=sm_90 kernel.cu -o kernel
ptxas -arch=sm_90 kernel.ptx -o kernel.cubin
```

## PTX Guidance Defaults

- Preserve the PTX documentation's exact operand order, qualifier order, type suffixes, and brace/optional notation when explaining or writing instructions.
- State required PTX ISA version and target architecture when an instruction or qualifier is version-gated.
- Be explicit about register widths and constraint letters in inline PTX; avoid relying on implicit C/C++ type sizes.
- Use braces in inline PTX blocks for temporary registers or labels when namespace collisions are possible.
- Add `volatile` or a memory clobber only when the inline asm has side effects or memory-ordering requirements; explain the reason.
- Treat PTX as a virtual ISA: do not promise a specific SASS instruction without disassembly.
