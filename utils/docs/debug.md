[<- Main](../../README.md)

---

#### Introduction

---

This document describes the debugging system. By default, the code compiles using the fused m16n16k16 library. Build modes and debugging tools are controlled via `run.sh` flags.

#### Build Flags

---

- `--mma-native` - use native CUDA `mma.h` with configurations (16x16x16, m32n8k16, m8n32k16). Sets `MMA_NATIVE=1` env var.
- `--mma-884` - use fused 8x8x4 MMA library. Sets `MMA_884=1` env var.
- `--debug` - enable debug mode: preserve build artifacts, add `-g`, `-DKERNEL_DEBUG`, `-Xptxas -v`, `--keep --keep-dir ./build` flags, generate assembly extraction script `asm_extract.sh`. Sets `ATTENTION_DEBUG=1` env var.

When the `--debug` flag is passed, `setup.py` detects `ATTENTION_DEBUG` and injects specific NVCC flags. The kernel headers (`debug.h`) activate debug macros only when `KERNEL_DEBUG` is defined.

#### Debug Output Format

---

All debug messages are printed from thread 0 of block 0 to avoid race conditions and excessive output. The format follows:

- `[DBG_ERR][B%d][STAGE]`: Error detected (inf/nan or mismatch). Includes coordinates `[r,c]` for matrices or indices for vectors.
- `[DBG_OK ][B%d][STAGE]`: Validation passed. Confirms the tile/vector was scanned without errors.
- `[DBG_MAT][B%d][T%d][STAGE]`: Matrix dump header. Followed by rows of formatted float values (`%7.3f`, `nan`, `-inf`).
- `[DBG_VEC][B%d][T%d][FIELD]`: Vector dump header. Lists elements separated by spaces.

#### Check and Print Macros

---

These macros are active only when `KERNEL_DEBUG` is defined. They operate on shared memory layouts defined by `Config::SmemLayout`.

- `__CHECK_INIT(FIELD_TAG, EXPECTED_VAL, VALID_ROWS)` - Verifies buffer zeroing (e.g., Q, K, O) in shared memory. It reads raw bytes via PTX `ld.shared.u16/u32` and compares against expected bit patterns. Prints `[DBG_ERR][B%d][INIT]` on mismatch.
- `__CHECK_ERRORS(STG, VM, VN, SC, WID, LID, TID)` - Scans shared memory matrices for inf/nan values at a given stage. Uses `stage_name()` for logging. Prints `[DBG_ERR][B%d][%s][r,c]` or `[DBG_OK ][B%d][%s]`. Only executes on thread 0 of block 0.
- `__PRINT_MATRIX(STG, VM, VN, SC, WID, LID, TID, TILE_IDX)` - Dumps 2D tile content to stdout. Supports stages like SQKT, DOVT, DQDSK, DVPTDO, DKDSTQ, DOPV. Prints `[DBG_MAT][B%d][T%d][%s]`. Only executes on thread 0/warp 0 of block 0.
- `__PRINT_RESULT(FIELD_TAG, VLEN, TILE_IDX)` - Traces 1D vectors (e.g., row_max, row_sum, lse, row_dot) for online softmax. Reads via PTX `ld.shared`. Prints `[DBG_VEC][B%d][T%d][FIELD]`. Only executes on thread 0 of block 0.
- `__ASM_DEBUG_BEGIN(STG, CTX)` - Inserts begin marker (`0xBEEF0001`) into PTX/SASS stream via inline asm comment `// DBG_PTX STG CTX BEGIN`.
- `__ASM_DEBUG_END(STG, CTX)` - Inserts end marker (`0xCAFE0002`) into PTX/SASS stream via inline asm comment `// DBG_PTX STG CTX END`.


#### Field Definitions

---

`debug.h` uses `DEFINE_FIELD` macro to create trait structs (`has_<field>`) and info structs (`field_info<Layout, TAG_<field>>`) for compile-time checks on shared memory layout fields. Examples include:
- Forward: `q_fwd`, `k_fwd`, `v_fwd`, `s_fwd`, `p_fwd`, `o_fwd`, `row_max_fwd`, `row_sum_fwd`
- Backward dQ: `q_dq`, `k_dq`, `v_dq`, `s_dq`, `dO_dq`, `dOV_dq`, `dS_dq`, `dQ_dq`
- Backward dKV: `k_dkv`, `v_dkv`, `q_dkv`, `dO_dkv`, `s_dkv`, `p_dkv`, `dS_dkv`, `dOV_dkv`, `dK_dkv`, `dV_dkv`
- Common: `lse`, `row_dot`

#### Assembly Extraction

---

When built with `--debug`, `run.sh` generates `./build/asm_extract.sh`. This script can extracts PTX or SASS blocks marked by `__ASM_DEBUG_BEGIN/END`.

**Usage:**
```bash
# Extract PTX block
./build/asm_extract.sh ./build/fused_mha_forward.ptx SQKT ptx

# Extract SASS block  
./build/asm_extract.sh ./build/fused_mha_backward.cubin DOPV sass
```

**Script Logic:**

- **PTX Mode**: The script invokes `cat` on the input `.ptx` file. It uses `awk` to search for case-insensitive substrings matching `dbg_ptx_<block_name_lower>_begin` and `dbg_ptx_<block_name_lower>_end`. These strings correspond to the inline assembly comments injected by `__ASM_DEBUG_BEGIN` and `__ASM_DEBUG_END` macros in `debug.h`.
- **SASS Mode**: The script invokes `cuobjdump --dump-sass` on the input `.cubin` file. It searches for the hexadecimal magic numbers `beef0001` (begin) and `cafe0002` (end) embedded in the SASS instruction stream via the `mov.u32` instructions defined in the `ASM_MARK` macro.

[<- Main](../../README.md)