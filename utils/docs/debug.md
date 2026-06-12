Debug
-------------
Now by default code will compile with fused m16n16k16 library. Youcan back or use

`--mma-native` Use native CUDA mma.h with (16x16x16, m32n8k16, m8n32k16)

`--mma-884` Use fused 8x8x4 MMA library

 `--debug` Enable debug mode, preserve build artifacts, generate assembly extraction tool


## Debug Markers

### Available Stages

| Stage | Description | Location |
|-------|-------------|----------|
| `SMEM` | Shared memory initialization check | After smem init, before any loads |
| `TILE` | Tile load validation (Q, K, V) | After each tile load from global memory |
| `SQKT` | Score matrix (Q·K^T) validation | After matrix multiplication, before softmax |
| `SOFTMAX` | Online softmax validation | After softmax computation |
| `DOVT` | dO·V^T validation (backward) | After gradient computation for dP |
| `DOPV` | Output accumulation (P·V) validation | After each P·V multiplication |
| `DQDSK` | dQ from dS·K validation (backward) | After dQ gradient computation |
| `DVPTDO` | dV from P^T·dO validation (backward) | After dV gradient computation |
| `DKDSTQ` | dK from S^T·dQ validation (backward) | After dK gradient computation |
| `ROWDQ` | Row-wise operations for dQ | During dQ reduction |
| `ROWDKV` | Row-wise operations for dK/dV | During dK/dV reduction |
| `WRITEO` | Final output write validation | Before writing O to global memory |
| `WRITEQ` | dQ write validation (backward) | Before writing dQ to global memory |
| `WRITEKV` | dK/dV write validation (backward) | Before writing dK/dV to global memory |
| `NONE` | No debug stage | Default/disabled state |

### Extracted Checks

- `__CHECK_INIT`      - Verifies buffer zeroing (Q, K, O)
- `__CHECK_ERRORS`    - Detects inf/nan in shared memory matrices
- `__PRINT_MATRIX`    - Dumps matrix content (optional)
- `__PRINT_RESULT`    - Traces row sums for online softmax
- `__ASM_DEBUG_BEGIN` - Put begin of ptx and sass insertion
- `__ASM_DEBUG_END`   - Put end of ptx and sass insertion

When running with `--debug` flag, the kernels insert assembly-level markers for PTX/SASS extraction using `asm_extract.sh` script.

```bash
# Extract PTX block
./build/asm_extract.sh ./build/fused_mha_forward.ptx SQKT ptx

# Extract SASS block  
./build/asm_extract.sh ./build/fused_mha_backward.cubin DOPV sass
