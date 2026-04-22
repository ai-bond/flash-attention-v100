#!/bin/bash

set -e

DEBUG=0
for arg in "$@"; do [[ "$arg" == "--debug" ]] && DEBUG=1; done

clear
rm -rf ./build

if [ "$DEBUG" -eq 1 ]; then
    export ATTENTION_DEBUG=1
    mkdir -p ./build
fi

pip install . --no-build-isolation -v

if [ "$DEBUG" -eq 1 ] && [ -d "./optimize" ]; then
    find ./build -maxdepth 1 -name "fused_mha_*" ! -name "*.ptx" ! -name "*.cubin" -type f -delete 2>/dev/null || true

    cat > ./build/asm_extract.sh << 'SCRIPT_EOF'
#!/bin/bash
set -eo pipefail
if [ $# -lt 3 ]; then
    echo "Usage: $0 <input_file> <block_name> [ptx|sass]" >&2
    echo "Example: $0 fused_mha_forward.ptx INIT_DO ptx" >&2
    echo "Example: $0 fused_mha_backward.cubin INIT_DO sass" >&2
    exit 1
fi
INPUT_FILE="$1"
BLOCK_NAME="$2"
MODE="${3:-ptx}"
[ ! -f "$INPUT_FILE" ] && { echo "Error: File '$INPUT_FILE' not found" >&2; exit 1; }
BLOCK_NAME_LOWER=$(echo "$BLOCK_NAME" | tr '[:upper:]' '[:lower:]')
if [ "$MODE" == "sass" ]; then
    BEGIN_PAT="beef0001"
    END_PAT="cafe0002"
    DUMP_CMD="cuobjdump --dump-sass"
else
    BEGIN_PAT="dbg_ptx_${BLOCK_NAME_LOWER}_begin"
    END_PAT="dbg_ptx_${BLOCK_NAME_LOWER}_end"
    DUMP_CMD="cat"
fi
echo "// ============================================================================"
echo "// Extracted Block: $BLOCK_NAME"
echo "// Source File:     $(basename "$INPUT_FILE")"
echo "// Mode:            $MODE"
echo "// Markers:         $BEGIN_PAT ... $END_PAT"
echo "// ============================================================================"
$DUMP_CMD "$INPUT_FILE" | awk -v bpat="$BEGIN_PAT" -v epat="$END_PAT" -v fname="$(basename "$INPUT_FILE")" '
BEGIN { inside = 0; found = 0; lines = 0; limit = 5000; }
{
    low = tolower($0)
    is_begin = (index(low, bpat) > 0)
    is_end   = (inside && index(low, epat) > 0)
    if (is_begin && !inside) { inside = 1; print $0; next }
    if (is_end) { print $0; found = 1; exit 0 }
    if (inside) {
        print $0; lines++
        if (lines > limit) { print "// [SAFETY BREAK] Line limit exceeded (" limit "). END marker not found." > "/dev/stderr"; exit 1 }
    }
}
END {
    if (!found) {
        print "// [ERROR] Block not closed. Marker \"" epat "\" not found after BEGIN in " fname > "/dev/stderr"
        print "// [HINT] 1) Check __ASM_DEBUG_END placement in source" > "/dev/stderr"
        print "// [HINT] 2) Ensure compilation used -g flag" > "/dev/stderr"
        exit 1
    }
}'
SCRIPT_EOF
    chmod +x ./build/asm_extract.sh
fi

CUDA_LAUNCH_BLOCKING=1 python test.py