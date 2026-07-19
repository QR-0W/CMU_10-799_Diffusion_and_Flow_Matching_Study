#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd -- "$CODE_ROOT/.." && pwd)"
ARTIFACTS_ROOT="$PROJECT_ROOT/artifacts"

METHOD="ddpm"
CHECKPOINT=""
DATASET_PATH="$ARTIFACTS_ROOT/datasets/celeba-subset/train"
METRICS="kid"
NUM_SAMPLES=1000
BATCH_SIZE=256
NUM_STEPS=1000
SEED=42
GENERATED_DIR=""
METRICS_FILE=""
PYTHON_BIN="${PYTHON_BIN:-$CODE_ROOT/.venv-cu126/bin/python}"
FIDELITY_BIN="${FIDELITY_BIN:-$CODE_ROOT/.venv-cu126/bin/fidelity}"

usage() {
    cat <<'EOF'
Usage:
  scripts/evaluate_torch_fidelity.sh --checkpoint PATH [OPTIONS]

Options:
  --method NAME
  --dataset-path PATH
  --metrics kid,fid,is
  --num-samples N
  --batch-size N
  --num-steps N
  --seed N
  --generated-dir PATH
  --metrics-file PATH
EOF
}

resolve_code_path() {
    if [[ "$1" = /* ]]; then
        printf '%s\n' "$1"
    else
        printf '%s\n' "$CODE_ROOT/$1"
    fi
}

while (( $# > 0 )); do
    case "$1" in
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --method) METHOD="$2"; shift 2 ;;
        --dataset-path) DATASET_PATH="$2"; shift 2 ;;
        --metrics) METRICS="$2"; shift 2 ;;
        --num-samples) NUM_SAMPLES="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --num-steps) NUM_STEPS="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --generated-dir) GENERATED_DIR="$2"; shift 2 ;;
        --metrics-file) METRICS_FILE="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 1 ;;
    esac
done

[[ -n "$CHECKPOINT" ]] || { printf 'Error: --checkpoint is required\n' >&2; exit 1; }
CHECKPOINT="$(resolve_code_path "$CHECKPOINT")"
DATASET_PATH="$(resolve_code_path "$DATASET_PATH")"
[[ -z "$GENERATED_DIR" ]] || GENERATED_DIR="$(resolve_code_path "$GENERATED_DIR")"
[[ -z "$METRICS_FILE" ]] || METRICS_FILE="$(resolve_code_path "$METRICS_FILE")"

[[ -f "$CHECKPOINT" ]] || { printf 'Error: checkpoint not found: %s\n' "$CHECKPOINT" >&2; exit 1; }
[[ -d "$DATASET_PATH" ]] || { printf 'Error: dataset not found: %s\n' "$DATASET_PATH" >&2; exit 1; }
[[ -x "$PYTHON_BIN" ]] || { printf 'Error: Python not found: %s\n' "$PYTHON_BIN" >&2; exit 1; }
[[ -x "$FIDELITY_BIN" ]] || { printf 'Error: fidelity not found: %s\n' "$FIDELITY_BIN" >&2; exit 1; }

IFS=',' read -ra requested_metrics <<<"$METRICS"
metrics_tag=""
for metric in "${requested_metrics[@]}"; do
    case "$metric" in
        fid|kid|is) metrics_tag="${metrics_tag:+${metrics_tag}+}${metric}" ;;
        *) printf 'Error: unsupported metric: %s\n' "$metric" >&2; exit 1 ;;
    esac
done

checkpoint_name="$(basename "${CHECKPOINT%.*}")"
experiment_name="$(basename "$(dirname "$CHECKPOINT")")"
run_name="${experiment_name}_${checkpoint_name}_steps${NUM_STEPS}_seed${SEED}_n${NUM_SAMPLES}"
GENERATED_DIR="${GENERATED_DIR:-$ARTIFACTS_ROOT/samples/evaluation/$run_name}"
METRICS_FILE="${METRICS_FILE:-$ARTIFACTS_ROOT/metrics/${run_name}_${metrics_tag}.txt}"
CACHE_DIR="$ARTIFACTS_ROOT/metrics/cache"

mkdir -p "$(dirname "$GENERATED_DIR")" "$(dirname "$METRICS_FILE")" "$CACHE_DIR"
if [[ -d "$GENERATED_DIR" ]]; then
    shopt -s nullglob dotglob
    existing_files=("$GENERATED_DIR"/*)
    shopt -u nullglob dotglob
    (( ${#existing_files[@]} == 0 )) || {
        printf 'Error: generated directory is not empty: %s\n' "$GENERATED_DIR" >&2
        exit 1
    }
fi
mkdir -p "$GENERATED_DIR"

printf 'Generating %s samples in %s\n' "$NUM_SAMPLES" "$GENERATED_DIR"
(
    cd "$CODE_ROOT"
    "$PYTHON_BIN" sample.py \
        --checkpoint "$CHECKPOINT" \
        --method "$METHOD" \
        --output_dir "$GENERATED_DIR" \
        --num_samples "$NUM_SAMPLES" \
        --batch_size "$BATCH_SIZE" \
        --num_steps "$NUM_STEPS" \
        --seed "$SEED"
)

fidelity_args=(
    --gpu 0
    --batch-size "$BATCH_SIZE"
    --cache-root "$CACHE_DIR"
    --input1 "$GENERATED_DIR"
    --input2 "$DATASET_PATH"
)
for metric in "${requested_metrics[@]}"; do
    case "$metric" in
        fid) fidelity_args+=(--fid) ;;
        kid) fidelity_args+=(--kid) ;;
        is) fidelity_args+=(--isc) ;;
        *) printf 'Error: unsupported metric: %s\n' "$metric" >&2; exit 1 ;;
    esac
done

printf 'Writing metrics to %s\n' "$METRICS_FILE"
"$FIDELITY_BIN" "${fidelity_args[@]}" | tee "$METRICS_FILE"
