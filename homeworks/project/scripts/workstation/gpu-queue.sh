#!/usr/bin/env bash
set -euo pipefail

umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
STATE_DIR="${GPU_QUEUE_DIR:-$PROJECT_ROOT/artifacts/queue}"
POLL_SECONDS="${GPU_QUEUE_POLL_SECONDS:-30}"
PROJECT_ID="$(printf '%s' "$PROJECT_ROOT" | cksum | cut -d' ' -f1)"
SESSION_PREFIX="gpuq-${PROJECT_ID}-"

usage() {
    cat <<'EOF'
Usage:
  gpu-queue init
  gpu-queue submit -- COMMAND [ARG ...]
  gpu-queue start [GPU]
  gpu-queue worker GPU
  gpu-queue status
  gpu-queue stop

The queue coordinates only jobs submitted from this workspace. It cannot
reserve a GPU against processes started outside this queue.
EOF
}

die() {
    printf 'gpu-queue: %s\n' "$*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

init_state() {
    mkdir -p \
        "$STATE_DIR/pending" \
        "$STATE_DIR/running" \
        "$STATE_DIR/done" \
        "$STATE_DIR/failed" \
        "$STATE_DIR/logs" \
        "$STATE_DIR/locks"
    chmod 700 "$STATE_DIR" "$STATE_DIR"/*
}

gpu_count() {
    nvidia-smi --query-gpu=index --format=csv,noheader | wc -l
}

validate_gpu() {
    local gpu="$1"
    local count

    [[ "$gpu" =~ ^[0-9]+$ ]] || die "invalid GPU index: $gpu"
    count="$(gpu_count)"
    (( gpu < count )) || die "GPU index $gpu is outside 0-$((count - 1))"
}

gpu_is_free() {
    local gpu="$1"
    local pids

    pids="$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')"
    [[ -z "$pids" ]]
}

submit_job() {
    local id tmp job

    [[ "${1:-}" == "--" ]] && shift
    (( $# > 0 )) || die "submit requires a command"

    init_state
    id="$(date -u +%Y%m%dT%H%M%SZ)-$$-$RANDOM"
    tmp="$STATE_DIR/pending/.${id}.tmp"
    job="$STATE_DIR/pending/${id}.job"

    {
        printf '#!/usr/bin/env bash\nset -euo pipefail\n'
        printf 'cd %q\n' "$PWD"
        printf 'exec'
        printf ' %q' "$@"
        printf '\n'
    } >"$tmp"
    chmod 600 "$tmp"
    mv "$tmp" "$job"
    printf '%s\n' "$id"
}

next_pending_job() {
    local jobs

    shopt -s nullglob
    jobs=("$STATE_DIR"/pending/*.job)
    shopt -u nullglob
    (( ${#jobs[@]} > 0 )) && printf '%s\n' "${jobs[0]}"
}

run_worker() {
    local gpu="$1"
    local lock job id running log destination exit_code

    require_command nvidia-smi
    require_command flock
    validate_gpu "$gpu"
    init_state

    lock="$STATE_DIR/locks/gpu-${gpu}.lock"
    exec 9>"$lock"
    flock -n 9 || die "worker for GPU $gpu is already running"

    printf 'worker GPU %s started; polling every %ss\n' "$gpu" "$POLL_SECONDS"
    while true; do
        if ! gpu_is_free "$gpu"; then
            sleep "$POLL_SECONDS"
            continue
        fi

        job="$(next_pending_job || true)"
        if [[ -z "$job" ]]; then
            sleep "$POLL_SECONDS"
            continue
        fi

        id="$(basename "$job" .job)"
        running="$STATE_DIR/running/gpu${gpu}--${id}.job"
        if ! mv "$job" "$running" 2>/dev/null; then
            continue
        fi

        log="$STATE_DIR/logs/${id}.log"
        printf 'starting %s on GPU %s\n' "$id" "$gpu"
        exit_code=0
        CUDA_VISIBLE_DEVICES="$gpu" GPU_QUEUE_JOB_ID="$id" bash "$running" >"$log" 2>&1 || exit_code=$?

        if (( exit_code == 0 )); then
            destination="$STATE_DIR/done/${id}.job"
        else
            destination="$STATE_DIR/failed/${id}.job"
            printf '%s\n' "$exit_code" >"$STATE_DIR/failed/${id}.exit-code"
        fi
        mv "$running" "$destination"
        printf 'finished %s with exit code %s\n' "$id" "$exit_code"
    done
}

start_workers() {
    local script gpu session
    local gpus=("$@")

    require_command tmux
    require_command nvidia-smi
    init_state
    script="$(readlink -f "$0")"

    if (( ${#gpus[@]} == 0 )); then
        gpus=(0)
    fi
    (( ${#gpus[@]} == 1 )) || die "this workspace supports one GPU worker"

    for gpu in "${gpus[@]}"; do
        validate_gpu "$gpu"
        session="${SESSION_PREFIX}${gpu}"
        if tmux has-session -t "$session" 2>/dev/null; then
            printf 'worker GPU %s already running\n' "$gpu"
            continue
        fi
        tmux new-session -d -s "$session" "$script" worker "$gpu"
        printf 'worker GPU %s started in tmux session %s\n' "$gpu" "$session"
    done
}

count_jobs() {
    local directory="$1"
    local jobs

    shopt -s nullglob
    jobs=("$STATE_DIR/$directory"/*.job)
    shopt -u nullglob
    printf '%s' "${#jobs[@]}"
}

show_status() {
    local directory

    init_state
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
    for directory in pending running done failed; do
        printf '%-7s %s\n' "$directory" "$(count_jobs "$directory")"
    done
}

stop_workers() {
    local gpu session count

    require_command tmux
    init_state
    count="$(count_jobs running)"
    (( count == 0 )) || die "refusing to stop while $count job(s) are running"

    while IFS= read -r gpu; do
        session="${SESSION_PREFIX}${gpu}"
        if tmux has-session -t "$session" 2>/dev/null; then
            tmux kill-session -t "$session"
            printf 'worker GPU %s stopped\n' "$gpu"
        fi
    done < <(nvidia-smi --query-gpu=index --format=csv,noheader)
}

command="${1:-}"
[[ -n "$command" ]] || {
    usage
    exit 1
}
shift

case "$command" in
    init)
        init_state
        printf '%s\n' "$STATE_DIR"
        ;;
    submit)
        submit_job "$@"
        ;;
    start)
        start_workers "$@"
        ;;
    worker)
        (( $# == 1 )) || die "worker requires one GPU index"
        run_worker "$1"
        ;;
    status)
        show_status
        ;;
    stop)
        stop_workers
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        usage >&2
        die "unknown command: $command"
        ;;
esac
