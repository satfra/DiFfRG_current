#!/usr/bin/env bash
# Run Wolfram example checks inside the dependency SIF while bind-mounting the
# host Wolfram/Form installation. The public CI image intentionally does not
# bake licensed Wolfram bits.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <dependency-image-or-sif>" >&2
  exit 2
fi

image="$1"
workspace="${GITHUB_WORKSPACE:-$(pwd)}"
log_dir="${workspace}/.ci/logs/wolfram"
summary_file="${workspace}/.ci/logs/wolfram-summary.md"
mkdir -p "${log_dir}" "$(dirname "${summary_file}")"

{
  echo "## Wolfram generation"
  echo
} > "${summary_file}"

binds=("${workspace}:/work")
seen_binds=" ${workspace}:/work "

add_bind_dir() {
  local source_path="$1"
  local dest_path="${2:-$1}"
  [[ -n "${source_path}" && -e "${source_path}" ]] || return 0

  case " ${seen_binds} " in
    *" ${source_path}:${dest_path} "*) return 0 ;;
  esac
  binds+=("${source_path}:${dest_path}")
  seen_binds+=" ${source_path}:${dest_path} "
}

add_bind_path() {
  local path="$1"
  local dest_path="${2:-}"
  [[ -n "${path}" && -e "${path}" ]] || return 0

  local source_path
  if [[ -d "${path}" ]]; then
    source_path="${path}"
  else
    source_path="$(dirname "${path}")"
  fi

  add_bind_dir "${source_path}" "${dest_path:-${source_path}}"
}

add_bind_resolved_path() {
  local path="$1"
  [[ -n "${path}" ]] || return 0

  local resolved
  resolved="$(readlink -f "${path}" 2>/dev/null || true)"
  add_bind_path "${resolved}"
}

add_bind_referenced_execs() {
  local path="$1"
  [[ -n "${path}" && -f "${path}" ]] || return 0

  local referenced_path
  while IFS= read -r referenced_path; do
    add_bind_path "${referenced_path}"
  done < <(grep -Eo '/[^[:space:]"'\'']+/wolframscript' "${path}" 2>/dev/null || true)
}

wolframscript_path="$(command -v wolframscript || true)"
if [[ -z "${wolframscript_path}" ]]; then
  {
    echo "Skipped: host \`wolframscript\` was not found on PATH."
    echo
    echo "No container run was attempted."
  } >> "${summary_file}"
  exit 0
fi

add_bind_path "${wolframscript_path}" /host-bin/wolfram-bin
add_bind_resolved_path "${wolframscript_path}"
add_bind_referenced_execs "${wolframscript_path}"

for candidate in \
  "${WOLFRAMSCRIPT_KERNELPATH:-}" \
  /home/software \
  /home/software/mathematica \
  /home/software/Mathematica \
  /home/software/Wolfram \
  /usr/local/Wolfram \
  /opt/Wolfram \
  /Applications/Wolfram.app \
  /Applications/Mathematica.app \
  "${HOME}/.Wolfram" \
  "${HOME}/.Mathematica" \
  "${HOME}/.WolframEngine"
do
  add_bind_path "${candidate}"
  add_bind_resolved_path "${candidate}"
done

bind_args=()
for bind in "${binds[@]}"; do
  bind_args+=(-b "${bind}")
done

{
  echo "Bound host paths:"
  for bind in "${binds[@]}"; do
    echo "- \`${bind}\`"
  done
  echo
} >> "${summary_file}"

echo "Bound host paths for Wolfram container:"
for bind in "${binds[@]}"; do
  echo "- ${bind}"
done

required="${REQUIRED_WOLFRAM_EXAMPLES:-}"
wolfram_timeout="${WOLFRAM_TIMEOUT:-1800}"
run_regressions="${RUN_EXAMPLE_REGRESSIONS:-1}"
update_baselines="${UPDATE_BASELINES:-0}"
required_regressions="${REQUIRED_EXAMPLE_REGRESSIONS:-}"
example_regressions="${EXAMPLE_REGRESSIONS:-}"

env \
  SINGULARITYENV_REQUIRED_WOLFRAM_EXAMPLES="${required}" \
  SINGULARITYENV_WOLFRAM_TIMEOUT="${wolfram_timeout}" \
  SINGULARITYENV_RUN_EXAMPLE_REGRESSIONS="${run_regressions}" \
  SINGULARITYENV_UPDATE_BASELINES="${update_baselines}" \
  SINGULARITYENV_REQUIRED_EXAMPLE_REGRESSIONS="${required_regressions}" \
  SINGULARITYENV_EXAMPLE_REGRESSIONS="${example_regressions}" \
  APPTAINERENV_REQUIRED_WOLFRAM_EXAMPLES="${required}" \
  APPTAINERENV_WOLFRAM_TIMEOUT="${wolfram_timeout}" \
  APPTAINERENV_RUN_EXAMPLE_REGRESSIONS="${run_regressions}" \
  APPTAINERENV_UPDATE_BASELINES="${update_baselines}" \
  APPTAINERENV_REQUIRED_EXAMPLE_REGRESSIONS="${required_regressions}" \
  APPTAINERENV_EXAMPLE_REGRESSIONS="${example_regressions}" \
  SINGULARITYENV_PREPEND_PATH="/host-bin/wolfram-bin" \
  APPTAINERENV_PREPEND_PATH="/host-bin/wolfram-bin" \
  bash containers/singularity-run.sh \
  "${bind_args[@]}" \
  -w /work \
  "${image}" \
  bash -lc '
    set -euo pipefail
    export PREPEND_PATH="/host-bin/wolfram-bin"
    export REQUIRED_WOLFRAM_EXAMPLES="'"${required}"'"
    export WOLFRAM_TIMEOUT="'"${wolfram_timeout}"'"
    export RUN_EXAMPLE_REGRESSIONS="'"${run_regressions}"'"
    export UPDATE_BASELINES="'"${update_baselines}"'"
    export REQUIRED_EXAMPLE_REGRESSIONS="'"${required_regressions}"'"
    export EXAMPLE_REGRESSIONS="'"${example_regressions}"'"
    wolfram_status=0
    bash containers/ci/wolfram-example-checks.sh || wolfram_status=$?
    if [[ ${wolfram_status} -eq 75 ]]; then
      echo "Wolfram preflight failed; cannot run generators or baseline regressions."
      exit 1
    fi
    if [[ ${wolfram_status} -ne 0 ]]; then
      exit "${wolfram_status}"
    fi
    if [[ "${RUN_EXAMPLE_REGRESSIONS:-1}" == "1" ]]; then
      bash containers/ci/build-examples.sh
      bash containers/ci/run-example-regressions.sh
    fi
  '
