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

tform_path="$(command -v tform || true)"
if [[ -n "${tform_path}" ]]; then
  add_bind_path "${tform_path}" /host-bin/tform-bin
fi

for candidate in \
  "${WOLFRAMSCRIPT_KERNELPATH:-}" \
  /usr/local/Wolfram \
  /opt/Wolfram \
  /Applications/Wolfram.app \
  /Applications/Mathematica.app \
  "${HOME}/.Wolfram" \
  "${HOME}/.Mathematica" \
  "${HOME}/.WolframEngine"
do
  add_bind_path "${candidate}"
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

expected="${EXPECTED_WOLFRAM_FAILURES:-ONfiniteT FourFermi QuarkMesonLPAprime}"
required="${REQUIRED_WOLFRAM_EXAMPLES:-}"
wolfram_timeout="${WOLFRAM_TIMEOUT:-1800}"
run_regressions="${RUN_EXAMPLE_REGRESSIONS:-1}"
update_baselines="${UPDATE_BASELINES:-0}"
required_regressions="${REQUIRED_EXAMPLE_REGRESSIONS:-}"
example_regressions="${EXAMPLE_REGRESSIONS:-}"

env \
  SINGULARITYENV_EXPECTED_WOLFRAM_FAILURES="${expected}" \
  SINGULARITYENV_REQUIRED_WOLFRAM_EXAMPLES="${required}" \
  SINGULARITYENV_WOLFRAM_TIMEOUT="${wolfram_timeout}" \
  SINGULARITYENV_RUN_EXAMPLE_REGRESSIONS="${run_regressions}" \
  SINGULARITYENV_UPDATE_BASELINES="${update_baselines}" \
  SINGULARITYENV_REQUIRED_EXAMPLE_REGRESSIONS="${required_regressions}" \
  SINGULARITYENV_EXAMPLE_REGRESSIONS="${example_regressions}" \
  APPTAINERENV_EXPECTED_WOLFRAM_FAILURES="${expected}" \
  APPTAINERENV_REQUIRED_WOLFRAM_EXAMPLES="${required}" \
  APPTAINERENV_WOLFRAM_TIMEOUT="${wolfram_timeout}" \
  APPTAINERENV_RUN_EXAMPLE_REGRESSIONS="${run_regressions}" \
  APPTAINERENV_UPDATE_BASELINES="${update_baselines}" \
  APPTAINERENV_REQUIRED_EXAMPLE_REGRESSIONS="${required_regressions}" \
  APPTAINERENV_EXAMPLE_REGRESSIONS="${example_regressions}" \
  SINGULARITYENV_PREPEND_PATH="/host-bin/wolfram-bin:/host-bin/tform-bin" \
  APPTAINERENV_PREPEND_PATH="/host-bin/wolfram-bin:/host-bin/tform-bin" \
  bash containers/singularity-run.sh \
  "${bind_args[@]}" \
  -w /work \
  "${image}" \
  bash -lc '
    set -euo pipefail
    wolfram_status=0
    bash containers/ci/wolfram-example-checks.sh || wolfram_status=$?
    if [[ ${wolfram_status} -eq 75 ]]; then
      echo "Wolfram preflight failed; skipping rebuild/run/baseline regression layer."
      exit 0
    fi
    if [[ ${wolfram_status} -ne 0 ]]; then
      exit "${wolfram_status}"
    fi
    if [[ "${RUN_EXAMPLE_REGRESSIONS:-1}" == "1" ]]; then
      bash containers/ci/build-examples.sh
      bash containers/ci/run-example-regressions.sh
    fi
  '
