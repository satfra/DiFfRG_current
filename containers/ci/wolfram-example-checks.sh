#!/usr/bin/env bash
# Executes inside the dependency container. Wolfram itself is expected to come
# from bind mounts prepared by run-wolfram-in-container.sh.
set -euo pipefail

workspace="${WORKSPACE:-/work}"
log_dir="${DIFFRG_WOLFRAM_LOG_DIR:-${workspace}/.ci/logs/wolfram}"
summary_file="${DIFFRG_WOLFRAM_SUMMARY:-${workspace}/.ci/logs/wolfram-summary.md}"
required_examples="${REQUIRED_WOLFRAM_EXAMPLES:-}"
timeout_seconds="${WOLFRAM_TIMEOUT:-1800}"

examples=(
  "ONfiniteT:Examples/ONfiniteT:ON.nb"
  "QuarkMesonLPAprime:Examples/QuarkMesonLPAprime:QuarkMesonLPAprime.m"
  "YangMills_SP:Examples/YangMills/SP:Yang-Mills.m"
  "YangMills_Full:Examples/YangMills/Full:Yang-Mills.m"
  "FourFermi:Examples/FourFermi:Four-Fermion.m"
)

mkdir -p "${log_dir}" "$(dirname "${summary_file}")"
path_prefix="${PREPEND_PATH:-}"
if [[ -d /host-bin/wolfram-bin ]]; then
  path_prefix="/host-bin/wolfram-bin${path_prefix:+:${path_prefix}}"
fi
export PATH="${path_prefix}:/usr/local/bin:/usr/bin:/bin:/opt/Wolfram/WolframEngine/Executables:/opt/Wolfram/Wolfram/Executables:${PATH:-}"

is_required() {
  local name="$1"
  case " ${required_examples} " in
    *" ${name} "*) return 0 ;;
    *) return 1 ;;
  esac
}

run_wolfram() {
  if command -v timeout >/dev/null 2>&1; then
    timeout "${timeout_seconds}" wolframscript "$@"
  else
    wolframscript "$@"
  fi
}

{
  echo "| Check | Result | Log | Note |"
  echo "| --- | --- | --- | --- |"
} >> "${summary_file}"

preflight_log="${log_dir}/preflight.log"
set +e
(
  echo "PATH: ${PATH}"
  echo "PREPEND_PATH: ${PREPEND_PATH:-}"
  echo "PWD: $(pwd)"
  echo "HOME: ${HOME:-}"
  echo
  echo "wolframscript path: $(command -v wolframscript || true)"
  if command -v wolframscript >/dev/null 2>&1; then
    ls -l "$(command -v wolframscript)" || true
  fi
  echo
  echo "Wolfram installation probes:"
  for candidate in \
    /home/software \
    /home/software/mathematica \
    /home/software/mathematica/Executables \
    /home/software/mathematica/Executables/wolframscript \
    /home/software/mathematica/Executables/WolframKernel \
    /home/software/mathematica/Executables/MathKernel \
    /home/software/mathematica/SystemFiles/Kernel/Binaries/Linux-x86-64 \
    /home/software/mathematica/SystemFiles/Kernel/Binaries/Linux-x86-64/WolframKernel \
    /home/software/mathematica/SystemFiles/Kernel/Binaries/Linux-x86-64/MathKernel
  do
    if [[ -e "${candidate}" ]]; then
      ls -ld "${candidate}" || true
      if [[ -f "${candidate}" ]]; then
        file "${candidate}" || true
        ldd "${candidate}" || true
      fi
    else
      echo "missing: ${candidate}"
    fi
  done
  echo
  echo "wolframscript version:"
  wolframscript -code '$VersionNumber'
  version_status=$?
  echo "wolframscript version exit: ${version_status}"
  echo
  echo "FunKit preflight:"
  wolframscript -code 'If[Length[PacletFind["FunKit"]] > 0, Print["FunKit found"]; Exit[0], Print["FunKit missing"]; Exit[2]]'
  funkit_status=$?
  echo "FunKit preflight exit: ${funkit_status}"
  echo
  echo "tform preflight:"
  if command -v tform >/dev/null 2>&1; then
    echo "tform path: $(command -v tform)"
    ls -l "$(command -v tform)" || true
    tform -v
    tform_status=$?
  else
    echo "tform missing"
    tform_status=127
  fi
  echo "tform preflight exit: ${tform_status}"
  exit $((version_status || funkit_status || tform_status))
) > "${preflight_log}" 2>&1
preflight_status=$?
set -e

if [[ ${preflight_status} -eq 0 ]]; then
  echo "Wolfram preflight passed."
  echo "| preflight | passed | \`${preflight_log#${workspace}/}\` | |" >> "${summary_file}"
else
  echo "Wolfram preflight failed with exit ${preflight_status}."
  echo "---- ${preflight_log#${workspace}/} ----"
  cat "${preflight_log}"
  echo "---- end ${preflight_log#${workspace}/} ----"
  echo "| preflight | failed | \`${preflight_log#${workspace}/}\` | Wolfram/FunKit/tform not fully available inside container |" >> "${summary_file}"
  echo >> "${summary_file}"
  echo "Preflight failed; generator execution skipped." >> "${summary_file}"
  exit 75
fi

required_failures=0

for item in "${examples[@]}"; do
  name="${item%%:*}"
  rest="${item#*:}"
  relpath="${rest%%:*}"
  entry="${rest#*:}"
  example_dir="${workspace}/${relpath}"
  log="${log_dir}/${name}.log"

  set +e
  if [[ "${entry}" == *.m ]]; then
    run_wolfram -code 'AppendTo[$Path, "/work/DiFfRG/Mathematica"]; SetDirectory["'"${example_dir}"'"]; Get["'"${entry}"'"]' > "${log}" 2>&1
    status=$?
  else
    {
      echo "No plain-text .m generator is available for ${name}; notebook execution is intentionally not attempted by default."
      echo "Entry point: ${relpath}/${entry}"
    } > "${log}"
    status=125
  fi
  set -e

  if [[ ${status} -eq 0 ]]; then
    echo "| ${name} | passed | \`${log#${workspace}/}\` | |" >> "${summary_file}"
  else
    echo "| ${name} | failed | \`${log#${workspace}/}\` | exit ${status} |" >> "${summary_file}"
    required_failures=$((required_failures + 1))
  fi
done

diff_log="${log_dir}/generated-flow-diff.patch"
set +e
git -C "${workspace}" diff -- 'Examples/**/flows/**' > "${diff_log}"
diff_status=$?
set -e
if [[ ${diff_status} -eq 0 && -s "${diff_log}" ]]; then
  echo "| generated-flow diff | failed | \`${diff_log#${workspace}/}\` | generated files changed |" >> "${summary_file}"
  required_failures=$((required_failures + 1))
else
  rm -f "${diff_log}"
  echo "| generated-flow diff | passed | | no generated flow drift |" >> "${summary_file}"
fi

if [[ ${required_failures} -ne 0 ]]; then
  echo "${required_failures} required Wolfram generator(s) failed." >&2
  exit 1
fi
