#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

DATA_ROOT="${LIBRISPEECH_ROOT:-${REPO_ROOT}/external/librispeech}"
SUBSET="${LIBRISPEECH_SUBSET:-train-clean-100}"
BASE_URL="${LIBRISPEECH_BASE_URL:-https://www.openslr.org/resources/12}"
CATALOG="${LIBRISPEECH_CATALOG:-${DATA_ROOT}/source_catalog_${SUBSET}.jsonl}"
REPORT="${LIBRISPEECH_REPORT:-${DATA_ROOT}/source_catalog_${SUBSET}.report.json}"

declare -A MD5=(
  [dev-clean]="42e2234ba48799c1f50f24a7926300a1"
  [dev-other]="c8d0bcc9cca99d4f8b62fcc847357931"
  [test-clean]="32fa31d27d2e1cad72775fee3f4849a9"
  [test-other]="fb5a50374b501bb3bac4815ee91d3135"
  [train-clean-100]="2a93770f6d5c6c964bc36631d331a522"
  [train-clean-360]="c0e676e450a7ff2f54aeade5171606fa"
  [train-other-500]="d1a0fd59409feb2c614ce4d30c387708"
)

if [[ -z "${MD5[${SUBSET}]+x}" ]]; then
  echo "Unsupported LIBRISPEECH_SUBSET=${SUBSET}" >&2
  echo "Choose one of: ${!MD5[*]}" >&2
  exit 2
fi
for command in curl md5sum tar python; do
  command -v "${command}" >/dev/null || {
    echo "Required command not found: ${command}" >&2
    exit 2
  }
done

mkdir -p "${DATA_ROOT}/archives"
ARCHIVE="${DATA_ROOT}/archives/${SUBSET}.tar.gz"
curl --fail --location --retry 5 --retry-delay 3 --continue-at - \
  "${BASE_URL}/${SUBSET}.tar.gz" --output "${ARCHIVE}"
printf '%s  %s\n' "${MD5[${SUBSET}]}" "${ARCHIVE}" | md5sum --check --status || {
  echo "Checksum mismatch for ${ARCHIVE}; remove the partial archive and retry." >&2
  exit 1
}

if [[ ! -d "${DATA_ROOT}/LibriSpeech/${SUBSET}" ]]; then
  tar -xzf "${ARCHIVE}" -C "${DATA_ROOT}"
fi

python -m sceneledger.cli.catalog_librispeech \
  --root "${DATA_ROOT}" \
  --subset "${SUBSET}" \
  --output "${CATALOG}" \
  --report "${REPORT}"

echo "LibriSpeech catalog: ${CATALOG}"
echo "Catalog report: ${REPORT}"
