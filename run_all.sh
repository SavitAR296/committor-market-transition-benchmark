#!/usr/bin/env bash
# Two-tier rebuild for the two companion papers.
#
#   ./run_all.sh quick   Tier 1 (minutes): rebuild all tables, text macros and figures from the
#                        shipped frozen CSVs, then recompile the manuscripts.
#   ./run_all.sh full    Tier 2 (hours, one core): recompute every simulation and fit from the
#                        raw data, then Tier 1.
#   ./run_all.sh check   Verify the shipped trees against the SHA-256 manifests.
#
# Paper 2 (IJF) is under double-blind review: its package is provided as anonymous supplementary
# material and enters the public tree only after acceptance. The Paper 2 steps below therefore run
# only if the package is present next to Paper 1 (e.g. a private checkout).
set -euo pipefail
cd "$(dirname "$0")"

P1=paper1_pre/reproducibility_pre
P2=paper2_ijf/reproducibility_p2

case "${1:-quick}" in
  quick|frozen)
    echo "== Paper 1 (PRE): quick rebuild from frozen CSVs =="
    make -C "$P1" -f Makefile_P1 quick
    if [ -d "$P2" ]; then
      echo "== Paper 2 (IJF): quick rebuild from frozen CSVs =="
      make -C "$P2" -f Makefile_P2 quick
    else
      echo "== Paper 2 package not present (private during review); skipped =="
    fi
    ;;
  full|all)
    echo "== Paper 1 (PRE): full rerun =="
    make -C "$P1" -f Makefile_P1 all
    if [ -d "$P2" ]; then
      echo "== Paper 2 (IJF): full rerun =="
      make -C "$P2" -f Makefile_P2 all
    else
      echo "== Paper 2 package not present (private during review); skipped =="
    fi
    ;;
  check)
    echo "== SHA-256 verification =="
    make -C "$P1" -f Makefile_P1 check
    if [ -d "$P2" ]; then
      make -C "$P2" -f Makefile_P2 check
    fi
    ;;
  *)
    echo "usage: $0 [quick|full|check]" >&2
    exit 1
    ;;
esac
