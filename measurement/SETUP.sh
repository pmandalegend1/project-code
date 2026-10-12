#!/usr/bin/env bash
# SportsM5 — one-command setup and smoke test.
#
# 
#   bash SETUP.sh
#
# Safe to re-run: every step checks whether it is already done.
# CPU only — no GPU anywhere in this pipeline.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT="$(pwd)"

bold() { printf '\n\033[1m%s\033[0m\n' "$*"; }
ok()   { printf '  \033[32mok\033[0m   %s\n' "$*"; }
warn() { printf '  \033[33mwarn\033[0m %s\n' "$*"; }
die()  { printf '  \033[31mFAIL\033[0m %s\n' "$*"; exit 1; }

bold "SportsM5 setup  —  $ROOT"

# ───────────────────────────── 1. install the corrected source
bold "1/5  Installing the corrected source files"
if [ -d "PPDA_FIX/FIXED_src/src" ]; then
    if [ ! -d "src_backup_prefix" ] && [ -d src ]; then
        cp -r src "src_backup_$(date +%Y%m%d_%H%M%S)" 2>/dev/null && ok "backed up src/"
    fi
    cp PPDA_FIX/FIXED_src/src/analysis/ppda.py             src/analysis/ppda.py
    cp PPDA_FIX/FIXED_src/src/analysis/did.py              src/analysis/did.py
    cp PPDA_FIX/FIXED_src/src/analysis/style_clustering.py src/analysis/style_clustering.py
    cp PPDA_FIX/FIXED_src/src/data/preprocessing.py        src/data/preprocessing.py
    cp PPDA_FIX/FIXED_src/src/run_did_count.py             src/run_did_count.py
    cp PPDA_FIX/FIXED_src/verify_fixes.py                  verify_fixes.py
    ok "5 corrected files + verify_fixes.py in place"
else
    warn "PPDA_FIX/FIXED_src not found — assuming src/ is already corrected"
fi

# quick proof the fixes really are in there
python3 - <<'PY' || exit 1
import pathlib, sys
checks = {
    "src/analysis/ppda.py":            'duel_type',
    "src/data/preprocessing.py":       'df["event_type"].eq("Duel")',
    "src/analysis/style_clustering.py":'FORMATION_WEIGHT',
    "src/analysis/did.py":             'run_did_count_regression',
}
bad = [f for f, token in checks.items()
       if not pathlib.Path(f).exists() or token not in pathlib.Path(f).read_text()]
if bad:
    print("  FAIL corrected code missing from:", *bad, sep="\n        ")
    sys.exit(1)
print("  ok   all four fixes verified present in the source")
PY

# ───────────────────────────── 2. python dependencies
bold "2/5  Python dependencies"
python3 - <<'PY'
import importlib, subprocess, sys
need = []
for mod, pkg in (("pandas","pandas"), ("numpy","numpy"), ("sklearn","scikit-learn"),
                 ("statsmodels","statsmodels"), ("socceraction","socceraction")):
    try:
        importlib.import_module(mod)
    except Exception:
        need.append(pkg)
if need:
    print("  installing:", " ".join(need))
    # multimethod>=2 breaks pandera, which socceraction imports
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "multimethod<2.0", *need])
else:
    print("  ok   all present")

import socceraction  # noqa: F401 - the real test
from socceraction.data.statsbomb import StatsBombLoader  # noqa: F401
print("  ok   socceraction imports cleanly")
PY
[ $? -eq 0 ] || die "dependency install failed — see the error above"

# ───────────────────────────── 3. folder structure
bold "3/5  Folder structure"
mkdir -p processed-data
ok "processed-data/"

# ───────────────────────────── 4. raw StatsBomb data
bold "4/5  StatsBomb open data"
if [ -d "open-data/data/events" ] && [ -f "open-data/data/competitions.json" ]; then
    N=$(ls open-data/data/events 2>/dev/null | wc -l | tr -d ' ')
    ok "already present — $N event files"
else
    echo "  cloning ~5 GB, this takes 10-20 minutes..."
    rm -rf open-data
    git clone --depth 1 https://github.com/statsbomb/open-data.git \
        || die "clone failed — check your network, then re-run: bash SETUP.sh"
    N=$(ls open-data/data/events 2>/dev/null | wc -l | tr -d ' ')
    ok "cloned — $N event files"
fi
[ -d "open-data/data/events" ]  || die "open-data/data/events missing"
[ -d "open-data/data/lineups" ] || die "open-data/data/lineups missing"
[ -d "open-data/data/matches" ] || die "open-data/data/matches missing"

# ───────────────────────────── 5. smoke test on 20 matches
bold "5/5  Smoke test — processing 20 matches"
echo "  (full run comes later; this is only to check the PPDA fix took)"
python3 src/process_dataset.py --limit 20 --output-dir processed-data-smoke 2>&1 | tail -22

bold "VERDICT"
python3 - <<'PY'
import sys, pathlib, pandas as pd

p = pathlib.Path("processed-data-smoke/match_ppda.csv")
if not p.exists():
    print("  FAIL  no match_ppda.csv — processing did not complete")
    sys.exit(1)

d = pd.read_csv(p)
v = d["ppda"].dropna()
if v.empty:
    print("  FAIL  every PPDA value is undefined")
    sys.exit(1)

med, undef = float(v.median()), float(d["ppda"].isna().mean())
print(f"  PPDA median         {med:6.2f}      (target 8-15, was ~2.4)")
print(f"  undefined rate      {undef:6.1%}      (was 22.8%)")
print(f"  matches processed   {d['game_id'].nunique():6d}")

good = 8.0 <= med <= 15.0
print()
if good:
    print("  \033[32mFIX CONFIRMED.\033[0m PPDA is in the published range.")
    print("  Next:")
    print("      rm -rf processed-data-smoke")
    print("      python3 src/process_dataset.py")
    print("      python3 src/run_ppda_diagnostics.py")
    print("      python3 src/run_style_clustering.py")
    print("      python3 verify_fixes.py --open-data open-data \\")
    print("          --ppda-15min processed-data/ppda_15min.csv \\")
    print("          --clusters   processed-data/team_style_clusters.csv \\")
    print("          --windows    processed-data/ppda-diagnostics/ppda_window_diagnostics.csv")
    print("      python3 src/run_did.py")
    print("      python3 src/run_did_count.py")
else:
    print("  \033[31mSTILL WRONG.\033[0m PPDA should be 8-15.")
    print("  Send this output back before running anything else.")
sys.exit(0 if good else 1)
PY
