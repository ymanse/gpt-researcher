#!/usr/bin/env bash
# Drive `gralph run dedup.yaml` until DONE, surviving Claude usage limits.
# Run from the harness-search dir:  ./run-dedup.sh
#
# Why an outer loop instead of a GALP `launcher:` — on Windows the launcher never receives
# the prompt (argv strips {{prompt}}'s braces) and its rate-limit pattern both misses the
# real banner and false-positives on a build's own output. gralph's own give-up (5 abnormal
# exits without cursor progress) is a SAFE stop — state and store are preserved — so all we
# need is to notice WHY it stopped and resume. See HARNESS.md.
#
# Stall detection is CURSOR-IDENTITY based (not linear stage index): the s6-benchmark gate
# legitimately routes BACKWARD to an earlier stage's impl on a below-baseline metric, so
# "index went down" is progress here, not a stall. Only an unchanged cursor across a whole
# round with no usage-limit banner counts toward STALL_LIMIT.
set -u
cd "$(dirname "$0")" || exit 1
command -v gralph >/dev/null 2>&1 || { echo "gralph not on PATH"; exit 127; }

# Nested `claude -p` sessions refuse to start when CLAUDECODE is inherited from a parent
# Claude Code session; Korean Windows (cp949) needs PYTHONUTF8 for subprocess output.
unset CLAUDECODE
export PYTHONUTF8=1

PROFILE="${PROFILE:-dedup.yaml}"
DIR="${DIR:-.gralph/dedup}"
LOG="${LOG:-$DIR/run-until-done.log}"
COOLDOWN="${COOLDOWN:-1800}"     # sleep when a usage limit is what stopped us
ITERS="${ITERS:-15}"             # gralph iterations per round (hard per-round seatbelt)
MAX_ROUNDS="${MAX_ROUNDS:-80}"
STALL_LIMIT="${STALL_LIMIT:-2}"  # consecutive same-cursor rounds before handing to a human

# Anchored on the Claude CLI's limit banner. Deliberately NOT matching bare
# "quota"/"rate limit"/"429": healthy research output prints those words.
LIMIT_RE="hit your [a-z0-9-]+ limit|usage limit reached|rate limit exceeded|Claude usage limit"

cursor() { python -c "import json;print(json.load(open('$DIR/state.json'))['cursor'])" 2>/dev/null || echo "?"; }
# Completion alarm — the loop and any interactive session are separate processes, so nothing
# surfaces DONE/STUCK on its own. Terminal bell + Windows toast (BurntToast -> msg * -> bell).
notify() { printf '\a'; powershell -NoProfile -Command "New-BurntToastNotification -Text '$1','$2'" >/dev/null 2>&1 || msg '*' "$1: $2" >/dev/null 2>&1 || true; }

stall=0
for round in $(seq 1 "$MAX_ROUNDS"); do
  [ "$(cursor)" = DONE ] && { echo "[loop] cursor=DONE after $((round-1)) round(s)"; notify "gralph dedup DONE" "flow complete"; exit 0; }

  before="$(cursor)"
  rm -f "$DIR/lock"
  rlog="$(mktemp)"
  gralph run "$PROFILE" --max-iterations "$ITERS" 2>&1 | tee -a "$LOG" | tee "$rlog"
  after="$(cursor)"
  echo "[loop] round $round: cursor $before -> $after"

  if [ "$after" = DONE ]; then echo "[loop] cursor=DONE"; rm -f "$rlog"; notify "gralph dedup DONE" "flow complete after $round round(s)"; exit 0; fi

  if grep -qiE "$LIMIT_RE" "$rlog"; then
    rm -f "$rlog"
    echo "[loop] stopped by a usage limit; sleeping ${COOLDOWN}s then resuming"
    sleep "$COOLDOWN"
    stall=0                      # a limit is not a stall
    continue
  fi
  rm -f "$rlog"

  if [ "$after" != "$before" ]; then stall=0; else stall=$((stall + 1)); fi
  if [ "$stall" -ge "$STALL_LIMIT" ]; then
    echo "[loop] $stall rounds with the cursor stuck at $after and no usage limit — a gate is genuinely stuck."
    echo "[loop] Inspect $LOG and $DIR/failures.json; not looping further."
    notify "gralph dedup STUCK" "gate stalled at cursor=$after; needs a human"
    exit 1
  fi
done

echo "[loop] hit MAX_ROUNDS=$MAX_ROUNDS without reaching DONE (cursor=$(cursor))"
notify "gralph dedup TIMEOUT" "hit MAX_ROUNDS=$MAX_ROUNDS, cursor=$(cursor)"
exit 1
