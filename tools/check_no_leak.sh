#!/usr/bin/env bash
# Pre-push guard. Fails loudly if anything that must stay private has crept in.
#
# Rules 1-3 are the original guard. Rules 4-5 and the wider path patterns were added after a
# release-tree audit: the guard checked only /Volumes/, so it would not have caught a home
# directory path or a file:// URL, and two scripts added to the mirror in the same audit did
# carry an absolute author-machine path. Every rule prints the offending file rather than a
# bare verdict — a guard that reports only pass/fail teaches you nothing when it is wrong.
#
# `scan` filters this file out of its own results by path instead of by grep --exclude,
# because --exclude is not honoured identically across grep implementations (BSD grep, GNU
# grep and ugrep all differ) and a guard that silently stops excluding is a guard that
# silently starts lying.
set -u
fail=0
say(){ printf '%s\n' "$*"; }

# `.git/` is filtered by path as well as by --exclude-dir. Neither --exclude-dir nor --exclude is
# honoured identically across grep implementations, and on a repository cloned from a local path
# .git/config holds that absolute path — so a guard trusting --exclude-dir reports a leak that is
# not in the published content, or worse, trusts it elsewhere and misses one that is.
scan(){ grep -rIlE -- "$1" . --exclude-dir=.git 2>/dev/null \
        | grep -v -e '^\./\.git/' -e 'tools/check_no_leak\.sh$' || true; }

report(){                       # $1 = pattern, $2 = message
  local hits; hits=$(scan "$1")
  if [ -n "$hits" ]; then say "FAIL: $2"; printf '  %s\n' $hits; fail=1; fi
}

# 1) No local absolute paths, and no file:// URLs.
#    `scan` uses grep -I, which skips binary files, so the figure PDFs are checked separately
#    below rather than being silently counted as clean.
report '/Volumes/'  'absolute path under /Volumes/ :'
report '/Users/'    'absolute path under /Users/ :'
report '(^|[^A-Za-z0-9])/home/' 'absolute path under /home/ :'
report 'file://'    'file:// URL :'

# 2) No reference that reaches into the private data root, and no 001/002 dataset name.
#
#    Scoped to references that could actually locate the private tree. The bare token also
#    appears in prose and in the documented fallback for the ARIS_RAW_ROOT environment
#    variable
#        RAW_ROOT = Path(os.environ.get("ARIS_RAW_ROOT", str(ROOT.parents[1] / "05data")))
#    which resolves outside this repository, ships no data, and is how a reader reaches the
#    avocado source after downloading it from its Mendeley DOI. Failing on that string would
#    force a rewrite of scripts that must stay byte-identical to the ones that produced the
#    published numbers, so it is allowed and nothing else is.
#
#    The first alternative requires at least one real directory name before the token, so an
#    absolute path such as /Volumes/<disk>/05data fails while the relative prose form
#    `<project root>/../../05data` does not; the second catches any concrete subpath below
#    it. Rule 1 independently catches every absolute form, so nothing rests on this alone.
report '/[A-Za-z0-9_ -]+(/[A-Za-z0-9_ -]+)*/05data|05data/[A-Za-z0-9_]' \
       'path reaching into the private data root :'
report '00[12]_apple' 'private dataset 001/002 named :'

# 3) No bulk/raw data blobs that could actually be pushed.
#
#    A blob fails if git tracks it, or if git does not ignore it. Both halves matter: the
#    tracked test catches a blob that was force-added or that predates the ignore rule, and the
#    not-ignored test catches someone deleting the ignore rule. Untracked *and* ignored is the
#    one benign case — several analysis scripts write an .xlsx workbook beside their JSON
#    output, so a reviewer who reproduces the results ends up with blobs in the working tree
#    that can never be pushed. Those are counted rather than ignored silently, so a guard run
#    right after a successful reproduction does not cry wolf.
blobs=''; ignored=0
while IFS= read -r b; do
  [ -n "$b" ] || continue
  if git ls-files --error-unmatch -- "$b" >/dev/null 2>&1; then blobs="$blobs $b"
  elif git check-ignore -q -- "$b" 2>/dev/null; then ignored=$((ignored+1))
  else blobs="$blobs $b"; fi
done <<EOF
$(find . -path ./.git -prune -o \( -name '*.npz' -o -name '*.npy' -o -name '*.xlsx' \
   -o -name '*.mat' -o -name '*.parquet' \) -print 2>/dev/null)
EOF
if [ -n "${blobs// /}" ]; then
  say "FAIL: raw/bulk data file present and not ignored:"
  printf '  %s\n' $blobs
  fail=1
fi
[ "$ignored" -gt 0 ] && say "note: ${ignored} generated workbook/blob file(s) present but git-ignored — reproduction side effects, not publishable content."

# 4) No credentials or tokens.
report 'ghp_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|BEGIN [A-Z ]*PRIVATE KEY' \
       'credential-shaped string :'

# 5) Report, without failing, the absolute interpreter paths recorded inside the archived run
#    receipts. They are provenance, not leakage: recording which interpreter produced a result
#    is the purpose of a receipt, those files are hash-registered in the shipped data card, and
#    redacting them would invalidate the published SHA-256 manifest. Counted so the number is
#    visible rather than silently tolerated.
interp=$(scan '/opt/miniconda3' | wc -l | tr -d ' ')
say "note: interpreter path recorded in ${interp} archived receipt/lock file(s) — provenance, hash-registered, not redacted."

# 6) Binary files, which rule 1 skips. Anything matching here is a path baked into a PDF or
#    other non-text artefact, which no amount of source-tree tidying would have caught.
binhits=$(grep -rlE -e '/Users/|/Volumes/|/home/|file://|05data|00[12]_apple' \
               -e 'ghp_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}' . --exclude-dir=.git 2>/dev/null \
          | grep -v -e '^\./\.git/' -e 'tools/check_no_leak\.sh$' \
          | while IFS= read -r f; do
              grep -qI . "$f" 2>/dev/null && continue          # text: rule 1 already covered it
              git check-ignore -q -- "$f" 2>/dev/null && continue   # e.g. __pycache__, never pushed
              printf '%s\n' "$f"
            done)
if [ -n "$binhits" ]; then
  say "FAIL: private path inside a binary file:"
  printf '  %s\n' $binhits
  fail=1
fi

[ "$fail" -eq 0 ] && say "OK: no local absolute paths, no private-root paths, no raw data blobs, no credentials, nothing in binaries."
exit "$fail"
