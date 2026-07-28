@echo off
REM Launch run-dedup.sh in a process tree that does NOT belong to the Claude Code
REM session. Measured three times: a `run_in_background` bash task is reaped when the
REM session rotates, killing gralph mid-round (the script dies during `gralph run`, so no
REM "[loop] round N:" line is ever printed and the exit status still reads 0).
REM
REM   run-dedup-detached.cmd                          <- from cmd
REM   cmd //c start "" /min run-dedup-detached.cmd    <- from git-bash
REM
REM Progress goes to .gralph/dedup/run-until-done.log as usual. The loop rings
REM notify() (bell + Windows toast) at DONE / STUCK / TIMEOUT — with no parent session to
REM re-invoke, that is the only completion signal a detached run gives you.
cd /d "%~dp0"
"C:\Program Files\Git\bin\bash.exe" run-dedup.sh
