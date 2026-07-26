# How Bun Ported Half a Million Lines of Zig to Rust in 11 Days

## Scale and timeline

Excluding comments, Bun is 535,496 lines of Zig [1]. The port ran as about 50 dynamic workflows in Claude Code run continuously over the course of 11 days [1]. The merge landed on May 14, 2026 after 6,502 commits [1]. Press coverage counted the total migrated line count differently, at 960,000 to 1,009,257 lines including generated Rust output — a broader scope than the primary source's Zig-only count.

## Harness design: worktree sharding

The work was split into 4 worktrees, each running 16 claudes, for 64 claudes in parallel [1]. Claude serialized the porting-rules discussion into a PORTING.md document, and the lifetime analysis of every struct field was recorded in a LIFETIMES.tsv table; every ported file had to match the PORTING.md and LIFETIMES.tsv [1]. Before asking Claude to translate all 1,448 .zig files to .rs files, the trial run started with just 3 [1].

## Review separation

Every line of code was reviewed by two separate adversarial reviewers and went through a round of fixes before committing [2]. The implementer doesn't review, and the reviewer doesn't implement; reviewers see only the diff and are told to assume the code is wrong [2]. Zig's creator dismissed the merged result as unreviewed slop, a criticism Bun answered by pointing at this adversarial review pipeline — both positions deserve mention in any honest account.

## Compile errors as a work queue

Fixing the cyclical dependencies revealed about 16,000 compiler errors [1]. cargo check wrote the errors to a file grouped by crate, and the workflow divvied them up among the 64 claudes as a work queue [1]. At peak Claude wrote about 1,300 lines of code per minute [2].

## Validation layers

Beyond the test suite with over a million assertions, the port went through 11 rounds of security review from Claude Code Security, and 24/7 coverage-guided fuzzing of every parser [1]. Fuzzilli fuzzes the runtime APIs around the clock.
