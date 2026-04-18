# EveryQuery refactor — collaboration summary

A record of how this refactor effort has been approached over the session, from a 25-line stale README to a Phase-1-complete + Phase-2-in-flight repository. Intended as context for future-you or a new collaborator picking up where we left off.

## The shape of the collaboration

The process has been a tight loop of:

1. **You** gave a direction — sometimes a specific task, sometimes a question, sometimes a judgment call on an open question I'd surfaced.
2. **I** executed or investigated, then came back with either a result or a concrete recommendation for a decision.
3. **You** or **GitHub collaborators** (Matthew, Payal, Greg) reviewed, pushed back, corrected, or approved.
4. **Loop** continued until the thing was either done or clearly blocked on a team-level decision.

A few patterns worth naming explicitly, because they shaped the work:

- **Small explicit corrections, often mid-flight.** You redirected me several times with short, precise notes — *"don't rebase, merge"*, *"revert the pretrain rename"*, *"scope `paper_experiments` is research-intent not infrastructure"*, *"close #88, open a narrower one"*. Each one re-aimed the work without requiring a restart. I saved several of those as feedback memories (`~/.claude/projects/.../memory/`) so the same correction wouldn't be needed twice.
- **Empower-then-verify.** You routinely gave me latitude ("proceed as far as you can on your own") but followed through on reviewing what came back. That combination kept momentum high without losing control of direction.
- **Trust the chain of review.** Copilot reviewed every PR; Payal and Matthew reviewed the substantive ones; you arbitrated when we disagreed. The multi-reviewer loop caught real bugs (the `do_overwrite=do_resume=True` edge case on #31, the entry-point missing `if __name__ == "__main__"` guard on #74) and also prevented a couple of scope-creep decisions from landing.

## The big structural decisions we made along the way

In rough chronological order, with the short version of each decision and how it got made:

1. **Use `gh api` with JSON payloads, not `gh pr create --body`, for multi-line PR/issue content.** Discovered early that `gh pr create/edit` + `gh issue create/comment` silently wrap backticks in backslashes — your "be careful" correction surfaced the bug, I saved it to memory, and the workaround became the default.
2. **Merge dev into feature branches, do not rebase.** Your correction after I reflexively rebased PR #76. Preserves commit SHAs, doesn't churn review context. Saved to memory.
3. **Refactor/rearrange only — stop at new functionality.** Your mid-session guardrail after several small fixes. Shaped the rest of the session: I prepped draft PRs up to the boundary and then stopped for team input on anything that crossed it (#80 schema, #81 predict, #83 evaluate).
4. **Eval stays in-repo, not a separate package.** Payal's NeurIPS-timing pushback, Matthew's clarification on *what kind of eval* stays: dataset-agnostic experiment code (ACES, composite, metrics) stays; cluster-specific ops code moves out. That framing changed Phase 2's plan substantially (we'd originally proposed moving eval to a separate repo).
5. **Paper experiments is a research-intent split, not a packaging one.** My misread — I initially framed `paper_experiments/` as outside `src/`, lower-CI. You corrected: it stays in `src/`, gets full CI + tests + PyPI, but is organized under a clearly-named submodule so the normal pipeline stays obvious.
6. **Keep `EQ_train` / `EQ_generate_tasks`, don't rename to pretrain / prepare_tasks.** Matthew's call on PR #90. Preserves the substantive submodule-restructure win while avoiding churn on stable-enough CLI names.
7. **`data/` is separate from `model/`.** Your direct question mid-restructure (*"why not two submodules?"*) made me reconsider. The decision rests on evolution cadence: the data-layer shape changes when the schema changes; the `nn.Module` doesn't. Separating gives a schema PR a tight diff.
8. **`TaskQuerySchema` extends MEDS `LabelSchema` — flat code + continuous duration, nothing more.** Your direction on #80, keeping the initial scope narrow. Mirrors meds-evaluation's `PredictionSchema` pattern.
9. **Scaffolding-heavy Phase 1 PR (#90) is the right size.** Your push: include everything — `train/`, `generate_tasks/`, `model/`, `data/`, `predict/external_tasks/`, `evaluate/`, `paper_experiments/sample_codes/`, top-level README rewrite. One PR that collaborators can see every relocation in. Subsumed the separate README PR (#69) and the narrower #88/#92 attempts.

## How we handled disagreement and uncertainty

- **Pushed back when pushback was warranted.** Several times I disagreed with a reviewer (or with Copilot) and said so, with reasoning. Examples: pushing back on #34 Copilot's "test is redundant with doctest" (the doctest covered the fix, the test covered the bug site — both had distinct value); pushing back on #97 Copilot's "file-mode inconsistency" (the inconsistency is intentional — content-hashed output paths vs. user-supplied output paths have different idempotency semantics).
- **Asked for sign-off before stepping into new-functionality territory.** #80 schema design had an explicit four-question list in the issue body; #81 predict has an explicit design-doc comment with four more. When the decisions were mine to make, I made them and flagged them with reasoning; when they were judgment calls that affected external contract, I left them for you + the team.
- **Filed upstream issues instead of diverging silently.** When Copilot surfaced code-quality concerns on the PR #74 preprocessing port that MEDS_EIC_AR inherits from, I filed `mmcdermott/MEDS_EIC_AR#112/#113/#114/#115` so the two repos don't drift. Then applied the fixes here once you clarified that divergence was permitted.

## The shape of the issue tracker at this snapshot

Opened this session: #79, #80, #81, #82, #83, #84, #85, #91.
Closed: #31, #32, #52 (earlier), #53 (earlier), #55, #59, #63 (earlier), #67 (earlier), #75 (earlier), #79, #84.
Paused: #64 (superseded pending Phase 2), #88 (closed without merging), #92 (closed without merging per your judgment).
Live: #54 (umbrella), #62, #66, #68, #81, #82, #83, #85, #91.
Upstream MEICAR: #112, #113, #114, #115.

Sub-issues on #54: the 15 (now 16) related issues are linked as GitHub sub-issues so #54's progress bar reflects the refactor's state.

## Open loops for when you pick this up next

These are the judgment calls I deliberately left for you:

1. **`EQ_predict` design doc** ([#81 comment](https://github.com/payalchandak/EveryQuery/issues/81)). Four open questions — output naming, embedding opt-in vs. out, label column pass-through, missing-code behavior. Signing off unblocks the implementation PR.
2. **Eval inventory sign-off** ([#82 comment](https://github.com/payalchandak/EveryQuery/issues/82)). I've classified every operation in `evaluate/`; Phase 2.4 (#83) waits on your agreement on the split, particularly (a) `select_model` moving to `paper_experiments/`, (b) dropping the `model_run_dirs` list, (c) where the eval-time per-subject index sampler lives post-consolidation.
3. **`sample_codes` entry-point registration**. I left them as `python -m ...` invocations rather than `EQ_sample_*` scripts. Easy one-line change if you'd prefer the top-level surface.
4. **Workflow approval for pending CI**. Several PRs' CI is in "action_required" state (Copilot-bot commits need repo-maintainer approval to run workflows). That approval is yours to give.

## Tools that worked well

- `gh api` for anything with backticks or cross-issue linking — more reliable than `gh pr`/`gh issue` wrappers.
- Sub-issues (new GitHub feature) for tying the refactor umbrella together visually.
- Copilot auto-review + auto-commit of small fixes (terminology nits, mdformat drift) — consistently useful, occasionally wrong in ways worth pushing back on.
- Feedback-memory entries under `~/.claude/projects/.../memory/` — persisted three corrections you made (don't rebase, don't use `gh pr --body` with backticks, read linked docs before hand-rolling) across sessions.

## Tools that didn't

- `sed` for complex edits inside files with active pre-commit hooks — hooks re-write the file, my in-session state goes stale, and the next Edit errors with *"file has been modified since read"*. Mostly worked around by reading first.
- Asking a web-fetching sub-agent for verbatim bytes — returns the intermediate model's paraphrase of the file, not the bytes. Had to fall back to `gh api .../contents/...` with base64 decode for `meds-evaluation`'s schema.

## What I'd flag for a fresh reviewer

- The whole refactor was scoped to be **completable in one focused session**. If Phase 2 + beyond drags out, the submodule scaffolding from Phase 1 stays useful even if Phase 2's `EQ_predict` / consolidated `EQ_evaluate` take months. The boundary between *"layout correct"* and *"behaviors consolidated"* is clean.
- **Copilot's suggestions are worth reading carefully, but aren't automatically right.** Several of its low-confidence nits were genuinely correct, a couple were wrong in defensible ways, and one (PR #98) actively duplicated work from a sibling PR in a way that would have created a merge conflict.
- **The `#54` umbrella + sub-issues structure is the canonical map of this work.** If you want to know what's done, in flight, or planned, `#54` has the live status. Individual PR / issue descriptions are honest about their scope boundaries and point at related work.
