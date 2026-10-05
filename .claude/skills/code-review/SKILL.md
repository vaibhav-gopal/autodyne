---
name: code-review
description: Review a change (a pull request, a branch, or what's uncommitted) for real bugs and for drift from the project's design, using Tend - the rules and decisions in force for the code it touches, the missteps made there before, the toolbox, the tests that reach it, and what the change leaves stale. Use when asked to review code or a PR, or before landing a change.
---

# Code review, with Tend

Adapted from Anthropic's `code-review` plugin (claude-plugins-official, Apache-2.0): the same parallel
reviewers and confidence scoring, with the project's design (through Tend's MCP tools) in place of
CLAUDE.md files.

Make a todo list first. Don't build, typecheck or run tests: CI does that.

## 1. What to review, and whether to

- A pull request (a number or URL): `gh pr view`, `gh pr diff`. Stop, and say why, if it's closed, a
  draft, automated or trivially fine, or you already reviewed it.
- Otherwise the branch against its base (`git diff <base>...HEAD`), or what's uncommitted (`git diff HEAD`).

## 2. What the design says about it (once, before any reviewer)

For each changed file, from Tend:

- `design_impact {file}`: the nodes that own it, what depends on them, the rules constraining a change,
  and the decisions in force.
- `design_rules {file}`, with the node each rule comes from.
- `design_journal {node, kind: "misstep"}` for each owning node: what went wrong here before.
- `design_node {node}` for each owning node: its checks, and whether each reaches its code.
- For each new or changed dependency: `env_dependency {package}` (used elsewhere already? its license
  allowed?) and `env_toolbox {name}` (approved, or to avoid, and what instead).
- `tend_conduct` for how this project works.

Put it into a short brief per file: owners, rules (with their node), decisions (with their `journal:`
address), missteps, toolbox entries, and the tests that reach it.

## 3. Review in parallel

Run six subagents at once. Give each the diff and only its part of the brief. Each returns its issues,
with why each was flagged and the evidence: a line, and a rule, decision or commit.

1. **Design:** the change breaks a rule of an owning node, or contradicts a decision in force without
   superseding it. Quote the rule or decision.
2. **Bugs:** a shallow scan of the diff for real bugs. Stay within the changes. Look for bugs that will
   bite, not nitpicks.
3. **History:** `code_history` for the touched files and symbols, and the owning nodes' missteps. Is a
   past mistake being made again? Is something being undone that was done for a reason?
4. **Toolbox and dependencies:** new imports against the toolbox (`avoid`, `instead_of`), and licenses
   against the policy.
5. **Documentation:** the changed symbols' doc comments still say what the code does, `tend:` tags still
   resolve and fit, and changed behavior a document describes is reflected (`code_docs`).
6. **Tests and freshness:** new behavior in a node that has tests but no test reaching it, and tests
   linked to a node that don't reach its code. Also what the change leaves stale: `tend_basis` on each
   owning node, and the document sections and decisions resting on them.

## 4. How sure

For each issue, a fresh, fast subagent gets the change, the issue and the brief, and scores it 0 to 100.
Give it this rubric word for word:

- 0: Not confident at all. This is a false positive that doesn't stand up to light scrutiny, or is a
  pre-existing issue.
- 25: Somewhat confident. This might be a real issue, but may also be a false positive. The agent wasn't
  able to verify that it's a real issue. If the issue is stylistic, it is one that was not explicitly
  called out in the design's rules, decisions or toolbox.
- 50: Moderately confident. The agent was able to verify this is a real issue, but it might be a nitpick
  or not happen very often in practice. Relative to the rest of the change, it's not very important.
- 75: Highly confident. The agent double checked the issue, and verified that it is very likely it is a
  real issue that will be hit in practice. The existing approach is insufficient. The issue is very
  important and will directly impact the code's functionality, or it is directly named by a rule or a
  decision in force.
- 100: Absolutely certain. The agent double checked the issue, and confirmed that it is definitely a real
  issue, that will happen frequently in practice. The evidence directly confirms this.

For an issue flagged by the design, the scorer checks that the rule or decision really says it, by
reading it through Tend (`design_rules`, `tend_explore journal:<id>`).

Keep the issues scoring 80 or more. For a pull request, check it's still eligible (step 1).

## 5. Report

- A pull request: comment with `gh pr comment`, brief and without emojis:

  ```
  ### Code review

  Found 2 issues:

  1. <what's wrong> (rule on Shop / Checkout: "<rule>")

  <link to the file and lines, with the full commit sha: https://github.com/<owner>/<repo>/blob/<sha>/<path>#L<a>-L<b>>

  2. <what's wrong> (bug: <file and code>)

  <link>
  ```

  Or `No issues found. Checked for bugs and against the design's rules, decisions and toolbox.`
  Links need the full sha (no shell substitution: the comment is rendered as written) and a line of
  context on each side.
- Local: the same list, with `file:line` and the addresses (`design:`, `journal:`, `toolbox:`) it rests on.

Then ask before recording anything. Design drift that isn't a bug can become a todo (`design_propose`
action `todo`, with `refs` to the code and the node). A mistake the change shipped can become a misstep
in the journal, with its lesson.

## Not issues

- Pre-existing problems, and real issues on lines the change doesn't touch.
- What a linter, compiler or type checker catches; formatting.
- Style the design doesn't name (no rule, decision or toolbox pattern says it).
- General quality (coverage, docs) unless a rule asks for it.
- Something a rule names that the code explicitly silences, with its reason.
- Changes in behavior that are clearly the point of the change.
