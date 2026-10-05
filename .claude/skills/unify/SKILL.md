---
name: unify
description: Simplify recently changed code and make it consistent with how the rest of the repo does the same thing - error handling, logging, configuration, paths and IO, async, serialization, CLI output, naming, module layout, tests. The accepted way comes from the toolbox's patterns and the design's rules and decisions first, then from what most and most recent code does; behavior never changes. Use after finishing a change, when asked to simplify or clean up, or to bring a part of the design (a node) in line.
---

# Unify: simpler code, and one way of doing each thing

Adapted from Anthropic's `code-simplifier` (claude-plugins-official, Apache-2.0). Consistency across the
whole repo is added with Tend: the design tree says what code belongs together, the toolbox says which
way is accepted, and the code graph and history say how the code does it now and which way it's moving.

## Scope

By default, the code changed in this session or on this branch (`git diff`). Or a design node's code
(`design_impact {node}` lists its files). Or the whole repo when asked, done one node at a time.

## 1. Simplify, without changing behavior

- Never change what the code does: its outputs, errors, side effects and timing stay the same.
- Less nesting and indirection. Remove redundant code and abstractions that serve one caller. Bring
  related logic together.
- Clear names. No nested conditional expressions where a `match`, a `switch` or `if`/`else` reads better.
  Explicit over compact.
- Remove comments that restate the code. Keep the ones that say why.
- Don't over-simplify. Keep the abstractions that organize the code. Don't merge concerns. Don't trade
  readability for fewer lines.

## 2. The concerns in scope

Name the recurring concerns the code in scope touches. Common ones:

- errors: their types, how they propagate, their messages;
- logging and what's printed;
- configuration and environment variables;
- paths, files and IO;
- running processes;
- serialization (JSON, TOML);
- async and concurrency;
- CLI output;
- how similar things are named;
- how modules and files are organized;
- tests: helpers, naming, fixtures.

## 3. How the repo does each, with Tend

The accepted way first:

- `env_toolbox {query: "<concern> in <language>"}`: patterns (what each is `for`, `avoid_when`, the tools it
  `uses`, an `example`), and tools to use or avoid.
- `design_rules` for the nodes owning the code.
- `design_journal`: decisions in force that name the concern, and missteps about it.
- `tend_find {query: "<concern>"}`: anything else that says it.

Then the code:

- `tend_find`, `code_refs` and `code_symbol` for the concern's markers (an error crate, `unwrap`,
  `map_err`, a logger, `println!`, a config loader).
- `code_outline` of a few files that are typical of the repo.
- `code_graph {node}` to see which parts of the design do it which way.
- `code_history` for when each way appeared. The newer one is the way the repo is moving.

Write it down as a table: each variant, the files and nodes using it, how many, its newest use, and
whether a toolbox pattern, rule or decision names it.

## 4. Which way is accepted

1. A toolbox pattern, or a design rule, that names it.
2. A decision in force that names it.
3. What most of the code does, weighted toward what newer code does.
4. The baseline's conventions (`env_baseline`).

If two ways are close, or what's written down contradicts what most code does, stop and ask the user
which way to go. Offer to record their answer (step 6).

## 5. Unify

- Change the outliers in scope to the accepted way, keeping their behavior.
- Work in small batches, one node at a time.
- Run that node's checks (`design_node` lists them), or say which you couldn't run.
- Leave outliers outside the scope alone and list them. With the user's agreement, file a todo for each
  group (`design_propose` action `todo`, `refs` to the code and to `toolbox:pattern:<name>`).

## 6. Record it, so the next run agrees

If the accepted way wasn't written down anywhere:

- Propose it for this project as a toolbox pattern on the root node: `design_propose` action `toolbox`
  with `kind: pattern`, `name`, `for`, `uses`, `example`, `avoid_when`.
- Record a journal decision that `rests_on` the files that show it.
- If it would serve every project, suggest it for the baseline: `graft_file` to the baseline project
  (`graft_look` first), or `tend baseline promote` for a lesson in the journal.

## Never

- Change behavior.
- Edit outside the scope in bulk without asking.
- Break a rule or the conduct to make code look the same.
