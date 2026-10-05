---
name: adopt
description: Bring an existing repo under Tend - follow `tend adopt`'s steps (the user's trust steps and choices, Tend's first draft and documents), then refine it node by node, its root first, until the design carries the code (owned, tested, checked, documented), recording the decisions visible in its history. Use when asked to set up Tend in a repo, when it has no .tend/ directory, or to fill in a thin design.
---

# Adopt Tend in a repo

Adopting is a fixed order of steps, and `tend adopt --check` says where the repo stands on each (it only
reads: run it). `tend adopt` itself is the user's to run: it takes Tend's steps and stops at the next that's
theirs, with its command.

1. **Where it stands:** `tend adopt --check`. The steps before refining aren't yours: the trust steps
   (`tend init`, the hub signing its history, rendering the MCP server and hooks, passkeys) and the user's
   choices (the baseline and its profiles, the license, the code's scope for a large repo). Name the one it
   stops at, with its command, and wait. The first draft (a part per folder of unowned code, filed as a
   proposal) and the documents (the repo's Markdown held where it is, its TODO list made todos) are Tend's:
   `tend adopt` takes them when the user runs it.
2. **Refine the root:** `design_refine` with the root node. It reads what the code shows, without a model:
   the parts its folders suggest, what each uses, its tests, the rules its code keeps, its packages and
   their licenses, what's unowned, and its parity (owned, tested, checked, documented).
3. **Propose what it found**, in one `design_propose`: its mechanical changes as they are, and the drafts
   you judge right (rules its code keeps, parts the first draft missed). Write what only prose can say, from
   its todos: each part's intent (what it's for, not how), with status `done` for what works and `building`
   for what's partial. Relations the code shows aren't declared: they're read from the code. Declare only
   what code can't show (planned parts, `conflicts_with`, `replaces`, other projects).
4. **Work down:** once the user accepts it, `design_refine` each part, and propose again. A part is adopted
   when its parity is full or what's missing is a todo. In a large repo the code's scope `design` reads only
   what's refined so far; widen it a folder at a time.
5. **Decisions already made:** from commit history (`code_history`) and the docs, record the big ones as
   journal decisions, with what they rest on. Where a document's section describes a node, it can name it in
   a `<!-- rests on: design:... -->` comment, so it's flagged when the node changes.
6. **Check it:** `design_review` and `impl_score`. Leave gaps in docs and tests as todos rather than
   writing everything now.

Keep the first design small and true. It grows with the work.
