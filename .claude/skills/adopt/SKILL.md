---
name: adopt
description: Bring an existing repo under Tend - start its design from what the code already is (areas and components from its structure, linked to their code), record the decisions visible in its history and docs, attach its tests, and read its README in as a document. Use when asked to set up Tend in a repo that has no .tend/ directory, or to fill in a thin design.
---

# Adopt Tend in a repo

The user runs `tend init --name <project>` (it registers the project with their hub). Then:

1. **Read the repo's shape.** `design_review`'s `drift_from_code.draft`: a first draft derived from the code
   without a model (a component per folder of unowned code, linked to it). `code_graph` for how they
   connect. The README, and any architecture docs.
2. **Propose a first design** in one `design_propose`, starting from the draft:
   - an area per major part, a component per cohesive module, each with a one-line intent saying what
     it's for (not how);
   - each node linked to its code (`link` with a path or glob), so ownership covers the repo
     (`design_impact` with no arguments lists unowned folders);
   - relations where the code shows them (`depends_on`, `uses`);
   - status `done` for what works, `building` for what's partial.
3. **Decisions already made:** from commit history (`code_history`) and the docs, record the big ones as
   journal decisions, with what they rest on.
4. **Tests:** `check` entries for the test commands, with `code` naming the test files, so their reach to
   each node's code is verified.
5. **Documents:** `tend init` read the README in as `readme`; read in any design docs too (`doc_import`).
   Where a section describes a node, it can name it in a `<!-- rests on: design:... -->` comment, so it's
   flagged when the node changes.
6. **Check it:** `design_review` and `impl_score`. Fix what doesn't resolve. Leave gaps in docs and tests
   as todos rather than writing everything now.

Keep the first design small and true. It grows with the work.
