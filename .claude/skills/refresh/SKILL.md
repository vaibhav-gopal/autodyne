---
name: refresh
description: Find what no longer holds in a Tend project - document sections, decisions, readings, scores and test results resting on ground that moved, and baseline changes that affect this repo - and bring it up to date in order, cheapest first. Use when tend_basis or the session context reports stale ground, after a large change, or when asked whether the docs or decisions are still current.
---

# Refresh what's stale

Refreshing is refine's stale-knowledge part: for one part of the design, `design_refine` lists what under
it is stale with the rest of what would bring it level with its code. For the whole project:

1. `tend_basis` with `order: true`: the remake list, everything not fresh in the order to make it again
   (each after what it rests on, what costs least first), code facts first since they cost no tokens.
   Each says what remaking it costs (lazy, run, agent, person) and the call that does it.
2. Take them in that order:
   - **run:** a measured score with `impl_score`. A check result is the user's to rerun
     (`tend code checks <node> --run`): say so.
   - **agent:** a reading (`doc_reading`, then `doc_check`), a judged score (`impl_score {fresh: true}`),
     a design document written again from `doc_write`'s outline. These spend a fresh model's tokens:
     ask first when there are many.
   - **person:** a decision resting on what moved. Read what moved (`tend_basis {address}` gives the chain),
     then propose a superseding decision resting on what holds now, or say why it still holds. A
     written document's section: rewrite it from what moved. A citation: read the source again and
     record the version read (`cite_basis`).
3. For anything you don't refresh, say why (it's fine as it is, or it needs the user).
4. `tend_basis` again: what's left, and why.
