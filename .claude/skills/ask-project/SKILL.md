---
name: ask-project
description: Ask another of the user's projects for something it should own - a feature, a fix, a change, an answer - as a Tend graft, after checking that project's plan for what already covers it (planned, done, retired, declined). Use when the work you need belongs in another repo, instead of editing that repo or working around it.
---

# Ask another project

1. **Look first.** `graft_look {to, query}`: what in that project may already cover the ask, and how each
   stands.
   - planned or in progress (a node, revision or todo): rely on it. If your work waits on it, record that:
     a todo here with `waits_on: ["graft:<id>"]`, or with what it says.
   - done: use it; maybe nothing to ask.
   - retired, abandoned, dropped or declined: it was refused for a reason (read it). Ask the user before
     asking again.
   - nothing close: ask.
2. **Ask** with `graft_file`:
   - `kind` and a `title`;
   - `need`: concretely what, not how;
   - `use_case`: what you'll do with it, so they can design for it;
   - `refs`: the addresses here it's for;
   - `rests_on`: what the ask is based on (a spec at its version, a design node here);
   - `waits_on`: anything that must happen first.

   If it stops because something there looks close, either rely on that, or file again with `despite`
   saying how yours differs.
3. **Don't wait.** Nobody has to be running there. Record what depends on the answer as a todo here that
   waits on the graft. `graft_list` shows the answer when it comes, and what carries it.
4. Never edit another project's code or design. Its own agents and the user do.
