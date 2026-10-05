---
name: land
description: Land a change the way a Tend project expects - find where it belongs and what binds it before writing code, then record it in the design (the node, its status, its code, decisions with what they rest on), check how far the code carries it, and catch what it left stale. Use when starting a feature or fix in a repo with a .tend/ directory, and again before saying it's done.
---

# Land a change, with Tend

## Before writing code

1. `design_locate {description}`: which node it belongs under. If the evidence is thin, say so and ask.
2. `design_rules {node}`, and `design_impact` for each file you'll touch: who owns it, what depends on it,
   and the decisions in force.
3. `env_toolbox {query}` before choosing a library, tool or pattern; `env_dependency {package}` before
   adding one (is it already used, and is its license allowed?).
4. `design_plan`: is it already planned (a node, a revision, a todo)? If so, work from that.

## While writing

- Follow the rules and the toolbox. If one should change, propose that (`design_propose`) rather than
  quietly not following it.
- Document new public items, and tag code that implements a node with a `tend: <node path>` comment line
  where the link isn't obvious from the code links.

## After

1. `design_propose` in one call:
   - the node (`add`), or its status (`status`), and its code (`link`);
   - checks for its tests (`check`, with `code` naming the test file so its reach can be verified);
   - journal entries: decisions with `rests_on` (the readings, sections or nodes they rest on), missteps
     with the lesson, obstacles.
2. `impl_score {node}`: how far its code, documentation and tests carry it. Fix what's lost: in the code,
   its docs, its tests, or the design.
3. `tend_basis` (no address): did the change leave a document section, decision or score stale? Update
   it, or say what's stale and why it's fine.
4. Commit with a trailer naming the node (`Tend-Node: <path>`), so `code_history` and `design_history`
   connect the two.

Say what's proposed and what's waiting for the user (`tend accept <id>`); never accept your own proposals.
