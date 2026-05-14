# Claude memory snapshot

Snapshot of `~/.claude/projects/-Users-mem4-Documents-tradefm/memory/` for
cross-device portability. Claude's per-project memory does **not** sync
automatically — copy this directory into the active memory location on a new
device to preserve continuity:

```bash
mkdir -p ~/.claude/projects/-Users-mem4-Documents-tradefm/memory
cp -R docs/.claude_memory_snapshot/. ~/.claude/projects/-Users-mem4-Documents-tradefm/memory/
```

After Claude has run for a while on the new machine, refresh the snapshot:

```bash
cp -R ~/.claude/projects/-Users-mem4-Documents-tradefm/memory/. docs/.claude_memory_snapshot/
git add docs/.claude_memory_snapshot/ && git commit -m "Refresh memory snapshot"
```

The snapshot is a point-in-time copy. It is not used by Claude directly; only
the live memory dir is read at runtime.
