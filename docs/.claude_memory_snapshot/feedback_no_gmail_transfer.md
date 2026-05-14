---
name: Don't use Gmail (or any auto-send) to deliver archives or files to colleagues
description: User explicitly asked not to send archives via Gmail 2026-05-13. Build the archive locally and let user handle delivery.
type: feedback
originSessionId: 28eafb54-f262-4642-b4e4-695fffb66865
---
When the user asks for a code archive (e.g. `tradefm.zip`) to take to the cluster or share with a colleague, **build the file locally and stop there**. Do not create Gmail drafts, do not attempt to send, do not propose Drive/Dropbox uploads automatically.

**Why:** stated preference, 2026-05-13. The Gmail flow was used briefly earlier but the user now wants to handle file delivery themselves.

**How to apply:**
- After `zip`/`tar`, report the file path + size and stop.
- If the user explicitly re-enables a transfer channel in a later session ("send via X"), follow that, but don't volunteer it.
- Existing Gmail drafts I created in past sessions can stay — user will clean them up.
