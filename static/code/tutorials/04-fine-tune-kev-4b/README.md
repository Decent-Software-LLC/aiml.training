# Fine-Tune Kev-4B — code files

This directory holds the reproducible inputs for the [Kev-4B tutorial](../../../../docs/tutorials/fine-tune-kev-4b-decision-model.md):

| File | What it is |
|------|-----------|
| `launch-kev-ft.sh` | The exact training command, wrapped in tmux (paths scrubbed — point `KEV_DIR` / `KEV_OUT` at your own directories) |
| `kev-train.sample.jsonl` | One sample record in Kev's native JSONL format: ticket text as `state`, one `choice` question, 11 options with `criteria`, and the `label` |

## Not here on purpose

The `kev` package itself — `kev.train`, `kev.benchmark`, the pointer head, the
FastAPI server — is **System One's open-source package**
(`jaredpalmer/kev-4b` on the Hugging Face Hub). It is not part of this site's
code: install/clone it from its own repository and drive it as a module:

```bash
uv run python -m kev.train --data kev-train.jsonl --init_from jaredpalmer/kev-4b ...
uv run python -m kev.benchmark --run ./kev-fin --data kev-heldout.jsonl --out ./kev-fin/eval-ft
```

The full commands, with every flag explained, are in the
[tutorial](../../../../docs/tutorials/fine-tune-kev-4b-decision-model.md).
