---
title: About
description: "What aiml.training is, how the tutorials are made, and the standards they hold to."
headerLead: "A small, independent library of practical AI/ML tutorials for working engineers — not a course platform, news site, or vendor blog. It is a focused set of deep, hands-on guides."
---

## What we optimize for

- **Usefulness over volume.** A few guides that take you from start to a working, verified result beat a hundred that stop at the setup step.
- **Reproducibility.** Every install block is version-pinned. If you copy the commands, you get the environment the author tested.
- **Honesty.** We state the hardware, the versions, and the numbers we actually measured. Where a figure is an estimate, we say so. We do not publish claims we cannot stand behind.

## How a tutorial is made

1. **Pick a real, specific problem.** A concrete task with a searchable constraint — e.g. “fine-tune a 4B model with LoRA on a single 12 GB GPU” — not a broad topic.
2. **Build it end to end.** Run the full pipeline on the target hardware and record what actually happens: VRAM, latency, failures.
3. **Write the gotchas.** The architecture-specific quirks and failure modes that fragment the existing documentation are the most valuable part of the guide.
4. **Pin and verify.** Lock the versions, re-run, and confirm the numbers before publishing.

## The current focus

The site is building one coherent cluster around **practical LLM engineering on consumer hardware**:

- **Fine-tune** — LoRA fine-tuning of a 4B model on a single 12 GB GPU (including a calibrated decision model).
- **Serve** — packaging the fine-tuned model as a load-tested, OpenAI-compatible API.
- **Evaluate** — measuring whether the fine-tune actually improved the model.

## What this is not

No affiliate links padding the content, no “top 10 tools” listicles, no paywalled gate on the core guides, and no fabricated testimonials or credentials. If you find something that is wrong or outdated, [let us know on GitHub](https://github.com/Decent-Software-LLC/aiml.training) — corrections are the point.

## Contact

- **GitHub:** [github.com/Decent-Software-LLC/aiml.training](https://github.com/Decent-Software-LLC/aiml.training)
