# aiml.training

[![Deploy to GitHub Pages](https://github.com/Decent-Software-LLC/aiml.training/actions/workflows/pages.yaml/badge.svg)](https://github.com/Decent-Software-LLC/aiml.training/actions/workflows/pages.yaml)

Practical, code-first AI/ML tutorials for working engineers — version-pinned pipelines, real hardware, and the numbers you need before you run.

🌐 **Live site:** [https://aiml.training](https://aiml.training)

Every tutorial is built end-to-end on a single 12 GB GPU (NVIDIA RTX 3080 Ti), with exact model versions, exact commands, and the real wall-clock times, memory peaks, and eval results from the run — so you know what to expect before you spend the hours.

## Tutorials

1. **Fine-Tune Qwen/Qwen3.5-4B with LoRA on a Single 12 GB GPU** — the full loop: quantized base model, 4-bit LoRA training, what fits in VRAM and what doesn't.
2. **Serving a Fine-Tuned LLM with vLLM (and the LoRA Trap)** — get a LoRA adapter serving with vLLM, and the dtype/adapter pitfalls that make it silently fail.
3. **Evaluating a Fine-Tuned LLM: Perplexity, Win Rate, and a Judge** — how to actually measure whether the fine-tune improved anything, not just whether it trained.
4. **Fine-Tune Kev-4B: A Calibrated Decision Model on a 12 GB GPU** — fine-tuning an open decision model (not a chat model) from a published checkpoint, warm-starting from a published LoRA, and running a statistically-significant paired comparison against the stock model.

## Layout

- `content/docs/tutorials/` — the tutorial pages (one `.md` per tutorial)
- `content/docs/_index.md` — landing page
- `layouts/` — Hextra theme overrides
- `hugo.yaml` — site config
- `.github/workflows/pages.yaml` — GitHub Pages build and deploy

## Local development

Pre-requisites: [Hugo](https://gohugo.io/getting-started/installing/) **0.117.0 extended**, and [Git](https://git-scm.com).

```shell
git clone git@github.com:Decent-Software-LLC/aiml.training.git
cd aiml.training
hugo mod tidy
hugo server --logLevel debug --disableFastRender -p 1313
```

The site builds at [http://localhost:1313](http://localhost:1313). Drafts (`draft: true` front matter) only render with `hugo server --buildDrafts`.

### Update the Hextra theme

```shell
hugo mod get -u
hugo mod tidy
```

## Deployment

The site deploys to [https://aiml.training](https://aiml.training) from the `main` branch via GitHub Pages. [`.github/workflows/pages.yaml`](./.github/workflows/pages.yaml) builds with Hugo 0.117.0 and deploys via the `actions/deploy-pages` action. In repo settings, the Pages source must be set to **GitHub Actions**, and the custom domain `aiml.training` is wired through `CNAME`.

## License

[MIT](./LICENSE) — the tutorials and site code are MIT-licensed. The model checkpoints and datasets referenced in the tutorials are subject to their own licenses (noted in each tutorial's version-pins section).
