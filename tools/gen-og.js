// Generates 1200x630 Open Graph / social preview images as PNG using resvg.
// Usage: node tools/gen-og.js
// Requires: @resvg/resvg-js (in /tmp/ogtest/node_modules)
const fs = require('fs');
const path = require('path');
const { Resvg } = require('@resvg/resvg-js');

const W = 1200, H = 630;
const FONT_DIR = path.join(__dirname, 'fonts'); // Inter TTFs live in tools/fonts/

function esc(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function wrap(text, maxChars) {
  const words = text.split(' ');
  const lines = [];
  let cur = '';
  for (const w of words) {
    if ((cur + ' ' + w).trim().length > maxChars && cur) {
      lines.push(cur.trim());
      cur = w;
    } else {
      cur = (cur + ' ' + w).trim();
    }
  }
  if (cur) lines.push(cur.trim());
  return lines;
}

function svg({ title, kicker, domain }) {
  const titleLines = wrap(title, 28).slice(0, 3);
  const fontSize = titleLines.length >= 3 ? 64 : 76;
  const lineH = Math.round(fontSize * 1.12);
  const titleY = 310;

  const titleSpans = titleLines.map((ln, i) =>
    `<tspan x="96" dy="${i === 0 ? 0 : lineH}">${esc(ln)}</tspan>`
  ).join('');

  const gridV = Array.from({ length: 19 }, (_, i) =>
    `<line x1="${(i + 1) * 60}" y1="0" x2="${(i + 1) * 60}" y2="${H}"/>`
  ).join('');
  const gridH = Array.from({ length: 10 }, (_, i) =>
    `<line x1="0" y1="${(i + 1) * 60}" x2="${W}" y2="${(i + 1) * 60}"/>`
  ).join('');

  return `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#0b1220"/>
      <stop offset="1" stop-color="#0f172a"/>
    </linearGradient>
    <radialGradient id="glow" cx="0.85" cy="0.1" r="0.9">
      <stop offset="0" stop-color="#3b82f6" stop-opacity="0.22"/>
      <stop offset="0.5" stop-color="#8b5cf6" stop-opacity="0.08"/>
      <stop offset="1" stop-color="#0b1220" stop-opacity="0"/>
    </radialGradient>
    <linearGradient id="txt" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#ffffff"/>
      <stop offset="1" stop-color="#94a3b8"/>
    </linearGradient>
    <linearGradient id="mark" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#60a5fa"/>
      <stop offset="1" stop-color="#a78bfa"/>
    </linearGradient>
  </defs>

  <rect width="${W}" height="${H}" fill="url(#bg)"/>
  <rect width="${W}" height="${H}" fill="url(#glow)"/>

  <g stroke="#1e293b" stroke-width="1" opacity="0.4">
    ${gridV}${gridH}
  </g>

  <g transform="translate(96,84)">
    <rect x="0" y="0" width="44" height="44" rx="10" fill="url(#mark)"/>
    <text x="22" y="32" text-anchor="middle" font-size="28" font-weight="700" fill="#0b1220" font-family="Inter">A</text>
    <text x="58" y="31" font-size="22" font-weight="700" fill="#e2e8f0" font-family="Inter">aiml.training</text>
  </g>

  <text x="96" y="${titleY - 56}" font-size="22" font-weight="400" fill="#60a5fa" letter-spacing="2" font-family="Inter">${esc(kicker.toUpperCase())}</text>

  <text x="96" y="${titleY}" font-size="${fontSize}" font-weight="700" fill="url(#txt)" font-family="Inter">${titleSpans}</text>

  <text x="96" y="${H - 72}" font-size="20" font-weight="400" fill="#64748b" font-family="Inter">${esc(domain)}</text>
  <text x="${W - 96}" y="${H - 72}" text-anchor="end" font-size="20" font-weight="400" fill="#475569" font-family="Inter">Code-first · Version-pinned · Single-GPU</text>
</svg>`;
}

function render(file, opts, outDir) {
  const svgStr = svg(opts);
  const out = path.join(outDir, file + '.png');
  const r = new Resvg(svgStr, {
    fitTo: { mode: 'width', value: W },
    font: { fontDirs: [FONT_DIR], defaultFontFamily: 'Inter' },
  });
  fs.writeFileSync(out, r.render().asPng());
  const stat = fs.statSync(out);
  console.log(`  ${file}.png  ${stat.size} bytes`);
}

const outDir = '/home/agent/aiml.training/static/og';
fs.mkdirSync(outDir, { recursive: true });

const items = [
  { file: 'default', opts: {
    title: 'Practical LLM engineering on hardware you already own',
    kicker: 'aiml.training',
    domain: 'aiml.training',
  }},
  { file: 'lora-fine-tuning-qwen3.5-4b', opts: {
    title: 'Fine-Tune Qwen/Qwen3.5-4B with LoRA on a Single 12 GB GPU',
    kicker: 'Tutorial · Fine-tune',
    domain: 'aiml.training',
  }},
  { file: 'serve-fine-tuned-llm-vllm', opts: {
    title: 'Serving a Fine-Tuned LLM with vLLM (and the LoRA Trap)',
    kicker: 'Tutorial · Serve',
    domain: 'aiml.training',
  }},
  { file: 'evaluate-fine-tuned-llm', opts: {
    title: 'Evaluating a Fine-Tuned LLM: Perplexity, Win Rate, and a Judge',
    kicker: 'Tutorial · Evaluate',
    domain: 'aiml.training',
  }},
  { file: 'fine-tune-kev-4b-decision-model', opts: {
    title: 'Fine-Tune Kev-4B: A Calibrated Decision Model on a 12 GB GPU',
    kicker: 'Tutorial · Fine-tune',
    domain: 'aiml.training',
  }},
  { file: 'end-to-end-lora-pipeline', opts: {
    title: 'End to End: Train, Verify, Merge, and Serve a 4B LoRA Model',
    kicker: 'Tutorial · Pipeline',
    domain: 'aiml.training',
  }},
  { file: 'fine-tune-decision-router', opts: {
    title: 'Teach a Decision Model Its Tribal Knowledge: LoRA From Chance to 100%',
    kicker: 'Tutorial · Fine-tune',
    domain: 'aiml.training',
  }},
  { file: 'serve-decision-model-vllm-systemone', opts: {
    title: "Inside vLLM's /v1/systemone: Decision Serving on One GPU",
    kicker: 'Tutorial · Serve',
    domain: 'aiml.training',
  }},
  { file: 'silent-failure-handbook', opts: {
    title: 'The Silent Failure Handbook: 10 Bugs That Faked Success on One GPU',
    kicker: 'Field guide · Verification',
    domain: 'aiml.training',
  }},
];

for (const it of items) render(it.file, it.opts, outDir);
console.log('done');
