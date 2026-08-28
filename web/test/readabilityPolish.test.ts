import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const mainSource = readFileSync(new URL("../src/main.tsx", import.meta.url), "utf8");
const cssSource = readFileSync(
  new URL("../src/styles/readability-polish.css", import.meta.url),
  "utf8"
);
const tutorialCss = readFileSync(
  new URL("../src/tutorial/tutorial.css", import.meta.url),
  "utf8"
);

const polishImport = 'import "./styles/readability-polish.css";';
assert.ok(mainSource.includes(polishImport));
assert.ok(
  mainSource.lastIndexOf(polishImport) > mainSource.lastIndexOf('import "./workflow-completion.css";'),
  "readability polish must load after domain/workflow CSS"
);

function pixelFontSizes(source: string): number[] {
  return [...source.matchAll(/font-size:\s*(\d+)px/g)].map((match) => Number(match[1]));
}

for (const [name, source] of [
  ["readability layer", cssSource],
  ["tutorial", tutorialCss]
] as const) {
  const sizes = pixelFontSizes(source);
  assert.ok(sizes.length > 0);
  assert.ok(
    sizes.every((size) => size >= 11),
    `${name} introduced visible text below 11px: ${sizes.join(", ")}`
  );
}

assert.match(cssSource, /--ui-micro-font-size:\s*11px/);
assert.match(cssSource, /--ui-caption-font-size:\s*12px/);
assert.match(cssSource, /--ui-table-font-size:\s*14px/);
assert.match(cssSource, /\.busy-card:has\(\.execution-progress-panel\)[\s\S]*760px/);
assert.match(cssSource, /\.execution-progress-head > strong[\s\S]*white-space:\s*normal/);
assert.match(cssSource, /\.execution-stage-list small[\s\S]*white-space:\s*normal/);
assert.match(cssSource, /\.table-wrap,[\s\S]*overflow-x:\s*auto/);
assert.match(cssSource, /@media \(max-width: 760px\)[\s\S]*\.simple-default-grid,[\s\S]*grid-template-columns:\s*minmax\(0, 1fr\)/);
assert.match(cssSource, /@media \(max-width: 760px\)[\s\S]*\.execution-progress-head[\s\S]*grid-template-columns:\s*minmax\(0, 1fr\)/);
assert.match(cssSource, /\.status-chip[\s\S]*overflow-wrap:\s*anywhere/);
assert.match(tutorialCss, /\.tutorial-choice-grid[\s\S]*repeat\(2, minmax\(0, 1fr\)\)/);
assert.match(tutorialCss, /@media \(max-width: 820px\)[\s\S]*\.tutorial-choice-grid[\s\S]*grid-template-columns:\s*1fr/);
