from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
file_path = ROOT / ".github/workflows/web-lse-settings-smoke.yml"
text = file_path.read_text(encoding="utf-8")
old = '''      - uses: actions/setup-node@v4
        with:
          node-version: "20"
          cache: npm
          cache-dependency-path: web/package-lock.json
      - name: Build Web
        working-directory: web
        run: |
          npm ci
          npm run build
'''
new = '''      - uses: actions/setup-node@v4
        with:
          node-version: "24"
      - name: Build Web
        working-directory: web
        run: |
          npm install --no-audit --no-fund
          npm run build
'''
if old in text:
    file_path.write_text(text.replace(old, new, 1), encoding="utf-8")
elif new not in text:
    raise RuntimeError("Expected Web LSE smoke Node block was not found")

print("Web LSE smoke aligned with canonical Web build")
