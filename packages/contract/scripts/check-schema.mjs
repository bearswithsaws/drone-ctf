import { execFileSync } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const packageRoot = fileURLToPath(new URL("..", import.meta.url));
const repositoryRoot = fileURLToPath(new URL("../../..", import.meta.url));
const worktreePython = join(repositoryRoot, ".venv", "bin", "python");
const python = process.env.PYTHON ?? (existsSync(worktreePython) ? worktreePython : "python3");

execFileSync(python, [join(repositoryRoot, "tools", "contract_schema.py"), "--check"], {
  cwd: repositoryRoot,
  stdio: "inherit",
});

const temporaryDirectory = mkdtempSync(join(tmpdir(), "commander-contract-"));
const generatedPath = join(temporaryDirectory, "schema.generated.ts");

try {
  execFileSync(
    join(packageRoot, "node_modules", ".bin", "json2ts"),
    [
      "--input",
      join(packageRoot, "schema", "contract-v1.schema.json"),
      "--output",
      generatedPath,
    ],
    { cwd: packageRoot, stdio: "inherit" },
  );

  const expected = readFileSync(join(packageRoot, "src", "schema.generated.ts"), "utf8");
  const actual = readFileSync(generatedPath, "utf8");
  if (expected !== actual) {
    throw new Error(
      "src/schema.generated.ts is stale; run `npm run generate` in packages/contract",
    );
  }
} finally {
  rmSync(temporaryDirectory, { recursive: true, force: true });
}
