import { execFileSync } from "node:child_process";
import { existsSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const repositoryRoot = fileURLToPath(new URL("../../..", import.meta.url));
const worktreePython = join(repositoryRoot, ".venv", "bin", "python");
const python = process.env.PYTHON ?? (existsSync(worktreePython) ? worktreePython : "python3");

execFileSync(python, [join(repositoryRoot, "tools", "contract_schema.py")], {
  cwd: repositoryRoot,
  stdio: "inherit",
});
