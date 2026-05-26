import { rmSync } from "node:fs";
import { join } from "node:path";

const nextDir = join(process.cwd(), ".next");

rmSync(nextDir, { recursive: true, force: true });
