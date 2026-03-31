// web/app/api/coreg/route.ts
//
// POST /api/coreg
// Body:
//   ref_path    : string   — T1w.nii    (relative to project root or absolute)
//   source_path : string   — meansub-02_4D.nii
//   other_paths : string[] — [arsub-02_4D.nii]
//   separation        ?: number[]  default [4, 2]
//   hist_smooth_fwhm  ?: number[]  default [7, 7]
//   interp_order      ?: number    default 4
//   wrap              ?: number[]  default [0,0,0]
//   mask              ?: boolean   default false
//   prefix            ?: string    default "r"
//
// Returns:
//   { params, M, output_files, time_estimate, time_reslice, stdout }

import { NextRequest, NextResponse } from "next/server";
import { execFile, execSync } from "child_process";
import { promisify } from "util";
import path from "path";
import fs from "fs";

const execFileAsync = promisify(execFile);

// ── Project layout ───────────────────────────────────────────────────
// <project>/
//   coreg/   ← Python scripts
//   web/     ← Next.js  (process.cwd() when running `next dev`)
const PROJECT_ROOT = path.resolve(process.cwd(), "..");
const COREG_DIR = path.join(PROJECT_ROOT, "coreg");
const SCRIPT_PATH = path.join(COREG_DIR, "coregister.py");

// ────────────────────────────────────────────────────────────────────
//  FIX 1 — Normalise Windows backslashes for Python
// ────────────────────────────────────────────────────────────────────
// Root cause of [Errno 22] Invalid argument:
//   Windows path  → D:\New folder (2)\fmri_preproc\converted\...
//   JSON.stringify → "D:\\New folder (2)\\fmri_preproc\\..."
//   Python reads  → D:\N  ew folder (2)\f  ... (\N \f = escape sequences!)
//
// Fix: replace every backslash with a forward slash before embedding
// any path into the Python script string.
// nibabel, numpy, and Python's open() all accept forward slashes on Windows.
const toPyPath = (p: string): string => p.replace(/\\/g, "/");

// ────────────────────────────────────────────────────────────────────
//  FIX 2 — Windows uses `python` not `python3`
// ────────────────────────────────────────────────────────────────────
// On Linux/Mac: python3 is the correct command
// On Windows:   python3 often doesn't exist; it's just `python`
// We detect once at module load time.
function detectPython(): string {
  const candidates = process.platform === "win32"
    ? ["python", "python3"]   // prefer `python` on Windows
    : ["python3", "python"];  // prefer `python3` on Linux/Mac

  for (const cmd of candidates) {
    try {
      execSync(`${cmd} --version`, { stdio: "pipe" });
      return cmd;
    } catch {
      // not available — try next
    }
  }
  throw new Error(
    "Python not found. Install Python and ensure it is in your PATH."
  );
}

let PYTHON_CMD: string;
try {
  PYTHON_CMD = detectPython();
} catch {
  PYTHON_CMD = "python"; // safe fallback — real error surfaces at runtime
}

// ────────────────────────────────────────────────────────────────────
//  Route handler
// ────────────────────────────────────────────────────────────────────

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();

    const {
      ref_path,
      source_path,
      other_paths = [],
      separation = [4.0, 2.0],
      hist_smooth_fwhm = [7.0, 7.0],
      interp_order = 4,
      wrap = [0, 0, 0],
      mask = false,
      prefix = "r",
    } = body as {
      ref_path: string;
      source_path: string;
      other_paths?: string[];
      separation?: number[];
      hist_smooth_fwhm?: number[];
      interp_order?: number;
      wrap?: number[];
      mask?: boolean;
      prefix?: string;
    };

    // ── Validate required fields ───────────────────────────────────
    if (!ref_path || !source_path) {
      return NextResponse.json(
        { error: "ref_path and source_path are required." },
        { status: 400 }
      );
    }

    // ── Check Python script exists ─────────────────────────────────
    if (!fs.existsSync(SCRIPT_PATH)) {
      return NextResponse.json(
        { error: `coregister.py not found at: ${SCRIPT_PATH}` },
        { status: 500 }
      );
    }

    // ── Resolve paths (handles relative + absolute) ────────────────
    const resolveP = (p: string) =>
      path.isAbsolute(p) ? p : path.resolve(PROJECT_ROOT, p);

    const absRef = resolveP(ref_path);
    const absSrc = resolveP(source_path);
    const absOthers = (other_paths as string[]).map(resolveP);

    // ── Check all input files exist on disk ────────────────────────
    const missing = [absRef, absSrc, ...absOthers].filter(
      (f) => !fs.existsSync(f)
    );
    if (missing.length > 0) {
      return NextResponse.json(
        {
          error: `Input file(s) not found:\n${missing
            .map((f) => `  ${f}`)
            .join("\n")}`,
        },
        { status: 404 }
      );
    }

    // ── FIX 1 applied: convert all paths to forward slashes ────────
    // Must be done BEFORE JSON.stringify embeds them in the Python source.
    const pyRef = toPyPath(absRef);
    const pySrc = toPyPath(absSrc);
    const pyOthers = absOthers.map(toPyPath);
    const pyCoreg = toPyPath(COREG_DIR);   // FIX 3: sys.path also needs it

    // ── Build inline Python script ────────────────────────────────
    // Paths are forward-slash normalised — JSON.stringify is safe.
    const wrapperScript = `
import sys, json, os

# Insert coreg/ directory so ALL three modules are findable:
#   coregister.py, coregister_estimate.py, coregister_reslice.py
# Must be done BEFORE any import, because coregister.py itself
# does 'from coregister_estimate import ...' at module load time.
_coreg_dir = ${JSON.stringify(pyCoreg)}
if _coreg_dir not in sys.path:
    sys.path.insert(0, _coreg_dir)

# Also change working directory to coreg/ so relative imports work
# as a final fallback on Windows where sys.path alone can sometimes
# fail to resolve same-directory modules.
os.chdir(_coreg_dir)

from coregister import run_coregistration
import numpy as np

try:
    result = run_coregistration(
        ref_path         = ${JSON.stringify(pyRef)},
        source_path      = ${JSON.stringify(pySrc)},
        other_paths      = ${JSON.stringify(pyOthers)},
        separation       = ${JSON.stringify(separation)},
        hist_smooth_fwhm = ${JSON.stringify(hist_smooth_fwhm)},
        interp_order     = ${interp_order},
        wrap             = ${JSON.stringify(wrap)},
        mask             = ${mask ? "True" : "False"},
        prefix           = ${JSON.stringify(prefix)},
        verbose          = True,
    )
    out = {
        "params"       : result["params"].tolist(),
        "M"            : result["M"].tolist(),
        "output_files" : result["output_files"],
        "time_estimate": round(result["time_estimate"], 2),
        "time_reslice" : round(result["time_reslice"],  2),
    }
    print("__RESULT__" + json.dumps(out))
except Exception as e:
    import traceback
    print("__ERROR__" + str(e))
    traceback.print_exc()
`;

    // ── FIX 2 applied: use detected Python command ─────────────────
    const { stdout, stderr } = await execFileAsync(
      PYTHON_CMD,                   // "python" on Windows, "python3" on Linux/Mac
      ["-c", wrapperScript],
      {
        timeout: 1200_000,          // 20 min max
        env: {
          ...process.env,
          PYTHONUNBUFFERED: "1",
          PYTHONIOENCODING: "utf-8",  // FIX: force UTF-8 for stdout/stderr
        },                              // prevents 'charmap' codec error on Windows
      }
    );

    // ── Parse output ───────────────────────────────────────────────
    const lines = stdout.split("\n");
    const resultLine = lines.find((l) => l.startsWith("__RESULT__"));
    const errorLine = lines.find((l) => l.startsWith("__ERROR__"));

    if (errorLine) {
      const msg = errorLine.replace("__ERROR__", "");
      return NextResponse.json(
        { error: msg, stdout, stderr },
        { status: 500 }
      );
    }

    if (!resultLine) {
      return NextResponse.json(
        { error: "No result from Python script.", stdout, stderr },
        { status: 500 }
      );
    }

    const data = JSON.parse(resultLine.replace("__RESULT__", ""));

    // Transform absolute paths to relative
    if (data.output_files) {
      const convertedDir = path.join(PROJECT_ROOT, "converted").toLowerCase().replace(/\\/g, "/");
      data.output_files = data.output_files.map((val: string) => {
        const normalizedVal = val.replace(/\\/g, "/");
        if (path.isAbsolute(normalizedVal) && normalizedVal.toLowerCase().startsWith(convertedDir)) {
          return path.relative(path.join(PROJECT_ROOT, "converted"), normalizedVal).replace(/\\/g, "/");
        }
        return normalizedVal;
      });
      // Add outputFile for convenience (usually the first resliced image)
      data.outputFile = data.output_files[0];
    }

    return NextResponse.json({ ...data, stdout });

  } catch (err: any) {
    const isTimeout = err?.code === "ETIMEDOUT" || err?.killed;
    return NextResponse.json(
      {
        error: isTimeout
          ? "Coregistration timed out (>20 min). Try with smaller input files."
          : (err?.message ?? "Internal server error"),
      },
      { status: 500 }
    );
  }
}