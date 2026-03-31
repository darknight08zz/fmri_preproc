// web/app/api/segment/route.ts
//
// POST /api/segment
// Body:
//   t1w_path     : string    — T1w.nii path (relative to project root or absolute)
//   bias_reg     ?: number   default 0.0001
//   bias_fwhm    ?: number   default 60
//   mrf_strength ?: number   default 1.0
//   sampling_mm  ?: number   default 3.0
//   n_gauss      ?: number[] default [1,1,2,3,4,2]
//   n_em_iter    ?: number   default 30
//
// TPM.nii is bundled at <project_root>/assets/TPM.nii — no user input needed.
//
// Returns:
//   { output_files, time_*, n_gm_voxels, n_wm_voxels, n_csf_voxels, stdout }

import { NextRequest, NextResponse } from "next/server";
import { execFile, execSync } from "child_process";
import { promisify } from "util";
import path from "path";
import fs from "fs";

const execFileAsync = promisify(execFile);

// ── Project layout ────────────────────────────────────────────────────
// <project_root>/
//   assets/      ← TPM.nii lives here (copied from SPM12/tpm/TPM.nii once)
//   segment/     ← Python scripts
//   web/         ← Next.js (process.cwd() = here when running `next dev`)
const PROJECT_ROOT = path.resolve(process.cwd(), "..");
const SEGMENT_DIR = path.join(PROJECT_ROOT, "segment");
const SCRIPT_PATH = path.join(SEGMENT_DIR, "segmentation.py");

// ── Bundled TPM — hardcoded, users never need to provide this ─────────
const TPM_PATH = path.join(PROJECT_ROOT, "assets", "TPM.nii");

// ── Windows backslash → forward slash for Python strings ─────────────
const toPyPath = (p: string): string => p.replace(/\\/g, "/");

// ── Detect correct Python executable (python3 on Linux, python on Win) ─
function detectPython(): string {
    const candidates = process.platform === "win32"
        ? ["python", "python3"]
        : ["python3", "python"];

    for (const cmd of candidates) {
        try {
            execSync(`${cmd} --version`, { stdio: "pipe" });
            return cmd;
        } catch { /* not found, try next */ }
    }
    throw new Error("Python not found. Install Python and add it to PATH.");
}

let PYTHON_CMD: string;
try { PYTHON_CMD = detectPython(); }
catch { PYTHON_CMD = "python"; }

// ─────────────────────────────────────────────────────────────────────
//  Route handler
// ─────────────────────────────────────────────────────────────────────

export async function POST(req: NextRequest) {
    try {
        const body = await req.json();

        const {
            t1w_path,
            bias_reg = 0.0001,
            bias_fwhm = 60.0,
            mrf_strength = 1.0,
            sampling_mm = 3.0,
            n_gauss = [1, 1, 2, 3, 4, 2],
            n_em_iter = 30,
        } = body as {
            t1w_path: string;
            bias_reg?: number;
            bias_fwhm?: number;
            mrf_strength?: number;
            sampling_mm?: number;
            n_gauss?: number[];
            n_em_iter?: number;
        };

        // ── Validate required field ────────────────────────────────────
        if (!t1w_path) {
            return NextResponse.json(
                { error: "t1w_path is required." },
                { status: 400 }
            );
        }

        // ── Check Python script exists ─────────────────────────────────
        if (!fs.existsSync(SCRIPT_PATH)) {
            return NextResponse.json(
                { error: `segmentation.py not found at: ${SCRIPT_PATH}` },
                { status: 500 }
            );
        }

        // ── Check bundled TPM exists ───────────────────────────────────
        if (!fs.existsSync(TPM_PATH)) {
            return NextResponse.json(
                {
                    error:
                        `TPM.nii not found at: ${TPM_PATH}\n` +
                        `Copy TPM.nii from your SPM12 installation:\n` +
                        `  <SPM12_dir>/tpm/TPM.nii  →  ${TPM_PATH}`,
                },
                { status: 500 }
            );
        }

        // ── Resolve T1w path (relative or absolute) ────────────────────
        const resolveP = (p: string) => {
            // Strip any accidental surrounding quotes from copy-paste
            const clean = p.replace(/^["']+|["']+$/g, "");
            return path.isAbsolute(clean)
                ? clean
                : path.resolve(PROJECT_ROOT, clean);
        };

        const absT1w = resolveP(t1w_path);

        // ── Check T1w file exists ──────────────────────────────────────
        if (!fs.existsSync(absT1w)) {
            return NextResponse.json(
                { error: `T1w file not found: ${absT1w}` },
                { status: 404 }
            );
        }

        // ── Normalise all paths for Python (forward slashes) ──────────
        const pyT1w = toPyPath(absT1w);
        const pyTpm = toPyPath(TPM_PATH);   // bundled — always the same
        const pySegDir = toPyPath(SEGMENT_DIR);

        // ── Build inline Python wrapper ────────────────────────────────
        const wrapperScript = `
import sys, json
sys.path.insert(0, ${JSON.stringify(pySegDir)})
from segmentation import run_segmentation

try:
    result = run_segmentation(
        t1w_path     = ${JSON.stringify(pyT1w)},
        tpm_path     = ${JSON.stringify(pyTpm)},
        bias_reg     = ${bias_reg},
        bias_fwhm    = ${bias_fwhm},
        mrf_strength = ${mrf_strength},
        sampling_mm  = ${sampling_mm},
        n_gauss      = ${JSON.stringify(n_gauss)},
        n_em_iter    = ${n_em_iter},
        verbose      = True,
    )
    print("__RESULT__" + json.dumps(result))
except Exception as e:
    import traceback
    print("__ERROR__" + str(e))
    traceback.print_exc()
`;

        // ── Run Python ─────────────────────────────────────────────────
        // Segmentation takes 15–25 min on CPU for a full T1w
        const { stdout, stderr } = await execFileAsync(
            PYTHON_CMD,
            ["-c", wrapperScript],
            {
                timeout: 2400_000,   // 40 min max
                env: {
                    ...process.env,
                    PYTHONUNBUFFERED: "1",
                    PYTHONIOENCODING: "utf-8",
                },
            }
        );

        // ── Parse Python output ────────────────────────────────────────
        const lines = stdout.split("\n");
        const resultLine = lines.find((l) => l.startsWith("__RESULT__"));
        const errorLine = lines.find((l) => l.startsWith("__ERROR__"));

        if (errorLine) {
            return NextResponse.json(
                { error: errorLine.replace("__ERROR__", ""), stdout, stderr },
                { status: 500 }
            );
        }

        if (!resultLine) {
            return NextResponse.json(
                { error: "No result returned from Python.", stdout, stderr },
                { status: 500 }
            );
        }

        const data = JSON.parse(resultLine.replace("__RESULT__", ""));

        // Transform absolute paths to relative
        if (data.output_files) {
            const convertedDir = path.join(PROJECT_ROOT, "converted").toLowerCase().replace(/\\/g, "/");
            const transformedOutputs: any = {};
            for (const [key, value] of Object.entries(data.output_files)) {
                let val = (value as string).replace(/\\/g, "/");
                if (path.isAbsolute(val) && val.toLowerCase().startsWith(convertedDir)) {
                    transformedOutputs[key] = path.relative(path.join(PROJECT_ROOT, "converted"), val).replace(/\\/g, "/");
                } else {
                    transformedOutputs[key] = val;
                }
            }
            data.output_files = transformedOutputs;
        }

        return NextResponse.json({ ...data, stdout });

    } catch (err: any) {
        const isTimeout = err?.code === "ETIMEDOUT" || err?.killed;
        return NextResponse.json(
            {
                error: isTimeout
                    ? "Segmentation timed out (>40 min). Normal for large T1w on CPU."
                    : (err?.message ?? "Internal server error"),
            },
            { status: 500 }
        );
    }
}