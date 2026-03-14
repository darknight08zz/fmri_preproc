// POST /api/extract-stc-params
// Body: { json_path: string }  ← path to _bold.json (relative to web/ or absolute)
// Returns: { TR, nSlices, nVolumes, TA, sliceOrder, refSlice, source }

import { NextRequest, NextResponse } from "next/server";
import { execFile } from "child_process";
import path from "path";
import fs from "fs";
import { promisify } from "util";

const execFileAsync = promisify(execFile);

export async function POST(req: NextRequest) {
    try {
        const body = await req.json();
        const { json_path } = body as { json_path: string };

        if (!json_path) {
            return NextResponse.json({ error: "json_path is required" }, { status: 400 });
        }

        const projectRoot = path.resolve(process.cwd(), "..");

        // Build list of candidate paths to try (in priority order)
        const filename = path.basename(json_path);
        const candidates: string[] = [];

        if (path.isAbsolute(json_path)) {
            candidates.push(json_path);
        } else {
            candidates.push(
                path.resolve(process.cwd(), json_path),           // relative to web/
                path.resolve(projectRoot, json_path),             // relative to project root
                path.resolve(projectRoot, "converted", json_path),// under converted/
            );
        }

        // Also search recursively inside converted/ by filename alone
        // (handles case where user browses and gets just the bare filename)
        const convertedDir = path.resolve(projectRoot, "converted");
        const findInConverted = (dir: string): string | null => {
            try {
                for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
                    const full = path.join(dir, entry.name);
                    if (entry.isDirectory()) {
                        const found = findInConverted(full);
                        if (found) return found;
                    } else if (entry.name === filename) {
                        return full;
                    }
                }
            } catch { /* ignore read errors */ }
            return null;
        };

        let absJsonPath = candidates.find((p) => fs.existsSync(p)) ?? null;
        if (!absJsonPath) absJsonPath = findInConverted(convertedDir);

        if (!absJsonPath) {
            return NextResponse.json(
                { error: `JSON file not found: "${filename}". Searched in project root and converted/ folder.` },
                { status: 404 }
            );
        }

        // Auto-detect paired NIfTI file
        const niiGzPath = absJsonPath.replace(/\.json$/, ".nii.gz");
        const niiPath   = absJsonPath.replace(/\.json$/, ".nii");
        const niftiPath = fs.existsSync(niiGzPath)
            ? niiGzPath
            : fs.existsSync(niiPath)
            ? niiPath
            : null;

        // Locate extractor script
        const scriptPath = path.resolve(projectRoot, "stc/extract_stc_params.py");

        if (!fs.existsSync(scriptPath)) {
            return NextResponse.json(
                { error: `Extractor script not found at: ${scriptPath}` },
                { status: 500 }
            );
        }

        // Build inline Python wrapper that prints JSON result
        const wrapperScript = `
import sys, json, os
sys.path.insert(0, ${JSON.stringify(path.dirname(scriptPath))})
from extract_stc_params import extract_stc_params

nifti    = ${niftiPath ? JSON.stringify(niftiPath) : "None"}
jsonfile = ${JSON.stringify(absJsonPath)}

try:
    params = extract_stc_params(
        nifti_path=nifti if nifti else jsonfile.replace(".json", ".nii"),
        json_path=jsonfile
    )
    source = "json+nifti" if (nifti and os.path.exists(nifti)) else "json_only"
    params["source"] = source
    print("__RESULT__" + json.dumps(params))
except Exception as e:
    print("__ERROR__" + str(e))
`;

        // Use 'python' on Windows (not 'python3')
        const pythonCmd = process.platform === "win32" ? "python" : "python3";
        const { stdout, stderr } = await execFileAsync(
            pythonCmd, ["-c", wrapperScript],
            {
                timeout: 30_000,
                env: { ...process.env, PYTHONIOENCODING: "utf-8", PYTHONUTF8: "1" },
            }
        );

        if (stderr) console.error("[extract-stc-params] stderr:", stderr);

        const resultLine = stdout.split("\n").find((l) => l.startsWith("__RESULT__"));
        const errorLine  = stdout.split("\n").find((l) => l.startsWith("__ERROR__"));

        if (errorLine) {
            return NextResponse.json(
                { error: errorLine.replace("__ERROR__", "") },
                { status: 500 }
            );
        }

        if (!resultLine) {
            return NextResponse.json(
                { error: "No result from Python script", stderr },
                { status: 500 }
            );
        }

        const params = JSON.parse(resultLine.replace("__RESULT__", ""));
        return NextResponse.json(params);

    } catch (err: any) {
        return NextResponse.json(
            { error: err?.message ?? "Internal server error" },
            { status: 500 }
        );
    }
}
