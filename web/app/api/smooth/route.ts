import { NextRequest, NextResponse } from "next/server";
import { spawn } from "child_process";
import path from "path";
import fs from "fs";

// ── Project layout ───────────────────────────────────────────────────
const PROJECT_ROOT = path.resolve(process.cwd(), "..");
const SMOOTH_DIR = path.join(PROJECT_ROOT, "smooth");
const SCRIPT_PATH = path.join(SMOOTH_DIR, "smooth.py");

function toPyPath(p: string): string {
    return p.replace(/\\/g, "/");
}

function detectPython(): string {
    const candidates =
        process.platform === "win32"
            ? ["python", "python3"]
            : ["python3", "python"];
    return candidates[0];
}

export async function POST(req: NextRequest) {
    try {
        const body = await req.json();
        const { funcPath, fwhm } = body as {
            funcPath: string;
            fwhm?: [number, number, number];
        };

        if (!funcPath) {
            return NextResponse.json(
                { error: "funcPath is required" },
                { status: 400 }
            );
        }

        // ── Resolve paths (handles relative + absolute) ────────────────
        const resolveP = (p: string) =>
            path.isAbsolute(p) ? p : path.resolve(PROJECT_ROOT, p);

        const absFunc = resolveP(funcPath);

        if (!fs.existsSync(absFunc)) {
            return NextResponse.json(
                { error: `Functional image not found: ${absFunc}` },
                { status: 400 }
            );
        }

        const python = detectPython();
        const fwhmArgs = fwhm ?? [6, 6, 6];

        const result = await new Promise<{
            success: boolean;
            output: string;
            error: string;
            outputFile?: string;
            timeTotal?: number;
        }>((resolve) => {
            const proc = spawn(
                python,
                [
                    SCRIPT_PATH,
                    "--func", toPyPath(absFunc),
                    "--fwhm", String(fwhmArgs[0]), String(fwhmArgs[1]), String(fwhmArgs[2]),
                ],
                {
                    cwd: SMOOTH_DIR,
                    env: { ...process.env, PYTHONIOENCODING: "utf-8" },
                }
            );

            let stdout = "";
            let stderr = "";

            proc.stdout.on("data", (d: Buffer) => { stdout += d.toString(); });
            proc.stderr.on("data", (d: Buffer) => { stderr += d.toString(); });

            proc.on("close", (code: number) => {
                if (code !== 0) {
                    resolve({ success: false, output: stdout, error: stderr });
                    return;
                }

                const resultLine = stdout
                    .split("\n")
                    .find((l) => l.startsWith("__RESULT__"));
                const outputFile = resultLine
                    ? resultLine.replace("__RESULT__", "").trim()
                    : undefined;

                const timeMatch = stdout.match(/Total\s*:\s*([\d.]+)s/);
                const timeTotal = timeMatch ? parseFloat(timeMatch[1]) : undefined;

                resolve({ success: true, output: stdout, error: stderr, outputFile, timeTotal });
            });
        });

        if (!result.success) {
            return NextResponse.json(
                { error: "Smoothing failed", details: result.error || result.output },
                { status: 500 }
            );
        }

        // ── Calculate relative path for frontend serving ────────────────
        const convertedDir = path.join(PROJECT_ROOT, "converted").toLowerCase().replace(/\\/g, "/");
        let outputFile = (result.outputFile || "").replace(/\\/g, "/");
        
        if (outputFile && path.isAbsolute(outputFile) && outputFile.toLowerCase().startsWith(convertedDir)) {
          outputFile = path.relative(path.join(PROJECT_ROOT, "converted"), outputFile).replace(/\\/g, "/");
        }

        return NextResponse.json({
            success: true,
            outputFile: outputFile,
            timeTotal: result.timeTotal,
            log: result.output,
        });
    } catch (err) {
        return NextResponse.json(
            { error: "Internal server error", details: String(err) },
            { status: 500 }
        );
    }
}