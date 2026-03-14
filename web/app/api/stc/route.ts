import { NextResponse } from "next/server";
import { exec } from "child_process";
import path from "path";
import util from "util";
import fs from "fs";

const execPromise = util.promisify(exec);

export async function POST(request: Request) {
    try {
        const body = await request.json();
        const { filename, tr, slices, slice_order, ref_slice, ta } = body;

        if (!filename) {
            return NextResponse.json(
                { error: "Filename is required" },
                { status: 400 }
            );
        }

        const scriptPath = path.resolve(process.cwd(), "..", "run_stc.py");

        // Resolve input file path — try multiple locations
        let resolvedInputPath = filename;
        const potentialPaths = [
            filename,
            path.resolve(process.cwd(), "..", filename),
            path.resolve(process.cwd(), "..", "converted", filename),
            path.resolve(process.cwd(), "..", "uploads", filename),
        ];

        for (const p of potentialPaths) {
            if (fs.existsSync(p)) {
                resolvedInputPath = p;
                break;
            }
        }

        let cmd = `python "${scriptPath}" "${resolvedInputPath}"`;
        if (slice_order) cmd += ` --slice_order ${slice_order}`;
        if (ref_slice !== undefined) cmd += ` --ref_slice ${ref_slice}`;
        if (ta) cmd += ` --ta ${ta}`;

        console.log("Executing STC command:", cmd);

        const { stdout, stderr } = await execPromise(cmd);

        if (stderr) {
            console.error("STC Stderr:", stderr);
        }

        console.log("STC Stdout:", stdout);

        try {
            const lines = stdout.trim().split("\n");
            const lastLine = lines[lines.length - 1];
            const result = JSON.parse(lastLine);

            if (result.status === "error") {
                return NextResponse.json({ error: result.message }, { status: 500 });
            }

            return NextResponse.json(result);
        } catch (parseError) {
            console.error("Error parsing STC output:", parseError);
            return NextResponse.json(
                { error: "Failed to parse STC output", details: stdout },
                { status: 500 }
            );
        }
    } catch (error: any) {
        console.error("API Error:", error);
        return NextResponse.json(
            { error: error.message || "Internal Server Error" },
            { status: 500 }
        );
    }
}