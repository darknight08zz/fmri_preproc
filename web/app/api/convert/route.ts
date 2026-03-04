import { NextRequest, NextResponse } from "next/server";
import { join } from "path";
import { exec } from "child_process";
import { promisify } from "util";
import { mkdir } from "fs/promises";
import { existsSync } from "fs";

const execAsync = promisify(exec);

export async function POST(request: NextRequest) {
    try {
        const body = await request.json();
        const { folderName } = body;

        if (!folderName) {
            return NextResponse.json(
                { error: "Folder name is required" },
                { status: 400 }
            );
        }

        // Define paths
        const baseDir = join(process.cwd(), ".."); // Go up one level from 'web'
        const inputDir = join(baseDir, "uploads", folderName);
        const outputDir = join(baseDir, "converted", folderName);
        const scriptPath = join(baseDir, "run_converter.py");

        // Validate input exists
        if (!existsSync(inputDir)) {
            return NextResponse.json(
                { error: "Study folder not found" },
                { status: 404 }
            );
        }

        // Ensure output directory exists (parent 'converted' + study folder)
        if (!existsSync(outputDir)) {
            await mkdir(outputDir, { recursive: true });
        }

        // Construct command
        // Using unbuffered python output (-u)
        const command = `python -u "${scriptPath}" "${inputDir}" "${outputDir}"`;

        console.log(`Executing: ${command}`);

        // Execute script
        const { stdout, stderr } = await execAsync(command);

        console.log("Conversion stdout:", stdout);
        if (stderr) console.error("Conversion stderr:", stderr);

        return NextResponse.json({ success: true, folderName, outputDir });

    } catch (error: any) {
        console.error("Conversion error:", error);
        return NextResponse.json(
            { error: error.message || "Error during conversion" },
            { status: 500 }
        );
    }
}
