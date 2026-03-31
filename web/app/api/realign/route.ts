
import { spawn } from "child_process";
import fs from "fs";
import { NextResponse } from "next/server";
import path from "path";

export async function POST(req: Request) {
    try {
        let { filename } = await req.json();

        if (!filename) {
            return NextResponse.json(
                { error: "Filename is required" },
                { status: 400 }
            );
        }

        filename = filename.trim();

        // Project root (assuming web is in a subdirectory)
        const projectRoot = path.resolve(process.cwd(), "..");
        const scriptPath = path.join(projectRoot, "run_realign.py");

        // Construct the full path to the input file (assuming it's in convert/ or wherever loader expects)
        // Adjust this if your loader expects relative or absolute in a specific way
        // Let's assume filename is relative to project root or absolute
        // For now, let's assume it's an absolute path from the frontend or relative to project root
        const inputFilePath = path.join(projectRoot, "converted", filename);

        // Check if file exists
        if (!fs.existsSync(inputFilePath)) {
            return NextResponse.json(
                { error: `File not found: ${inputFilePath}` },
                { status: 404 }
            );
        }

        console.log(`Running Realign on: ${inputFilePath}`);

        // Spawn Python process
        const pythonProcess = spawn("python", [scriptPath, inputFilePath]);

        let outputData = "";
        let errorData = "";

        pythonProcess.stdout.on("data", (data) => {
            const chunk = data.toString();
            outputData += chunk;
            console.log(`[Realign]: ${chunk}`);
        });

        pythonProcess.stderr.on("data", (data) => {
            errorData += data.toString();
            console.error(`[Realign Error]: ${data}`);
        });

        return new Promise((resolve) => {
            pythonProcess.on("close", (code) => {
                if (code !== 0) {
                    resolve(
                        NextResponse.json(
                            { error: "Realign process failed", details: errorData },
                            { status: 500 }
                        )
                    );
                } else {
                    try {
                        // Parse the last JSON line from output
                        const lines = outputData.trim().split("\n");
                        const lastLine = lines[lines.length - 1];
                        const result = JSON.parse(lastLine);
                        
                        // Transform absolute paths to relative
                        if (result.status === "completed" && result.outputs) {
                            const convertedDir = path.join(projectRoot, "converted").toLowerCase().replace(/\\/g, "/");
                            const transformedOutputs: any = {};
                            for (const [key, value] of Object.entries(result.outputs)) {
                                let val = (value as string).replace(/\\/g, "/");
                                if (path.isAbsolute(val) && val.toLowerCase().startsWith(convertedDir)) {
                                    // Use original case for the relative path, but normalized slashes
                                    transformedOutputs[key] = path.relative(path.join(projectRoot, "converted"), val).replace(/\\/g, "/");
                                } else {
                                    transformedOutputs[key] = val;
                                }
                            }
                            result.outputs = transformedOutputs;
                            // Add a top-level outputFile for convenience (resliced)
                            result.outputFile = transformedOutputs.resliced;
                        }

                        resolve(NextResponse.json(result));
                    } catch (e) {
                        resolve(
                            NextResponse.json(
                                { error: "Failed to parse Python output", raw: outputData },
                                { status: 500 }
                            )
                        );
                    }
                }
            });
        });

    } catch (error) {
        console.error("API Error:", error);
        return NextResponse.json(
            { error: "Internal Server Error" },
            { status: 500 }
        );
    }
}
