import { NextResponse } from "next/server";
import { readdir, stat } from "fs/promises";
import { join } from "path";
import { existsSync } from "fs";

export async function GET() {
    try {
        const convertedDir = join(process.cwd(), "..", "converted");

        if (!existsSync(convertedDir)) {
            return NextResponse.json({ folders: [] });
        }

        const items = await readdir(convertedDir);

        // Filter to only include directories
        const folders: { name: string, fileCount: number, files: string[] }[] = [];

        for (const item of items) {
            const itemPath = join(convertedDir, item);
            const stats = await stat(itemPath);
            if (stats.isDirectory()) {
                const files = await readdir(itemPath);
                // Filter only .nii or .nii.gz files
                const niftiFiles = files.filter(f => f.endsWith('.nii') || f.endsWith('.nii.gz'));

                folders.push({
                    name: item,
                    fileCount: niftiFiles.length,
                    files: niftiFiles // Include file names
                });
            }
        }

        return NextResponse.json({ folders });
    } catch (error) {
        console.error("List converted folders error:", error);
        return NextResponse.json(
            { error: "Error listing converted folders" },
            { status: 500 }
        );
    }
}
