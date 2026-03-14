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

        const folders: { name: string, fileCount: number, files: string[], jsonFiles: string[] }[] = [];

        for (const item of items) {
            const itemPath = join(convertedDir, item);
            const stats = await stat(itemPath);
            if (stats.isDirectory()) {
                const allNiftiFiles: string[] = [];
                const allJsonFiles: string[] = [];

                // 1. Look inside anat/ and func/ subdirectories
                const types = ["anat", "func"];
                for (const t of types) {
                    const typePath = join(itemPath, t);
                    if (existsSync(typePath)) {
                        const typeStats = await stat(typePath);
                        if (typeStats.isDirectory()) {
                            const files = await readdir(typePath);
                            const nifties = files.filter(f => f.endsWith('.nii') || f.endsWith('.nii.gz')).map(f => `${t}/${f}`);
                            const jsons = files.filter(f => f.endsWith('.json')).map(f => `${t}/${f}`);
                            allNiftiFiles.push(...nifties);
                            allJsonFiles.push(...jsons);
                        }
                    }
                }

                // 2. Also look in the root of the folder (for older studies before the update)
                const rootFiles = await readdir(itemPath);
                for (const f of rootFiles) {
                    const fPath = join(itemPath, f);
                    const fStats = await stat(fPath);
                    if (fStats.isFile()) {
                        if (f.endsWith('.nii') || f.endsWith('.nii.gz')) allNiftiFiles.push(f);
                        if (f.endsWith('.json')) allJsonFiles.push(f);
                    }
                }

                // Push folder if it contains files or subdirectories
                if (allNiftiFiles.length > 0 || allJsonFiles.length > 0 || existsSync(join(itemPath, "anat")) || existsSync(join(itemPath, "func"))) {
                    folders.push({
                        name: item,
                        fileCount: allNiftiFiles.length,
                        files: allNiftiFiles,
                        jsonFiles: allJsonFiles,
                    });
                }
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
