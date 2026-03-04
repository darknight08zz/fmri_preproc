import { NextResponse } from "next/server";
import { readdir, stat } from "fs/promises";
import { join } from "path";
import { existsSync } from "fs";

export async function GET() {
    try {
        const uploadDir = join(process.cwd(), "..", "uploads");

        if (!existsSync(uploadDir)) {
            return NextResponse.json({ folders: [] });
        }

        const items = await readdir(uploadDir);

        // Filter to only include directories
        const folders: { name: string, fileCount: number }[] = [];

        for (const item of items) {
            const itemPath = join(uploadDir, item);
            const stats = await stat(itemPath);
            if (stats.isDirectory()) {
                const files = await readdir(itemPath);
                folders.push({
                    name: item,
                    fileCount: files.length
                });
            }
        }

        return NextResponse.json({ folders });
    } catch (error) {
        console.error("List folders error:", error);
        return NextResponse.json(
            { error: "Error listing folders" },
            { status: 500 }
        );
    }
}
