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
        const folders: { name: string, anatCount: number, funcCount: number }[] = [];

        for (const item of items) {
            const itemPath = join(uploadDir, item);
            const stats = await stat(itemPath);
            if (stats.isDirectory()) {
                let anatCount = 0;
                let funcCount = 0;

                // Check anat/
                const anatPath = join(itemPath, "anat");
                if (existsSync(anatPath)) {
                    const files = await readdir(anatPath);
                    anatCount = files.length;
                }

                // Check func/
                const funcPath = join(itemPath, "func");
                if (existsSync(funcPath)) {
                    const files = await readdir(funcPath);
                    funcCount = files.length;
                }

                if (anatCount > 0 || funcCount > 0) {
                    folders.push({
                        name: item,
                        anatCount,
                        funcCount
                    });
                }
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
