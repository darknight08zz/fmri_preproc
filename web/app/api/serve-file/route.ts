import { NextRequest, NextResponse } from "next/server";
import { join } from "path";
import { readFile } from "fs/promises";
import { existsSync } from "fs";

export async function GET(request: NextRequest) {
    try {
        const searchParams = request.nextUrl.searchParams;
        const filePathParam = searchParams.get("path");

        if (!filePathParam) {
            return NextResponse.json(
                { error: "Path parameter is required" },
                { status: 400 }
            );
        }
        const baseDir = join(process.cwd(), "..", "converted");
        const fullPath = join(baseDir, filePathParam);

        if (!fullPath.startsWith(baseDir)) {
            return NextResponse.json(
                { error: "Invalid path" },
                { status: 403 }
            );
        }

        if (!existsSync(fullPath)) {
            return NextResponse.json(
                { error: "File not found" },
                { status: 404 }
            );
        }

        const fileBuffer = await readFile(fullPath);

        return new NextResponse(fileBuffer, {
            headers: {
                "Content-Type": "application/octet-stream",
                "Content-Disposition": `attachment; filename="${fullPath.split(/[\\/]/).pop()}"`,
            },
        });

    } catch (error) {
        console.error("Serve file error:", error);
        return NextResponse.json(
            { error: "Error serving file" },
            { status: 500 }
        );
    }
}
