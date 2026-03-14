import { NextRequest, NextResponse } from "next/server";
import { join } from "path";
import { writeFile, mkdir } from "fs/promises";
import { existsSync } from "fs";

export async function POST(request: NextRequest) {
    try {
        const formData = await request.formData();
        const files = formData.getAll("file") as File[];
        const folderName = formData.get("folderName") as string | null;
        const folderType = formData.get("folderType") as string | null;

        if (!files || files.length === 0) {
            return NextResponse.json(
                { error: "No files provided" },
                { status: 400 }
            );
        }

        if (!folderName || folderName.trim() === "") {
            return NextResponse.json(
                { error: "Study/Folder Name is required" },
                { status: 400 }
            );
        }
        
        const type = folderType === "anat" ? "anat" : "func";

        // Sanitize folder name (basic)
        const safeFolderName = folderName.replace(/[^a-zA-Z0-9_\-]/g, "_");

        // Ensure uploads directory exists: uploads/StudyName/type
        const baseUploadDir = join(process.cwd(), "..", "uploads");
        const targetDir = join(baseUploadDir, safeFolderName, type);

        if (!existsSync(targetDir)) {
            await mkdir(targetDir, { recursive: true });
        }

        const savedFiles: string[] = [];

        // Process each file
        for (const file of files) {
            const bytes = await file.arrayBuffer();
            const buffer = Buffer.from(bytes);
            const filePath = join(targetDir, file.name);
            await writeFile(filePath, buffer);
            savedFiles.push(file.name);
        }

        return NextResponse.json({ 
            success: true, 
            count: savedFiles.length, 
            folder: safeFolderName,
            type: type 
        });
    } catch (error) {
        console.error("Upload error:", error);
        return NextResponse.json(
            { error: "Error uploading files" },
            { status: 500 }
        );
    }
}
