"use client";

import { useState } from "react";
import FileUpload from "@/components/FileUpload";
import FileList from "@/components/FileList";
import ConvertedList from "@/components/ConvertedList";

export default function Home() {
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  const [conversionTrigger, setConversionTrigger] = useState(0);

  return (
    <div className="flex flex-col items-center justify-start min-h-[90vh] px-4 py-12 max-w-6xl mx-auto w-full">
      <div className="relative z-10 mb-12">
        <div className="absolute -inset-4 rounded-3xl bg-emerald-400 opacity-20 blur-2xl animate-pulse"></div>
        <h1 className="relative text-5xl md:text-7xl font-black text-black tracking-tighter text-center mb-4 drop-shadow-sm">
          fMRI <span className="text-emerald-600">Analytics</span>
        </h1>
        <p className="text-xl text-gray-600 text-center font-light leading-snug">
          Upload DICOM files, convert to NIfTI, and prepare for preprocessing.
        </p>
      </div>

      <div className="w-full grid md:grid-cols-2 gap-8 items-start">
        {/* Left Column: Upload */}
        <div className="w-full space-y-8">
          <FileUpload onUploadSuccess={() => setRefreshTrigger(prev => prev + 1)} />
        </div>

        {/* Right Column: Manage & Convert */}
        <div className="w-full space-y-8">
          <FileList
            refreshTrigger={refreshTrigger}
            onConversionSuccess={() => setConversionTrigger(prev => prev + 1)}
          />
          <ConvertedList refreshTrigger={conversionTrigger} />
        </div>
      </div>

      <div className="mt-16 flex gap-6 justify-center w-full">
        <div className="h-2 w-16 rounded-full bg-emerald-500 animate-bounce [animation-delay:-0.3s]"></div>
        <div className="h-2 w-16 rounded-full bg-teal-500 animate-bounce [animation-delay:-0.15s]"></div>
        <div className="h-2 w-16 rounded-full bg-emerald-700 animate-bounce"></div>
      </div>
    </div>
  );
}
