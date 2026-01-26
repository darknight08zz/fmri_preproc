import React, { useEffect, useRef, useState } from 'react';
import { Niivue } from '@niivue/niivue';
import { Loader2, Maximize2, Layers } from 'lucide-react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: (string | undefined | null | false)[]) {
    return twMerge(clsx(inputs));
}

interface NiftiViewerProps {
    url: string;
    className?: string;
}

export default function NiftiViewer({ url, className }: NiftiViewerProps) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [loading, setLoading] = useState(true);
    const [nv, setNv] = useState<Niivue | null>(null);

    useEffect(() => {
        if (!canvasRef.current) return;

        const niivue = new Niivue({
            show3Dcrosshair: true,
            loadingText: "Loading NIfTI...",
            backColor: [0, 0, 0, 1]
        });

        niivue.attachToCanvas(canvasRef.current);
        setNv(niivue);

        const loadVolume = async () => {
            setLoading(true);
            try {
                await niivue.loadVolumes([{ url }]);
            } catch (e) {
                console.error("Failed to load volume", e);
            } finally {
                setLoading(false);
            }
        };

        loadVolume();

        // Cleanup
        return () => {
            // Niivue cleanup if needed, usually reliable on detach
        };
    }, [url]);

    return (
        <div className={cn("relative bg-black rounded-xl overflow-hidden border border-slate-800", className)}>
            <div className="absolute top-4 left-4 z-10 flex gap-2">
                <div className="bg-slate-900/80 backdrop-blur px-3 py-1 rounded-md text-xs text-white font-mono border border-slate-700">
                    Niivue Viewer
                </div>
            </div>

            {loading && (
                <div className="absolute inset-0 flex items-center justify-center bg-slate-950/80 z-20">
                    <Loader2 className="w-8 h-8 text-cyan-400 animate-spin" />
                </div>
            )}

            <canvas ref={canvasRef} className="w-full h-full outline-none" />

            {/* Simple Controls Overlay */}
            <div className="absolute bottom-4 left-1/2 -translate-x-1/2 bg-slate-900/80 backdrop-blur p-2 rounded-lg border border-slate-700 flex gap-2">
                <button
                    onClick={() => nv?.setSliceType(nv.sliceTypeMultiplanar)}
                    className="p-2 hover:bg-slate-700 rounded text-slate-300 hover:text-white transition-colors"
                    title="Multiplanar View"
                >
                    <Layers className="w-4 h-4" />
                </button>
                <button
                    onClick={() => nv?.setSliceType(nv.sliceTypeRender)}
                    className="p-2 hover:bg-slate-700 rounded text-slate-300 hover:text-white transition-colors"
                    title="3D Render"
                >
                    <Maximize2 className="w-4 h-4" />
                </button>
            </div>
        </div>
    );
}
