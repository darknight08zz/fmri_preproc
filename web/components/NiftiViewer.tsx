"use client";

import { useEffect, useRef, useState } from "react";
import { Niivue } from "@niivue/niivue";

interface NiftiViewerProps {
    url: string;
}

interface ImageInfo {
    name: string;
    dims: number[];
    pixDims: number[];
    datatype: string;
}

export default function NiftiViewer({ url }: NiftiViewerProps) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const nvRef = useRef<Niivue | null>(null);

    const [crosshairMM, setCrosshairMM] = useState("0.0 0.0 0.0");
    const [crosshairVox, setCrosshairVox] = useState("0 0 0");
    const [intensity, setIntensity] = useState("0");
    const [imageInfo, setImageInfo] = useState<ImageInfo | null>(null);
    const [crosshairVisible, setCrosshairVisible] = useState(true);

    useEffect(() => {
        if (!canvasRef.current) return;

        const nv = new Niivue({
            dragAndDropEnabled: false,
            backColor: [0, 0, 0, 1],
            show3Dcrosshair: false,
        });

        nv.attachToCanvas(canvasRef.current);
        nvRef.current = nv;

        nv.onLocationChange = (data: any) => {
            if (!data) return;

            if (data.mm) {
                setCrosshairMM(
                    data.mm.map((v: number) => v.toFixed(1)).join(" ")
                );
            }

            if (data.vox) {
                setCrosshairVox(
                    data.vox.map((v: number) => v.toFixed(0)).join(" ")
                );
            }

            if (data.values?.length > 0) {
                const val = data.values[0];
                setIntensity(
                    typeof val === "number" ? val.toFixed(2) : String(val)
                );
            }
        };

        const loadVolume = async () => {
            try {
                await nv.loadVolumes([{ url }]);

                // ---------- SPM STYLE CONFIG ----------

                nv.setSliceType(nv.sliceTypeMultiplanar);

                // Remove 3D render panel
                nv.opts.multiplanarShowRender = 0;

                // Neurological convention (SPM style)
                nv.opts.isRadiologicalConvention = false;

                // Use world-space (mm)
                nv.setSliceMM(true);

                // Blue crosshair
                nv.opts.crosshairWidth = 1;
                nv.opts.crosshairColor = [0, 0, 1, 1];

                // Exact SPM quadrant layout
                nv.setMultiplanarLayout([
                    {
                        sliceType: nv.sliceTypeCoronal,
                        left: 0,
                        top: 0,
                        width: 0.5,
                        height: 0.5,
                    },
                    {
                        sliceType: nv.sliceTypeSagittal,
                        left: 0.5,
                        top: 0,
                        width: 0.5,
                        height: 0.5,
                    },
                    {
                        sliceType: nv.sliceTypeAxial,
                        left: 0,
                        top: 0.5,
                        width: 0.5,
                        height: 0.5,
                    },
                ] as any);

                // Reset zoom and center
                // nv.setPan2Dxyzmm([0, 0, 0]);
                // nv.setZoom(1);

                nv.updateGLVolume();

                const vol = (nv as any).volumes[0];

                setImageInfo({
                    name: vol.name || url.split("/").pop() || "Unknown",
                    dims: vol.dims,
                    pixDims: vol.pixDims,
                    datatype:
                        vol.hdr?.datatypeCode?.toString() || "unknown",
                });
            } catch (err) {
                console.error("Error loading NIfTI:", err);
            }
        };

        loadVolume();

        return () => {
            nvRef.current = null;
        };
    }, [url]);

    const toggleCrosshair = () => {
        if (!nvRef.current) return;

        const nv = nvRef.current;
        const newState = !crosshairVisible;

        nv.opts.crosshairWidth = newState ? 1 : 0;
        nv.updateGLVolume();

        setCrosshairVisible(newState);
    };

    const resetView = () => {
        if (!nvRef.current) return;

        const nv = nvRef.current;
        nv.setSliceType(nv.sliceTypeMultiplanar);
        // nv.setPan2Dxyzmm([0, 0, 0]);
        // nv.setZoom(1);
        nv.updateGLVolume();
    };

    return (
        <div className="flex flex-col min-h-screen bg-[#1e1e1e] text-[#d4d4d4]">

            {/* -------- SQUARE SPM VIEWER -------- */}
            <div className="flex justify-center bg-black border-b border-[#333] py-6">
                <div className="w-full max-w-[800px] aspect-square">
                    <canvas
                        ref={canvasRef}
                        className="w-full h-full block cursor-crosshair"
                    />
                </div>
            </div>

            {/* -------- BOTTOM PANEL -------- */}
            <div className="h-64 bg-[#2d2d2d] grid grid-cols-2 gap-px border-t border-[#444]">

                {/* Crosshair Info */}
                <div className="bg-[#252526] p-4 flex flex-col gap-3">
                    <h3 className="text-white font-bold">
                        Crosshair Position
                        <span className="text-xs text-gray-400 ml-2 bg-[#333] px-1 rounded">
                            Origin
                        </span>
                    </h3>

                    <div className="grid grid-cols-[80px_1fr] gap-2 text-sm font-mono">
                        <span className="text-right text-cyan-400">mm:</span>
                        <div className="bg-white text-black px-2 py-1 text-right border shadow-inner">
                            {crosshairMM}
                        </div>

                        <span className="text-right text-cyan-400">vx:</span>
                        <div className="bg-white text-gray-500 px-2 py-1 text-right border shadow-inner">
                            {crosshairVox}
                        </div>

                        <span className="text-right text-cyan-400">Intensity:</span>
                        <div className="bg-[#1e1e1e] text-white px-2 py-1 text-right border border-gray-600">
                            {intensity}
                        </div>
                    </div>
                </div>

                {/* Image Info */}
                <div className="bg-[#252526] p-4 relative">
                    <h3 className="text-white font-bold truncate">
                        File:
                        <span className="text-gray-300 font-normal ml-2">
                            {imageInfo?.name}
                        </span>
                    </h3>

                    <div className="mt-4 grid grid-cols-[100px_1fr] gap-y-1 text-xs font-mono text-gray-300">
                        <span className="text-right text-gray-400">Dimensions:</span>
                        <span>
                            {imageInfo?.dims
                                ? `${imageInfo.dims[1]} × ${imageInfo.dims[2]} × ${imageInfo.dims[3]}`
                                : "..."}
                        </span>

                        <span className="text-right text-gray-400">Datatype:</span>
                        <span>{imageInfo?.datatype}</span>

                        <span className="text-right text-gray-400">Vox size:</span>
                        <span>
                            {imageInfo?.pixDims
                                ? `${imageInfo.pixDims[1]?.toFixed(2)} × ${imageInfo.pixDims[2]?.toFixed(2)} × ${imageInfo.pixDims[3]?.toFixed(2)}`
                                : "..."}
                        </span>
                    </div>

                    {/* Buttons */}
                    <div className="absolute bottom-3 right-3 flex gap-2">
                        <button
                            onClick={resetView}
                            className="px-3 py-1 bg-[#3c3c3c] hover:bg-[#4c4c4c] text-xs text-white border rounded"
                        >
                            Full Volume
                        </button>

                        <button
                            onClick={toggleCrosshair}
                            className="px-3 py-1 bg-[#3c3c3c] hover:bg-[#4c4c4c] text-xs text-white border rounded"
                        >
                            {crosshairVisible
                                ? "Hide Crosshair"
                                : "Show Crosshair"}
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}
