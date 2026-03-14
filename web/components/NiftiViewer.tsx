"use client";

import {
    useEffect, useRef, useState, useCallback, memo
} from "react";
import { Niivue } from "@niivue/niivue";

// ─────────────────────────────────────────────
//  Types
// ─────────────────────────────────────────────

interface NiftiViewerProps {
    url: string;
    /** Optional overlays pre-loaded on mount (e.g. passed from segmentation page) */
    initialOverlays?: OverlayConfig[];
}

interface ImageInfo {
    name: string; dims: number[]; pixDims: number[];
    datatype: string; nVolumes: number;
}

interface CrosshairState {
    mm: number[]; vox: number[]; intensity: number | null;
}

// ── Overlay ──────────────────────────────────
interface OverlayConfig {
    id:      string;   // unique key
    url:     string;   // NIfTI file URL
    label:   string;   // display name  e.g. "c1 — GM"
    color:   string;   // Niivue colormap name
    opacity: number;   // 0–1
    visible: boolean;
}

// SPM default overlay colors — matches your screenshot exactly:
//   GM  → red    (cortex, basal ganglia)
//   WM  → yellow (deep white matter)
//   CSF → green  (ventricles, sulci)
const SPM_DEFAULTS = [
    { key: "c1", label: "c1 — Gray Matter",  color: "red",    hex: "#f87171" },
    { key: "c2", label: "c2 — White Matter", color: "yellow", hex: "#fbbf24" },
    { key: "c3", label: "c3 — CSF",          color: "green",  hex: "#4ade80" },
] as const;

// All colormaps available for overlays
const OVERLAY_COLORMAPS = [
    { name: "red",     hex: "#f87171" },
    { name: "yellow",  hex: "#fbbf24" },
    { name: "green",   hex: "#4ade80" },
    { name: "blue",    hex: "#60a5fa" },
    { name: "orange",  hex: "#fb923c" },
    { name: "cyan",    hex: "#22d3ee" },
    { name: "pink",    hex: "#f472b6" },
    { name: "hot",     hex: "#f59e0b" },
    { name: "cool",    hex: "#818cf8" },
    { name: "plasma",  hex: "#a78bfa" },
    { name: "viridis", hex: "#34d399" },
    { name: "inferno", hex: "#ef4444" },
] as const;

type SliceView = "multiplanar" | "axial" | "coronal" | "sagittal" | "render";
type SidebarTab = "info" | "controls" | "overlays";

const COLORMAP_OPTIONS = [
    "gray", "hot", "cool", "plasma", "viridis", "inferno", "jet",
] as const;

const VIEW_LABELS: Record<SliceView, string> = {
    multiplanar: "Multi", axial: "Ax", coronal: "Cor", sagittal: "Sag", render: "3D",
};

const DEFAULT_CROSSHAIR: CrosshairState = { mm: [0, 0, 0], vox: [0, 0, 0], intensity: null };

const hexOf = (colorName: string) =>
    OVERLAY_COLORMAPS.find((c) => c.name === colorName)?.hex ?? "#7b8494";

// ─────────────────────────────────────────────
//  Main component
// ─────────────────────────────────────────────

export default function NiftiViewer({ url, initialOverlays = [] }: NiftiViewerProps) {
    const canvasRef        = useRef<HTMLCanvasElement>(null);
    const nvRef            = useRef<Niivue | null>(null);
    const isInitialisedRef = useRef(false);
    const crosshairRawRef  = useRef<CrosshairState>(DEFAULT_CROSSHAIR);
    const rafPendingRef    = useRef(false);

    // ── Viewer state ──────────────────────────────────────────────────
    const [isLoading,     setIsLoading]     = useState(true);
    const [loadError,     setLoadError]     = useState<string | null>(null);
    const [imageInfo,     setImageInfo]     = useState<ImageInfo | null>(null);
    const [activeView,    setActiveView]    = useState<SliceView>("multiplanar");
    const [crosshairOn,   setCrosshairOn]   = useState(true);
    const [colormap,      setColormap]      = useState("gray");
    const [brightness,    setBrightness]    = useState(50);
    const [contrast,      setContrast]      = useState(50);
    const [currentVolume, setCurrentVolume] = useState(0);
    const [activeTab,     setActiveTab]     = useState<SidebarTab>("info");

    // ── Overlay state ─────────────────────────────────────────────────
    const [overlays, setOverlays] = useState<OverlayConfig[]>(initialOverlays);

    // ── Crosshair display (RAF-throttled) ─────────────────────────────
    const [crosshair, setCrosshair] = useState<CrosshairState>(DEFAULT_CROSSHAIR);

    // ── Available files for overlays ──────────────────────────────────
    const [availableFiles, setAvailableFiles] = useState<string[]>([]);

    useEffect(() => {
        const fetchFiles = async () => {
            try {
                const res = await fetch("/api/converted-files");
                const data = await res.json();
                if (data.folders) {
                    const flattened: string[] = [];
                    data.folders.forEach((folder: any) => {
                        if (folder.files) {
                            folder.files.forEach((file: string) => {
                                // The API already returns paths like "anat/file.nii"
                                // We need "folderName/anat/file.nii" for /api/serve-file?path=
                                flattened.push(`${folder.name}/${file}`);
                            });
                        }
                    });
                    setAvailableFiles(flattened);
                }
            } catch (err) {
                console.error("Failed to fetch available files:", err);
            }
        };
        fetchFiles();
    }, []);

    // ─────────────────────────────────────────
    //  Init Niivue
    // ─────────────────────────────────────────

    useEffect(() => {
        if (isInitialisedRef.current) return;
        const canvas = canvasRef.current;
        if (!canvas) return;
        let isCancelled = false;

        const initNiivue = async () => {
            if (isCancelled || isInitialisedRef.current) return;
            isInitialisedRef.current = true;

            const nv = new Niivue({
                dragAndDropEnabled: false,
                backColor:          [0.07, 0.07, 0.09, 1],
                show3Dcrosshair:    false,
                isColorbar:         false,
            });
            nvRef.current = nv;
            nv.attachToCanvas(canvas);

            // RAF-throttled crosshair (prevents 180 re-renders/sec shake)
            nv.onLocationChange = (data: any) => {
                if (!data) return;
                crosshairRawRef.current = {
                    mm:  data.mm  ?? crosshairRawRef.current.mm,
                    vox: data.vox ?? crosshairRawRef.current.vox,
                    intensity: data.values?.length > 0
                        ? (typeof data.values[0] === "number" ? data.values[0] : null)
                        : crosshairRawRef.current.intensity,
                };
                if (!rafPendingRef.current) {
                    rafPendingRef.current = true;
                    requestAnimationFrame(() => {
                        rafPendingRef.current = false;
                        setCrosshair({ ...crosshairRawRef.current });
                    });
                }
            };

            setIsLoading(true);
            setLoadError(null);

            try {
                // Base T1w + any initial overlays
                const volumes: any[] = [{ url, colormap: "gray" }];
                for (const ov of initialOverlays) {
                    if (ov.visible) volumes.push(buildVolumeSpec(ov));
                }

                await nv.loadVolumes(volumes);
                if (isCancelled) return;

                nv.setSliceType((nv as any).sliceTypeMultiplanar);
                nv.opts.multiplanarShowRender    = 0;
                nv.opts.isRadiologicalConvention = false;
                nv.setSliceMM(true);
                nv.opts.crosshairWidth = 0.15;
                nv.opts.crosshairColor = [0.2, 0.6, 1, 1];
                applyMultiplanarLayout(nv);
                nv.updateGLVolume();

                const vol = (nv as any).volumes?.[0];
                if (vol) setImageInfo({
                    name:     vol.name || url.split("/").pop() || "Unknown",
                    dims:     vol.dims    ?? [],
                    pixDims:  vol.pixDims ?? [],
                    datatype: vol.hdr?.datatypeCode?.toString() ?? "unknown",
                    nVolumes: vol.dims?.[4] ?? 1,
                });
            } catch (err: any) {
                if (!isCancelled) {
                    setLoadError(err?.message ?? "Failed to load NIfTI.");
                    console.error("NiftiViewer:", err);
                }
            } finally {
                if (!isCancelled) setIsLoading(false);
            }
        };

        const ro = new ResizeObserver((entries) => {
            const r = entries[0]?.contentRect;
            if (r && r.width > 0 && r.height > 0) { ro.disconnect(); initNiivue(); }
        });
        ro.observe(canvas);
        if (canvas.clientWidth > 0 && canvas.clientHeight > 0) { ro.disconnect(); initNiivue(); }

        return () => {
            isCancelled = true;
            ro.disconnect();
            rafPendingRef.current = false;
            nvRef.current = null;
        };
    }, [url]);

    // Reset on URL change
    useEffect(() => {
        isInitialisedRef.current = false;
        setCrosshair(DEFAULT_CROSSHAIR);
        crosshairRawRef.current = DEFAULT_CROSSHAIR;
        setIsLoading(true);
        setLoadError(null);
        setImageInfo(null);
        setCurrentVolume(0);
    }, [url]);

    // ─────────────────────────────────────────
    //  View / display helpers
    // ─────────────────────────────────────────

    const applyMultiplanarLayout = (nv: Niivue) => {
        (nv as any).setMultiplanarLayout([
            { sliceType: (nv as any).sliceTypeCoronal,  left: 0,   top: 0,   width: 0.5, height: 0.5 },
            { sliceType: (nv as any).sliceTypeSagittal, left: 0.5, top: 0,   width: 0.5, height: 0.5 },
            { sliceType: (nv as any).sliceTypeAxial,    left: 0,   top: 0.5, width: 0.5, height: 0.5 },
            { sliceType: (nv as any).sliceTypeRender,   left: 0.5, top: 0.5, width: 0.5, height: 0.5 },
        ]);
    };

    const switchView = useCallback((view: SliceView) => {
        const nv = nvRef.current; if (!nv) return;
        setActiveView(view);
        if (view === "multiplanar") {
            nv.setSliceType((nv as any).sliceTypeMultiplanar);
            nv.opts.multiplanarShowRender = 0;
            applyMultiplanarLayout(nv);
        } else {
            const map: Record<string, any> = {
                axial:    (nv as any).sliceTypeAxial,
                coronal:  (nv as any).sliceTypeCoronal,
                sagittal: (nv as any).sliceTypeSagittal,
                render:   (nv as any).sliceTypeRender,
            };
            nv.setSliceType(map[view]);
        }
        nv.updateGLVolume();
    }, []);

    const toggleCrosshair = useCallback(() => {
        const nv = nvRef.current; if (!nv) return;
        const next = !crosshairOn;
        nv.opts.crosshairWidth = next ? 0.15 : 0;
        nv.updateGLVolume();
        setCrosshairOn(next);
    }, [crosshairOn]);

    const applyColormap = useCallback((cm: string) => {
        const nv = nvRef.current;
        if (!nv || !(nv as any).volumes?.length) return;
        setColormap(cm);
        nv.setColormap((nv as any).volumes[0].id, cm);
        nv.updateGLVolume();
    }, []);

    const applyBrightnessContrast = useCallback((b: number, c: number) => {
        const vol = (nvRef.current as any)?.volumes?.[0]; if (!vol) return;
        const min = vol.global_min ?? 0, max = vol.global_max ?? 4096;
        const range  = max - min;
        const center = min + range * (b / 100);
        const half   = range * (1 - c / 100) * 0.5 + range * 0.05;
        vol.cal_min  = center - half;
        vol.cal_max  = center + half;
        nvRef.current!.updateGLVolume();
    }, []);

    const handleBrightness = (v: number) => { setBrightness(v); applyBrightnessContrast(v, contrast); };
    const handleContrast   = (v: number) => { setContrast(v);   applyBrightnessContrast(brightness, v); };

    const handleVolumeChange = useCallback((t: number) => {
        const nv = nvRef.current;
        if (!nv || !(nv as any).volumes?.length) return;
        setCurrentVolume(t);
        nv.setFrame4D((nv as any).volumes[0].id, t);
        nv.updateGLVolume();
    }, []);

    const resetView = useCallback(() => {
        const nv = nvRef.current; if (!nv) return;
        setActiveView("multiplanar"); setBrightness(50); setContrast(50);
        setCrosshairOn(true); setColormap("gray");
        nv.setSliceType((nv as any).sliceTypeMultiplanar);
        nv.opts.multiplanarShowRender = 0;
        nv.opts.crosshairWidth = 0.15;
        nv.opts.crosshairColor = [0.2, 0.6, 1, 1];
        const vol = (nv as any).volumes?.[0];
        if (vol) { vol.cal_min = vol.global_min; vol.cal_max = vol.global_max; nv.setColormap(vol.id, "gray"); }
        applyMultiplanarLayout(nv);
        nv.updateGLVolume();
    }, []);

    // ─────────────────────────────────────────
    //  Overlay engine
    // ─────────────────────────────────────────

    /**
     * Re-sync ALL visible overlays into Niivue.
     *
     * Strategy: reload volumes[0] (T1w base) + every visible overlay.
     * This is the cleanest approach — Niivue doesn't expose a
     * remove-single-volume API reliably across versions.
     * SPM does the same: it redraws the full stack on every overlay change.
     */
    const syncToNiivue = useCallback(async (next: OverlayConfig[]) => {
        const nv = nvRef.current; if (!nv) return;
        setIsLoading(true);
        try {
            const volumes: any[] = [{ url, colormap }];
            for (const ov of next) {
                if (ov.visible) volumes.push(buildVolumeSpec(ov));
            }
            await nv.loadVolumes(volumes);
            nv.opts.crosshairWidth = crosshairOn ? 0.15 : 0;
            nv.updateGLVolume();
        } catch (err) {
            console.error("Overlay sync:", err);
        } finally {
            setIsLoading(false);
        }
    }, [url, colormap, crosshairOn]);

    // Add one overlay by URL + auto-detect label/color from filename
    const addOverlay = useCallback(async (ovUrl: string, label: string, color: string) => {
        if (!ovUrl.trim()) return;
        
        // Rewrite incorrect /api/files/ paths to the actual file serving endpoint
        let finalUrl = ovUrl.trim();
        if (finalUrl.startsWith("/api/files/")) {
            finalUrl = `/api/serve-file?path=${finalUrl.replace("/api/files/", "")}`;
        }

        const newOv: OverlayConfig = {
            id:      `ov-${Date.now()}-${Math.random().toString(36).slice(2)}`,
            url:     finalUrl,
            label,
            color,
            opacity: 0.5,
            visible: true,
        };
        const next = [...overlays, newOv];
        setOverlays(next);
        await syncToNiivue(next);
    }, [overlays, syncToNiivue]);

    const toggleOverlay = useCallback(async (id: string) => {
        const next = overlays.map((o) => o.id === id ? { ...o, visible: !o.visible } : o);
        setOverlays(next);
        await syncToNiivue(next);
    }, [overlays, syncToNiivue]);

    const changeColor = useCallback(async (id: string, color: string) => {
        const next = overlays.map((o) => o.id === id ? { ...o, color } : o);
        setOverlays(next);
        await syncToNiivue(next);
    }, [overlays, syncToNiivue]);

    const changeOpacity = useCallback(async (id: string, opacity: number) => {
        const next = overlays.map((o) => o.id === id ? { ...o, opacity } : o);
        setOverlays(next);
        await syncToNiivue(next);
    }, [overlays, syncToNiivue]);

    const removeOverlay = useCallback(async (id: string) => {
        const next = overlays.filter((o) => o.id !== id);
        setOverlays(next);
        await syncToNiivue(next);
    }, [overlays, syncToNiivue]);

    const removeAllOverlays = useCallback(async () => {
        setOverlays([]);
        await syncToNiivue([]);
    }, [syncToNiivue]);

    const is4D = imageInfo && imageInfo.nVolumes > 1;
    const visibleCount = overlays.filter((o) => o.visible).length;

    // ─────────────────────────────────────────
    //  Render
    // ─────────────────────────────────────────

    return (
        <div
            style={{ fontFamily: "'JetBrains Mono','Fira Code',monospace" }}
            className="flex flex-col min-h-screen bg-[#0d0e11] text-[#c8cdd6] select-none"
        >
            {/* ── Top bar ──────────────────────────────────────────── */}
            <header className="flex items-center justify-between px-4 py-2 bg-[#13141a] border-b border-[#1f2130]">
                <div className="flex items-center gap-2 min-w-0">
                    <div className="w-6 h-6 rounded-sm bg-[#2563eb] flex-shrink-0 flex items-center justify-center">
                        <span className="text-white text-[10px] font-bold leading-none">N</span>
                    </div>
                    <span className="text-[11px] font-semibold tracking-widest uppercase text-[#7b8494] flex-shrink-0">
                        NIfTI Viewer
                    </span>
                    {imageInfo && (
                        <span className="text-[11px] text-[#4a5568] truncate max-w-[180px]">
                            / {imageInfo.name}
                        </span>
                    )}
                    {/* Active overlay pills in top bar */}
                    {overlays.filter((o) => o.visible).map((ov) => (
                        <span key={ov.id}
                            className="hidden md:flex items-center gap-1 px-1.5 py-0.5
                                       bg-[#1a1b22] border border-[#2d2f3d] rounded
                                       text-[8px] font-mono text-[#7b8494] flex-shrink-0">
                            <span className="w-1.5 h-1.5 rounded-full"
                                  style={{ background: hexOf(ov.color) }} />
                            {ov.label.split("—")[0].trim()}
                        </span>
                    ))}
                </div>

                {/* View switcher */}
                <div className="flex gap-px bg-[#1a1b22] rounded border border-[#252835] overflow-hidden flex-shrink-0">
                    {(Object.keys(VIEW_LABELS) as SliceView[]).map((v) => (
                        <button key={v} onClick={() => switchView(v)}
                            className={`px-3 py-1 text-[10px] tracking-widest uppercase transition-all ${
                                activeView === v
                                    ? "bg-[#2563eb] text-white"
                                    : "text-[#5a6477] hover:text-[#9ba8bb] hover:bg-[#1f2232]"
                            }`}
                        >{VIEW_LABELS[v]}</button>
                    ))}
                </div>

                <button onClick={resetView}
                    className="flex-shrink-0 px-3 py-1 text-[10px] tracking-widest uppercase
                               text-[#5a6477] hover:text-white border border-[#252835] rounded
                               hover:border-[#2563eb] transition-all"
                >Reset</button>
            </header>

            {/* ── Main ─────────────────────────────────────────────── */}
            <div className="flex flex-1 overflow-hidden">

                {/* Canvas */}
                <div className="relative flex-1 bg-[#0d0e11]" style={{ contain: "layout size" }}>
                    <canvas
                        ref={canvasRef}
                        className="w-full h-full block cursor-crosshair"
                        style={{ minHeight: 480 }}
                    />

                    {isLoading && (
                        <div className="absolute inset-0 flex flex-col items-center justify-center
                                        bg-[#0d0e11]/90 z-10 pointer-events-none">
                            <div className="w-8 h-8 border-2 border-[#2563eb] border-t-transparent
                                            rounded-full animate-spin mb-3" />
                            <span className="text-[11px] tracking-widest uppercase text-[#5a6477]">
                                {overlays.length > 0 ? "Updating overlays…" : "Loading volume…"}
                            </span>
                        </div>
                    )}

                    {loadError && (
                        <div className="absolute inset-0 flex flex-col items-center justify-center
                                        bg-[#0d0e11]/90 z-10 pointer-events-none">
                            <div className="text-red-400 text-[11px] tracking-wider uppercase mb-2">Load Error</div>
                            <div className="text-[#5a6477] text-xs max-w-xs text-center">{loadError}</div>
                        </div>
                    )}

                    {!isLoading && !loadError && <CrosshairHUD crosshair={crosshair} />}

                    {is4D && !isLoading && (
                        <div className="absolute top-3 right-3 z-10 bg-[#0d0e11]/80
                                        border border-[#1f2130] rounded px-3 py-1.5
                                        text-[10px] font-mono pointer-events-none">
                            <span className="text-[#4a5568]">t = </span>
                            <span className="text-[#fbbf24]">{currentVolume}</span>
                            <span className="text-[#4a5568]"> / {imageInfo!.nVolumes - 1}</span>
                        </div>
                    )}
                </div>

                {/* Sidebar */}
                <Sidebar
                    imageInfo={imageInfo}
                    crosshair={crosshair}
                    activeTab={activeTab}
                    setActiveTab={setActiveTab}
                    crosshairOn={crosshairOn}
                    colormap={colormap}
                    brightness={brightness}
                    contrast={contrast}
                    currentVolume={currentVolume}
                    is4D={!!is4D}
                    nVolumes={imageInfo?.nVolumes ?? 1}
                    overlays={overlays}
                    visibleCount={visibleCount}
                    onToggleCrosshair={toggleCrosshair}
                    onColormap={applyColormap}
                    onBrightness={handleBrightness}
                    onContrast={handleContrast}
                    onVolumeChange={handleVolumeChange}
                    onAddOverlay={addOverlay}
                    onToggleOverlay={toggleOverlay}
                    onChangeColor={changeColor}
                    onChangeOpacity={changeOpacity}
                    onRemoveOverlay={removeOverlay}
                    onRemoveAll={removeAllOverlays}
                    availableFiles={availableFiles}
                />
            </div>
        </div>
    );
}

// ─────────────────────────────────────────────
//  Helper — build Niivue volume spec for an overlay
// ─────────────────────────────────────────────

function buildVolumeSpec(ov: OverlayConfig) {
    return {
        url:      ov.url,
        colormap: ov.color,
        opacity:  ov.opacity,
        // cal_min=0.01 hides near-zero background voxels (same as SPM threshold)
        cal_min:  0.01,
        cal_max:  1.0,
    };
}

// ─────────────────────────────────────────────
//  CrosshairHUD
// ─────────────────────────────────────────────

const CrosshairHUD = memo(function CrosshairHUD({ crosshair }: { crosshair: CrosshairState }) {
    const fmt = (v: number) => (v >= 0 ? ` ${v.toFixed(1)}` : v.toFixed(1));
    return (
        <div
            className="absolute top-3 left-3 z-10 bg-[#0d0e11]/80 border border-[#1f2130]
                        rounded px-3 py-2 text-[10px] font-mono tabular-nums space-y-0.5"
            style={{ pointerEvents: "none", willChange: "contents" }}
        >
            <div className="text-[#2563eb] uppercase tracking-widest text-[9px] mb-1">Crosshair</div>
            <div>
                <span className="text-[#4a5568] w-6 inline-block">mm</span>
                <span className="text-[#e2e8f0]">{crosshair.mm.map(fmt).join("  ")}</span>
            </div>
            <div>
                <span className="text-[#4a5568] w-6 inline-block">vx</span>
                <span className="text-[#94a3b8]">
                    {crosshair.vox.map((v) => String(Math.round(v)).padStart(3)).join("  ")}
                </span>
            </div>
            <div>
                <span className="text-[#4a5568] w-6 inline-block">I</span>
                <span className="text-[#fbbf24]">
                    {crosshair.intensity !== null ? crosshair.intensity.toFixed(2) : "—"}
                </span>
            </div>
        </div>
    );
});

// ─────────────────────────────────────────────
//  Sidebar
// ─────────────────────────────────────────────

interface SidebarProps {
    imageInfo:     ImageInfo | null;
    crosshair:     CrosshairState;
    activeTab:     SidebarTab;
    setActiveTab:  (t: SidebarTab) => void;
    crosshairOn:   boolean;
    colormap:      string;
    brightness:    number;
    contrast:      number;
    currentVolume: number;
    is4D:          boolean;
    nVolumes:      number;
    overlays:      OverlayConfig[];
    visibleCount:  number;
    onToggleCrosshair: () => void;
    onColormap:        (cm: string) => void;
    onBrightness:      (v: number) => void;
    onContrast:        (v: number) => void;
    onVolumeChange:    (v: number) => void;
    onAddOverlay:      (url: string, label: string, color: string) => void;
    onToggleOverlay:   (id: string) => void;
    onChangeColor:     (id: string, color: string) => void;
    onChangeOpacity:   (id: string, opacity: number) => void;
    onRemoveOverlay:   (id: string) => void;
    onRemoveAll:       () => void;
    availableFiles:    string[];
}

const Sidebar = memo(function Sidebar({
    imageInfo, crosshair, activeTab, setActiveTab,
    crosshairOn, colormap, brightness, contrast, currentVolume, is4D, nVolumes,
    overlays, visibleCount,
    onToggleCrosshair, onColormap, onBrightness, onContrast, onVolumeChange,
    onAddOverlay, onToggleOverlay, onChangeColor, onChangeOpacity,
    onRemoveOverlay, onRemoveAll, availableFiles,
}: SidebarProps) {
    return (
        <aside className="w-64 min-w-[256px] flex-shrink-0 bg-[#10111a] border-l border-[#1a1b26] flex flex-col text-[11px]">

            {/* Tabs */}
            <div className="flex border-b border-[#1a1b26]">
                {(["info", "controls", "overlays"] as SidebarTab[]).map((tab) => (
                    <button key={tab} onClick={() => setActiveTab(tab)}
                        className={`flex-1 py-2 uppercase tracking-widest text-[9px]
                                    transition-all relative ${
                            activeTab === tab
                                ? "text-[#2563eb] border-b border-[#2563eb]"
                                : "text-[#4a5568] hover:text-[#7b8494]"
                        }`}
                    >
                        {tab}
                        {/* Badge on overlay tab */}
                        {tab === "overlays" && visibleCount > 0 && (
                            <span className="absolute top-1 right-1.5 w-3.5 h-3.5 bg-[#2563eb]
                                             rounded-full text-[7px] text-white flex items-center
                                             justify-center leading-none font-bold">
                                {visibleCount}
                            </span>
                        )}
                    </button>
                ))}
            </div>

            {/* ── INFO ─────────────────────────────── */}
            {activeTab === "info" && (
                <div className="flex-1 overflow-y-auto p-3 space-y-4">
                    {imageInfo ? (
                        <>
                            <Section label="File">
                                <Row k="Name"  v={imageInfo.name} truncate />
                                <Row k="Type"  v={imageInfo.nVolumes > 1 ? `4D (${imageInfo.nVolumes} vols)` : "3D"} />
                                <Row k="Dtype" v={imageInfo.datatype} />
                            </Section>
                            <Section label="Geometry">
                                <Row k="Dims"
                                    v={imageInfo.dims[1] !== undefined
                                        ? `${imageInfo.dims[1]}×${imageInfo.dims[2]}×${imageInfo.dims[3]}` : "—"} />
                                <Row k="Vox"
                                    v={imageInfo.pixDims[1] !== undefined
                                        ? `${imageInfo.pixDims[1]?.toFixed(2)}×${imageInfo.pixDims[2]?.toFixed(2)}×${imageInfo.pixDims[3]?.toFixed(2)} mm` : "—"} />
                            </Section>
                            <Section label="Cursor">
                                <Row k="mm" v={crosshair.mm.map((v) => v.toFixed(1)).join(", ")} />
                                <Row k="vx" v={crosshair.vox.map((v) => Math.round(v)).join(", ")} />
                                <Row k="I"  v={crosshair.intensity !== null ? crosshair.intensity.toFixed(4) : "—"} highlight />
                            </Section>
                        </>
                    ) : (
                        <div className="text-[#3a4050] text-center py-8 uppercase tracking-widest text-[9px]">
                            No volume loaded
                        </div>
                    )}
                </div>
            )}

            {/* ── CONTROLS ─────────────────────────── */}
            {activeTab === "controls" && (
                <div className="flex-1 overflow-y-auto p-3 space-y-5">
                    <Section label="Crosshair">
                        <button onClick={onToggleCrosshair}
                            className={`w-full py-1.5 text-[9px] uppercase tracking-widest
                                        rounded border transition-all ${crosshairOn
                                ? "bg-[#1e3a6e] border-[#2563eb] text-[#93c5fd]"
                                : "bg-[#1a1b22] border-[#252835] text-[#4a5568]"
                            }`}
                        >{crosshairOn ? "Visible" : "Hidden"}</button>
                    </Section>

                    <Section label="Colormap">
                        <div className="grid grid-cols-2 gap-1">
                            {COLORMAP_OPTIONS.map((cm) => (
                                <button key={cm} onClick={() => onColormap(cm)}
                                    className={`py-1 text-[9px] uppercase tracking-wider rounded
                                                border transition-all ${colormap === cm
                                            ? "bg-[#1e3a6e] border-[#2563eb] text-[#93c5fd]"
                                            : "border-[#1f2232] text-[#4a5568] hover:text-[#7b8494] hover:border-[#2d3250]"
                                        }`}
                                >{cm}</button>
                            ))}
                        </div>
                    </Section>

                    <Section label="Brightness">
                        <SliderControl value={brightness} onChange={onBrightness} min={0} max={100} />
                    </Section>
                    <Section label="Contrast">
                        <SliderControl value={contrast} onChange={onContrast} min={0} max={100} />
                    </Section>

                    {is4D && (
                        <Section label={`Volume  t=${currentVolume}`}>
                            <SliderControl value={currentVolume} onChange={onVolumeChange}
                                min={0} max={nVolumes - 1} step={1} accent="#fbbf24" />
                            <div className="flex justify-between text-[9px] text-[#3a4050] mt-1">
                                <span>0</span><span>{nVolumes - 1}</span>
                            </div>
                        </Section>
                    )}
                </div>
            )}

            {/* ── OVERLAYS ─────────────────────────── */}
            {activeTab === "overlays" && (
                <div className="flex-1 overflow-y-auto p-3 space-y-4">

                    {/* ── Quick-add: Segmentation outputs ── */}
                    <Section label="Add Segmentation Overlay">
                        <p className="text-[9px] text-[#3a4050] mb-3 leading-relaxed">
                            Enter the URL/path for each tissue map from your segmentation output.
                            Default colors match SPM.
                        </p>

                        {SPM_DEFAULTS.map((t) => (
                            <QuickAddRow
                                key={t.key}
                                label={t.label}
                                color={t.color}
                                hex={t.hex}
                                availableFiles={availableFiles}
                                onAdd={(url, color) => onAddOverlay(url, t.label, color)}
                            />
                        ))}
                    </Section>

                    {/* ── Active overlays ── */}
                    {overlays.length > 0 && (
                        <Section label={`Active Overlays (${overlays.length})`}>
                            {overlays.length > 1 && (
                                <button onClick={onRemoveAll}
                                    className="w-full mb-2 py-1 text-[8px] uppercase tracking-widest
                                               text-[#ef4444] border border-[#3a1515] rounded
                                               hover:bg-[#2d0a0a] transition-colors">
                                    Remove All
                                </button>
                            )}
                            <div className="space-y-3">
                                {overlays.map((ov) => (
                                    <OverlayRow
                                        key={ov.id}
                                        overlay={ov}
                                        onToggle={() => onToggleOverlay(ov.id)}
                                        onColorChange={(c) => onChangeColor(ov.id, c)}
                                        onOpacityChange={(o) => onChangeOpacity(ov.id, o)}
                                        onRemove={() => onRemoveOverlay(ov.id)}
                                    />
                                ))}
                            </div>
                        </Section>
                    )}

                    {overlays.length === 0 && (
                        <div className="text-center py-6 text-[9px] uppercase tracking-widest text-[#2d3350]">
                            No overlays added
                        </div>
                    )}
                </div>
            )}

            <div className="p-3 border-t border-[#1a1b26] text-[9px] text-[#2d3350]
                            uppercase tracking-widest text-center">
                Powered by NiiVue
            </div>
        </aside>
    );
});

// ─────────────────────────────────────────────
//  QuickAddRow — one row per SPM tissue (c1/c2/c3)
//  URL input + color selector + Add button
// ─────────────────────────────────────────────

function QuickAddRow({ label, color: defaultColor, hex, availableFiles, onAdd }: {
    label: string;
    color: string;
    hex:   string;
    availableFiles: string[];
    onAdd: (url: string, color: string) => void;
}) {
    const [url,      setUrl]      = useState("");
    const [selColor, setSelColor] = useState(defaultColor);

    return (
        <div className="mb-3 space-y-1.5">
            {/* Label + color swatch row */}
            <div className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-sm flex-shrink-0"
                      style={{ background: hexOf(selColor) }} />
                <span className="text-[9px] text-[#7b8494] flex-1 font-mono">{label}</span>
            </div>

            {/* Color picker */}
            <div className="flex flex-wrap gap-1 pl-4">
                {OVERLAY_COLORMAPS.map((cm) => (
                    <button
                        key={cm.name}
                        onClick={() => setSelColor(cm.name)}
                        title={cm.name}
                        className={`w-4 h-4 rounded-sm transition-all border ${
                            selColor === cm.name
                                ? "border-white scale-125 shadow-lg"
                                : "border-transparent hover:border-[#4a5568]"
                        }`}
                        style={{ background: cm.hex }}
                    />
                ))}
            </div>

            {/* URL dropdown + Add button */}
            <div className="flex gap-1 pl-4">
                <select
                    value={url}
                    onChange={(e) => setUrl(e.target.value)}
                    className="flex-1 px-2 py-1 bg-[#0d0e11] border border-[#1f2130]
                               rounded text-[8px] font-mono text-[#7b8494]
                               focus:outline-none focus:border-[#2563eb] focus:text-white"
                >
                    <option value="">-- select file --</option>
                    {availableFiles.map((file) => (
                        <option key={file} value={`/api/serve-file?path=${file}`}>
                            {file}
                        </option>
                    ))}
                </select>
                <button
                    onClick={() => url.trim() && onAdd(url, selColor)}
                    disabled={!url.trim()}
                    className="px-2.5 py-1 rounded text-[8px] font-bold transition-all
                               disabled:opacity-30 disabled:cursor-not-allowed"
                    style={{
                        background:  url.trim() ? hexOf(selColor) + "33" : undefined,
                        borderWidth: 1,
                        borderStyle: "solid",
                        borderColor: url.trim() ? hexOf(selColor) + "88" : "#1f2130",
                        color:       url.trim() ? hexOf(selColor)         : "#4a5568",
                    }}
                >
                    Add
                </button>
            </div>
        </div>
    );
}

// ─────────────────────────────────────────────
//  OverlayRow — controls for one active overlay
// ─────────────────────────────────────────────

function OverlayRow({ overlay, onToggle, onColorChange, onOpacityChange, onRemove }: {
    overlay:         OverlayConfig;
    onToggle:        () => void;
    onColorChange:   (color: string) => void;
    onOpacityChange: (opacity: number) => void;
    onRemove:        () => void;
}) {
    return (
        <div className={`rounded border p-2 space-y-2 transition-all ${
            overlay.visible
                ? "border-[#2d2f3d] bg-[#0d0e11]"
                : "border-[#1a1b22] bg-[#0a0b0e] opacity-40"
        }`}>
            {/* Row 1: visibility toggle + label + remove */}
            <div className="flex items-center gap-1.5">
                {/* Checkbox-style visibility toggle */}
                <button
                    onClick={onToggle}
                    title={overlay.visible ? "Hide" : "Show"}
                    className="w-4 h-4 rounded-sm border flex-shrink-0 flex items-center
                               justify-center transition-all"
                    style={{
                        background:  overlay.visible ? hexOf(overlay.color) : "transparent",
                        borderColor: overlay.visible ? hexOf(overlay.color) : "#3a4050",
                    }}
                >
                    {overlay.visible && (
                        <span className="text-[7px] font-bold leading-none"
                              style={{ color: "#000" }}>✓</span>
                    )}
                </button>

                <span className="flex-1 text-[9px] font-mono truncate"
                      style={{ color: overlay.visible ? hexOf(overlay.color) : "#4a5568" }}>
                    {overlay.label}
                </span>

                <button onClick={onRemove}
                    className="text-[#3a4050] hover:text-[#f87171] text-xs leading-none
                               px-0.5 transition-colors flex-shrink-0"
                    title="Remove">×</button>
            </div>

            {/* Row 2: Color swatches */}
            <div>
                <div className="text-[8px] text-[#2d3350] uppercase tracking-widest mb-1">
                    Color
                </div>
                <div className="flex flex-wrap gap-1">
                    {OVERLAY_COLORMAPS.map((cm) => (
                        <button
                            key={cm.name}
                            onClick={() => onColorChange(cm.name)}
                            title={cm.name}
                            className={`w-4 h-4 rounded-sm border transition-all ${
                                overlay.color === cm.name
                                    ? "border-white scale-125"
                                    : "border-transparent hover:border-[#4a5568]"
                            }`}
                            style={{ background: cm.hex }}
                        />
                    ))}
                </div>
            </div>

            {/* Row 3: Opacity slider */}
            <div>
                <div className="flex justify-between text-[8px] mb-1">
                    <span className="text-[#2d3350] uppercase tracking-widest">Opacity</span>
                    <span className="text-[#4a5568]">{Math.round(overlay.opacity * 100)}%</span>
                </div>
                <SliderControl
                    value={overlay.opacity * 100}
                    onChange={(v) => onOpacityChange(v / 100)}
                    min={0} max={100}
                    accent={hexOf(overlay.color)}
                />
            </div>
        </div>
    );
}

// ─────────────────────────────────────────────
//  Sub-components
// ─────────────────────────────────────────────

function Section({ label, children }: { label: string; children: React.ReactNode }) {
    return (
        <div>
            <div className="text-[9px] uppercase tracking-widest text-[#2d3350] mb-2 pb-1
                            border-b border-[#1a1b26]">
                {label}
            </div>
            <div className="space-y-1">{children}</div>
        </div>
    );
}

function Row({ k, v, truncate = false, highlight = false }: {
    k: string; v: string; truncate?: boolean; highlight?: boolean;
}) {
    return (
        <div className="flex gap-2 items-baseline">
            <span className="text-[#3a4050] w-10 flex-shrink-0 text-right">{k}</span>
            <span className={`font-mono tabular-nums text-[10px] ${highlight ? "text-[#fbbf24]" : "text-[#7b8494]"}
                             ${truncate ? "truncate" : ""}`}
                title={v}>{v}</span>
        </div>
    );
}

function SliderControl({ value, onChange, min, max, step = 1, accent = "#2563eb" }: {
    value: number; onChange: (v: number) => void;
    min: number; max: number; step?: number; accent?: string;
}) {
    return (
        <input type="range" min={min} max={max} step={step} value={value}
            onChange={(e) => onChange(Number(e.target.value))}
            className="w-full h-1 rounded appearance-none cursor-pointer"
            style={{
                background: `linear-gradient(to right, ${accent} ${((value - min) / (max - min)) * 100}%, #1f2232 0%)`,
                accentColor: accent,
            }}
        />
    );
}