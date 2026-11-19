import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  Upload, Sparkles, Layers, Sliders,
  ArrowLeftRight, RefreshCw, TrendingUp, TrendingDown,
  CheckCircle, AlertTriangle, Info
} from 'lucide-react';

/* ─── Category Config ─────────────────────────────────────────── */
const CAT_CFG = [
  {
    key: 'appeared',
    label: 'Appeared',
    icon: TrendingUp,
    color: '#22c55e',
    bg: 'rgba(34,197,94,0.1)',
    border: 'rgba(34,197,94,0.35)',
    emoji: '🆕',
    desc: 'Objects / structures present in T2 (After) but absent in T1 (Before)'
  },
  {
    key: 'disappeared',
    label: 'Disappeared',
    icon: TrendingDown,
    color: '#ef4444',
    bg: 'rgba(239,68,68,0.1)',
    border: 'rgba(239,68,68,0.35)',
    emoji: '❌',
    desc: 'Objects / structures present in T1 (Before) but missing in T2 (After)'
  },
  {
    key: 'changed',
    label: 'Changed',
    icon: RefreshCw,
    color: '#f59e0b',
    bg: 'rgba(245,158,11,0.1)',
    border: 'rgba(245,158,11,0.35)',
    emoji: '🔄',
    desc: 'Existing objects present in both but with visible modifications'
  },
  {
    key: 'unchanged',
    label: 'Unchanged',
    icon: CheckCircle,
    color: '#6b7280',
    bg: 'rgba(107,114,128,0.1)',
    border: 'rgba(107,114,128,0.3)',
    emoji: '✅',
    desc: 'Objects or geographical features visually consistent in both'
  }
];

/* ─── Component ───────────────────────────────────────────────── */
export default function BiTemporalViewer({ onAnalysisComplete }) {
  const [uploadMode, setUploadMode] = useState('single');

  // Single-image state
  const [singleFile, setSingleFile] = useState(null);
  const [previewSingle, setPreviewSingle] = useState(null);

  // Dual-image state
  const [fileT1, setFileT1] = useState(null);
  const [fileT2, setFileT2] = useState(null);
  const [previewT1, setPreviewT1] = useState(null);
  const [previewT2, setPreviewT2] = useState(null);

  // Slider state
  const [sliderPos, setSliderPos] = useState(50);
  const [isDraggingSlider, setIsDraggingSlider] = useState(false);
  const sliderContainerRef = useRef(null);

  // Result state
  const [analyzing, setAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [showBoxes, setShowBoxes] = useState(true);
  const [activeCategory, setActiveCategory] = useState(null);

  // Track the actual rendered pixel dimensions of the base image
  // so SVG boxes can be placed over the image content only (no letterbox bleed)
  const [imgRenderedRect, setImgRenderedRect] = useState(null); // {left,top,width,height} in px
  const baseImgRef = useRef(null);        // ref on the T2 base <img>
  const viewportWrapRef = useRef(null);   // ref on the viewport container

  const inputSingleRef = useRef(null);
  const inputT1Ref = useRef(null);
  const inputT2Ref = useRef(null);

  /* ── Upload handlers ─────────────────────────────────────────── */
  const handleSelectSingle = async (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSingleFile(file);
      setPreviewSingle(URL.createObjectURL(file));
      setAnalysisResult(null);
      await executeSingleImageAnalysis(file);
    }
  };

  const executeSingleImageAnalysis = async (file) => {
    setAnalyzing(true);
    try {
      const fd = new FormData();
      fd.append('file', file);
      const res = await fetch('/api/bitemporal/upload-single', { method: 'POST', body: fd });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Failed to analyze image');
      }
      const data = await res.json();
      setAnalysisResult(data);
      setPreviewT1(data.t1_url);
      setPreviewT2(data.t2_url);
      if (onAnalysisComplete) onAnalysisComplete(data);
    } catch (err) {
      alert('Bi-Temporal analysis error: ' + err.message);
    } finally {
      setAnalyzing(false);
    }
  };

  const handleSelectT1 = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFileT1(e.target.files[0]);
      setPreviewT1(URL.createObjectURL(e.target.files[0]));
      setAnalysisResult(null);
    }
  };

  const handleSelectT2 = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFileT2(e.target.files[0]);
      setPreviewT2(URL.createObjectURL(e.target.files[0]));
      setAnalysisResult(null);
    }
  };

  const handleRunDualAnalysis = async () => {
    if (!fileT1 || !fileT2) return;
    setAnalyzing(true);
    try {
      const fd = new FormData();
      fd.append('file_t1', fileT1);
      fd.append('file_t2', fileT2);
      const upRes = await fetch('/api/bitemporal/upload', { method: 'POST', body: fd });
      if (!upRes.ok) throw new Error('Upload failed');
      const upData = await upRes.json();
      const anaRes = await fetch('/api/bitemporal/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          t1_filename: upData.t1.filename,
          t2_filename: upData.t2.filename,
          include_ai_report: true
        })
      });
      if (!anaRes.ok) throw new Error('Analysis failed');
      const anaData = await anaRes.json();
      setAnalysisResult(anaData);
      if (onAnalysisComplete) onAnalysisComplete(anaData);
    } catch (err) {
      alert('Dual analysis error: ' + err.message);
    } finally {
      setAnalyzing(false);
    }
  };

  /* ── Slider drag ─────────────────────────────────────────────── */
  const updateSliderPos = useCallback((clientX) => {
    if (!sliderContainerRef.current) return;
    const rect = sliderContainerRef.current.getBoundingClientRect();
    if (rect.width <= 0) return;
    const x = Math.max(0, Math.min(clientX - rect.left, rect.width));
    const percent = Math.max(0, Math.min(100, (x / rect.width) * 100));
    setSliderPos(percent);
  }, []);

  const handleMouseMove = useCallback((e) => {
    if (!isDraggingSlider) return;
    updateSliderPos(e.clientX);
  }, [isDraggingSlider, updateSliderPos]);

  const handleTouchMove = useCallback((e) => {
    if (!isDraggingSlider || !e.touches || !e.touches[0]) return;
    updateSliderPos(e.touches[0].clientX);
  }, [isDraggingSlider, updateSliderPos]);

  const handleMouseUp = useCallback(() => {
    setIsDraggingSlider(false);
  }, []);

  useEffect(() => {
    if (isDraggingSlider) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
      window.addEventListener('touchmove', handleTouchMove, { passive: false });
      window.addEventListener('touchend', handleMouseUp);
    }
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
      window.removeEventListener('touchmove', handleTouchMove);
      window.removeEventListener('touchend', handleMouseUp);
    };
  }, [isDraggingSlider, handleMouseMove, handleTouchMove, handleMouseUp]);

  /* ── Track actual rendered image rect for SVG box alignment ─── */
  const updateImgRect = useCallback(() => {
    const img = baseImgRef.current;
    const wrap = viewportWrapRef.current;
    if (!img || !wrap) return;
    // naturalWidth/Height give the true image dims for aspect ratio calc
    const nw = img.naturalWidth || 1;
    const nh = img.naturalHeight || 1;
    const { width: cw, height: ch } = wrap.getBoundingClientRect();
    // object-fit: contain letterboxing calculation
    const imgAspect = nw / nh;
    const conAspect = cw / ch;
    let rendW, rendH;
    if (imgAspect > conAspect) {
      rendW = cw;
      rendH = cw / imgAspect;
    } else {
      rendH = ch;
      rendW = ch * imgAspect;
    }
    const left = (cw - rendW) / 2;
    const top  = (ch - rendH) / 2;
    setImgRenderedRect({ left, top, width: rendW, height: rendH });
  }, []);

  useEffect(() => {
    const img = baseImgRef.current;
    if (!img) return;
    img.addEventListener('load', updateImgRect);
    const ro = new ResizeObserver(updateImgRect);
    if (viewportWrapRef.current) ro.observe(viewportWrapRef.current);
    updateImgRect();
    return () => {
      img.removeEventListener('load', updateImgRect);
      ro.disconnect();
    };
  }, [previewT2, updateImgRect]);

  /* ── Reset ───────────────────────────────────────────────────── */
  const handleReset = () => {
    setAnalysisResult(null);
    setSingleFile(null);
    setPreviewSingle(null);
    setPreviewT1(null);
    setPreviewT2(null);
    setFileT1(null);
    setFileT2(null);
    setActiveCategory(null);
    setSliderPos(50);
  };

  /* ── Helpers ─────────────────────────────────────────────────── */
  const cats = analysisResult?.change_categories || {};
  const hasResult = !!analysisResult;
  const hasSlider = hasResult && previewT1 && previewT2;

  /* ═══════════════════════════════════════════════════════════════
     RENDER
  ═══════════════════════════════════════════════════════════════ */
  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      background: 'var(--bg-panel)',
      borderRadius: '16px',
      border: '1px solid var(--border-subtle)',
      overflow: 'hidden'
    }}>

      {/* ── Top Bar ───────────────────────────────────────────── */}
      <div style={{
        padding: '10px 16px',
        borderBottom: '1px solid var(--border-subtle)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        background: 'rgba(0,0,0,0.3)',
        flexShrink: 0
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{
            width: '30px', height: '30px', borderRadius: '8px',
            background: 'rgba(244,63,94,0.15)',
            border: '1px solid rgba(244,63,94,0.35)',
            display: 'flex', alignItems: 'center', justifyContent: 'center'
          }}>
            <ArrowLeftRight size={15} color="#f43f5e" />
          </div>
          <div>
            <h4 style={{ fontSize: '0.88rem', fontWeight: '600', color: '#fff', margin: 0 }}>
              Bi-Temporal Change Analysis
            </h4>
            <p style={{ fontSize: '0.68rem', color: 'var(--text-muted)', margin: 0 }}>
              Upload 1 side-by-side image → auto-split T1/T2 → heatmap + AI report
            </p>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {/* Mode switcher */}
          <div style={{
            display: 'flex', background: 'rgba(255,255,255,0.05)',
            borderRadius: '8px', padding: '2px',
            border: '1px solid var(--border-subtle)'
          }}>
            {[['single', '1 Image', '#06b6d4'], ['dual', '2 Images', '#f43f5e']].map(([m, label, col]) => (
              <button
                key={m}
                onClick={() => { setUploadMode(m); handleReset(); }}
                style={{
                  padding: '3px 10px', borderRadius: '6px', border: 'none',
                  fontSize: '0.72rem', cursor: 'pointer',
                  background: uploadMode === m ? `rgba(${m === 'single' ? '6,182,212' : '244,63,94'},0.22)` : 'transparent',
                  color: uploadMode === m ? col : 'var(--text-muted)',
                  fontWeight: uploadMode === m ? '600' : '400'
                }}
              >{label}</button>
            ))}
          </div>

          {/* Overlay toggles (only when result available) */}
          {hasResult && (
            <>
              <button
                onClick={() => setShowHeatmap(!showHeatmap)}
                title="Toggle heatmap overlay"
                style={{
                  display: 'flex', alignItems: 'center', gap: '4px',
                  padding: '4px 9px', borderRadius: '6px', fontSize: '0.72rem',
                  border: '1px solid var(--border-glow)',
                  background: showHeatmap ? 'rgba(239,68,68,0.22)' : 'rgba(15,23,42,0.6)',
                  color: showHeatmap ? '#ef4444' : 'var(--text-secondary)', cursor: 'pointer'
                }}
              >
                <Layers size={12} /> Heatmap
              </button>
              <button
                onClick={() => setShowBoxes(!showBoxes)}
                title="Toggle change boxes"
                style={{
                  display: 'flex', alignItems: 'center', gap: '4px',
                  padding: '4px 9px', borderRadius: '6px', fontSize: '0.72rem',
                  border: '1px solid var(--border-glow)',
                  background: showBoxes ? 'rgba(6,182,212,0.22)' : 'rgba(15,23,42,0.6)',
                  color: showBoxes ? '#06b6d4' : 'var(--text-secondary)', cursor: 'pointer'
                }}
              >
                <Sliders size={12} /> Boxes
              </button>
            </>
          )}

          {/* Dual: Analyze button */}
          {uploadMode === 'dual' && (
            <button
              onClick={handleRunDualAnalysis}
              disabled={!fileT1 || !fileT2 || analyzing}
              style={{
                display: 'flex', alignItems: 'center', gap: '5px',
                padding: '5px 12px', borderRadius: '8px', fontSize: '0.76rem', fontWeight: '600',
                border: 'none',
                background: (!fileT1 || !fileT2) ? 'rgba(255,255,255,0.1)' : 'linear-gradient(135deg,#f43f5e,#e11d48)',
                color: '#fff',
                cursor: (!fileT1 || !fileT2 || analyzing) ? 'not-allowed' : 'pointer',
                opacity: (!fileT1 || !fileT2) ? 0.5 : 1
              }}
            >
              <Sparkles size={13} /> {analyzing ? 'Computing…' : 'Analyze Pair'}
            </button>
          )}
        </div>
      </div>

      {/* ── Body: two-panel horizontal layout ─────────────────── */}
      <div style={{
        flex: 1, display: 'flex', flexDirection: 'row', overflow: 'hidden', position: 'relative'
      }}>

        {/* ════════════════ LEFT PANEL (image viewer) ═══════════ */}
        <div style={{
          flex: hasResult ? '0 0 55%' : '1',
          display: 'flex', flexDirection: 'column',
          borderRight: hasResult ? '1px solid var(--border-subtle)' : 'none',
          overflow: 'hidden', position: 'relative', background: '#060c18'
        }}>

          {/* ── UPLOAD STATE ─────────────────────────────────── */}
          {!hasResult && !analyzing && (
            <div style={{
              flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
              padding: '32px'
            }}>
              {uploadMode === 'single' ? (
                <>
                  <input type="file" ref={inputSingleRef} onChange={handleSelectSingle}
                    accept="image/*" style={{ display: 'none' }} />
                  <div
                    onClick={() => inputSingleRef.current?.click()}
                    style={{
                      width: '100%', maxWidth: '480px', height: '300px',
                      borderRadius: '16px', border: '2px dashed rgba(6,182,212,0.4)',
                      background: 'rgba(15,23,42,0.5)', display: 'flex',
                      flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
                      cursor: 'pointer', padding: '28px', textAlign: 'center',
                      transition: 'all 0.2s ease'
                    }}
                    onMouseEnter={e => e.currentTarget.style.borderColor = '#06b6d4'}
                    onMouseLeave={e => e.currentTarget.style.borderColor = 'rgba(6,182,212,0.4)'}
                  >
                    <div style={{
                      width: '56px', height: '56px', borderRadius: '50%',
                      background: 'rgba(6,182,212,0.12)', display: 'flex',
                      alignItems: 'center', justifyContent: 'center',
                      marginBottom: '14px', border: '1px solid var(--border-glow)'
                    }}>
                      <Upload size={26} color="#06b6d4" />
                    </div>
                    <h3 style={{ fontSize: '1.05rem', fontWeight: '600', color: '#fff', margin: '0 0 7px 0' }}>
                      Upload 1 Bi-Temporal Image
                    </h3>
                    <p style={{
                      fontSize: '0.8rem', color: 'var(--text-secondary)', margin: '0 0 14px 0',
                      lineHeight: '1.45', maxWidth: '340px'
                    }}>
                      Upload any satellite scene containing <strong style={{ color: '#06b6d4' }}>Before</strong> and <strong style={{ color: '#f43f5e' }}>After</strong> views side-by-side.
                      SatAssist auto-splits, co-registers, and detects changes.
                    </p>
                    <div style={{ display: 'flex', gap: '7px', flexWrap: 'wrap', justifyContent: 'center' }}>
                      {['Side-by-side', 'Auto-Split', 'SSIM Heatmap', 'AI Report'].map(t => (
                        <span key={t} className="mono-tag">{t}</span>
                      ))}
                    </div>
                  </div>
                </>
              ) : (
                /* Dual upload dropzones */
                <div style={{ display: 'flex', gap: '20px', width: '100%', maxWidth: '700px' }}>
                  <input type="file" ref={inputT1Ref} onChange={handleSelectT1}
                    accept="image/*" style={{ display: 'none' }} />
                  <input type="file" ref={inputT2Ref} onChange={handleSelectT2}
                    accept="image/*" style={{ display: 'none' }} />

                  {[
                    { ref: inputT1Ref, preview: previewT1, label: 'T1 · BEFORE', color: '#06b6d4', border: 'rgba(6,182,212,0.4)' },
                    { ref: inputT2Ref, preview: previewT2, label: 'T2 · AFTER', color: '#f43f5e', border: 'rgba(244,63,94,0.4)' }
                  ].map(({ ref, preview, label, color, border }) => (
                    <div
                      key={label}
                      onClick={() => ref.current?.click()}
                      style={{
                        flex: 1, height: '260px', borderRadius: '12px',
                        border: `2px dashed ${border}`, background: 'rgba(15,23,42,0.4)',
                        display: 'flex', flexDirection: 'column',
                        alignItems: 'center', justifyContent: 'center',
                        cursor: 'pointer', padding: '16px', textAlign: 'center'
                      }}
                    >
                      {preview ? (
                        <img src={preview} alt={label}
                          style={{ maxWidth: '100%', maxHeight: '200px', borderRadius: '8px', objectFit: 'contain' }} />
                      ) : (
                        <>
                          <Upload size={22} color={color} style={{ marginBottom: '10px' }} />
                          <span style={{ color, fontWeight: '700', fontSize: '0.78rem', letterSpacing: '0.08em' }}>{label}</span>
                          <span style={{ color: 'var(--text-muted)', fontSize: '0.72rem', marginTop: '5px' }}>
                            {label.includes('BEFORE') ? 'Baseline satellite scene' : 'Recent comparison scene'}
                          </span>
                        </>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* ── SPLIT SLIDER (after result) ───────────────────── */}
          {hasSlider && (
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
              {/* Label bar */}
              <div style={{
                display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                padding: '6px 14px', background: 'rgba(0,0,0,0.5)', flexShrink: 0
              }}>
                <span style={{
                  fontSize: '0.68rem', fontWeight: '700', letterSpacing: '0.08em',
                  color: '#06b6d4', fontFamily: 'var(--font-mono)'
                }}>◀ T1: BEFORE</span>

                {/* Slider scrub control */}
                <div style={{
                  fontSize: '0.65rem', color: 'var(--text-muted)',
                  display: 'flex', alignItems: 'center', gap: '8px'
                }}>
                  <ArrowLeftRight size={11} color="var(--accent-cyan)" />
                  <span style={{ userSelect: 'none' }}>Drag slider to compare:</span>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={Math.round(sliderPos)}
                    onChange={(e) => setSliderPos(Number(e.target.value))}
                    style={{
                      width: '100px',
                      height: '4px',
                      accentColor: '#06b6d4',
                      cursor: 'pointer'
                    }}
                    title="Slide to compare Before and After"
                  />
                  <span style={{
                    fontFamily: 'var(--font-mono)',
                    fontSize: '0.68rem',
                    color: '#06b6d4',
                    minWidth: '34px',
                    fontWeight: '600'
                  }}>
                    {Math.round(sliderPos)}%
                  </span>
                </div>

                <span style={{
                  fontSize: '0.68rem', fontWeight: '700', letterSpacing: '0.08em',
                  color: '#f43f5e', fontFamily: 'var(--font-mono)'
                }}>T2: AFTER ▶</span>
              </div>

              {/* Slider viewport */}
              <div
                ref={viewportWrapRef}
                style={{
                  flex: 1, position: 'relative', overflow: 'hidden',
                  userSelect: 'none', display: 'flex', alignItems: 'center', justifyContent: 'center',
                  background: '#060c18'
                }}
              >
                {/*
                  INNER WRAPPER — sized exactly to the rendered image dimensions.
                  All overlays (heatmap, boxes, slider) are placed inside this wrapper
                  so they never bleed into the letterbox area around the image.
                */}
                <div
                  ref={sliderContainerRef}
                  onMouseDown={(e) => {
                    e.preventDefault();
                    setIsDraggingSlider(true);
                    updateSliderPos(e.clientX);
                  }}
                  onTouchStart={(e) => {
                    if (e.touches && e.touches[0]) {
                      setIsDraggingSlider(true);
                      updateSliderPos(e.touches[0].clientX);
                    }
                  }}
                  style={{
                    position: 'relative',
                    width:  imgRenderedRect ? `${imgRenderedRect.width}px`  : '100%',
                    height: imgRenderedRect ? `${imgRenderedRect.height}px` : '100%',
                    flexShrink: 0,
                    overflow: 'hidden',
                    cursor: isDraggingSlider ? 'ew-resize' : 'col-resize',
                    touchAction: 'none'
                  }}
                >
                  {/* T2 (After) base image — fills the inner wrapper exactly */}
                  <img
                    ref={baseImgRef}
                    src={previewT2} alt="T2 After"
                    onLoad={updateImgRect}
                    draggable={false}
                    style={{
                      display: 'block', width: '100%', height: '100%',
                      objectFit: 'fill', pointerEvents: 'none', userSelect: 'none'
                    }}
                  />

                  {/* T1 (Before) clipped by slider */}
                  <div style={{
                    position: 'absolute', inset: 0,
                    clipPath: `inset(0 ${100 - sliderPos}% 0 0)`,
                    pointerEvents: 'none'
                  }}>
                    <img
                      src={previewT1} alt="T1 Before"
                      draggable={false}
                      style={{
                        display: 'block', width: '100%', height: '100%',
                        objectFit: 'fill', pointerEvents: 'none', userSelect: 'none'
                      }}
                    />
                  </div>

                  {/* Heatmap overlay — also sized to inner wrapper */}
                  {showHeatmap && analysisResult?.heatmap_url && (
                    <img
                      src={analysisResult.heatmap_url} alt="Heatmap"
                      draggable={false}
                      style={{
                        position: 'absolute', inset: 0, width: '100%', height: '100%',
                        objectFit: 'fill', pointerEvents: 'none', mixBlendMode: 'screen', userSelect: 'none'
                      }}
                    />
                  )}

                  {/* Change boxes — SVG viewBox 0 0 1 1, inside inner wrapper only */}
                  {showBoxes && analysisResult?.change_boxes?.length > 0 && (
                    <svg
                      style={{
                        position: 'absolute', inset: 0, width: '100%', height: '100%',
                        pointerEvents: 'none', overflow: 'visible'
                      }}
                      viewBox="0 0 1 1"
                      preserveAspectRatio="none"
                    >
                      {analysisResult.change_boxes.map((box, idx) => {
                        const bType = box.type || box.change_type || 'modified';
                        const isMatch = !activeCategory || activeCategory === bType;
                        const strokeColor = bType === 'appeared' ? '#22c55e' : bType === 'disappeared' ? '#ef4444' : '#f59e0b';
                        const fillColor = bType === 'appeared' ? 'rgba(34,197,94,0.16)' : bType === 'disappeared' ? 'rgba(239,68,68,0.16)' : 'rgba(245,158,11,0.16)';
                        const bw = Math.max(0, box.xmax - box.xmin);
                        const bh = Math.max(0, box.ymax - box.ymin);
                        const opacity = isMatch ? 1 : 0.18;
                        const isSelectedCat = activeCategory === bType;
                        const strokeWidth = isSelectedCat ? '0.0055' : '0.0035';
                        const labelText = box.label || (bType === 'appeared' ? 'New Construction' : bType === 'disappeared' ? 'Demolished / Cleared' : 'Surface Shift');
                        const confText = box.confidence ? `${Math.round(box.confidence * 100)}%` : '';

                        return (
                          <g
                            key={idx}
                            style={{
                              opacity,
                              transition: 'opacity 0.2s ease',
                              pointerEvents: 'auto',
                              cursor: 'pointer'
                            }}
                          >
                            <title>{`${bType.toUpperCase()}: ${labelText} ${confText ? `(${confText})` : ''} - [${Math.round(box.xmin * 100)}%, ${Math.round(box.ymin * 100)}%]`}</title>
                            <rect
                              x={box.xmin} y={box.ymin}
                              width={bw}
                              height={bh}
                              fill={fillColor}
                              stroke={strokeColor}
                              strokeWidth={strokeWidth}
                              strokeDasharray={isSelectedCat ? 'none' : '0.012,0.006'}
                              rx="0.004"
                            />
                          </g>
                        );
                      })}
                    </svg>
                  )}

                  {/* Slider divider line and draggable handle */}
                  <div
                    style={{
                      position: 'absolute', top: 0, bottom: 0,
                      left: `${sliderPos}%`,
                      width: '36px',
                      transform: 'translateX(-50%)',
                      cursor: 'ew-resize',
                      zIndex: 25,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      touchAction: 'none'
                    }}
                    onMouseDown={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      setIsDraggingSlider(true);
                    }}
                    onTouchStart={(e) => {
                      e.stopPropagation();
                      setIsDraggingSlider(true);
                    }}
                  >
                    {/* Vertical line indicator */}
                    <div style={{
                      position: 'absolute', top: 0, bottom: 0,
                      width: '3px',
                      background: '#fff',
                      boxShadow: '0 0 10px rgba(0,0,0,0.9), 0 0 6px #06b6d4',
                      pointerEvents: 'none'
                    }} />

                    {/* Circular handle badge */}
                    <div style={{
                      position: 'relative',
                      width: '32px', height: '32px', borderRadius: '50%',
                      background: '#fff',
                      border: '2px solid #06b6d4',
                      boxShadow: '0 2px 10px rgba(0,0,0,0.7), 0 0 8px rgba(6,182,212,0.4)',
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      color: '#0891b2',
                      pointerEvents: 'none',
                      transition: 'transform 0.1s ease',
                      transform: isDraggingSlider ? 'scale(1.15)' : 'scale(1)'
                    }}>
                      <ArrowLeftRight size={14} />
                    </div>
                  </div>
                </div>
              </div>

              {/* Stats strip at bottom of image panel */}
              <div style={{
                display: 'flex', gap: '16px', padding: '7px 14px',
                background: 'rgba(0,0,0,0.6)', borderTop: '1px solid var(--border-subtle)',
                flexShrink: 0, flexWrap: 'wrap', alignItems: 'center'
              }}>
                <StatChip label="Change Extent" value={`${analysisResult.change_percentage}%`} color="#ef4444" />
                <StatChip label="SSIM Score" value={analysisResult.structural_similarity} color="#06b6d4" />
                <StatChip label="Change Zones" value={`${analysisResult.change_boxes?.length || 0} clusters`} color="#f59e0b" />
                {analysisResult.summary_counts && (
                  <>
                    <StatChip label="Appeared" value={`${analysisResult.summary_counts.appeared || 0}`} color="#22c55e" />
                    <StatChip label="Disappeared" value={`${analysisResult.summary_counts.disappeared || 0}`} color="#ef4444" />
                    <StatChip label="Modified" value={`${analysisResult.summary_counts.modified || 0}`} color="#f59e0b" />
                  </>
                )}
                <button
                  onClick={handleReset}
                  style={{
                    marginLeft: 'auto', background: 'transparent', border: 'none',
                    color: 'var(--accent-cyan)', fontSize: '0.73rem',
                    cursor: 'pointer', textDecoration: 'underline',
                    display: 'flex', alignItems: 'center', gap: '4px'
                  }}
                >
                  <Upload size={11} /> Upload Another
                </button>
              </div>
            </div>
          )}

          {/* ── LOADING OVERLAY ───────────────────────────────── */}
          {analyzing && (
            <div style={{
              position: 'absolute', inset: 0,
              background: 'rgba(6,9,19,0.88)', backdropFilter: 'blur(10px)',
              display: 'flex', flexDirection: 'column',
              alignItems: 'center', justifyContent: 'center', zIndex: 30
            }}>
              <div style={{
                width: '44px', height: '44px',
                border: '3px solid rgba(6,182,212,0.2)',
                borderTopColor: '#06b6d4', borderRadius: '50%',
                animation: 'spin 0.8s linear infinite', marginBottom: '14px'
              }} />
              <p style={{ color: '#fff', fontWeight: '600', fontSize: '0.9rem', margin: 0 }}>
                Analyzing Temporal Changes…
              </p>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.75rem', marginTop: '4px' }}>
                Splitting T1/T2 · SSIM heatmap · AI Intelligence Brief
              </p>
            </div>
          )}
        </div>

        {/* ════════════════ RIGHT PANEL (analysis) ══════════════ */}
        {hasResult && (
          <div style={{
            flex: '0 0 45%', display: 'flex', flexDirection: 'column',
            overflow: 'hidden', background: 'var(--bg-panel)'
          }}>
            {/* Panel header */}
            <div style={{
              padding: '10px 16px', borderBottom: '1px solid var(--border-subtle)',
              background: 'rgba(0,0,0,0.25)', display: 'flex', alignItems: 'center', gap: '8px',
              flexShrink: 0
            }}>
              <Sparkles size={15} color="#f59e0b" />
              <span style={{ fontSize: '0.84rem', fontWeight: '600', color: '#fff' }}>
                Change Intelligence Report
              </span>
              <span style={{
                marginLeft: 'auto', fontSize: '0.68rem', color: 'var(--text-muted)',
                fontFamily: 'var(--font-mono)'
              }}>AI · Qwen2-VL</span>
            </div>

            {/* Scrollable content */}
            <div style={{ flex: 1, overflowY: 'auto', padding: '14px' }}>

              {/* ── Original uploaded image preview (single mode) ── */}
              {previewSingle && (
                <div style={{ marginBottom: '14px' }}>
                  <div style={{
                    fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: '6px',
                    display: 'flex', alignItems: 'center', gap: '5px'
                  }}>
                    <Info size={11} /> Uploaded composite image (auto-split at centre)
                  </div>
                  <img
                    src={previewSingle} alt="Uploaded composite"
                    style={{
                      width: '100%', borderRadius: '8px', objectFit: 'contain',
                      border: '1px solid var(--border-subtle)', maxHeight: '160px'
                    }}
                  />
                </div>
              )}

              {/* ── AI Summary ─────────────────────────────────── */}
              {analysisResult.ai_report && (
                <div style={{
                  marginBottom: '14px', padding: '10px 13px',
                  borderRadius: '10px', background: 'rgba(244,63,94,0.08)',
                  border: '1px solid rgba(244,63,94,0.3)',
                  borderLeft: '3px solid #f43f5e'
                }}>
                  <div style={{
                    fontSize: '0.68rem', fontWeight: '700', color: '#f43f5e',
                    letterSpacing: '0.07em', marginBottom: '5px'
                  }}>AI INTELLIGENCE BRIEF</div>
                  <p style={{ margin: 0, fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: '1.5' }}>
                    {analysisResult.ai_report}
                  </p>
                </div>
              )}

              {/* ── 4-Category Grid ────────────────────────────── */}
              <div style={{
                display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px',
                marginBottom: '14px'
              }}>
                {CAT_CFG.map(cat => {
                  const rawItems = cats[cat.key] || [];
                  const isNone = rawItems.length === 0 || (rawItems.length === 1 && rawItems[0].toLowerCase().includes('no significant changes detected'));
                  const displayItems = isNone ? [] : rawItems;
                  const displayCount = displayItems.length;
                  const Icon = cat.icon;
                  const isActive = activeCategory === cat.key;
                  return (
                    <div
                      key={cat.key}
                      onClick={() => setActiveCategory(isActive ? null : cat.key)}
                      style={{
                        borderRadius: '10px', padding: '10px 12px',
                        background: isActive ? cat.bg : 'rgba(255,255,255,0.03)',
                        border: `1px solid ${isActive ? cat.border : 'var(--border-subtle)'}`,
                        cursor: 'pointer', transition: 'all 0.18s ease'
                      }}
                    >
                      <div style={{
                        display: 'flex', alignItems: 'center', gap: '7px', marginBottom: '3px'
                      }}>
                        <div style={{
                          width: '24px', height: '24px', borderRadius: '6px',
                          background: cat.bg, border: `1px solid ${cat.border}`,
                          display: 'flex', alignItems: 'center', justifyContent: 'center'
                        }}>
                          <Icon size={13} color={cat.color} />
                        </div>
                        <span style={{ fontSize: '0.78rem', fontWeight: '600', color: '#fff' }}>
                          {cat.emoji} {cat.label}
                        </span>
                        <span style={{
                          marginLeft: 'auto', fontSize: '0.7rem', fontWeight: '700',
                          color: displayCount > 0 ? cat.color : 'var(--text-muted)',
                          background: displayCount > 0 ? cat.bg : 'rgba(255,255,255,0.05)',
                          padding: '1px 7px',
                          borderRadius: '10px',
                          border: `1px solid ${displayCount > 0 ? cat.border : 'var(--border-subtle)'}`
                        }}>
                          {displayCount}
                        </span>
                      </div>
                      <div style={{ fontSize: '0.64rem', color: 'var(--text-muted)', marginBottom: '8px', lineHeight: '1.25' }}>
                        {cat.desc}
                      </div>
                      {isNone ? (
                        <p style={{
                          margin: 0, fontSize: '0.72rem', color: 'var(--text-muted)',
                          fontStyle: 'italic', padding: '2px 0'
                        }}>No significant changes detected.</p>
                      ) : (
                        <ul style={{ margin: 0, padding: '0 0 0 14px' }}>
                          {displayItems.slice(0, isActive ? undefined : 3).map((item, i) => (
                            <li key={i} style={{
                              fontSize: '0.74rem', color: 'var(--text-secondary)',
                              marginBottom: '3px', lineHeight: '1.35'
                            }}>
                              {item}
                            </li>
                          ))}
                          {!isActive && displayItems.length > 3 && (
                            <li style={{ fontSize: '0.7rem', color: cat.color, listStyle: 'none', marginTop: '3px' }}>
                              + {displayItems.length - 3} more (click to expand)
                            </li>
                          )}
                        </ul>
                      )}
                    </div>
                  );
                })}
              </div>

              {/* ── Legend ─────────────────────────────────────── */}
              <div style={{
                padding: '10px 12px', borderRadius: '10px',
                background: 'rgba(255,255,255,0.03)',
                border: '1px solid var(--border-subtle)'
              }}>
                <div style={{
                  fontSize: '0.68rem', fontWeight: '700', color: 'var(--text-muted)',
                  letterSpacing: '0.07em', marginBottom: '8px'
                }}>VISUAL OVERLAY LEGEND</div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '5px' }}>
                  <LegendRow color="#22c55e" label="Green dashed boxes" desc="Appeared (New construction / road expansion)" />
                  <LegendRow color="#ef4444" label="Red dashed boxes" desc="Disappeared (Demolished / cleared terrain)" />
                  <LegendRow color="#f59e0b" label="Amber dashed boxes" desc="Modified (Land-use / surface shift)" />
                  <LegendRow color="rgba(239,68,68,0.5)" label="Heatmap overlay" desc="Pixel-level structural change intensity" />
                  <LegendRow color="#06b6d4" label="Cyan — T1 (Before)" desc="Baseline comparison scene" />
                  <LegendRow color="#f43f5e" label="Red — T2 (After)" desc="Recent comparison scene" />
                </div>
              </div>

              {/* ── Change boxes count chip ─────────────────────── */}
              {analysisResult.change_boxes?.length > 0 && (
                <div style={{
                  marginTop: '12px', padding: '8px 12px', borderRadius: '8px',
                  background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.25)',
                  display: 'flex', alignItems: 'center', gap: '8px'
                }}>
                  <AlertTriangle size={14} color="#ef4444" />
                  <span style={{ fontSize: '0.76rem', color: 'var(--text-secondary)' }}>
                    <strong style={{ color: '#ef4444' }}>{analysisResult.change_boxes.length} change clusters</strong> detected
                    — toggle <em>Boxes</em> in the top bar to show/hide on viewer.
                  </span>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

/* ─── Small reusable components ──────────────────────────────── */
function StatChip({ label, value, color }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column' }}>
      <span style={{ fontSize: '0.6rem', color: 'var(--text-muted)', letterSpacing: '0.07em' }}>{label}</span>
      <span style={{ fontSize: '0.78rem', fontWeight: '700', color }}>{value}</span>
    </div>
  );
}

function LegendRow({ color, label, desc }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
      <div style={{
        width: '28px', height: '10px', borderRadius: '3px',
        background: color, flexShrink: 0,
        border: color.startsWith('rgba') ? `1.5px dashed ${color.replace(/[\d.]+\)$/, '1)')}` : 'none'
      }} />
      <span style={{ fontSize: '0.72rem', color: 'var(--text-secondary)' }}>
        <strong style={{ color: '#fff' }}>{label}</strong> — {desc}
      </span>
    </div>
  );
}
