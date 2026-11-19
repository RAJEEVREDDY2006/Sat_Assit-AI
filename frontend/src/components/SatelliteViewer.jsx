import React, { useState, useRef } from 'react';
import { Upload, ZoomIn, ZoomOut, Maximize2, Eye, EyeOff, Crosshair, Image as ImageIcon, Sparkles, Download } from 'lucide-react';

export default function SatelliteViewer({
  session,
  onUpload,
  uploading,
  boundingBoxes,
  highlightedBoxIndex,
  onHoverBox
}) {
  const [zoom, setZoom] = useState(1);
  const [showBoxes, setShowBoxes] = useState(true);
  const [isDragging, setIsDragging] = useState(false);
  const [exporting, setExporting] = useState(false);
  const fileInputRef = useRef(null);

  const handleExportAnnotated = async () => {
    if (!session || !boundingBoxes || boundingBoxes.length === 0) return;
    setExporting(true);
    try {
      const res = await fetch('/api/annotate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: session.session_id,
          bounding_boxes: boundingBoxes,
          label: 'Grounding Detections'
        })
      });
      if (!res.ok) throw new Error('Annotation request failed');
      const data = await res.json();
      // Trigger download
      const link = document.createElement('a');
      link.href = data.download_url;
      link.download = `${session.session_id}_annotated.png`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } catch (err) {
      alert('Could not export annotated image: ' + err.message);
    } finally {
      setExporting(false);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onUpload(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      onUpload(e.target.files[0]);
    }
  };

  return (
    <div className="glass-panel" style={{
      flex: '1.2',
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      position: 'relative',
      overflow: 'hidden'
    }}>
      {/* Viewer Header / Toolbar */}
      <div style={{
        padding: '12px 18px',
        borderBottom: '1px solid var(--border-subtle)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        background: 'rgba(0, 0, 0, 0.2)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Crosshair size={16} color="var(--accent-cyan)" />
          <span style={{ fontSize: '0.85rem', fontWeight: '600', letterSpacing: '0.04em', textTransform: 'uppercase', color: 'var(--text-secondary)' }}>
            Satellite Viewport
          </span>
          {session && (
            <span className="mono-tag" style={{ marginLeft: '4px' }}>
              {session.filename || 'Active Scene'}
            </span>
          )}
        </div>

        {session && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            {/* Toggle Bounding Boxes Overlay */}
            {boundingBoxes && boundingBoxes.length > 0 && (
              <>
                <button
                  onClick={() => setShowBoxes(!showBoxes)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '5px',
                    padding: '5px 10px',
                    borderRadius: '6px',
                    background: showBoxes ? 'rgba(6, 182, 212, 0.2)' : 'rgba(255, 255, 255, 0.06)',
                    border: showBoxes ? '1px solid var(--accent-cyan)' : '1px solid var(--border-subtle)',
                    color: showBoxes ? 'var(--accent-cyan)' : 'var(--text-muted)',
                    fontSize: '0.75rem',
                    fontWeight: '500'
                  }}
                >
                  {showBoxes ? <Eye size={13} /> : <EyeOff size={13} />}
                  <span>Grounding ({boundingBoxes.length})</span>
                </button>

                <button
                  onClick={handleExportAnnotated}
                  disabled={exporting}
                  title="Export and download annotated satellite image with bounded boxes"
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '5px',
                    padding: '5px 10px',
                    borderRadius: '6px',
                    background: exporting ? 'rgba(255,255,255,0.08)' : 'rgba(99, 102, 241, 0.25)',
                    border: '1px solid #6366f1',
                    color: '#a5b4fc',
                    fontSize: '0.75rem',
                    fontWeight: '600',
                    cursor: exporting ? 'wait' : 'pointer'
                  }}
                >
                  <Download size={13} />
                  <span>{exporting ? 'Exporting…' : 'Export PNG'}</span>
                </button>
              </>
            )}

            {/* Zoom Controls */}
            <div style={{ display: 'flex', background: 'rgba(255, 255, 255, 0.05)', borderRadius: '6px', padding: '2px', border: '1px solid var(--border-subtle)' }}>
              <button
                onClick={() => setZoom(Math.max(0.6, zoom - 0.2))}
                style={{ background: 'transparent', border: 'none', color: '#fff', padding: '4px 8px', borderRadius: '4px' }}
                title="Zoom Out"
              >
                <ZoomOut size={14} />
              </button>
              <button
                onClick={() => setZoom(1)}
                style={{ background: 'transparent', border: 'none', color: 'var(--text-secondary)', padding: '4px 6px', fontSize: '0.72rem', fontFamily: 'var(--font-mono)' }}
                title="Reset Zoom"
              >
                {Math.round(zoom * 100)}%
              </button>
              <button
                onClick={() => setZoom(Math.min(2.5, zoom + 0.2))}
                style={{ background: 'transparent', border: 'none', color: '#fff', padding: '4px 8px', borderRadius: '4px' }}
                title="Zoom In"
              >
                <ZoomIn size={14} />
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Main View Area */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        style={{
          flex: 1,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          position: 'relative',
          overflow: 'auto',
          padding: '24px',
          background: isDragging ? 'rgba(6, 182, 212, 0.06)' : 'radial-gradient(circle, rgba(15, 23, 42, 0.4) 0%, rgba(6, 9, 19, 0.9) 100%)'
        }}
      >
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileChange}
          accept=".jpg,.jpeg,.png,.webp,.tif,.tiff"
          style={{ display: 'none' }}
        />

        {session ? (
          <div style={{
            position: 'relative',
            transform: `scale(${zoom})`,
            transformOrigin: 'center center',
            transition: 'transform 0.15s ease-out',
            boxShadow: '0 20px 50px rgba(0,0,0,0.6)',
            borderRadius: '12px',
            overflow: 'hidden',
            border: '1px solid rgba(255, 255, 255, 0.15)'
          }}>
            <img
              src={session.image_url}
              alt="Satellite Target Scene"
              style={{
                display: 'block',
                maxWidth: '100%',
                maxHeight: '62vh',
                objectFit: 'contain',
                borderRadius: '12px'
              }}
            />

            {/* Dynamic Visual Grounding SVG Overlay */}
            {showBoxes && boundingBoxes && boundingBoxes.length > 0 && (() => {
              const getObjectColor = (label = '') => {
                const l = label.toLowerCase();
                if (l.includes('river') || l.includes('water') || l.includes('lake') || l.includes('canal') || l.includes('ocean') || l.includes('sea')) {
                  return { stroke: '#06b6d4', fill: 'rgba(6, 182, 212, 0.22)', chip: '#06b6d4' };
                }
                if (l.includes('bridge')) {
                  return { stroke: '#f43f5e', fill: 'rgba(244, 63, 94, 0.22)', chip: '#f43f5e' };
                }
                if (l.includes('ship') || l.includes('vessel') || l.includes('boat')) {
                  return { stroke: '#10b981', fill: 'rgba(16, 185, 129, 0.20)', chip: '#10b981' };
                }
                if (l.includes('aircraft') || l.includes('airplane') || l.includes('plane') || l.includes('runway')) {
                  return { stroke: '#fbbf24', fill: 'rgba(251, 191, 36, 0.20)', chip: '#fbbf24' };
                }
                if (l.includes('vehicle') || l.includes('car') || l.includes('truck') || l.includes('traffic')) {
                  return { stroke: '#38bdf8', fill: 'rgba(56, 189, 248, 0.18)', chip: '#38bdf8' };
                }
                if (l.includes('building') || l.includes('structure')) {
                  return { stroke: '#818cf8', fill: 'rgba(129, 140, 248, 0.18)', chip: '#818cf8' };
                }
                if (l.includes('road')) {
                  return { stroke: '#f87171', fill: 'rgba(248, 113, 113, 0.18)', chip: '#f87171' };
                }
                if (l.includes('tank') || l.includes('storage')) {
                  return { stroke: '#a78bfa', fill: 'rgba(167, 139, 250, 0.18)', chip: '#a78bfa' };
                }
                return { stroke: '#06b6d4', fill: 'rgba(6, 182, 212, 0.18)', chip: '#06b6d4' };
              };

              const isDense = boundingBoxes.length > 8;

              return (
                <svg
                  style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    width: '100%',
                    height: '100%',
                    pointerEvents: 'none'
                  }}
                  viewBox="0 0 100 100"
                  preserveAspectRatio="none"
                >
                  {boundingBoxes.map((box, idx) => {
                    const x = box.xmin * 100;
                    const y = box.ymin * 100;
                    const width = Math.max(0.6, (box.xmax - box.xmin) * 100);
                    const height = Math.max(0.6, (box.ymax - box.ymin) * 100);
                    const isHighlighted = highlightedBoxIndex === idx;
                    const colors = getObjectColor(box.label);

                    const boxFill = isHighlighted ? 'rgba(244, 63, 94, 0.35)' : colors.fill;
                    const boxStroke = isHighlighted ? '#f43f5e' : colors.stroke;
                    const chipColor = isHighlighted ? '#f43f5e' : colors.chip;

                    // Suppress overlapping chip text on dense scenes; show on hover, or for prominent boxes
                    const showChip = !isDense || isHighlighted || (width >= 3.5 && (idx % 3 === 0 || idx < 3));

                    return (
                      <g
                        key={idx}
                        style={{ pointerEvents: 'auto', cursor: 'pointer' }}
                        onMouseEnter={() => onHoverBox && onHoverBox(idx)}
                        onMouseLeave={() => onHoverBox && onHoverBox(null)}
                      >
                        {/* Bounding Box Rectangle */}
                        <rect
                          x={`${x}%`}
                          y={`${y}%`}
                          width={`${width}%`}
                          height={`${height}%`}
                          fill={boxFill}
                          stroke={boxStroke}
                          strokeWidth={isHighlighted ? '0.8%' : '0.45%'}
                          strokeDasharray={isHighlighted ? 'none' : '2, 1'}
                          style={{
                            transition: 'all 0.15s ease',
                            filter: isHighlighted
                              ? 'drop-shadow(0 0 6px #f43f5e)'
                              : `drop-shadow(0 0 3px ${boxStroke})`
                          }}
                        />

                        {/* Label Chip */}
                        {showChip && (
                          <>
                            <rect
                              x={`${x}%`}
                              y={`${Math.max(0, y - 3.2)}%`}
                              width={`${Math.min(24, Math.max(6, width + 2))}%`}
                              height="3.2%"
                              fill={chipColor}
                              rx="0.5"
                            />
                            <text
                              x={`${x + 0.5}%`}
                              y={`${Math.max(2.4, y - 0.8)}%`}
                              fill="#000000"
                              fontSize="2.4"
                              fontWeight="700"
                              fontFamily="sans-serif"
                            >
                              {box.label || `Object #${idx + 1}`}
                            </text>
                          </>
                        )}
                      </g>
                    );
                  })}
                </svg>
              );
            })()}

            {/* Corner HUD Reticles */}
            <div style={{ position: 'absolute', top: 8, left: 8, width: 12, height: 12, borderTop: '2px solid var(--accent-cyan)', borderLeft: '2px solid var(--accent-cyan)' }} />
            <div style={{ position: 'absolute', top: 8, right: 8, width: 12, height: 12, borderTop: '2px solid var(--accent-cyan)', borderRight: '2px solid var(--accent-cyan)' }} />
            <div style={{ position: 'absolute', bottom: 8, left: 8, width: 12, height: 12, borderBottom: '2px solid var(--accent-cyan)', borderLeft: '2px solid var(--accent-cyan)' }} />
            <div style={{ position: 'absolute', bottom: 8, right: 8, width: 12, height: 12, borderBottom: '2px solid var(--accent-cyan)', borderRight: '2px solid var(--accent-cyan)' }} />
          </div>
        ) : (
          /* Empty / Upload Dropzone State */
          <div
            onClick={() => fileInputRef.current?.click()}
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              padding: '48px 32px',
              borderRadius: '16px',
              border: '2px dashed rgba(6, 182, 212, 0.4)',
              background: 'rgba(15, 23, 42, 0.5)',
              cursor: 'pointer',
              textAlign: 'center',
              maxWidth: '440px',
              transition: 'all 0.3s ease'
            }}
          >
            <div style={{
              width: '64px',
              height: '64px',
              borderRadius: '50%',
              background: 'rgba(6, 182, 212, 0.12)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              marginBottom: '18px',
              border: '1px solid var(--border-glow)'
            }}>
              <Upload size={30} color="var(--accent-cyan)" />
            </div>

            <h3 style={{ fontSize: '1.15rem', fontWeight: '600', marginBottom: '8px', color: '#fff' }}>
              Upload Satellite Image
            </h3>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '18px', lineHeight: '1.4' }}>
              Drag & drop any remote-sensing scene here, or browse from local disk.
            </p>

            <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', justifyContent: 'center' }}>
              <span className="mono-tag">JPG</span>
              <span className="mono-tag">PNG</span>
              <span className="mono-tag">TIFF / GeoTIFF</span>
              <span className="mono-tag">512×512 to 2k</span>
            </div>
          </div>
        )}

        {/* Uploading Overlay */}
        {uploading && (
          <div style={{
            position: 'absolute',
            inset: 0,
            background: 'rgba(6, 9, 19, 0.85)',
            backdropFilter: 'blur(8px)',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 30
          }}>
            <div style={{
              width: '48px',
              height: '48px',
              border: '3px solid rgba(6, 182, 212, 0.2)',
              borderTopColor: 'var(--accent-cyan)',
              borderRadius: '50%',
              animation: 'spin 0.8s linear infinite',
              marginBottom: '16px'
            }} />
            <p style={{ color: '#fff', fontWeight: '500', fontSize: '0.95rem' }}>Processing Satellite Image...</p>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem', marginTop: '4px' }}>Running contrast stretching & RGB normalization</p>
          </div>
        )}
      </div>

      {/* Viewport Footer Bar */}
      {session && session.metadata && (
        <div style={{
          padding: '8px 18px',
          borderTop: '1px solid var(--border-subtle)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          fontSize: '0.78rem',
          color: 'var(--text-muted)',
          background: 'rgba(0, 0, 0, 0.25)',
          fontFamily: 'var(--font-mono)'
        }}>
          <div>Dimensions: <span style={{ color: 'var(--accent-cyan)' }}>{session.metadata.width}×{session.metadata.height}</span> ({session.metadata.mode})</div>
          <div>Domain: <span style={{ color: '#fff' }}>Aerial / Satellite</span></div>
          <div>Session: <span style={{ color: 'var(--text-secondary)' }}>{session.session_id}</span></div>
        </div>
      )}
    </div>
  );
}
