import React from 'react';
import { Satellite, Database, RefreshCw, Layers, ArrowLeftRight, Image as ImageIcon } from 'lucide-react';

export default function Header({
  health,
  activeSession,
  onResetSession,
  onToggleSamples,
  showSamples,
  activeMode = 'single',
  onSwitchMode
}) {
  return (
    <header style={{
      height: '68px',
      padding: '0 24px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      borderBottom: '1px solid var(--border-subtle)',
      background: 'rgba(9, 14, 26, 0.85)',
      backdropFilter: 'blur(20px)',
      position: 'sticky',
      top: 0,
      zIndex: 50
    }}>
      {/* Brand & Logo */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
        <div style={{
          width: '42px',
          height: '42px',
          borderRadius: '12px',
          background: 'linear-gradient(135deg, #06b6d4 0%, #3b82f6 50%, #6366f1 100%)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          boxShadow: '0 0 20px rgba(6, 182, 212, 0.4)'
        }}>
          <Satellite size={24} color="#ffffff" />
        </div>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <h1 style={{ fontSize: '1.25rem', fontWeight: '700', letterSpacing: '-0.02em', color: '#fff' }}>
              SatAssist <span style={{ color: 'var(--accent-cyan)' }}>AI</span>
            </h1>
            <span className="mono-tag" style={{ fontSize: '0.65rem', padding: '2px 6px' }}>VRSBench-5k</span>
          </div>
          <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
            Remote Sensing VQA, Grounding & Bi-Temporal System
          </p>
        </div>
      </div>

      {/* Mode Switcher Tabs (Single Image VQA vs Bi-Temporal Comparison) */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        background: 'rgba(0, 0, 0, 0.35)',
        padding: '3px',
        borderRadius: '10px',
        border: '1px solid var(--border-subtle)'
      }}>
        <button
          onClick={() => onSwitchMode && onSwitchMode('single')}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            padding: '6px 14px',
            borderRadius: '8px',
            border: 'none',
            fontSize: '0.8rem',
            fontWeight: '600',
            cursor: 'pointer',
            background: activeMode === 'single' ? 'rgba(6, 182, 212, 0.2)' : 'transparent',
            color: activeMode === 'single' ? '#06b6d4' : 'var(--text-secondary)',
            borderBottom: activeMode === 'single' ? '2px solid #06b6d4' : '2px solid transparent',
            transition: 'all 0.2s'
          }}
        >
          <ImageIcon size={14} />
          <span>VQA & Grounding</span>
        </button>

        <button
          onClick={() => onSwitchMode && onSwitchMode('bitemporal')}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            padding: '6px 14px',
            borderRadius: '8px',
            border: 'none',
            fontSize: '0.8rem',
            fontWeight: '600',
            cursor: 'pointer',
            background: activeMode === 'bitemporal' ? 'rgba(244, 63, 94, 0.2)' : 'transparent',
            color: activeMode === 'bitemporal' ? '#f43f5e' : 'var(--text-secondary)',
            borderBottom: activeMode === 'bitemporal' ? '2px solid #f43f5e' : '2px solid transparent',
            transition: 'all 0.2s'
          }}
        >
          <ArrowLeftRight size={14} />
          <span>Bi-Temporal Change</span>
        </button>
      </div>

      {/* Hardware & Dataset Telemetry */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
        {/* Dataset Counter */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          padding: '6px 14px',
          borderRadius: '20px',
          background: 'rgba(255, 255, 255, 0.04)',
          border: '1px solid var(--border-subtle)',
          fontSize: '0.8rem',
          color: 'var(--text-secondary)'
        }}>
          <Database size={15} color="var(--accent-cyan)" />
          <span>VRSBench 5k subset</span>
        </div>

        {/* Preset Samples Toggle Button */}
        <button
          onClick={onToggleSamples}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            padding: '7px 14px',
            borderRadius: '10px',
            background: showSamples ? 'rgba(6, 182, 212, 0.2)' : 'rgba(255, 255, 255, 0.06)',
            border: showSamples ? '1px solid var(--accent-cyan)' : '1px solid var(--border-subtle)',
            color: showSamples ? 'var(--accent-cyan)' : '#fff',
            fontSize: '0.82rem',
            fontWeight: '500'
          }}
        >
          <Layers size={15} />
          <span>Preset Samples</span>
        </button>

        {/* Reset Active Session Button */}
        {activeSession && (
          <button
            onClick={onResetSession}
            title="Reset active image session"
            style={{
              padding: '7px 12px',
              borderRadius: '10px',
              background: 'rgba(244, 63, 94, 0.12)',
              border: '1px solid rgba(244, 63, 94, 0.3)',
              color: 'var(--accent-rose)',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              fontSize: '0.8rem'
            }}
          >
            <RefreshCw size={14} />
            <span>New Image</span>
          </button>
        )}
      </div>
    </header>
  );
}
