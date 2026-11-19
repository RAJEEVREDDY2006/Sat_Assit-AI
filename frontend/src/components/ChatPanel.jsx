import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Sparkles, MapPin, Compass, HelpCircle, FileText, CornerDownLeft, Download, Image as ImageIcon } from 'lucide-react';

export default function ChatPanel({
  session,
  messages,
  onSendMessage,
  loading,
  suggestedQuestions,
  onHoverBox
}) {
  const [input, setInput] = useState('');
  const [taskMode, setTaskMode] = useState('auto');
  const [annotatingIdx, setAnnotatingIdx] = useState(null);
  const [annotatedUrls, setAnnotatedUrls] = useState({});
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  const handleSubmit = (e) => {
    e?.preventDefault();
    if (!input.trim() || loading || !session) return;
    onSendMessage(input.trim(), taskMode);
    setInput('');
  };

  const handleQuickQuestion = (q) => {
    if (loading || !session) return;
    onSendMessage(q, taskMode);
  };

  // Request annotated image from backend
  const handleAnnotate = async (msgIndex, boxes) => {
    if (!session || !boxes || boxes.length === 0) return;
    setAnnotatingIdx(msgIndex);
    try {
      const res = await fetch('/api/annotate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: session.session_id,
          bounding_boxes: boxes,
          label: 'Detected Objects'
        })
      });
      if (!res.ok) throw new Error('Annotation failed');
      const data = await res.json();
      setAnnotatedUrls(prev => ({ ...prev, [msgIndex]: data.annotated_image_url }));
    } catch (err) {
      alert('Could not generate annotated image: ' + err.message);
    } finally {
      setAnnotatingIdx(null);
    }
  };

  const defaultSuggestions = suggestedQuestions && suggestedQuestions.length > 0 ? suggestedQuestions : [
    "Describe this satellite image in detail.",
    "Are there any aircraft or runways visible?",
    "Where are the main buildings located?",
    "What type of terrain or area is shown?"
  ];

  return (
    <div className="glass-panel" style={{
      flex: '1',
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      position: 'relative',
      overflow: 'hidden'
    }}>
      {/* Chat Header & Task Mode Selector */}
      <div style={{
        padding: '12px 18px',
        borderBottom: '1px solid var(--border-subtle)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        background: 'rgba(0, 0, 0, 0.2)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Bot size={18} color="var(--accent-cyan)" />
          <span style={{ fontSize: '0.85rem', fontWeight: '600', letterSpacing: '0.04em', textTransform: 'uppercase', color: 'var(--text-secondary)' }}>
            AI Assistant
          </span>
        </div>

        {/* Task Mode Pills */}
        <div style={{ display: 'flex', gap: '4px', background: 'rgba(255, 255, 255, 0.04)', padding: '3px', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
          {[
            { id: 'auto', label: 'Auto' },
            { id: 'caption', label: 'Caption' },
            { id: 'vqa', label: 'VQA' },
            { id: 'grounding', label: 'Locate' }
          ].map((mode) => (
            <button
              key={mode.id}
              onClick={() => setTaskMode(mode.id)}
              style={{
                padding: '4px 9px',
                borderRadius: '6px',
                border: 'none',
                background: taskMode === mode.id ? 'var(--accent-cyan)' : 'transparent',
                color: taskMode === mode.id ? '#000' : 'var(--text-muted)',
                fontSize: '0.72rem',
                fontWeight: taskMode === mode.id ? '700' : '500'
              }}
            >
              {mode.label}
            </button>
          ))}
        </div>
      </div>

      {/* Messages Scroll Area */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        padding: '18px',
        display: 'flex',
        flexDirection: 'column',
        gap: '16px'
      }}>
        {/* Empty State / Welcome */}
        {messages.length === 0 && (
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            height: '100%',
            textAlign: 'center',
            color: 'var(--text-muted)',
            padding: '24px'
          }}>
            <div style={{
              width: '52px',
              height: '52px',
              borderRadius: '14px',
              background: 'rgba(6, 182, 212, 0.1)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              marginBottom: '14px',
              border: '1px solid var(--border-glow)'
            }}>
              <Sparkles size={24} color="var(--accent-cyan)" />
            </div>
            <h4 style={{ color: '#fff', fontSize: '1rem', fontWeight: '600', marginBottom: '6px' }}>
              Satellite AI Analysis Ready
            </h4>
            <p style={{ fontSize: '0.82rem', lineHeight: '1.4', maxWidth: '320px', marginBottom: '18px' }}>
              {session
                ? "Ask natural-language questions about this satellite image or choose a prompt below."
                : "Upload a satellite image or select a preset sample from the gallery to begin."}
            </p>

            {/* Quick Prompt Pills */}
            {session && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', width: '100%', maxWidth: '360px' }}>
                {defaultSuggestions.map((q, idx) => (
                  <button
                    key={idx}
                    onClick={() => handleQuickQuestion(q)}
                    style={{
                      textAlign: 'left',
                      padding: '8px 14px',
                      borderRadius: '8px',
                      background: 'rgba(255, 255, 255, 0.04)',
                      border: '1px solid var(--border-subtle)',
                      color: 'var(--text-primary)',
                      fontSize: '0.8rem',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between'
                    }}
                  >
                    <span>{q}</span>
                    <CornerDownLeft size={13} color="var(--accent-cyan)" />
                  </button>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Rendered Conversation Messages */}
        {messages.map((msg, index) => (
          <div
            key={index}
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: msg.role === 'user' ? 'flex-end' : 'flex-start',
              gap: '6px'
            }}
          >
            {/* Sender Badge */}
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              fontSize: '0.72rem',
              color: msg.role === 'user' ? 'var(--text-muted)' : 'var(--accent-cyan)',
              padding: '0 4px'
            }}>
              {msg.role === 'user' ? <User size={12} /> : <Bot size={12} />}
              <span>{msg.role === 'user' ? 'You' : 'SatAssist AI'}</span>
            </div>

            {/* Message Bubble */}
            <div style={{
              maxWidth: '88%',
              padding: '12px 16px',
              borderRadius: msg.role === 'user' ? '14px 14px 2px 14px' : '14px 14px 14px 2px',
              background: msg.role === 'user'
                ? 'linear-gradient(135deg, #0284c7 0%, #0369a1 100%)'
                : 'rgba(18, 26, 47, 0.95)',
              border: msg.role === 'user' ? 'none' : '1px solid var(--border-glow)',
              color: '#fff',
              fontSize: '0.88rem',
              lineHeight: '1.5',
              boxShadow: msg.role === 'user' ? '0 4px 12px rgba(2, 132, 199, 0.25)' : 'var(--shadow-card)'
            }}>
              {msg.content}

              {/* Direct Inline Annotated Image if provided */}
              {(msg.annotated_image_url || annotatedUrls[index]) && (
                <div style={{ width: '100%', marginTop: '10px' }}>
                  <img
                    src={msg.annotated_image_url || annotatedUrls[index]}
                    alt="Annotated satellite image"
                    style={{
                      width: '100%',
                      maxHeight: '340px',
                      objectFit: 'contain',
                      borderRadius: '8px',
                      border: '1px solid #6366f1',
                      display: 'block',
                      marginBottom: '6px',
                      background: 'rgba(0,0,0,0.5)'
                    }}
                  />
                  <a
                    href={`/api/annotate/download/${(msg.annotated_image_url || annotatedUrls[index]).split('/').pop()}`}
                    download
                    style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '5px',
                      padding: '4px 12px',
                      borderRadius: '6px',
                      background: 'rgba(99, 102, 241, 0.3)',
                      border: '1px solid #6366f1',
                      color: '#a5b4fc',
                      fontSize: '0.75rem',
                      textDecoration: 'none'
                    }}
                  >
                    <Download size={12} />
                    Download Annotated PNG
                  </a>
                </div>
              )}

              {/* Visual Grounding Detections Chips */}
              {msg.boxes && msg.boxes.length > 0 && (
                <div style={{
                  marginTop: '10px',
                  paddingTop: '8px',
                  borderTop: '1px solid rgba(255, 255, 255, 0.1)',
                  display: 'flex',
                  flexWrap: 'wrap',
                  gap: '6px'
                }}>
                  {msg.boxes.slice(0, 15).map((box, bIdx) => (
                    <span
                      key={bIdx}
                      onMouseEnter={() => onHoverBox && onHoverBox(bIdx)}
                      onMouseLeave={() => onHoverBox && onHoverBox(null)}
                      style={{
                        cursor: 'pointer',
                        padding: '3px 8px',
                        borderRadius: '4px',
                        background: 'rgba(6, 182, 212, 0.2)',
                        border: '1px solid var(--accent-cyan)',
                        color: 'var(--accent-cyan)',
                        fontSize: '0.72rem',
                        fontFamily: 'var(--font-mono)',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '4px'
                      }}
                    >
                      <MapPin size={11} />
                      <span>{box.label || `Box #${bIdx + 1}`}</span>
                    </span>
                  ))}
                  {msg.boxes.length > 15 && (
                    <span style={{
                      padding: '3px 8px',
                      borderRadius: '4px',
                      background: 'rgba(255, 255, 255, 0.08)',
                      color: 'var(--text-muted)',
                      fontSize: '0.72rem',
                      fontFamily: 'var(--font-mono)'
                    }}>
                      +{msg.boxes.length - 15} more
                    </span>
                  )}
                </div>
              )}
            </div>
          </div>
        ))}

        {/* Thinking / Typing Radar Indicator */}
        {loading && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', padding: '10px 14px', background: 'rgba(6, 182, 212, 0.08)', borderRadius: '12px', border: '1px solid var(--border-glow)', width: 'fit-content' }}>
            <div style={{ width: '12px', height: '12px', borderRadius: '50%', background: 'var(--accent-cyan)', animation: 'radar-ping 1.2s infinite ease-in-out' }} />
            <span style={{ fontSize: '0.82rem', color: 'var(--accent-cyan)' }}>Analyzing satellite image with VLM...</span>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Suggested Chips Bar (When session is active and messages exist) */}
      {session && messages.length > 0 && (
        <div style={{
          padding: '6px 14px',
          display: 'flex',
          gap: '6px',
          overflowX: 'auto',
          borderTop: '1px solid var(--border-subtle)',
          background: 'rgba(0, 0, 0, 0.15)'
        }}>
          {defaultSuggestions.slice(0, 3).map((q, idx) => (
            <button
              key={idx}
              onClick={() => handleQuickQuestion(q)}
              disabled={loading}
              style={{
                whiteSpace: 'nowrap',
                padding: '4px 10px',
                borderRadius: '12px',
                background: 'rgba(255, 255, 255, 0.04)',
                border: '1px solid var(--border-subtle)',
                color: 'var(--text-secondary)',
                fontSize: '0.74rem'
              }}
            >
              {q}
            </button>
          ))}
        </div>
      )}

      {/* Input Box Bar */}
      <form
        onSubmit={handleSubmit}
        style={{
          padding: '14px 18px',
          borderTop: '1px solid var(--border-subtle)',
          background: 'rgba(9, 14, 26, 0.95)',
          display: 'flex',
          gap: '10px',
          alignItems: 'center'
        }}
      >
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={session ? "Ask anything about this satellite scene..." : "Please upload or select an image first..."}
          disabled={!session || loading}
          style={{
            flex: 1,
            padding: '11px 16px',
            borderRadius: '10px',
            background: 'rgba(255, 255, 255, 0.06)',
            border: '1px solid var(--border-subtle)',
            color: '#fff',
            fontSize: '0.88rem',
            fontFamily: 'var(--font-main)',
            outline: 'none',
            transition: 'border 0.2s ease'
          }}
          onFocus={(e) => e.target.style.borderColor = 'var(--accent-cyan)'}
          onBlur={(e) => e.target.style.borderColor = 'var(--border-subtle)'}
        />

        <button
          type="submit"
          disabled={!input.trim() || !session || loading}
          style={{
            padding: '11px 18px',
            borderRadius: '10px',
            background: input.trim() && session && !loading
              ? 'linear-gradient(135deg, #06b6d4 0%, #0284c7 100%)'
              : 'rgba(255, 255, 255, 0.08)',
            border: 'none',
            color: input.trim() && session && !loading ? '#000' : 'var(--text-muted)',
            fontWeight: '600',
            fontSize: '0.88rem',
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            boxShadow: input.trim() && session ? '0 0 15px rgba(6, 182, 212, 0.4)' : 'none'
          }}
        >
          <span>Ask</span>
          <Send size={15} />
        </button>
      </form>
    </div>
  );
}
