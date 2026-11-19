import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import SatelliteViewer from './components/SatelliteViewer';
import ChatPanel from './components/ChatPanel';
import SampleGallery from './components/SampleGallery';
import BiTemporalViewer from './components/BiTemporalViewer';

export default function App() {
  const [health, setHealth] = useState(null);
  const [session, setSession] = useState(null);
  const [messages, setMessages] = useState([]);
  const [boundingBoxes, setBoundingBoxes] = useState([]);
  const [highlightedBoxIndex, setHighlightedBoxIndex] = useState(null);
  const [samples, setSamples] = useState([]);
  const [suggestedQuestions, setSuggestedQuestions] = useState([]);
  const [showSamplesModal, setShowSamplesModal] = useState(false);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [activeMode, setActiveMode] = useState('single'); // 'single' | 'bitemporal'

  // 1. Fetch Backend Telemetry and Samples on Mount
  useEffect(() => {
    fetchHealth();
    fetchSamples();
    const interval = setInterval(fetchHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  const fetchHealth = async () => {
    try {
      const res = await fetch('/api/health');
      if (res.ok) {
        const data = await res.json();
        setHealth(data);
      }
    } catch (err) {
      console.warn('Backend health check failed:', err);
    }
  };

  const fetchSamples = async () => {
    try {
      const res = await fetch('/api/samples?count=16');
      if (res.ok) {
        const data = await res.json();
        setSamples(data.samples || []);
      }
    } catch (err) {
      console.warn('Failed to load dataset samples:', err);
    }
  };

  // 2. Handle Satellite Image Upload
  const handleUpload = async (file) => {
    setUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Upload failed');
      }

      const data = await res.json();
      setSession(data);
      setMessages([]);
      setBoundingBoxes([]);
      setSuggestedQuestions([
        "Describe this satellite image.",
        "What are the main structures visible in this scene?",
        "Are there any vehicles or aircraft?",
        "Where are the key facilities located?"
      ]);
    } catch (err) {
      alert(`Upload error: ${err.message}`);
    } finally {
      setUploading(false);
    }
  };

  // 3. Handle Selecting a Sample from VRSBench Dataset
  const handleSelectSample = async (imageFile) => {
    setUploading(true);
    try {
      const res = await fetch(`/api/samples/select/${imageFile}`, {
        method: 'POST'
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Failed to select sample');
      }

      const data = await res.json();
      setSession(data);
      setMessages([]);
      setBoundingBoxes([]);

      // Find sample questions
      const matching = samples.find(s => s.image_file === imageFile);
      if (matching && matching.suggested_questions) {
        setSuggestedQuestions(matching.suggested_questions);
      }
    } catch (err) {
      alert(`Sample selection error: ${err.message}`);
    } finally {
      setUploading(false);
    }
  };

  // 4. Handle Submitting Chat / VQA / Grounding Questions
  const handleSendMessage = async (text, taskMode = 'auto') => {
    if (!session) return;

    // Optimistically add user message
    const userMsg = { role: 'user', content: text };
    setMessages(prev => [...prev, userMsg]);
    setLoading(true);

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: session.session_id,
          message: text,
          task: taskMode
        })
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Inference failed');
      }

      const data = await res.json();
      const aiMsg = {
        role: 'assistant',
        content: data.answer,
        boxes: data.bounding_boxes || [],
        annotated_image_url: data.annotated_image_url || null
      };

      setMessages(prev => [...prev, aiMsg]);
      if (data.bounding_boxes && data.bounding_boxes.length > 0) {
        setBoundingBoxes(data.bounding_boxes);
      }
    } catch (err) {
      setMessages(prev => [
        ...prev,
        { role: 'assistant', content: `⚠️ Error during analysis: ${err.message}`, boxes: [] }
      ]);
    } finally {
      setLoading(false);
    }
  };

  // 5. Reset Session
  const handleResetSession = () => {
    if (session) {
      fetch(`/api/session/${session.session_id}`, { method: 'DELETE' }).catch(() => {});
    }
    setSession(null);
    setMessages([]);
    setBoundingBoxes([]);
    setSuggestedQuestions([]);
  };

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100vh',
      overflow: 'hidden',
      background: 'var(--bg-main)'
    }}>
      {/* Top Navbar */}
      <Header
        health={health}
        activeSession={session}
        onResetSession={handleResetSession}
        onToggleSamples={() => setShowSamplesModal(true)}
        showSamples={showSamplesModal}
        activeMode={activeMode}
        onSwitchMode={setActiveMode}
      />

      {/* Main Workspace Layout */}
      <main style={{
        flex: 1,
        display: 'flex',
        gap: '16px',
        padding: '16px',
        overflow: 'hidden'
      }}>
        {activeMode === 'single' ? (
          <>
            {/* Left: Satellite Image Viewport & Grounding Canvas */}
            <SatelliteViewer
              session={session}
              onUpload={handleUpload}
              uploading={uploading}
              boundingBoxes={boundingBoxes}
              highlightedBoxIndex={highlightedBoxIndex}
              onHoverBox={setHighlightedBoxIndex}
            />

            {/* Right: AI VQA & Multi-Turn Chatbot */}
            <ChatPanel
              session={session}
              messages={messages}
              onSendMessage={handleSendMessage}
              loading={loading}
              suggestedQuestions={suggestedQuestions}
              onHoverBox={setHighlightedBoxIndex}
            />
          </>
        ) : (
          /* Fullscreen Bi-Temporal Change Detection Viewport */
          <div style={{ flex: 1, height: '100%' }}>
            <BiTemporalViewer />
          </div>
        )}
      </main>

      {/* VRSBench Preset Sample Gallery Modal */}
      <SampleGallery
        isOpen={showSamplesModal}
        onClose={() => setShowSamplesModal(false)}
        samples={samples}
        onSelectSample={handleSelectSample}
        selectedSampleId={session?.filename}
      />
    </div>
  );
}
