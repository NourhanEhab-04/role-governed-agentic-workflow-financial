// src/App.jsx

import { useState, useCallback } from "react";
import InputPanel               from "./components/InputPanel";
import ChatPanel                from "./components/ChatPanel";
import SummaryStrip             from "./components/SummaryStrip";
import { useAssessment }        from "./hooks/useAssessment";
import { mapStateToMessages }   from "./utils/mapStateToMessages";

export default function App() {
  const { run, isLoading, error, state } = useAssessment();

  // Derive messages from state every render — pure, no extra useState needed
  const messages = state ? mapStateToMessages(state) : [];

  const handleRun = useCallback(
    (clientInput, productInput) => {
      run(clientInput, productInput);
    },
    [run]
  );

  return (
    <div className="flex h-screen bg-gray-50 text-gray-900">

      {/* ── Left panel — fixed width ── */}
      <aside className="w-96 flex-shrink-0 flex flex-col border-r border-gray-200 bg-white">

        <div className="px-4 py-3 border-b border-gray-100">
          <h1 className="font-semibold text-sm tracking-wide">MiFID II Suitability</h1>
          <p className="text-xs text-gray-400 mt-0.5">Multi-agent assessment pipeline</p>
        </div>

        <div className="flex-1 overflow-y-auto">
          <InputPanel onRun={handleRun} isLoading={isLoading} />
        </div>

        {/* Error banner — inside left panel, below input */}
        {error && (
          <div className="mx-3 mb-3 rounded-lg bg-red-50 border border-red-200 px-3 py-2 text-xs text-red-700">
            <span className="font-semibold">Error: </span>{error}
          </div>
        )}

        <div className="border-t border-gray-100 overflow-y-auto max-h-[55vh]">
          <SummaryStrip state={state} />
        </div>

      </aside>

      {/* ── Right panel — scrollable chat ── */}
      <main className="flex-1 flex flex-col overflow-hidden">

        <div className="px-4 py-3 border-b border-gray-100 bg-white">
          <h2 className="text-sm font-medium text-gray-500">Agent conversation</h2>
        </div>

        <div className="flex-1 overflow-y-auto">
          <ChatPanel messages={messages} />
        </div>

      </main>

    </div>
  );
}