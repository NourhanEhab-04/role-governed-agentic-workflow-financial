// src/hooks/useAssessment.js

import { useState, useCallback } from "react";

const FIELD_LABELS = {
  investment_amount:    "investment amount (e.g. \"investing €10,000\")",
  liquid_assets:        "liquid assets (e.g. \"€50,000 in savings\")",
  income:               "annual income (e.g. \"earns €60,000 per year\")",
  risk_tolerance_score: "risk tolerance score (1–10)",
  investment_horizon:   "investment horizon (e.g. \"5-year horizon\")",
  financial_knowledge:  "financial knowledge level (none / basic / moderate / advanced)",
  can_afford_total_loss:"whether they can afford a total loss",
  financial_vulnerability: "vulnerability level (low / medium / high)",
};

function friendlyError(raw) {
  const pydanticMatch = raw.match(/validation error for \w+\s+([\w_]+)/);
  if (pydanticMatch) {
    const field = pydanticMatch[1];
    const label = FIELD_LABELS[field] ?? field.replace(/_/g, " ");
    return `Your client description is missing the ${label}. Please add it and try again.`;
  }
  const haltMatch = raw.match(/Stage \w+ failed[^:]*:\s*(.+)/);
  if (haltMatch) return `Pipeline error: ${haltMatch[1]}`;
  return raw;
}

const API_STREAM_URL = "http://localhost:8000/assess/stream";

export function useAssessment() {
  const [isLoading, setIsLoading]       = useState(false);
  const [error, setError]               = useState(null);   // string | null
  const [state, setState]               = useState(null);   // raw backend state | null
  const [currentStage, setCurrentStage] = useState(null);  // latest event type

  const run = useCallback(async (clientInput, productInput) => {
    setIsLoading(true);
    setError(null);
    setState(null);
    setCurrentStage("start");

    try {
      const res = await fetch(API_STREAM_URL, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({
          client_input:  clientInput,
          product_input: productInput,
        }),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail ?? `Server error: ${res.status}`);
      }

      const reader  = res.body.getReader();
      const decoder = new TextDecoder();
      let   buffer  = "";

      outer: while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // Process all complete SSE messages (each ends with \n\n)
        let boundary = buffer.indexOf("\n\n");
        while (boundary !== -1) {
          const chunk = buffer.slice(0, boundary);
          buffer      = buffer.slice(boundary + 2);

          for (const line of chunk.split("\n")) {
            if (!line.startsWith("data: ")) continue;
            const data = line.slice(6).trim();

            if (data === "[DONE]") break outer;

            try {
              const event = JSON.parse(data);

              if (event.type === "error") {
                setError(friendlyError(event.detail ?? "Unknown pipeline error"));
                continue;
              }

              if (event.state) {
                setState(prev => ({ ...(prev ?? {}), ...event.state }));
                setCurrentStage(event.type);

                if (event.state.halt && event.state.halt_reason) {
                  setError(friendlyError(event.state.halt_reason));
                }
              }
            } catch {
              // skip malformed event
            }
          }

          boundary = buffer.indexOf("\n\n");
        }
      }
    } catch (err) {
      setError(friendlyError(err.message ?? "Unknown error"));
    } finally {
      setIsLoading(false);
      setCurrentStage(null);
    }
  }, []);

  return { run, isLoading, error, state, currentStage };
}
