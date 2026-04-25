// src/hooks/useAssessment.js

import { useState, useCallback } from "react";

const FIELD_LABELS = {
  investment_amount:    "investment amount (e.g. \"investing €10,000\")",
  liquid_assets:        "liquid assets (e.g. \"€50,000 in savings\")",
  income:               "annual income (e.g. \"earns €60,000 per year\")",
  age:                  "age (e.g. \"45-year-old\")",
  risk_tolerance_score: "risk tolerance score (1–10)",
  investment_horizon:   "investment horizon (e.g. \"5-year horizon\")",
  financial_knowledge:  "financial knowledge level (none / basic / moderate / advanced)",
  can_afford_total_loss:"whether they can afford a total loss",
  financial_vulnerability: "vulnerability level (low / medium / high)",
};

function friendlyError(raw) {
  // Pydantic: "... validation error for ClientProfile investment_amount Input should be ..."
  const pydanticMatch = raw.match(/validation error for \w+\s+([\w_]+)/);
  if (pydanticMatch) {
    const field = pydanticMatch[1];
    const label = FIELD_LABELS[field] ?? field.replace(/_/g, " ");
    return `Your client description is missing the ${label}. Please add it and try again.`;
  }

  // Pipeline halt message
  const haltMatch = raw.match(/Stage \w+ failed[^:]*:\s*(.+)/);
  if (haltMatch) return `Pipeline error: ${haltMatch[1]}`;

  return raw;
}

const API_URL = "http://localhost:8000/assess";

export function useAssessment() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError]         = useState(null);   // string | null
  const [state, setState]         = useState(null);   // raw backend state | null

  const run = useCallback(async (clientInput, productInput) => {
    setIsLoading(true);
    setError(null);
    setState(null);

    try {
      const res = await fetch(API_URL, {
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

      const data = await res.json();
      setState(data);

      // Pipeline returned 200 but halted mid-way (e.g. validation failure)
      if (data.halt && data.halt_reason) {
        setError(friendlyError(data.halt_reason));
      }

    } catch (err) {
      // Network failure (backend down) or non-OK response
      setError(friendlyError(err.message ?? "Unknown error"));
    } finally {
      setIsLoading(false);
    }
  }, []);

  return { run, isLoading, error, state };
}