// src/lib/api.js (or wherever you call fetch/axios)
export const API_BASE = import.meta.env.VITE_API_BASE;

export async function generateImage(prompt) {
  const res = await fetch(`${API_BASE}/generate-image`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt }),
  });
  return res.json();
}

export async function generateCode(prompt, model = "llama3") {
  const res = await fetch(`${API_BASE}/generate-code`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, model }),
  });
  return res.json();
}

export async function webSearch(query) {
  const res = await fetch(`${API_BASE}/api/search?query=${encodeURIComponent(query)}`);
  return res.json();
}
