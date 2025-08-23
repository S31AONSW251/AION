const API_BASE = "https://aion-server.onrender.com";

/**
 * Send a chat prompt to AION backend
 */
export async function sendPrompt(prompt) {
  try {
    const res = await fetch(`${API_BASE}/ai`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt }),
    });
    return await res.json();
  } catch (err) {
    console.error("API error:", err);
    return { reply: "⚠️ Failed to connect to AI server." };
  }
}

/**
 * Placeholders for future routes
 * Uncomment once backend has these endpoints
 */

// export async function generateImage(prompt) {
//   return { reply: "Image generation not implemented yet." };
// }

// export async function generateCode(prompt) {
//   return { reply: "Code generation not implemented yet." };
// }
