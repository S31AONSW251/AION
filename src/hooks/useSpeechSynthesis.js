// src/hooks/useSpeechSynthesis.js
import { useState, useEffect } from "react";

export const useSpeechSynthesis = () => {
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isSpeechSupported, setIsSpeechSupported] = useState(false);

  useEffect(() => {
    if ("speechSynthesis" in window) {
      setIsSpeechSupported(true);
    }
  }, []);

  const speak = (text, voice = null) => {
    if (!isSpeechSupported || !text) {
      return;
    }

    const utterance = new SpeechSynthesisUtterance(text);
    if (voice) {
      utterance.voice = voice;
    }

    utterance.onstart = () => {
      setIsSpeaking(true);
    };

    utterance.onend = () => {
      setIsSpeaking(false);
    };

    utterance.onerror = (event) => {
      console.error("Speech synthesis error:", event.error);
      setIsSpeaking(false);
    };

    window.speechSynthesis.speak(utterance);
  };

  const stop = () => {
    if (isSpeechSupported) {
      window.speechSynthesis.cancel();
    }
  };

  return { speak, stop, isSpeaking, isSpeechSupported };
};