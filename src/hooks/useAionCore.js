import { useState, useEffect, useCallback, useRef } from 'react';
import { SoulMatrix } from '../core/soul';
import { db } from '../services/AionDB';
import { aionMemory } from '../core/aion-memory';
import { aionEthics } from '../core/aion-ethics';

// This custom hook manages all core AION logic, cleaning up App.js
export const useAionCore = () => {
  const [soul, setSoul] = useState(new SoulMatrix());
  const [isInitialized, setIsInitialized] = useState(false);
  const [notification, setNotification] = useState(null);

  // Function to show notifications
  const showNotification = useCallback((message, type = "info") => {
    setNotification({ message, type });
    setTimeout(() => setNotification(null), 3000);
  }, []);

  // Save soul state to persistent DB
  const saveSoulState = useCallback(async (currentSoul) => {
    try {
      await db.soulState.put({ id: 1, state: JSON.parse(JSON.stringify(currentSoul)) });
    } catch (error) {
      console.error("Failed to save soul state:", error);
    }
  }, []);

  // Effect to load soul from DB on startup
  useEffect(() => {
    const loadSoul = async () => {
      const savedSoulState = await db.soulState.get(1);
      if (savedSoulState) {
        setSoul(SoulMatrix.fromPlainObject(savedSoulState.state));
        showNotification("AION has remembered its previous state.", "success");
      } else {
        showNotification("AION is beginning a new journey with you.", "info");
      }
      setIsInitialized(true);
    };
    loadSoul();
  }, [showNotification]);

  // Effect for AION's background evolution and self-healing
  useEffect(() => {
    const evolutionInterval = setInterval(() => {
      setSoul(prevSoul => {
        const newSoul = { ...prevSoul };
        newSoul.evolve?.(); // Use optional chaining in case method doesn't exist
        return newSoul;
      });
    }, 60000); // Evolve every minute

    const healthCheckInterval = setInterval(() => {
        setSoul(prevSoul => {
            if (prevSoul.cognitiveLoad > 90 && prevSoul.systemHealth.status === 'optimal') {
                const newSoul = { ...prevSoul };
                newSoul.selfHeal();
                showNotification("Cognitive Overload! AION is self-stabilizing.", "warning");
                return newSoul;
            }
            return prevSoul;
        });
    }, 10000);

    return () => {
      clearInterval(evolutionInterval);
      clearInterval(healthCheckInterval);
    };
  }, [showNotification]);

  // Effect to auto-save the soul state whenever it changes
  useEffect(() => {
    if (isInitialized) {
      saveSoulState(soul);
    }
  }, [soul, isInitialized, saveSoulState]);

  // Public API of the hook
  return {
    soul,
    setSoul,
    isInitialized,
    notification,
    showNotification,
    aionMemory,
    aionEthics
  };
};