import React, { useState, useEffect } from 'react';
import MemoryGalaxy from '../visualizations/MemoryGalaxy';
import { aionMemory } from '../../core/aion-memory';

const MindPanel = () => {
    const [memories, setMemories] = useState([]);

    useEffect(() => {
        const fetchMemories = async () => {
            const allMemories = await aionMemory.getAllMemoriesForVisualization();
            setMemories(allMemories);
        };
        fetchMemories();
    }, []);

    return (
        <div className="mind-panel">
            {memories.length > 0 ? (
                <MemoryGalaxy memories={memories} />
            ) : (
                <div className="no-galaxy">
                    <p>No long-term memories to visualize yet. Interact with AION to build its mind.</p>
                </div>
            )}
        </div>
    );
};

export default MindPanel;