import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Points, PointMaterial } from '@react-three/drei';
import * as THREE from 'three';

const GalaxyPoints = ({ memories }) => {
  const ref = useRef();

  const positions = useMemo(() => {
    const points = new Float32Array(memories.length * 3);
    memories.forEach((mem, i) => {
      // Use the first 3 dimensions of the vector for position
      const x = (mem.vector[0] || 0) * 10 - 5;
      const y = (mem.vector[1] || 0) * 10 - 5;
      const z = (mem.vector[2] || 0) * 10 - 5;
      points.set([x, y, z], i * 3);
    });
    return points;
  }, [memories]);

  useFrame((state, delta) => {
    ref.current.rotation.x -= delta / 15;
    ref.current.rotation.y -= delta / 20;
  });

  return (
    <Points ref={ref} positions={positions} stride={3} frustumCulled={false}>
      <PointMaterial
        transparent
        color="#ffffff"
        size={0.05}
        sizeAttenuation={true}
        depthWrite={false}
      />
    </Points>
  );
};

const MemoryGalaxy = ({ memories }) => {
  return (
    <div className="memory-galaxy-container">
      <Canvas camera={{ position: [0, 0, 5] }}>
        <ambientLight intensity={0.5} />
        {memories.length > 0 && <GalaxyPoints memories={memories} />}
      </Canvas>
      <div className="galaxy-overlay">
        <h3>AION's Mind</h3>
        <p>{memories.length} memories visualized. Similar thoughts cluster together.</p>
      </div>
    </div>
  );
};

export default MemoryGalaxy;