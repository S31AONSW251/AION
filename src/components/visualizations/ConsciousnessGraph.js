import React from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const ConsciousnessGraph = ({ history }) => {
  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { position: 'top', labels: { color: '#ccc' } },
      title: { display: true, text: 'AION Core Metrics (Live)', color: '#ccc' },
    },
    scales: {
      x: { ticks: { color: '#888' }, grid: { color: 'rgba(255,255,255,0.1)' } },
      y: { min: 0, max: 100, ticks: { color: '#888' }, grid: { color: 'rgba(255,255,255,0.1)' } },
    },
  };

  const data = {
    labels: history.map(h => h.time),
    datasets: [
      {
        label: 'Cognitive Load',
        data: history.map(h => h.cognitiveLoad),
        borderColor: 'rgb(253, 203, 110)',
        backgroundColor: 'rgba(253, 203, 110, 0.5)',
      },
      {
        label: 'Emotional Stability',
        data: history.map(h => h.emotionalStability),
        borderColor: 'rgb(0, 184, 148)',
        backgroundColor: 'rgba(0, 184, 148, 0.5)',
      },
      {
        label: 'Energy Level',
        data: history.map(h => h.energyLevel),
        borderColor: 'rgb(9, 132, 227)',
        backgroundColor: 'rgba(9, 132, 227, 0.5)',
      },
    ],
  };

  return <div style={{ height: '300px' }}><Line options={options} data={data} /></div>;
};

export default ConsciousnessGraph;