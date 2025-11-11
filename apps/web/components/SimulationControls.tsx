/**
 * SimulationControls Component
 *
 * Controls for creating new simulations
 */

'use client';

import { useState } from 'react';

interface SimulationControlsProps {
  onSimulationComplete: () => void;
}

export default function SimulationControls({ onSimulationComplete }: SimulationControlsProps) {
  const [targetRevenue, setTargetRevenue] = useState(50);
  const [k, setK] = useState(5);
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState<string>('');
  const [error, setError] = useState<string | null>(null);

  const runSimulation = async () => {
    setRunning(true);
    setError(null);
    setStatus('Creating simulation...');

    try {
      // Create simulation
      const response = await fetch('http://localhost:8000/simulations', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          target_revenue_bn: targetRevenue,
          k: k,
          epsilon: 0.02,
          max_candidates: 1000
        })
      });

      if (!response.ok) {
        throw new Error('Failed to create simulation');
      }

      const simulation = await response.json();
      setStatus(`Simulation ${simulation.simulation_id} running...`);

      // Poll for completion
      const checkStatus = async () => {
        const statusResponse = await fetch(`http://localhost:8000/simulations/${simulation.simulation_id}`);
        const statusData = await statusResponse.json();

        if (statusData.status === 'complete') {
          setStatus('Simulation complete!');
          setRunning(false);
          onSimulationComplete();
        } else if (statusData.status === 'failed') {
          setError(statusData.error || 'Simulation failed');
          setRunning(false);
        } else {
          setStatus(`Running... (${statusData.progress}%)`);
          setTimeout(checkStatus, 2000);  // Poll every 2 seconds
        }
      };

      setTimeout(checkStatus, 2000);

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      setRunning(false);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6 mb-8">
      <h2 className="text-lg font-semibold text-slate-900 mb-4">
        Run New Simulation
      </h2>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
        <div>
          <label className="block text-sm font-medium text-slate-700 mb-2">
            Target Revenue (£bn)
          </label>
          <input
            type="number"
            value={targetRevenue}
            onChange={(e) => setTargetRevenue(parseFloat(e.target.value))}
            disabled={running}
            className="w-full px-3 py-2 border border-slate-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500 disabled:bg-slate-100"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-slate-700 mb-2">
            Number of Options (k)
          </label>
          <select
            value={k}
            onChange={(e) => setK(parseInt(e.target.value))}
            disabled={running}
            className="w-full px-3 py-2 border border-slate-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500 disabled:bg-slate-100"
          >
            <option value={3}>3</option>
            <option value={5}>5</option>
            <option value={7}>7</option>
          </select>
        </div>

        <div className="flex items-end">
          <button
            onClick={runSimulation}
            disabled={running}
            className="w-full px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 disabled:bg-slate-400 disabled:cursor-not-allowed transition-colors"
          >
            {running ? 'Running...' : 'Run Simulation'}
          </button>
        </div>
      </div>

      {/* Status */}
      {status && (
        <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded text-sm text-blue-800">
          {status}
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded text-sm text-red-800">
          {error}
        </div>
      )}

      {/* Info */}
      <div className="mt-4 text-xs text-slate-500">
        This will generate {k} diverse, lawful, near-optimal policy options to raise £{targetRevenue}bn.
        Simulation typically takes 30-60 seconds.
      </div>
    </div>
  );
}
