/**
 * ALBION Main Page
 *
 * Displays the five diverse policy options in a card grid
 */

'use client';

import { useEffect, useState } from 'react';
import PolicyCard from '../components/PolicyCard';
import SimulationControls from '../components/SimulationControls';

interface Plan {
  id: string;
  levers: any;
  impacts: {
    distribution: {
      decile_deltas_pct: number[];
      gini_delta: number;
    };
    regional: Record<string, any>;
    climate: {
      emissions_delta_mtco2e: number;
      trajectory_2024_2037: number[];
      budget_compliance: {
        status: string;
        headroom_or_overshoot_mtco2e: number;
      };
    };
    macro: {
      gdp_delta_pct: number;
      employment_delta_thousands: number;
      debt_ratio_year5_pct: number;
    };
  };
  objective_value: number;
  signature?: any;
}

export default function Home() {
  const [plans, setPlans] = useState<Plan[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load plans from API
  const loadPlans = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:8000/plans');
      const data = await response.json();

      if (data.plans && data.plans.length > 0) {
        // Load full plan details
        const fullPlans = await Promise.all(
          data.plans.map(async (p: any) => {
            const planResponse = await fetch(`http://localhost:8000/plans/${p.plan_id}`);
            return await planResponse.json();
          })
        );

        setPlans(fullPlans);
      }
    } catch (err) {
      setError('Failed to load plans. Make sure the API is running (http://localhost:8000)');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadPlans();
  }, []);

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-slate-900">
                ALBION
              </h1>
              <p className="text-sm text-slate-600 mt-1">
                The National Options Engine
              </p>
            </div>
            <div className="text-right">
              <div className="text-sm text-slate-500">
                Lawful · Diverse · Near-Optimal
              </div>
              <div className="text-xs text-slate-400 mt-1">
                With Public Receipts
              </div>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Controls */}
        <SimulationControls onSimulationComplete={loadPlans} />

        {/* Loading State */}
        {loading && (
          <div className="flex items-center justify-center py-20">
            <div className="text-center">
              <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600"></div>
              <p className="mt-4 text-slate-600">Loading plans...</p>
            </div>
          </div>
        )}

        {/* Error State */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
            <p className="text-red-800">{error}</p>
          </div>
        )}

        {/* Plans Grid */}
        {!loading && plans.length > 0 && (
          <>
            <div className="mb-6">
              <h2 className="text-2xl font-semibold text-slate-900">
                Five Diverse Options
              </h2>
              <p className="text-slate-600 mt-2">
                Each option is lawful, near-optimal, and mathematically proven to be diverse.
              </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
              {plans.map((plan, index) => (
                <PolicyCard
                  key={plan.id || index}
                  plan={plan}
                  planNumber={index + 1}
                />
              ))}
            </div>
          </>
        )}

        {/* Empty State */}
        {!loading && !error && plans.length === 0 && (
          <div className="text-center py-20">
            <div className="inline-block p-6 bg-white rounded-full shadow-sm mb-4">
              <svg className="w-12 h-12 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            </div>
            <h3 className="text-lg font-medium text-slate-900 mb-2">
              No Plans Available
            </h3>
            <p className="text-slate-600 mb-6">
              Run a simulation to generate diverse policy options
            </p>
          </div>
        )}
      </div>

      {/* Footer */}
      <footer className="border-t border-slate-200 mt-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="text-center text-sm text-slate-500">
            <p>
              <strong>ALBION</strong>: Lawful, diverse, near-optimal policy options with public receipts.
            </p>
            <p className="mt-2 text-xs">
              "Show your working. Sign your work. Serve the public."
            </p>
          </div>
        </div>
      </footer>
    </main>
  );
}
