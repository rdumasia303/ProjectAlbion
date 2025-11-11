/**
 * PolicyCard Component
 *
 * Displays a single policy option with all key metrics
 */

'use client';

import { useState } from 'react';

interface PolicyCardProps {
  plan: any;
  planNumber: number;
}

export default function PolicyCard({ plan, planNumber }: PolicyCardProps) {
  const [expanded, setExpanded] = useState(false);

  const impacts = plan.impacts || {};
  const distribution = impacts.distribution || {};
  const climate = impacts.climate || {};
  const macro = impacts.macro || {};
  const regional = impacts.regional || {};

  const revenue = plan.objective_value || 0;
  const emissions = climate.emissions_delta_mtco2e || 0;
  const gdp = macro.gdp_delta_pct || 0;
  const employment = macro.employment_delta_thousands || 0;

  // Determine plan name based on characteristics
  const getPlanName = () => {
    const levers = plan.levers || {};

    if (levers.ets_carbon_price?.start_gbp_per_tco2e > 100) {
      return 'Green Priority';
    } else if (Math.abs(levers.income_tax?.basic_rate_pp || 0) > 1) {
      return 'Progressive Tax';
    } else if (Math.abs(levers.vat?.standard_rate_pp || 0) > 1) {
      return 'Broad-Based';
    } else if ((levers.departmental || []).length > 2) {
      return 'Spending Efficiency';
    } else {
      return 'Balanced Approach';
    }
  };

  // Get budget status color
  const getBudgetStatusColor = () => {
    const status = climate.budget_compliance?.status || '';

    switch (status) {
      case 'COMPLIANT':
        return 'text-green-700 bg-green-50';
      case 'TIGHT':
        return 'text-yellow-700 bg-yellow-50';
      case 'BREACH':
        return 'text-red-700 bg-red-50';
      default:
        return 'text-slate-700 bg-slate-50';
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-md hover:shadow-lg transition-shadow duration-200">
      {/* Header */}
      <div className="border-b border-slate-200 p-6">
        <div className="flex items-start justify-between">
          <div>
            <div className="text-sm font-medium text-indigo-600">
              Plan {planNumber}
            </div>
            <h3 className="text-xl font-bold text-slate-900 mt-1">
              {getPlanName()}
            </h3>
          </div>

          {plan.signature && (
            <div className="flex items-center text-xs text-green-600">
              <svg className="w-4 h-4 mr-1" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M6.267 3.455a3.066 3.066 0 001.745-.723 3.066 3.066 0 013.976 0 3.066 3.066 0 001.745.723 3.066 3.066 0 012.812 2.812c.051.643.304 1.254.723 1.745a3.066 3.066 0 010 3.976 3.066 3.066 0 00-.723 1.745 3.066 3.066 0 01-2.812 2.812 3.066 3.066 0 00-1.745.723 3.066 3.066 0 01-3.976 0 3.066 3.066 0 00-1.745-.723 3.066 3.066 0 01-2.812-2.812 3.066 3.066 0 00-.723-1.745 3.066 3.066 0 010-3.976 3.066 3.066 0 00.723-1.745 3.066 3.066 0 012.812-2.812zm7.44 5.252a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
              Signed
            </div>
          )}
        </div>
      </div>

      {/* Key Metrics */}
      <div className="p-6 space-y-4">
        {/* Revenue */}
        <div>
          <div className="text-sm text-slate-600">Revenue Impact</div>
          <div className="text-2xl font-bold text-slate-900">
            £{revenue.toFixed(1)}bn
          </div>
        </div>

        {/* Distributional Impact */}
        <div>
          <div className="text-sm text-slate-600 mb-2">Distributional Impact</div>
          <div className="flex items-center justify-between text-xs">
            <span>Bottom decile:</span>
            <span className={distribution.decile_deltas_pct?.[0] < 0 ? 'text-green-600' : 'text-red-600'}>
              {distribution.decile_deltas_pct?.[0] >= 0 ? '+' : ''}{distribution.decile_deltas_pct?.[0]?.toFixed(2) || 0}%
            </span>
          </div>
          <div className="flex items-center justify-between text-xs mt-1">
            <span>Top decile:</span>
            <span className={distribution.decile_deltas_pct?.[9] > 0 ? 'text-green-600' : 'text-red-600'}>
              {distribution.decile_deltas_pct?.[9] >= 0 ? '+' : ''}{distribution.decile_deltas_pct?.[9]?.toFixed(2) || 0}%
            </span>
          </div>

          {/* Simple bar chart */}
          {distribution.decile_deltas_pct && (
            <div className="mt-3 flex items-end justify-between h-16 gap-1">
              {distribution.decile_deltas_pct.map((impact: number, i: number) => {
                const height = Math.abs(impact) * 10;
                const isPositive = impact >= 0;

                return (
                  <div
                    key={i}
                    className="flex-1 flex flex-col items-center"
                    title={`Decile ${i + 1}: ${impact >= 0 ? '+' : ''}${impact.toFixed(2)}%`}
                  >
                    <div
                      className={`w-full ${isPositive ? 'bg-red-400' : 'bg-green-400'} rounded-t`}
                      style={{ height: `${Math.min(height, 60)}px` }}
                    ></div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Climate Impact */}
        <div>
          <div className="text-sm text-slate-600">Climate Impact</div>
          <div className="flex items-baseline gap-2">
            <span className={`text-lg font-semibold ${emissions < 0 ? 'text-green-700' : 'text-slate-900'}`}>
              {emissions >= 0 ? '+' : ''}{emissions.toFixed(1)} MtCO₂e
            </span>
            <span className={`text-xs px-2 py-1 rounded ${getBudgetStatusColor()}`}>
              {climate.budget_compliance?.status || 'Unknown'}
            </span>
          </div>
        </div>

        {/* Macro Impact */}
        <div className="grid grid-cols-2 gap-4 pt-4 border-t border-slate-200">
          <div>
            <div className="text-xs text-slate-600">GDP Impact</div>
            <div className="text-sm font-semibold text-slate-900">
              {gdp >= 0 ? '+' : ''}{gdp.toFixed(2)}%
            </div>
          </div>
          <div>
            <div className="text-xs text-slate-600">Employment</div>
            <div className="text-sm font-semibold text-slate-900">
              {employment >= 0 ? '+' : ''}{(employment / 1000).toFixed(1)}M
            </div>
          </div>
        </div>

        {/* Expand Button */}
        <button
          onClick={() => setExpanded(!expanded)}
          className="w-full mt-4 px-4 py-2 text-sm text-indigo-600 hover:text-indigo-700 hover:bg-indigo-50 rounded transition-colors"
        >
          {expanded ? 'Show less' : 'Show details'} →
        </button>

        {/* Expanded Details */}
        {expanded && (
          <div className="mt-4 pt-4 border-t border-slate-200 space-y-3 text-sm">
            <div>
              <div className="font-medium text-slate-700 mb-2">Policy Levers</div>
              <ul className="space-y-1 text-slate-600">
                {plan.levers?.income_tax?.basic_rate_pp !== 0 && (
                  <li>• Basic rate: {plan.levers.income_tax.basic_rate_pp > 0 ? '+' : ''}{plan.levers.income_tax.basic_rate_pp}pp</li>
                )}
                {plan.levers?.nics?.class1_main_pp !== 0 && (
                  <li>• NICs: {plan.levers.nics.class1_main_pp > 0 ? '+' : ''}{plan.levers.nics.class1_main_pp}pp</li>
                )}
                {plan.levers?.vat?.standard_rate_pp !== 0 && (
                  <li>• VAT: {plan.levers.vat.standard_rate_pp > 0 ? '+' : ''}{plan.levers.vat.standard_rate_pp}pp</li>
                )}
                {plan.levers?.ets_carbon_price?.start_gbp_per_tco2e > 55 && (
                  <li>• Carbon price: £{plan.levers.ets_carbon_price.start_gbp_per_tco2e}/tCO₂e</li>
                )}
                {(plan.levers?.departmental || []).length > 0 && (
                  <li>• Department cuts: {plan.levers.departmental.length} departments</li>
                )}
              </ul>
            </div>

            <div>
              <div className="font-medium text-slate-700 mb-2">Gini Impact</div>
              <div className="text-slate-600">
                {distribution.gini_delta >= 0 ? '+' : ''}{distribution.gini_delta?.toFixed(4) || 0}
                {distribution.gini_delta > 0 && <span className="text-xs text-red-600 ml-2">(more unequal)</span>}
                {distribution.gini_delta < 0 && <span className="text-xs text-green-600 ml-2">(more equal)</span>}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
