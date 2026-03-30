// frontend/src/pages/index.tsx
// Main dashboard — NYC Gentrification Early Warning System

import { useState, useEffect, useCallback } from "react";
import dynamic from "next/dynamic";
import Head from "next/head";
import { AlertTriangle, TrendingUp, MapPin, Activity, Info } from "lucide-react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from "recharts";

// Leaflet must be dynamically imported (no SSR)
const MapView = dynamic(() => import("../components/MapView"), { ssr: false });

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ── Types ──────────────────────────────────────────────────────────────────
interface Neighborhood {
  name: string;
  borough: string;
  lat: number;
  lng: number;
  risk_score: number;
  trend: string;
  top_signals?: string[];
  rent_12m?: number[];
  description?: string;
}

// ── Risk color helpers ─────────────────────────────────────────────────────
export function riskColor(score: number) {
  if (score >= 0.7) return "#FF4444";
  if (score >= 0.45) return "#FF9900";
  return "#22C55E";
}

export function riskLabel(score: number) {
  if (score >= 0.7) return "HIGH";
  if (score >= 0.45) return "MEDIUM";
  return "LOW";
}

// ── Risk Gauge ─────────────────────────────────────────────────────────────
function RiskGauge({ score }: { score: number }) {
  const pct = Math.round(score * 100);
  const color = riskColor(score);
  const circumference = 2 * Math.PI * 38;
  const strokeDash = (pct / 100) * circumference;

  return (
    <div className="flex flex-col items-center gap-1">
      <svg width="100" height="100" viewBox="0 0 100 100">
        <circle cx="50" cy="50" r="38" fill="none" stroke="#1e293b" strokeWidth="8" />
        <circle
          cx="50" cy="50" r="38"
          fill="none"
          stroke={color}
          strokeWidth="8"
          strokeDasharray={`${strokeDash} ${circumference}`}
          strokeLinecap="round"
          transform="rotate(-90 50 50)"
          style={{ transition: "stroke-dasharray 0.6s ease" }}
        />
        <text x="50" y="46" textAnchor="middle" fill="white" fontSize="18" fontWeight="bold">
          {pct}
        </text>
        <text x="50" y="60" textAnchor="middle" fill="#94a3b8" fontSize="9">
          RISK SCORE
        </text>
      </svg>
      <span
        className="text-xs font-bold tracking-widest px-2 py-0.5 rounded"
        style={{ color, background: `${color}22`, border: `1px solid ${color}44` }}
      >
        {riskLabel(score)} RISK
      </span>
    </div>
  );
}

// ── Rent Trend Chart ───────────────────────────────────────────────────────
function RentTrend({ data, color }: { data: number[]; color: string }) {
  const months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
  const chartData = data.map((v, i) => ({ month: months[i], rent: v }));

  return (
    <ResponsiveContainer width="100%" height={90}>
      <LineChart data={chartData}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
        <XAxis dataKey="month" tick={{ fontSize: 9, fill: "#64748b" }} />
        <YAxis domain={["auto", "auto"]} tick={{ fontSize: 9, fill: "#64748b" }} width={40}
          tickFormatter={(v) => `$${(v/1000).toFixed(1)}k`} />
        <Tooltip
          contentStyle={{ background: "#0f172a", border: "1px solid #334155", borderRadius: 6 }}
          labelStyle={{ color: "#94a3b8", fontSize: 10 }}
          formatter={(v: number) => [`$${v.toLocaleString()}`, "Median Rent"]}
        />
        <Line type="monotone" dataKey="rent" stroke={color} strokeWidth={2} dot={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

// ── Sidebar ────────────────────────────────────────────────────────────────
function Sidebar({ hood, onClose }: { hood: Neighborhood; onClose: () => void }) {
  const color = riskColor(hood.risk_score);
  const trendIcon = hood.trend === "accelerating"
    ? "🔴" : hood.trend === "rising" ? "🟠" : "🟢";

  return (
    <div
      className="absolute top-0 right-0 h-full w-80 z-[1000] flex flex-col overflow-y-auto"
      style={{ background: "#0a0f1e", borderLeft: "1px solid #1e293b" }}
    >
      {/* Header */}
      <div className="flex items-start justify-between p-5 pb-3"
           style={{ borderBottom: "1px solid #1e293b" }}>
        <div>
          <div className="flex items-center gap-2 mb-1">
            <MapPin size={12} color={color} />
            <span className="text-xs text-slate-400">{hood.borough}</span>
          </div>
          <h2 className="text-xl font-bold text-white leading-tight">{hood.name}</h2>
          <div className="flex items-center gap-2 mt-1">
            <span className="text-xs text-slate-400">{trendIcon} {hood.trend}</span>
          </div>
        </div>
        <button onClick={onClose} className="text-slate-500 hover:text-white text-lg leading-none mt-1">×</button>
      </div>

      {/* Gauge */}
      <div className="flex justify-center py-4" style={{ borderBottom: "1px solid #1e293b" }}>
        <RiskGauge score={hood.risk_score} />
      </div>

      {/* Description */}
      {hood.description && (
        <div className="px-5 py-4" style={{ borderBottom: "1px solid #1e293b" }}>
          <div className="flex items-center gap-1.5 mb-2">
            <Info size={11} color="#64748b" />
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Analysis</span>
          </div>
          <p className="text-sm text-slate-300 leading-relaxed">{hood.description}</p>
        </div>
      )}

      {/* Signals */}
      {hood.top_signals && (
        <div className="px-5 py-4" style={{ borderBottom: "1px solid #1e293b" }}>
          <div className="flex items-center gap-1.5 mb-3">
            <Activity size={11} color="#64748b" />
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Top Signals</span>
          </div>
          <div className="flex flex-col gap-2">
            {hood.top_signals.map((s, i) => (
              <div key={i} className="flex items-center gap-2">
                <div className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ background: color }} />
                <span className="text-sm text-slate-300">{s}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Rent Trend */}
      {hood.rent_12m && (
        <div className="px-5 py-4">
          <div className="flex items-center gap-1.5 mb-3">
            <TrendingUp size={11} color="#64748b" />
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">12-Month Rent Trend</span>
          </div>
          <RentTrend data={hood.rent_12m} color={color} />
        </div>
      )}
    </div>
  );
}

// ── Main Page ──────────────────────────────────────────────────────────────
export default function Home() {
  const [neighborhoods, setNeighborhoods] = useState<Neighborhood[]>([]);
  const [selected, setSelected] = useState<Neighborhood | null>(null);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<"all" | "high" | "medium" | "low">("all");

  useEffect(() => {
    fetch(`${API_BASE}/neighborhoods`)
      .then((r) => r.json())
      .then(setNeighborhoods)
      .catch(() => {
        // Fallback mock data if API is not running
        setNeighborhoods([
          { name: "Bushwick", borough: "Brooklyn", lat: 40.6944, lng: -73.9213, risk_score: 0.82, trend: "accelerating",
            top_signals: ["rent_yoy (+14%)", "new_licenses (+8)", "permit_intensity high"],
            rent_12m: [2100,2120,2145,2160,2185,2210,2230,2255,2280,2310,2345,2390],
            description: "Rent YoY outpacing borough median by 2.3×. High permit activity signals gut renovations." },
          { name: "Mott Haven", borough: "Bronx", lat: 40.8092, lng: -73.9236, risk_score: 0.78, trend: "accelerating",
            top_signals: ["rent_yoy (+11%)", "new_licenses surging", "housing_complaints declining"],
            rent_12m: [1650,1670,1695,1720,1750,1780,1810,1840,1875,1910,1950,1995],
            description: "South Bronx waterfront development driving displacement pressure." },
          { name: "Ridgewood", borough: "Queens", lat: 40.7059, lng: -73.9088, risk_score: 0.69, trend: "rising",
            top_signals: ["rent_3m_momentum +6%", "new_licenses +5", "demo_shift +0.15"],
            rent_12m: [1800,1815,1830,1850,1870,1890,1910,1935,1960,1985,2010,2045],
            description: "Spillover from Bushwick driving rapid appreciation." },
          { name: "East New York", borough: "Brooklyn", lat: 40.6528, lng: -73.8826, risk_score: 0.71, trend: "rising",
            top_signals: ["permit_intensity rising", "income_index +12%", "demo_shift accelerating"],
            rent_12m: [1500,1510,1525,1530,1545,1560,1580,1590,1610,1630,1660,1695],
            description: "Early signals emerging post-2022. Rezoning activity driving interest." },
          { name: "Jackson Heights", borough: "Queens", lat: 40.7557, lng: -73.8830, risk_score: 0.65, trend: "rising",
            top_signals: ["permit_intensity high", "income_index rising", "rent_vs_nyc +0.8"],
            rent_12m: [1750,1760,1775,1790,1810,1830,1850,1870,1895,1920,1945,1975],
            description: "Transit hub status attracting displacement pressure from LIC." },
          { name: "Washington Heights", borough: "Manhattan", lat: 40.8417, lng: -73.9394, risk_score: 0.55, trend: "steady",
            top_signals: ["rent_yoy moderate", "permit_intensity moderate", "demo_shift stable"],
            rent_12m: [2000,2010,2020,2030,2045,2055,2065,2080,2090,2105,2115,2130],
            description: "Moderate and stable risk. Community land trusts providing buffer." },
          { name: "South Bronx", borough: "Bronx", lat: 40.8122, lng: -73.9198, risk_score: 0.48, trend: "steady",
            top_signals: ["housing_complaints high", "income_index low", "permit_intensity low"],
            rent_12m: [1400,1405,1415,1420,1430,1440,1450,1460,1470,1480,1490,1500],
            description: "Lower risk currently but infrastructure investment could accelerate signals." },
          { name: "Flatbush", borough: "Brooklyn", lat: 40.6421, lng: -73.9616, risk_score: 0.44, trend: "stable",
            top_signals: ["rent_yoy modest", "new_licenses moderate", "demo_shift low"],
            rent_12m: [1850,1855,1860,1870,1875,1885,1890,1900,1910,1920,1930,1940],
            description: "Strong community organizations keeping displacement pressure lower." },
        ]);
      })
      .finally(() => setLoading(false));
  }, []);

  const selectNeighborhood = useCallback(async (hood: Neighborhood) => {
    try {
      const res = await fetch(`${API_BASE}/neighborhoods/${hood.name}`);
      if (res.ok) {
        const detail = await res.json();
        setSelected({ ...hood, ...detail });
      } else {
        setSelected(hood);
      }
    } catch {
      setSelected(hood);
    }
  }, []);

  const filtered = neighborhoods.filter((n) => {
    if (filter === "high") return n.risk_score >= 0.7;
    if (filter === "medium") return n.risk_score >= 0.45 && n.risk_score < 0.7;
    if (filter === "low") return n.risk_score < 0.45;
    return true;
  });

  const highRisk = neighborhoods.filter((n) => n.risk_score >= 0.7).length;
  const medRisk  = neighborhoods.filter((n) => n.risk_score >= 0.45 && n.risk_score < 0.7).length;
  const lowRisk  = neighborhoods.filter((n) => n.risk_score < 0.45).length;

  return (
    <>
      <Head>
        <title>NYC Gentrification Watch</title>
        <meta name="description" content="Early warning system for gentrification risk across NYC neighborhoods" />
        <link rel="icon" href="/favicon.ico" />
        <link
          rel="stylesheet"
          
        />
      </Head>

      <div
        className="flex flex-col h-screen text-white"
        style={{ background: "#060d1a", fontFamily: "'DM Sans', 'Inter', sans-serif" }}
      >
        {/* Top Bar */}
        <header
          className="flex items-center justify-between px-6 py-3 flex-shrink-0"
          style={{ background: "#0a0f1e", borderBottom: "1px solid #1e293b" }}
        >
          <div className="flex items-center gap-3">
            <AlertTriangle size={18} color="#FF4444" />
            <div>
              <h1 className="text-sm font-bold tracking-wide">NYC GENTRIFICATION WATCH</h1>
              <p className="text-xs text-slate-500">LSTM-powered early warning · Updated monthly</p>
            </div>
          </div>

          {/* Stats pills */}
          <div className="hidden md:flex items-center gap-3">
            {[
              { label: "HIGH RISK", count: highRisk, color: "#FF4444", filterKey: "high" as const },
              { label: "MEDIUM",    count: medRisk,  color: "#FF9900", filterKey: "medium" as const },
              { label: "STABLE",    count: lowRisk,  color: "#22C55E", filterKey: "low" as const },
            ].map(({ label, count, color, filterKey }) => (
              <button
                key={label}
                onClick={() => setFilter(filter === filterKey ? "all" : filterKey)}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-bold transition-all"
                style={{
                  background: filter === filterKey ? `${color}22` : "transparent",
                  border: `1px solid ${filter === filterKey ? color : "#1e293b"}`,
                  color: filter === filterKey ? color : "#64748b",
                }}
              >
                <span
                  className="w-2 h-2 rounded-full"
                  style={{ background: color }}
                />
                {count} {label}
              </button>
            ))}
          </div>

          <div className="text-xs text-slate-600">
            Data: NYC Open Data · ACS Census · Zillow ZORI
          </div>
        </header>

        {/* Main: Map + Sidebar */}
        <div className="flex flex-1 relative overflow-hidden">
          {/* Map */}
          <div className="flex-1 relative">
            {loading ? (
              <div className="flex items-center justify-center h-full">
                <div className="text-slate-500 text-sm">Loading neighborhoods...</div>
              </div>
            ) : (
              <MapView
                neighborhoods={filtered}
                selected={selected}
                onSelect={selectNeighborhood}
              />
            )}

            {/* Legend */}
            <div
              className="absolute bottom-6 left-6 z-[999] p-3 rounded-lg text-xs"
              style={{ background: "#0a0f1eee", border: "1px solid #1e293b" }}
            >
              <div className="text-slate-400 font-semibold mb-2 uppercase tracking-wider text-[10px]">Risk Level</div>
              {[
                { color: "#FF4444", label: "High  ≥ 70" },
                { color: "#FF9900", label: "Medium 45–70" },
                { color: "#22C55E", label: "Low  < 45" },
              ].map(({ color, label }) => (
                <div key={label} className="flex items-center gap-2 mb-1">
                  <div className="w-3 h-3 rounded-full" style={{ background: color }} />
                  <span className="text-slate-300">{label}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Sidebar */}
          {selected && (
            <div className="relative w-80 flex-shrink-0 hidden md:block"
                 style={{ borderLeft: "1px solid #1e293b" }}>
              <Sidebar hood={selected} onClose={() => setSelected(null)} />
            </div>
          )}
        </div>

        {/* Mobile selected card */}
        {selected && (
          <div className="md:hidden fixed bottom-0 left-0 right-0 z-[1000] p-4"
               style={{ background: "#0a0f1e", borderTop: "1px solid #1e293b" }}>
            <div className="flex items-center justify-between">
              <div>
                <div className="font-bold text-white">{selected.name}</div>
                <div className="text-xs text-slate-400">{selected.borough}</div>
              </div>
              <RiskGauge score={selected.risk_score} />
              <button onClick={() => setSelected(null)} className="text-slate-400 text-xl">×</button>
            </div>
          </div>
        )}
      </div>
    </>
  );
}
