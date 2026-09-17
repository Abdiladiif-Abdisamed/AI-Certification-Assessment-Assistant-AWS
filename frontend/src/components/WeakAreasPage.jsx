import { useEffect, useMemo, useState } from "react";
import { ArrowLeft, ArrowRight, Target } from "lucide-react";
import { api } from "../services/api";
import "./WeakAreasPage.css";

function WeakAreasPage({ setActiveTab }) {
  const [analysis, setAnalysis] = useState(null);
  const [filter, setFilter] = useState("all");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  useEffect(() => { api.weakAreas().then(setAnalysis).catch((requestError) => setError(requestError.message)).finally(() => setLoading(false)); }, []);
  const areas = useMemo(() => analysis?.weak_areas || [], [analysis]);
  const filtered = filter === "all" ? areas : areas.filter((area) => area.priority === filter);
  const average = areas.length ? Math.round(areas.reduce((sum, area) => sum + area.accuracy, 0) / areas.length) : 0;
  if (loading) return <div className="page-state"><div className="loading-spinner" /><p>Analyzing your latest result…</p></div>;
  if (!analysis) return <div className="weak-areas-page"><div className="empty-state"><h2>No weak-area analysis yet</h2><p>{error || "Complete an assessment to identify topics for review."}</p><button type="button" onClick={() => setActiveTab("Home")}>Take an assessment</button></div></div>;
  return <div className="weak-areas-page"><header className="weak-header"><div><h1>Weak Areas</h1><p className="subtitle">Deterministic topic accuracy from your latest assessment.</p></div><button className="back-results-btn" type="button" onClick={() => setActiveTab("Home")}><ArrowLeft aria-hidden="true" />Back to Dashboard</button></header><div className="weak-top-stats-grid"><div className="stat-card"><p className="card-title">Topics to Review</p><p className="stat-value dark-text">{areas.length}</p></div><div className="stat-card"><p className="card-title">High Priority</p><p className="stat-value red-text">{areas.filter((area) => area.priority === "high").length}</p></div><div className="stat-card"><p className="card-title">Average Weak Accuracy</p><p className="stat-value orange-text">{average}%</p></div><div className="stat-card"><p className="card-title">Strongest Domains</p><p className="stat-value dark-text">{analysis.strongest_domains.length}</p></div></div><section className="weak-list-section"><div className="weak-list-header"><h2>Topics That Need Attention</h2><div className="review-filters">{["all", "high", "medium", "low"].map((value) => <button key={value} type="button" className={`filter-tab ${filter === value ? "active" : ""}`} aria-pressed={filter === value} onClick={() => setFilter(value)}>{value[0].toUpperCase() + value.slice(1)}</button>)}</div></div>{filtered.length ? <div className="weak-area-list">{filtered.map((area) => <article className="weak-area-card" key={`${area.domain}-${area.topic}`}><div className="weak-area-top"><div><div className="weak-area-title-row"><h3>{area.topic}</h3><span className={`priority-badge priority-${area.priority}`}>{area.priority}</span></div><p className="weak-area-desc">{area.domain}</p></div><div className="weak-area-score"><span className="weak-score-number">{area.accuracy}%</span><span className="weak-score-label">Accuracy</span></div></div><div className="weak-progress-track"><div className={`weak-progress-fill bar-${area.priority}`} style={{ width: `${area.accuracy}%` }} /></div><div className="weak-area-footer"><div className="weak-area-meta"><span><Target aria-hidden="true" />Review priority: {area.priority}</span></div><button className="practice-btn" type="button" onClick={() => setActiveTab("Home")}>Practice this topic<ArrowRight aria-hidden="true" /></button></div></article>)}</div> : <div className="empty-state"><h3>No topics in this filter</h3><p>Your selected priority has no weak areas.</p></div>}</section></div>;
}
export default WeakAreasPage;

