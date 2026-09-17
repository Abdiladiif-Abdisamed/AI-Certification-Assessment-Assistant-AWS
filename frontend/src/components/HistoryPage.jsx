import { useEffect, useState } from "react";
import { ArrowLeft, Eye } from "lucide-react";
import { api } from "../services/api";
import "./HistoryPage.css";

function HistoryPage({ setActiveTab, onOpenResult }) {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [opening, setOpening] = useState("");
  useEffect(() => { api.history().then(setHistory).catch((requestError) => setError(requestError.message)).finally(() => setLoading(false)); }, []);
  const open = async (id) => { setOpening(id); setError(""); try { await onOpenResult(id); } catch (requestError) { setError(requestError.message); setOpening(""); } };
  if (loading) return <div className="page-state"><div className="loading-spinner" /><p>Loading assessment history…</p></div>;
  const completed = history.filter((item) => item.status === "completed");
  const average = completed.length ? Math.round(completed.reduce((sum, item) => sum + item.score_percentage, 0) / completed.length) : 0;
  return <div className="history-page"><header className="history-header"><div><h1>Assessment History</h1><p className="subtitle">Every saved local attempt and result.</p></div><button className="back-results-btn" type="button" onClick={() => setActiveTab("Home")}><ArrowLeft aria-hidden="true" />Back to Dashboard</button></header>{error && <div className="inline-error" role="alert">{error}</div>}<div className="history-top-stats-grid"><div className="stat-card"><p className="card-title">Total Attempts</p><p className="stat-value dark-text">{history.length}</p></div><div className="stat-card"><p className="card-title">Completed</p><p className="stat-value green-text">{completed.length}</p></div><div className="stat-card"><p className="card-title">Average Score</p><p className="stat-value dark-text">{average}%</p></div></div><section className="history-list-section"><h2>Attempts</h2>{history.length ? <div className="history-table"><div className="history-table-head"><span>Date</span><span>Certification</span><span>Difficulty</span><span>Score</span><span>Readiness</span><span>Action</span></div>{history.map((item) => <div className="history-table-row" key={item.assessment_id}><span>{new Date(item.created_at).toLocaleDateString()}</span><span>{item.certification_name}</span><span className="difficulty-badge">{item.difficulty}</span><span className={item.score_percentage >= 80 ? "score-high" : item.score_percentage >= 60 ? "score-mid" : "score-low"}>{item.score_percentage == null ? "—" : `${item.score_percentage}%`}</span><span>{item.readiness || item.status}</span><button type="button" className="view-history-btn" disabled={item.status !== "completed" || opening === item.assessment_id} onClick={() => open(item.assessment_id)}><Eye aria-hidden="true" />{opening === item.assessment_id ? "Opening…" : "View"}</button></div>)}</div> : <div className="empty-state"><h3>No attempts yet</h3><p>Start your first assessment from the dashboard.</p></div>}</section></div>;
}
export default HistoryPage;
