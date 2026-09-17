import { useEffect, useState } from "react";
import { ArrowLeft, Award, BookOpen, ChevronDown, ExternalLink } from "lucide-react";
import { api } from "../services/api";
import "./ResultsPage.css";

function ResultsPage({ result, setActiveTab }) {
  const [fetchedResult, setFetchedResult] = useState(null);
  const [loading, setLoading] = useState(!result);
  const [error, setError] = useState("");
  const [filter, setFilter] = useState("All");
  const [expanded, setExpanded] = useState(new Set());

  useEffect(() => {
    if (result) return;
    api.latestResult().then(setFetchedResult).catch((requestError) => setError(requestError.message)).finally(() => setLoading(false));
  }, [result]);

  const data = result || fetchedResult;

  if (loading) return <div className="page-state"><div className="loading-spinner" /><p>Loading your latest result…</p></div>;
  if (!data) return <div className="results-page"><div className="empty-state"><h2>No completed result yet</h2><p>{error || "Complete a practice assessment to unlock scoring and readiness."}</p><button type="button" onClick={() => setActiveTab("Home")}>Start an assessment</button></div></div>;

  const score = data.result.score_percentage;
  const reviews = data.question_review || [];
  const filtered = reviews.filter((item) => filter === "All" || (filter === "Correct" ? item.is_correct : !item.is_correct));
  const counts = { All: reviews.length, Correct: data.result.correct, Incorrect: data.result.incorrect };
  const toggle = (id) => setExpanded((current) => { const next = new Set(current); if (next.has(id)) next.delete(id); else next.add(id); return next; });

  return <div className="results-page">
    <header className="results-header"><div><h1>Assessment Results</h1><p className="subtitle">{data.certification_name} · {data.difficulty}</p></div><button className="back-results-btn" type="button" onClick={() => setActiveTab("Home")}><ArrowLeft aria-hidden="true" />Back to Dashboard</button></header>
    <div className="top-stats-grid"><div className="stat-card score-summary-card"><p className="card-title">Overall Score</p><div className="score-circle-wrapper" style={{ "--score": `${score * 3.6}deg` }}><div className="score-inner-text"><span className="big-percent">{score}%</span><span className="fraction-text">{data.result.correct} / {data.result.total_questions}</span></div></div><p className="score-comment">{data.readiness.classification}</p></div><div className="stat-card"><p className="card-title">Correct Answers</p><p className="stat-value green-text">{data.result.correct}</p></div><div className="stat-card"><p className="card-title">Incorrect Answers</p><p className="stat-value red-text">{data.result.incorrect}</p></div><div className="stat-card"><p className="card-title">Unanswered</p><p className="stat-value dark-text">{reviews.filter((item) => !item.selected_answer).length}</p></div><div className="stat-card"><p className="card-title">Practice only</p><p className="time-value">Verified</p><p className="time-sub">New grounded questions</p></div></div>

    <div className="analytics-grid results-live-grid"><section className="analytics-card"><h2>Performance by Domain</h2>{Object.entries(data.result.performance_by_domain).map(([domain, performance]) => <div className="domain-item" key={domain}><div className="domain-info"><span>{domain}</span><span>{performance.accuracy}%</span></div><div className="domain-track"><div className="domain-fill" style={{ width: `${performance.accuracy}%` }} /></div></div>)}</section><section className="analytics-card"><h2>Recommended Review</h2>{data.recommendations.recommendations.length ? <div className="recommendation-list">{data.recommendations.recommendations.map((item) => <article key={`${item.what_to_study}-${item.priority}`} className="recommendation-row"><BookOpen aria-hidden="true" /><div><h3>{item.what_to_study}</h3><p>{item.reason}</p>{item.recommended_source && <a href={item.recommended_source} target="_blank" rel="noreferrer">Official source <ExternalLink aria-hidden="true" /></a>}</div><span className={`priority-chip ${item.priority}`}>{item.priority}</span></article>)}</div> : <p className="analytics-empty">No weak topics were detected in this assessment.</p>}</section><section className="analytics-card center-card"><h2>Readiness Level</h2><div className="badge-illustration"><div className="medal-circle"><Award aria-hidden="true" /></div></div><h3 className="readiness-status">{data.readiness.classification}</h3><p className="readiness-desc">Score {data.readiness.score_percentage}% · Ready at {data.readiness.ready_threshold}%</p><button type="button" className="study-plan-link" onClick={() => setActiveTab("Study Plan")}>Open study plan</button></section></div>

    <section className="question-review-section" aria-labelledby="review-title"><h2 id="review-title">Question Review</h2><div className="review-filters" role="group" aria-label="Question result filter">{Object.keys(counts).map((tab) => <button key={tab} type="button" className={`filter-tab ${filter === tab ? "active" : ""}`} aria-pressed={filter === tab} onClick={() => setFilter(tab)}>{tab} ({counts[tab]})</button>)}</div><div className="review-list">{filtered.map((item) => <article className="review-item-card" key={item.question_id}><div className="review-item-top"><span className={item.is_correct ? "badge-correct" : "badge-incorrect"}>{item.is_correct ? "Correct" : "Incorrect"}</span><h3 className="question-code">Q{item.position}. {item.question}</h3></div><div className="review-answers-box"><p className="user-ans"><span>Your answer:</span> {item.selected_answer ? `${item.selected_answer}. ${item.options[item.selected_answer]}` : "Unanswered"}</p><p className="correct-ans"><span>Correct answer:</span> {item.correct_answer}. {item.options[item.correct_answer]}</p></div><button className="view-explanation-btn" type="button" aria-expanded={expanded.has(item.question_id)} onClick={() => toggle(item.question_id)}>Explanation and sources <ChevronDown aria-hidden="true" /></button>{expanded.has(item.question_id) && <div className="explanation-panel"><p>{item.explanation}</p><div className="source-links">{item.source_urls.map((url) => <a key={url} href={url} target="_blank" rel="noreferrer">Official source <ExternalLink aria-hidden="true" /></a>)}</div><p className="practice-label">Practice-only question · Evidence: {item.source_chunk_ids.join(", ")}</p></div>}</article>)}</div></section>
  </div>;
}

export default ResultsPage;
