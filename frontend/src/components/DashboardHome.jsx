import { useEffect, useState } from "react";
import { ArrowRight, Award, BarChart3, Bell, CheckCircle2, ClipboardCheck, Moon, Sparkles, Sun } from "lucide-react";
import { api } from "../services/api";
import "./DashboardHome.css";

function ThemeSwitcher({ theme, setTheme }) {
  const options = [{ id: "light", label: "Light", Icon: Sun }, { id: "dark", label: "Dark", Icon: Moon }, { id: "yellow", label: "Warm", Icon: Sparkles }];
  return <div className="theme-switcher" aria-label="Color theme">{options.map(({ id, label, Icon }) => <button key={id} type="button" aria-pressed={theme === id} className={`theme-option theme-${id} ${theme === id ? "active" : ""}`} onClick={() => setTheme(id)}><span className="theme-icon"><Icon aria-hidden="true" /></span><span>{label}</span></button>)}</div>;
}

function ProgressRing({ percent = 0 }) {
  const radius = 42;
  const circumference = 2 * Math.PI * radius;
  return <div className="readiness-progress-circle" aria-label={`Latest readiness score ${percent}%`}><svg viewBox="0 0 100 100" className="progress-ring-svg" aria-hidden="true"><circle className="progress-ring-bg" cx="50" cy="50" r={radius} /><circle className="progress-ring-fill" cx="50" cy="50" r={radius} style={{ strokeDasharray: circumference, strokeDashoffset: circumference * (1 - percent / 100) }} /></svg><span className="progress-ring-text">{percent}%</span></div>;
}

const statIcons = { assessments: ClipboardCheck, average: BarChart3, best: Award, week: CheckCircle2 };
function StatCard({ type, title, value }) { const Icon = statIcons[type]; return <div className="stat-card"><span className="stat-icon"><Icon aria-hidden="true" /></span><div><p className="stat-title">{title}</p><p className="stat-number">{value}</p></div></div>; }

function DashboardHome({ user, theme, setTheme, onStartAssessment }) {
  const [certifications, setCertifications] = useState([]);
  const [summary, setSummary] = useState({ assessments_taken: 0, average_score: 0, best_score: 0, latest_score: 0, readiness_classification: "Not assessed", completed_this_week: 0 });
  const [config, setConfig] = useState({ certification_id: "", difficulty: "medium", question_count: 3, available_days: 7 });
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    Promise.all([api.certifications(), api.dashboard()])
      .then(([catalog, dashboard]) => {
        setCertifications(catalog);
        setSummary(dashboard);
        const ready = catalog.find((item) => item.question_generation_ready);
        if (ready) setConfig((current) => ({ ...current, certification_id: ready.certification_id }));
      })
      .catch((requestError) => setError(requestError.message))
      .finally(() => setLoading(false));
  }, []);

  const selected = certifications.find((item) => item.certification_id === config.certification_id);
  const start = async () => {
    setError("");
    setGenerating(true);
    try { await onStartAssessment(config); }
    catch (requestError) { setError(requestError.message); }
    finally { setGenerating(false); }
  };

  return (
    <div className="dashboard-home">
      <ThemeSwitcher theme={theme} setTheme={setTheme} />
      <div className="dashboard-header-top"><div><h1>Welcome back, {user.full_name.split(" ")[0]}</h1><p className="subtitle">Ready to build your certification confidence today?</p></div><button type="button" className="header-notification" aria-label="Notifications"><Bell aria-hidden="true" /><span className="notif-badge">1</span></button></div>

      <div className="readiness-stats-grid">
        <div className="readiness-card"><div className="readiness-info"><h2>Overall Readiness</h2><p className="readiness-label">{summary.readiness_classification}</p><p>{summary.assessments_taken ? "Based on your latest completed assessment" : "Complete an assessment to calculate readiness"}</p><button className="take-assessment-btn" type="button" onClick={() => document.getElementById("quick-start")?.scrollIntoView({ behavior: "smooth" })}>Take Assessment</button></div><ProgressRing percent={summary.latest_score} /></div>
        <div className="small-stats-grid"><StatCard type="assessments" title="Assessments Taken" value={summary.assessments_taken} /><StatCard type="average" title="Average Score" value={`${summary.average_score}%`} /><StatCard type="best" title="Best Score" value={`${summary.best_score}%`} /><StatCard type="week" title="Completed This Week" value={summary.completed_this_week} /></div>
      </div>

      <section id="quick-start" className="section-container quick-start-container" aria-labelledby="quick-start-title">
        <h2 id="quick-start-title">Quick Start</h2><p className="section-desc">Generate a new practice-only assessment from retrieved official content.</p>
        {loading ? <div className="page-state compact"><div className="loading-spinner" /><p>Loading certification catalog…</p></div> : <div className="quick-start-form">
          <div className="form-group"><label htmlFor="certification">Certification</label><select id="certification" value={config.certification_id} onChange={(event) => setConfig({ ...config, certification_id: event.target.value })}>{certifications.map((cert) => <option key={cert.certification_id} value={cert.certification_id} disabled={!cert.question_generation_ready}>{cert.certification_name}{cert.question_generation_ready ? "" : " — content not ready"}</option>)}</select></div>
          <div className="form-group"><label htmlFor="difficulty">Difficulty</label><select id="difficulty" value={config.difficulty} onChange={(event) => setConfig({ ...config, difficulty: event.target.value })}><option value="easy">Easy</option><option value="medium">Medium</option><option value="hard">Hard</option></select></div>
          <div className="form-group"><label htmlFor="question-count">Questions</label><select id="question-count" value={config.question_count} onChange={(event) => setConfig({ ...config, question_count: Number(event.target.value) })}><option value="3">3 Questions</option><option value="5">5 Questions</option><option value="10">10 Questions</option></select></div>
          <div className="form-group"><label htmlFor="study-days">Study plan days</label><select id="study-days" value={config.available_days} onChange={(event) => setConfig({ ...config, available_days: Number(event.target.value) })}><option value="3">3 Days</option><option value="7">7 Days</option><option value="14">14 Days</option></select></div>
          <button className="start-assessment-btn" type="button" disabled={!selected?.question_generation_ready || generating} onClick={start}>{generating ? "Retrieving and generating…" : <>Start Assessment <ArrowRight aria-hidden="true" /></>}</button>
        </div>}
        {selected && !selected.question_generation_ready && <p className="corpus-warning">{selected.warning}</p>}
        {generating && <p className="generation-note" role="status">Grounded generation performs schema, citation, lexical, and semantic checks. This may take a few minutes.</p>}
        {error && <div className="inline-error" role="alert">{error} Confirm the backend and OpenAI key are available, then retry.</div>}
      </section>

      <section className="section-container" aria-labelledby="catalog-title"><h2 id="catalog-title">Certification Corpus</h2><div className="certifications-grid">{certifications.map((cert) => <article className="cert-card" key={cert.certification_id}><div className="cert-header-row"><div className={`cert-icon cert-icon-${cert.provider.toLowerCase().replaceAll(" ", "-")}`}><span>{cert.provider.slice(0, 1)}</span></div><h3>{cert.certification_name}</h3></div><span className={`active-badge ${cert.question_generation_ready ? "" : "not-ready"}`}>{cert.question_generation_ready ? "Generation ready" : "Needs detailed content"}</span><p className="cert-subtitle">{cert.provider}</p><div className="cert-footer-info"><span>{cert.domains} domains</span><span>{cert.detailed_chunks}/{cert.chunks} detailed chunks</span></div></article>)}</div></section>
    </div>
  );
}

export default DashboardHome;
