import { useEffect, useState } from "react";
import { ArrowLeft, BookOpen, CheckCircle2, ExternalLink, Target } from "lucide-react";
import { api } from "../services/api";
import "./StudyPlan.css";

function StudyPlan({ setActiveTab }) {
  const [plan, setPlan] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  useEffect(() => { api.studyPlan().then(setPlan).catch((requestError) => setError(requestError.message)).finally(() => setLoading(false)); }, []);
  if (loading) return <div className="page-state"><div className="loading-spinner" /><p>Loading your study plan…</p></div>;
  if (!plan) return <div className="study-plan-page"><div className="empty-state"><h2>No study plan yet</h2><p>{error || "A plan is created from weak areas after you complete an assessment."}</p><button type="button" onClick={() => setActiveTab("Home")}>Take an assessment</button></div></div>;
  if (!plan.days.length) return <div className="study-plan-page"><div className="empty-state"><h2>No weak-topic review is required</h2><p>Your latest assessment did not produce any topics below the review threshold. Take a fresh assessment when you want to verify consistency.</p><button type="button" onClick={() => setActiveTab("Home")}>Take another assessment</button></div></div>;
  return <div className="study-plan-page"><header className="sp-header"><div><h1>My Study Plan</h1><p className="subtitle">A deterministic plan built from your latest weak topics and official sources.</p></div><button className="sp-back-btn" type="button" onClick={() => setActiveTab("Home")}><ArrowLeft aria-hidden="true" />Back to Dashboard</button></header><div className="sp-content-grid"><section className="sp-banner"><div className="sp-banner-info"><span className="sp-badge"><Target aria-hidden="true" />{plan.available_days}-day focus plan</span><h2>{plan.certification}</h2><p>Work from highest priority topics first, then generate a fresh practice assessment to measure improvement.</p></div><button className="sp-primary-btn" type="button" onClick={() => setActiveTab("Home")}>Take Practice Assessment</button></section><section className="schedule-container"><h2>Daily learning milestones</h2><div className="schedule-list">{plan.days.map((day) => <article className="schedule-item" key={day.day}><div className="sch-status"><CheckCircle2 aria-hidden="true" /></div><div className="sch-info"><h3>Day {day.day}: {day.topic}</h3><ul>{day.activities.map((activity) => <li key={activity}>{activity}</li>)}</ul><div className="study-resources">{day.resources.map((url) => <a key={url} href={url} target="_blank" rel="noreferrer"><BookOpen aria-hidden="true" />Official source<ExternalLink aria-hidden="true" /></a>)}</div></div><span className={`sch-tag ${day.priority === "high" ? "current" : "pending-tag"}`}>{day.priority}</span></article>)}</div></section></div></div>;
}
export default StudyPlan;
