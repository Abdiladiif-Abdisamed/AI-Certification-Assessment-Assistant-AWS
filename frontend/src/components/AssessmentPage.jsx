import { useMemo, useState } from "react";
import { ArrowLeft, ArrowRight, Bookmark, Check, LogOut, Send } from "lucide-react";
import "./AssessmentPage.css";

function AssessmentPage({ assessment, onSubmit, onBookmark, setActiveTab }) {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [answers, setAnswers] = useState({});
  const [saved, setSaved] = useState(new Set());
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  const questions = assessment?.questions || [];
  const question = questions[currentIndex];
  const answeredCount = Object.keys(answers).length;
  const progress = questions.length ? Math.round((answeredCount / questions.length) * 100) : 0;
  const answerRows = useMemo(() => Object.entries(answers).map(([question_id, selected_answer]) => ({ question_id, selected_answer })), [answers]);

  if (!assessment || !question) return <div className="assessment-page"><div className="empty-state"><h2>No assessment in progress</h2><p>Use Quick Start to generate a new grounded practice assessment.</p><button type="button" onClick={() => setActiveTab("Home")}>Go to Quick Start</button></div></div>;

  const submit = async () => {
    if (answeredCount < questions.length && !window.confirm(`${questions.length - answeredCount} question(s) are unanswered and will count as incorrect. Submit anyway?`)) return;
    setError(""); setSubmitting(true);
    try { await onSubmit(assessment.assessment_id, answerRows); }
    catch (requestError) { setError(requestError.message); setSubmitting(false); }
  };
  const bookmark = async () => {
    setError("");
    try { await onBookmark(assessment.assessment_id, question.question_id); setSaved((current) => new Set(current).add(question.question_id)); }
    catch (requestError) { setError(requestError.message); }
  };

  return <div className="assessment-page">
    <header className="assessment-top-bar"><div className="top-bar-title"><span className="title-indicator" /><div><h1>Practice Assessment</h1><p>{assessment.certification_name}</p></div></div><button className="action-btn pause-btn" type="button" onClick={() => setActiveTab("Home")}><LogOut aria-hidden="true" />Exit to Dashboard</button></header>
    {error && <div className="inline-error" role="alert">{error} Try the action again.</div>}
    <div className="assessment-main-grid">
      <section className="exam-card" aria-labelledby="question-heading">
        <div className="exam-card-header"><div className="practice-brand"><span>AI</span></div><div className="exam-details"><h2>{assessment.certification_name}</h2><p>{assessment.difficulty} · {assessment.question_count} new practice questions</p></div></div>
        <div className="main-progress-track" aria-hidden="true"><div className="main-progress-fill" style={{ width: `${((currentIndex + 1) / questions.length) * 100}%` }} /></div><p className="question-counter">Question {currentIndex + 1} of {questions.length}</p>
        <div className="question-content-box"><p className="question-context">{question.domain} · {question.topic}</p><h3 id="question-heading" className="question-text">{question.question}</h3><div className="options-container" role="radiogroup" aria-labelledby="question-heading">{Object.entries(question.options).map(([label, text]) => { const selected = answers[question.question_id] === label; return <button key={label} type="button" role="radio" aria-checked={selected} className={`option-card ${selected ? "selected" : ""}`} onClick={() => setAnswers({ ...answers, [question.question_id]: label })}><span className={selected ? "option-radio-selected" : "option-badge"}>{selected ? <span className="radio-inner-dot" /> : label}</span><span className={`option-label ${selected ? "font-medium" : ""}`}>{text}</span></button>; })}</div><button className="bookmark-btn" type="button" disabled={saved.has(question.question_id)} onClick={bookmark}>{saved.has(question.question_id) ? <Check aria-hidden="true" /> : <Bookmark aria-hidden="true" />}{saved.has(question.question_id) ? "Saved to Bookmarks" : "Save for later"}</button></div>
        <div className="exam-footer-nav"><button className="nav-btn prev-btn" type="button" disabled={currentIndex === 0} onClick={() => setCurrentIndex((index) => index - 1)}><ArrowLeft aria-hidden="true" />Previous</button>{currentIndex < questions.length - 1 ? <button className="nav-btn next-btn" type="button" onClick={() => setCurrentIndex((index) => index + 1)}>Next<ArrowRight aria-hidden="true" /></button> : <button className="nav-btn next-btn" type="button" disabled={submitting} onClick={submit}>{submitting ? "Scoring…" : <>Submit Assessment<Send aria-hidden="true" /></>}</button>}</div>
      </section>
      <aside className="assessment-sidebar" aria-label="Question navigation and progress"><div className="sidebar-card"><h2>Question Navigator</h2><div className="navigator-grid">{questions.map((item, index) => <button key={item.question_id} type="button" aria-label={`Go to question ${index + 1}${answers[item.question_id] ? ", answered" : ""}`} aria-current={index === currentIndex ? "step" : undefined} className={`nav-number ${answers[item.question_id] ? "answered" : ""} ${index === currentIndex ? "current" : ""}`} onClick={() => setCurrentIndex(index)}>{index + 1}</button>)}</div><div className="navigator-legend"><div className="legend-row"><span className="legend-dot answered" />Answered</div><div className="legend-row"><span className="legend-dot current" />Current</div><div className="legend-row"><span className="legend-dot not-answered" />Not answered</div></div></div><div className="sidebar-card"><p className="timer-title">Assessment Progress</p><p className="timer-clock">{answeredCount} / {questions.length}</p><div className="progress-info-row"><span>Answered</span><span className="progress-percent">{progress}%</span></div><div className="progress-bar-track"><div className="progress-bar-fill" style={{ width: `${progress}%` }} /></div><button className="submit-side-button" type="button" disabled={submitting} onClick={submit}>{submitting ? "Scoring…" : "Submit assessment"}</button></div></aside>
    </div>
  </div>;
}

export default AssessmentPage;
