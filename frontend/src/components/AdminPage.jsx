import { useEffect, useState } from "react";
import {
  Users, ClipboardList, BarChart3, BookOpen, ShieldCheck,
  ChevronDown, X, CheckCircle2, XCircle,
} from "lucide-react";
import { api } from "../services/api";
import "./AdminPage.css";

const TABS = [
  { id: "overview", label: "Overview" },
  { id: "attempts", label: "Attempts" },
  { id: "users", label: "Users" },
  { id: "exams", label: "Exams" },
];

function StatCard({ Icon, title, value }) {
  return (
    <div className="admin-stat-card">
      <span className="admin-stat-icon"><Icon aria-hidden="true" /></span>
      <div>
        <p className="admin-stat-title">{title}</p>
        <p className="admin-stat-number">{value}</p>
      </div>
    </div>
  );
}

function ReadinessTag({ value }) {
  const tone = value === "Ready" ? "ready" : value === "Nearly Ready" ? "nearly" : "needs";
  return <span className={`admin-tag admin-tag-${tone}`}>{value}</span>;
}

function fmtDate(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? "—"
    : d.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}

function AdminPage() {
  const [tab, setTab] = useState("overview");
  const [overview, setOverview] = useState(null);
  const [attempts, setAttempts] = useState(null);
  const [users, setUsers] = useState(null);
  const [exams, setExams] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  const [detail, setDetail] = useState(null);       // { attempt, result }
  const [detailLoading, setDetailLoading] = useState(false);
  const [openCert, setOpenCert] = useState(null);   // certification_id
  const [certDetail, setCertDetail] = useState({}); // id -> detail

  useEffect(() => {
    Promise.all([api.adminOverview(), api.adminAttempts(), api.adminUsers(), api.adminExams()])
      .then(([o, a, u, e]) => {
        setOverview(o);
        setAttempts(a.attempts);
        setUsers(u.users);
        setExams(e.exams);
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  const viewAttempt = async (attempt) => {
    setDetail({ attempt, result: null });
    setDetailLoading(true);
    try {
      const result = await api.adminAssessment(attempt.assessment_id);
      setDetail({ attempt, result });
    } catch (err) {
      setError(err.message);
      setDetail(null);
    } finally {
      setDetailLoading(false);
    }
  };

  const toggleCert = async (id) => {
    if (openCert === id) { setOpenCert(null); return; }
    setOpenCert(id);
    if (!certDetail[id]) {
      try {
        const d = await api.adminCertification(id);
        setCertDetail((prev) => ({ ...prev, [id]: d }));
      } catch (err) { setError(err.message); }
    }
  };

  if (loading) return <div className="admin-page"><div className="admin-boot"><div className="loading-spinner" /><p>Loading admin console…</p></div></div>;
  if (error) return <div className="admin-page"><div className="admin-error" role="alert">{error}</div></div>;

  return (
    <div className="admin-page">
      <header className="admin-header">
        <span className="admin-badge-icon"><ShieldCheck aria-hidden="true" /></span>
        <div><h1>Admin Console</h1><p className="admin-subtitle">Manage certifications, users, and every learner's results.</p></div>
      </header>

      <div className="admin-tabs" role="tablist">
        {TABS.map((t) => (
          <button key={t.id} role="tab" aria-selected={tab === t.id}
            className={`admin-tab ${tab === t.id ? "active" : ""}`} onClick={() => setTab(t.id)}>
            {t.label}
          </button>
        ))}
      </div>

      {tab === "overview" && overview && (
        <div className="admin-section">
          <div className="admin-stats-grid">
            <StatCard Icon={Users} title="Total users" value={overview.stats.total_users} />
            <StatCard Icon={ClipboardList} title="Assessments taken" value={overview.stats.total_assessments} />
            <StatCard Icon={BarChart3} title="Average score" value={`${overview.stats.average_score}%`} />
          </div>
          <h2 className="admin-h2">By certification</h2>
          <div className="admin-card">
            <table className="admin-table">
              <thead><tr><th>Certification</th><th className="num">Attempts</th><th className="num">Avg score</th></tr></thead>
              <tbody>
                {overview.per_certification.length === 0
                  ? <tr><td colSpan={3} className="admin-empty">No activity yet.</td></tr>
                  : overview.per_certification.map((c) => (
                    <tr key={c.certification_id}><td>{c.certification_name}</td><td className="num">{c.attempts}</td><td className="num">{c.average_score}%</td></tr>
                  ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {tab === "attempts" && attempts && (
        <div className="admin-section">
          <p className="admin-note">Every learner's exam. Open one to see their full result and answers.</p>
          <div className="admin-card">
            <table className="admin-table">
              <thead><tr><th>Learner</th><th>Certification</th><th>Difficulty</th><th className="num">Score</th><th>Readiness</th><th>Date</th><th></th></tr></thead>
              <tbody>
                {attempts.length === 0 ? <tr><td colSpan={7} className="admin-empty">No attempts yet.</td></tr>
                  : attempts.map((a) => (
                    <tr key={a.assessment_id}>
                      <td><div className="admin-learner"><span>{a.user_name}</span><span className="admin-learner-email">{a.user_email}</span></div></td>
                      <td>{a.certification_name}</td>
                      <td className="admin-cap">{a.difficulty}</td>
                      <td className="num">{a.status === "completed" ? `${a.score_percentage ?? 0}%` : "—"}</td>
                      <td>{a.status === "completed" ? <ReadinessTag value={a.readiness} /> : <span className="admin-tag admin-tag-nearly">in progress</span>}</td>
                      <td>{fmtDate(a.completed_at || a.created_at)}</td>
                      <td>{a.status === "completed" && <button className="admin-link-btn" onClick={() => viewAttempt(a)}>View</button>}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {tab === "users" && users && (
        <div className="admin-section">
          <div className="admin-card">
            <table className="admin-table">
              <thead><tr><th>Email</th><th>Name</th><th className="num">Tests</th><th className="num">Best</th><th className="num">Avg</th><th></th></tr></thead>
              <tbody>
                {users.map((u) => (
                  <tr key={u.id}>
                    <td>{u.email}{u.is_admin && <span className="admin-pill">admin</span>}{!u.is_active && <span className="admin-pill inactive">inactive</span>}</td>
                    <td>{u.full_name}</td>
                    <td className="num">{u.attempts}</td>
                    <td className="num">{u.best_score}%</td>
                    <td className="num">{u.average_score}%</td>
                    <td>{u.attempts > 0 && <button className="admin-link-btn" onClick={() => { setTab("attempts"); }}>See attempts</button>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {tab === "exams" && exams && (
        <div className="admin-section admin-exams">
          {exams.map((e) => {
            const open = openCert === e.certification_id;
            const cd = certDetail[e.certification_id];
            return (
              <div key={e.certification_id} className="admin-card admin-exam-card">
                <button className="admin-exam-head-btn" onClick={() => toggleCert(e.certification_id)} aria-expanded={open}>
                  <div className="admin-exam-head">
                    <span className="admin-exam-icon"><BookOpen aria-hidden="true" /></span>
                    <div>
                      <h3>{e.certification_name}</h3>
                      <p className="admin-exam-meta">{e.provider} · {e.domains} domains · {e.topics} topics · {e.attempts} attempts</p>
                    </div>
                  </div>
                  <div className="admin-exam-right">
                    <span className={`admin-tag ${e.question_generation_ready ? "admin-tag-ready" : "admin-tag-needs"}`}>{e.question_generation_ready ? "ready" : "not ready"}</span>
                    <ChevronDown className={`admin-chevron ${open ? "open" : ""}`} aria-hidden="true" />
                  </div>
                </button>
                {open && (
                  <div className="admin-exam-domains">
                    {!cd ? <p className="admin-empty">Loading domains…</p>
                      : cd.domains.length === 0 ? <p className="admin-empty">No domain breakdown available.</p>
                        : cd.domains.map((d) => (
                          <div key={d.name} className="admin-domain-row">
                            <span className="admin-domain-name">{d.name}</span>
                            <span className="admin-domain-topics">{d.topics} topics</span>
                            <span className="admin-domain-weight">{d.weight || "—"}</span>
                          </div>
                        ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {detail && (
        <ResultModal
          detail={detail}
          loading={detailLoading}
          onClose={() => setDetail(null)}
        />
      )}
    </div>
  );
}

function ResultModal({ detail, loading, onClose }) {
  const { attempt, result } = detail;
  return (
    <div className="admin-modal-scrim" onClick={onClose}>
      <div className="admin-modal" role="dialog" aria-modal="true" onClick={(e) => e.stopPropagation()}>
        <div className="admin-modal-head">
          <div>
            <h2>{attempt.user_name}&rsquo;s result</h2>
            <p className="admin-modal-sub">{attempt.user_email} · {attempt.certification_name}</p>
          </div>
          <button className="admin-modal-close" aria-label="Close" onClick={onClose}><X aria-hidden="true" /></button>
        </div>

        <div className="admin-modal-stats">
          <div><span className="admin-modal-num">{attempt.score_percentage ?? 0}%</span><span className="admin-modal-lbl">Score</span></div>
          <div><ReadinessTag value={attempt.readiness} /><span className="admin-modal-lbl">Readiness</span></div>
          <div><span className="admin-modal-num admin-cap">{attempt.difficulty}</span><span className="admin-modal-lbl">Difficulty</span></div>
        </div>

        <div className="admin-modal-body">
          {loading || !result ? <div className="admin-boot"><div className="loading-spinner" /><p>Loading answers…</p></div>
            : (
              <ol className="admin-review">
                {result.question_review.map((q) => (
                  <li key={q.question_id} className={`admin-review-item ${q.is_correct ? "correct" : "incorrect"}`}>
                    <div className="admin-review-top">
                      {q.is_correct ? <CheckCircle2 className="ic-ok" aria-hidden="true" /> : <XCircle className="ic-bad" aria-hidden="true" />}
                      <span className="admin-review-q">{q.position}. {q.question}</span>
                    </div>
                    <div className="admin-review-meta">{q.domain} · {q.topic}</div>
                    <div className="admin-review-answers">
                      <span>Selected: <b>{q.selected_answer || "—"}</b></span>
                      <span>Correct: <b>{q.correct_answer}</b></span>
                    </div>
                    {q.explanation && <p className="admin-review-exp">{q.explanation}</p>}
                  </li>
                ))}
              </ol>
            )}
        </div>
      </div>
    </div>
  );
}

export default AdminPage;
