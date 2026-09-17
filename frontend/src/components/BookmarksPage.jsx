import { useEffect, useState } from "react";
import { ArrowLeft, Bookmark, Trash2 } from "lucide-react";
import { api } from "../services/api";
import "./BookmarksPage.css";

function BookmarksPage({ setActiveTab }) {
  const [bookmarks, setBookmarks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  useEffect(() => { api.bookmarks().then(setBookmarks).catch((requestError) => setError(requestError.message)).finally(() => setLoading(false)); }, []);
  const remove = async (id) => { try { await api.deleteBookmark(id); setBookmarks((items) => items.filter((item) => item.id !== id)); } catch (requestError) { setError(requestError.message); } };
  if (loading) return <div className="page-state"><div className="loading-spinner" /><p>Loading bookmarks…</p></div>;
  return <div className="bookmarks-page"><header className="bookmarks-header"><div><h1>Bookmarked Questions</h1><p className="subtitle">Practice-only questions saved from your assessments.</p></div><button className="back-results-btn" type="button" onClick={() => setActiveTab("Home")}><ArrowLeft aria-hidden="true" />Back to Dashboard</button></header>{error && <div className="inline-error" role="alert">{error}</div>}<div className="bookmarks-top-stats-grid"><div className="stat-card"><p className="card-title">Total Bookmarked</p><p className="stat-value dark-text">{bookmarks.length}</p></div><div className="stat-card"><p className="card-title">Topics Covered</p><p className="stat-value green-text">{new Set(bookmarks.map((item) => item.question.topic)).size}</p></div><div className="stat-card"><p className="card-title">Certifications</p><p className="stat-value dark-text">{new Set(bookmarks.map((item) => item.certification_id)).size}</p></div></div><section className="bookmark-list-section"><h2>Saved Questions</h2>{bookmarks.length ? <div className="bookmark-list">{bookmarks.map((item) => <article className="bookmark-card" key={item.id}><div className="bookmark-top"><div className="bookmark-tags"><span className="category-tag">{item.question.topic}</span><span className="cert-tag">{item.question.certification}</span><span className="status-badge badge-correct">Practice only</span></div><button className="remove-bookmark-btn" type="button" aria-label="Remove bookmark" onClick={() => remove(item.id)}><Trash2 aria-hidden="true" /></button></div><h3 className="bookmark-question">{item.question.question}</h3><div className="bookmark-options">{Object.entries(item.question.options).map(([label, text]) => <p key={label}><span>{label}</span>{text}</p>)}</div>{item.note && <p className="bookmark-note">{item.note}</p>}</article>)}</div> : <div className="empty-bookmarks"><Bookmark aria-hidden="true" /><h3>No bookmarks yet</h3><p>Save a question during an assessment and it will appear here.</p></div>}</section></div>;
}
export default BookmarksPage;

