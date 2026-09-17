import { useMemo, useState } from "react";
import { ArrowLeft, ChevronDown, CircleHelp, Search } from "lucide-react";
import "./HelpPage.css";

const faqs = [
  { category: "Getting Started", q: "How do I start an assessment?", a: "Open Home, choose a generation-ready certification, difficulty, question count, and study-plan length, then select Start Assessment." },
  { category: "Assessments", q: "Why is only AWS AIF-C01 generation-ready?", a: "The current AWS JSON contains detailed official source content. Other certification files currently contain scope labels, so the system blocks question generation instead of inventing knowledge." },
  { category: "Assessments", q: "Are these real exam questions?", a: "No. Every item is a new practice-only question generated only from retrieved repository context, with source chunk IDs and official URLs preserved." },
  { category: "Results", q: "How is readiness calculated?", a: "Readiness is deterministic: 80% or higher is Ready, 60–79% is Nearly Ready, and below 60% is Needs Improvement. The thresholds come from environment settings." },
  { category: "Results", q: "How are weak areas selected?", a: "Python calculates topic and domain accuracy from known answer keys. Topics below the review threshold are prioritized; the language model does not change the scores." },
  { category: "Technical", q: "The app cannot reach the API. What should I check?", a: "Confirm the FastAPI server is running at 127.0.0.1:8000 and VITE_API_URL points to its /api path. Then retry the request." },
  { category: "Technical", q: "Question generation timed out or failed. What should I do?", a: "Confirm OPENAI_API_KEY is in the root .env, the vector store is ready, and internet access is available. A failed grounding review may safely reject a batch; retry with fewer questions." },
];

function HelpPage({ setActiveTab }) {
  const [search, setSearch] = useState("");
  const [open, setOpen] = useState(null);
  const filtered = useMemo(() => faqs.filter((item) => `${item.q} ${item.a} ${item.category}`.toLowerCase().includes(search.trim().toLowerCase())), [search]);
  return <div className="help-page"><header className="help-header"><div><h1>Help & Project Guidance</h1><p className="subtitle">Accurate answers for the current local MVP.</p></div><button className="back-results-btn" type="button" onClick={() => setActiveTab("Home")}><ArrowLeft aria-hidden="true" />Back to Dashboard</button></header><div className="help-search-card"><Search aria-hidden="true" /><label className="sr-only" htmlFor="help-search">Search help</label><input id="help-search" type="search" placeholder="Search for answers…" value={search} onChange={(event) => { setSearch(event.target.value); setOpen(null); }} /></div><div className="help-layout help-layout-single"><main className="help-content"><section className="help-card"><div className="help-card-title"><CircleHelp aria-hidden="true" /><div><h2>Frequently asked questions</h2><p className="section-desc">{filtered.length} matching answer{filtered.length === 1 ? "" : "s"}</p></div></div>{filtered.length ? <div className="faq-list">{filtered.map((item, index) => <article className="faq-item" key={item.q}><button className="faq-question" type="button" aria-expanded={open === index} onClick={() => setOpen(open === index ? null : index)}><span><small>{item.category}</small>{item.q}</span><ChevronDown className={open === index ? "open" : ""} aria-hidden="true" /></button>{open === index && <div className="faq-answer"><p>{item.a}</p></div>}</article>)}</div> : <div className="empty-help"><CircleHelp aria-hidden="true" /><h3>No matching answers</h3><p>Try “API”, “readiness”, or “AWS”.</p></div>}</section><section className="help-card"><h2>Current scope boundary</h2><p className="scope-copy">Full browser end-to-end automation and deployment/production security hardening are intentionally deferred. Local auth, database persistence, RAG/OpenAI generation, assessment analytics, and frontend/API integration are implemented now.</p></section></main></div></div>;
}
export default HelpPage;
