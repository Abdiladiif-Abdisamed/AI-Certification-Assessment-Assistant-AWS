import { useState } from "react";
import { Bot, Eye, EyeOff, LockKeyhole, Mail, UserRound } from "lucide-react";
import { api, setToken } from "../services/api";
import "./AuthPage.css";

function AuthPage({ onAuthenticated }) {
  const [mode, setMode] = useState("login");
  const [form, setForm] = useState({ full_name: "", email: "", password: "" });
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const submit = async (event) => {
    event.preventDefault();
    setError("");
    setLoading(true);
    try {
      const payload = mode === "login"
        ? await api.login({ email: form.email, password: form.password })
        : await api.register(form);
      setToken(payload.access_token);
      onAuthenticated(payload.user);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="auth-page">
      <section className="auth-hero" aria-labelledby="auth-title">
        <div className="auth-brand-mark"><Bot aria-hidden="true" /></div>
        <p className="auth-eyebrow">AI Exam Assistant</p>
        <h1 id="auth-title">Grounded practice. Clear progress. Real readiness.</h1>
        <p>Build new practice assessments from official certification material, then turn every result into a focused study plan.</p>
        <ul>
          <li>Official-source RAG retrieval</li>
          <li>Deterministic scoring and readiness</li>
          <li>Personalized weak-area review</li>
        </ul>
      </section>

      <section className="auth-panel" aria-labelledby="auth-form-title">
        <div className="auth-card">
          <div className="auth-card-heading">
            <h2 id="auth-form-title">{mode === "login" ? "Welcome back" : "Create your account"}</h2>
            <p>{mode === "login" ? "Sign in to continue your study progress." : "Start with a secure local learner account."}</p>
          </div>

          <div className="auth-mode-switch" role="tablist" aria-label="Authentication mode">
            <button type="button" role="tab" aria-selected={mode === "login"} className={mode === "login" ? "active" : ""} onClick={() => { setMode("login"); setError(""); }}>Sign in</button>
            <button type="button" role="tab" aria-selected={mode === "register"} className={mode === "register" ? "active" : ""} onClick={() => { setMode("register"); setError(""); }}>Register</button>
          </div>

          <form onSubmit={submit} className="auth-form">
            {mode === "register" && (
              <div className="auth-field">
                <label htmlFor="full-name">Full name</label>
                <div className="auth-input-wrap"><UserRound aria-hidden="true" /><input id="full-name" autoComplete="name" value={form.full_name} onChange={(event) => setForm({ ...form, full_name: event.target.value })} minLength="2" required /></div>
              </div>
            )}
            <div className="auth-field">
              <label htmlFor="email">Email address</label>
              <div className="auth-input-wrap"><Mail aria-hidden="true" /><input id="email" type="email" autoComplete="email" value={form.email} onChange={(event) => setForm({ ...form, email: event.target.value })} required /></div>
            </div>
            <div className="auth-field">
              <label htmlFor="password">Password</label>
              <div className="auth-input-wrap"><LockKeyhole aria-hidden="true" /><input id="password" type={showPassword ? "text" : "password"} autoComplete={mode === "login" ? "current-password" : "new-password"} value={form.password} onChange={(event) => setForm({ ...form, password: event.target.value })} minLength={mode === "register" ? 8 : 1} required /><button type="button" className="password-toggle" aria-label={showPassword ? "Hide password" : "Show password"} onClick={() => setShowPassword(!showPassword)}>{showPassword ? <EyeOff aria-hidden="true" /> : <Eye aria-hidden="true" />}</button></div>
              {mode === "register" && <p className="field-help">Use at least 8 characters. Password paste and password managers are supported.</p>}
            </div>
            {error && <div className="auth-error" role="alert">{error} Check your details and try again.</div>}
            <button className="auth-submit" type="submit" disabled={loading}>{loading ? "Please wait…" : mode === "login" ? "Sign in" : "Create account"}</button>
          </form>
          <p className="auth-local-note">This account is stored in your local project database.</p>
        </div>
      </section>
    </main>
  );
}

export default AuthPage;

