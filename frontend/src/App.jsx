import { useEffect, useState } from "react";
import { Menu } from "lucide-react";
import Sidebar from "./components/Sidebar";
import AuthPage from "./components/AuthPage";
import DashboardHome from "./components/DashboardHome";
import AssessmentPage from "./components/AssessmentPage";
import ResultsPage from "./components/ResultsPage";
import StudyPlan from "./components/StudyPlan";
import WeakAreasPage from "./components/WeakAreasPage";
import BookmarksPage from "./components/BookmarksPage";
import HistoryPage from "./components/HistoryPage";
import SettingsPage from "./components/SettingsPage";
import HelpPage from "./components/HelpPage";
import AdminPage from "./components/AdminPage";
import { api, getToken, setToken } from "./services/api";
import "./App.css";

const TABS = ["Home", "Assessments", "Results", "Study Plan", "Weak Areas", "Bookmarks", "History", "Settings", "Help", "Admin"];
const tabHash = (tab) => `#${tab.toLowerCase().replaceAll(" ", "-")}`;
const tabFromHash = () => TABS.find((tab) => tabHash(tab) === window.location.hash) || "Home";

function App() {
  const [user, setUser] = useState(null);
  const [checkingSession, setCheckingSession] = useState(Boolean(getToken()));
  const [activeTab, setActiveTabState] = useState(tabFromHash);
  const [theme, setTheme] = useState(() => localStorage.getItem("ai-cert-theme") || "light");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [activeAssessment, setActiveAssessment] = useState(null);
  const [latestResult, setLatestResult] = useState(null);
  const [notice, setNotice] = useState("");

  useEffect(() => {
    if (!getToken()) return;
    api.me().then(setUser).catch(() => setToken("")).finally(() => setCheckingSession(false));
  }, []);

  useEffect(() => {
    const onHashChange = () => setActiveTabState(tabFromHash());
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  useEffect(() => localStorage.setItem("ai-cert-theme", theme), [theme]);

  useEffect(() => {
    if (!notice) return undefined;
    const timer = window.setTimeout(() => setNotice(""), 4500);
    return () => window.clearTimeout(timer);
  }, [notice]);

  const setActiveTab = (tab) => {
    setActiveTabState(tab);
    window.history.pushState(null, "", tabHash(tab));
    setSidebarOpen(false);
    window.requestAnimationFrame(() => document.querySelector(".main-content")?.focus());
  };

  const logout = () => {
    setToken("");
    setUser(null);
    setActiveAssessment(null);
    setLatestResult(null);
  };

  const startAssessment = async (configuration) => {
    const assessment = await api.createAssessment(configuration);
    setActiveAssessment(assessment);
    setLatestResult(null);
    setActiveTab("Assessments");
    return assessment;
  };

  const submitAssessment = async (assessmentId, answers) => {
    const result = await api.submitAssessment(assessmentId, answers);
    setLatestResult(result);
    setActiveAssessment(null);
    setActiveTab("Results");
    return result;
  };

  const bookmarkQuestion = async (assessmentId, questionId) => {
    await api.addBookmark({ assessment_id: assessmentId, question_id: questionId });
    setNotice("Question saved to Bookmarks.");
  };

  if (checkingSession) {
    return <main className="app-boot" aria-live="polite"><div className="loading-spinner" /><p>Restoring your session…</p></main>;
  }
  if (!user) return <AuthPage onAuthenticated={setUser} />;

  const pageProps = { setActiveTab };
  const renderPage = () => {
    switch (activeTab) {
      case "Home": return <DashboardHome {...pageProps} user={user} theme={theme} setTheme={setTheme} onStartAssessment={startAssessment} />;
      case "Assessments": return <AssessmentPage {...pageProps} assessment={activeAssessment} onSubmit={submitAssessment} onBookmark={bookmarkQuestion} />;
      case "Results": return <ResultsPage {...pageProps} result={latestResult} />;
      case "Study Plan": return <StudyPlan {...pageProps} />;
      case "Weak Areas": return <WeakAreasPage {...pageProps} />;
      case "Bookmarks": return <BookmarksPage {...pageProps} />;
      case "History": return <HistoryPage {...pageProps} onOpenResult={(id) => api.result(id).then((result) => { setLatestResult(result); setActiveTab("Results"); })} />;
      case "Settings": return <SettingsPage {...pageProps} user={user} theme={theme} setTheme={setTheme} />;
      case "Help": return <HelpPage {...pageProps} />;
      case "Admin":
        return user.is_admin
          ? <AdminPage />
          : <DashboardHome {...pageProps} user={user} theme={theme} setTheme={setTheme} onStartAssessment={startAssessment} />;
      default: return <DashboardHome {...pageProps} user={user} theme={theme} setTheme={setTheme} onStartAssessment={startAssessment} />;
    }
  };

  return (
    <div className="app-container" data-theme={theme}>
      <a className="skip-link" href="#main-content">Skip to main content</a>
      <button className="mobile-menu-button" type="button" aria-label="Open navigation" aria-expanded={sidebarOpen} onClick={() => setSidebarOpen(true)}><Menu aria-hidden="true" /></button>
      {sidebarOpen && <button className="sidebar-scrim" aria-label="Close navigation" onClick={() => setSidebarOpen(false)} />}
      <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} user={user} onLogout={logout} isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />
      <main id="main-content" className="main-content" tabIndex="-1">{renderPage()}</main>
      {notice && <div className="app-toast" role="status" aria-live="polite">{notice}</div>}
    </div>
  );
}

export default App;

