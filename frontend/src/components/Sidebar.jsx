import {
  AlertTriangle, BarChart2, Bookmark, BookOpen, Bot, ChevronRight,
  ClipboardList, HelpCircle, History, Home, LogOut, Settings, ShieldCheck, User, X,
} from "lucide-react";
import "./Sidebar.css";

const menuItems = [
  { name: "Home", icon: Home }, { name: "Assessments", icon: ClipboardList },
  { name: "Results", icon: BarChart2 }, { name: "Study Plan", icon: BookOpen },
  { name: "Weak Areas", icon: AlertTriangle }, { name: "Bookmarks", icon: Bookmark },
  { name: "History", icon: History }, { name: "Settings", icon: Settings },
  { name: "Help", icon: HelpCircle },
];

function Sidebar({ activeTab, setActiveTab, user, onLogout, isOpen, onClose }) {
  const items = user.is_admin
    ? [...menuItems, { name: "Admin", icon: ShieldCheck }]
    : menuItems;
  return (
    <aside className={`sidebar ${isOpen ? "mobile-open" : ""}`} aria-label="Primary navigation">
      <div className="sidebar-header">
        <div className="logo-icon"><Bot size={20} aria-hidden="true" /></div>
        <div className="logo-text"><h2>AI Exam Assistant</h2><p>Your AI-Powered Study Companion</p></div>
        <button type="button" className="sidebar-close" aria-label="Close navigation" onClick={onClose}><X aria-hidden="true" /></button>
      </div>
      <nav aria-label="Application sections">
        <ul className="sidebar-menu">
          {items.map((item) => {
            const Icon = item.icon;
            const active = activeTab === item.name;
            return <li key={item.name}><button type="button" className={active ? "active" : ""} aria-current={active ? "page" : undefined} onClick={() => setActiveTab(item.name)}><span className="icon"><Icon size={18} aria-hidden="true" /></span>{item.name}</button></li>;
          })}
        </ul>
      </nav>
      <div className="sidebar-footer">
        <div className="user-profile-card"><div className="avatar-placeholder"><User size={18} aria-hidden="true" /></div><div className="user-info"><p className="user-name">{user.full_name}</p><p className="user-role">Student</p></div><ChevronRight size={18} className="arrow" aria-hidden="true" /></div>
        <button className="logout-button" type="button" onClick={onLogout}><LogOut size={17} aria-hidden="true" />Sign out</button>
      </div>
    </aside>
  );
}

export default Sidebar;
