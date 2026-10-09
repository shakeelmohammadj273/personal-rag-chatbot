import { useEffect, useRef, useState } from "react";
import { PROFILE } from "./profile";

const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const SUGGESTIONS = [
  "What is your current job?",
  "Tell me about yourself",
  "What are your technical skills?",
  "What projects have you worked on?",
  "What is your educational background?",
];

const Icon = ({ children }) => (
  <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor"
       strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{children}</svg>
);
const MailIcon = () => <Icon><path d="M4 4h16a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2z" /><polyline points="22,6 12,13 2,6" /></Icon>;
const LinkedInIcon = () => <Icon><path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-4 0v7h-4v-7a6 6 0 0 1 6-6z" /><rect x="2" y="9" width="4" height="12" /><circle cx="4" cy="4" r="2" /></Icon>;
const GitHubIcon = () => <Icon><path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22" /></Icon>;
const DownloadIcon = () => <Icon><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="7 10 12 15 17 10" /><line x1="12" y1="15" x2="12" y2="3" /></Icon>;
const SunIcon = () => <Icon><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></Icon>;
const MoonIcon = () => <Icon><path d="M21 12.8A9 9 0 1 1 11.2 3 7 7 0 0 0 21 12.8z" /></Icon>;

function getInitialTheme() {
  try {
    const saved = localStorage.getItem("theme");
    if (saved === "light" || saved === "dark") return saved;
  } catch { /* storage blocked */ }
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [theme, setTheme] = useState(getInitialTheme);
  const endRef = useRef(null);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem("theme", theme); } catch { /* ignore */ }
  }, [theme]);

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  const updateLast = (fn) => setMessages((m) => [...m.slice(0, -1), fn(m[m.length - 1])]);

  async function send(text) {
    const question = text.trim();
    if (!question || loading) return;

    setInput("");
    setLoading(true);
    setMessages((m) => [...m, { role: "user", text: question }, { role: "bot", text: "" }]);

    try {
      const res = await fetch(`${API}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, session_id: sessionId }),
      });
      if (!res.ok) {
        throw new Error(
          res.status === 429 ? "Too many questions. Please wait a minute."
          : res.status === 422 ? "Please keep your question under 500 characters."
          : "Something went wrong. Please try again."
        );
      }
      setSessionId(res.headers.get("X-Session-Id"));

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        updateLast((last) => ({ ...last, text: last.text + chunk }));
      }
    } catch (err) {
      updateLast(() => ({
        role: "bot", error: true,
        text: err instanceof TypeError ? "The assistant is offline right now. Please try again later." : err.message,
      }));
    } finally {
      setLoading(false);
    }
  }

  function newChat() {
    if (sessionId) {
      fetch(`${API}/chat/reset`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId }),
      }).catch(() => {});
    }
    setMessages([]);
    setSessionId(null);
  }

  const last = messages[messages.length - 1];
  const thinking = loading && last?.role === "bot" && last.text === "";
  const initials = PROFILE.name.split(" ").map((w) => w[0]).join("");

  const links = [
    PROFILE.email && { href: `mailto:${PROFILE.email}`, label: PROFILE.email, icon: <MailIcon /> },
    PROFILE.linkedin && { href: PROFILE.linkedin, label: "LinkedIn", icon: <LinkedInIcon />, external: true },
    PROFILE.github && { href: PROFILE.github, label: "GitHub", icon: <GitHubIcon />, external: true },
  ].filter(Boolean);

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="avatar">{initials}</div>
        <h2>{PROFILE.name}</h2>
        <p className="role">{PROFILE.title}</p>
        <p className="bio">{PROFILE.bio}</p>

        <nav className="links">
          {links.map((l) => (
            <a key={l.label} href={l.href} {...(l.external ? { target: "_blank", rel: "noopener noreferrer" } : {})}>
              {l.icon}<span>{l.label}</span>
            </a>
          ))}
        </nav>

        <a className="resume" href={PROFILE.resume} download="Mohammad_Shakeel_Resume.pdf">
          <DownloadIcon /> Download resume
        </a>
      </aside>

      <section className="chat">
        <header>
          <div>
            <h1>Ask about {PROFILE.name.split(" ")[0]}</h1>
            <p>AI assistant answering from his resume and profile</p>
          </div>
          <div className="actions">
            <button className="ghost" onClick={newChat} disabled={loading || !messages.length}>New chat</button>
            <button className="icon" onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
                    aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
                    title={theme === "dark" ? "Light mode" : "Dark mode"}>
              {theme === "dark" ? <SunIcon /> : <MoonIcon />}
            </button>
          </div>
        </header>

        <main>
          {!messages.length && (
            <div className="welcome">
              <h3>Hi, I'm Mohammad's AI assistant 👋</h3>
              <p>Ask about his experience, projects, skills or education.</p>
              <div className="chips">
                {SUGGESTIONS.map((s) => <button key={s} onClick={() => send(s)}>{s}</button>)}
              </div>
            </div>
          )}

          {messages.map((m, i) => (
            <div key={i} className={`row ${m.role}`}>
              {m.role === "bot" && <div className="bot-avatar">AI</div>}
              <div className={`msg ${m.error ? "error" : ""}`}>
                {thinking && i === messages.length - 1 ? <span className="dots">Thinking</span> : m.text}
              </div>
            </div>
          ))}
          <div ref={endRef} />
        </main>

        <form onSubmit={(e) => { e.preventDefault(); send(input); }}>
          <input value={input} onChange={(e) => setInput(e.target.value)}
                 placeholder="Ask a question..." maxLength={500} disabled={loading} />
          <button type="submit" disabled={loading || !input.trim()}>Send</button>
        </form>
      </section>
    </div>
  );
}