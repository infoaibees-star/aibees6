"""Presentation layer: theme and reusable render helpers for the AI Bees PII demo.

Design notes
------------
Palette and motif come from the AI Bees identity: honeyed ivory paper, warm
near-black ink, brand gold and rust. Colour is load-bearing, never decorative -
rust always means "exposed", green always means "protected". The header is a
dark ink slab so the app opens with contrast instead of a wash of pale grey,
and the signature element is the redaction bar itself: the thing this demo
exists to show.

Public API unchanged: inject_theme(), header(), badge(), status_panel(), pill().
Additive helpers: info_card(), section_label(), note(), redacted().
"""

from __future__ import annotations

import streamlit as st

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,600;12..96,800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {
    /* paper + ink */
    --paper:        #FCF8F0;
    --paper-tint:   #F5EDDC;
    --surface:      #FFFDF8;
    --ink:          #171310;
    --ink-slab:     #1C1712;
    --ink-soft:     #5C5347;
    --ink-faint:    #8C8272;
    --rule:         #E4D9C3;
    --rule-strong:  #CDBE9F;

    /* brand */
    --gold:         #D9A427;
    --gold-bright:  #F0BC3C;
    --gold-wash:    #FAF0D6;
    --rust:         #CC4B28;
    --rust-wash:    #FBEAE4;
    --rust-rule:    #EDB9A6;

    /* protected state */
    --moss:         #1B6B4C;
    --moss-wash:    #E4F2EA;
    --moss-rule:    #A6D2BC;

    --radius:       12px;
    --shadow:       0 1px 2px rgba(23,19,16,.05), 0 4px 14px rgba(23,19,16,.05);
    --hex: url("data:image/svg+xml,%3Csvg width='56' height='64' viewBox='0 0 56 64' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M28 0L56 16v32L28 64 0 48V16z' fill='none' stroke='%23D9A427' stroke-opacity='.14' stroke-width='1.5'/%3E%3C/svg%3E");
}

/* ---------------- base ---------------- */

html, body, [class*="css"], .stApp, button, input, textarea, select {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

.stApp { background: var(--paper); color: var(--ink); }
.block-container { padding-top: 2rem; max-width: 1160px; }

h1, h2, h3, h4 {
    font-family: 'Bricolage Grotesque', 'Inter', sans-serif;
    color: var(--ink);
    letter-spacing: -.02em;
}
p, li, label { color: var(--ink); }
hr, [data-testid="stDivider"] { border-color: var(--rule); }

/* ---------------- header: the dark slab ---------------- */

.demo-header {
    position: relative;
    overflow: hidden;
    background: var(--ink-slab);
    background-image: var(--hex);
    background-size: 56px 64px;
    border-radius: var(--radius);
    padding: 30px 34px 28px;
    margin-bottom: 26px;
    box-shadow: var(--shadow);
}
.demo-header::after {
    content: "";
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    background: linear-gradient(180deg, var(--gold-bright), var(--rust));
}
.demo-header .eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: .68rem;
    font-weight: 700;
    letter-spacing: .18em;
    text-transform: uppercase;
    color: var(--gold);
    margin: 0 0 10px;
}
.demo-header h1 {
    font-family: 'Bricolage Grotesque', sans-serif;
    font-size: 2.1rem;
    font-weight: 800;
    line-height: 1.08;
    letter-spacing: -.035em;
    color: #FFFDF8;
    margin: 0;
}
.demo-header h1 .plus { color: var(--gold); font-weight: 600; }
.demo-header p {
    font-size: .92rem;
    line-height: 1.6;
    color: #B9AE9C;
    margin: 12px 0 0;
    max-width: 62ch;
}

/* signature: the redaction strip */
.redact-strip {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 9px;
    margin-top: 20px;
    padding-top: 18px;
    border-top: 1px solid rgba(217,164,39,.22);
    font-family: 'JetBrains Mono', monospace;
    font-size: .78rem;
    color: #8C8272;
}
.redact-strip .field { color: #B9AE9C; }
.redact-strip .clear { color: var(--rust); }
.redact-strip .bar {
    display: inline-block;
    background: repeating-linear-gradient(90deg, var(--gold) 0 6px, #A87C15 6px 12px);
    border-radius: 3px;
    color: transparent;
    user-select: none;
}

/* ---------------- badge ---------------- */

.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 5px;
    font-family: 'JetBrains Mono', monospace;
    font-size: .68rem;
    font-weight: 700;
    letter-spacing: .09em;
    text-transform: uppercase;
    line-height: 1.6;
}
.badge-on  { background: var(--moss-wash); color: var(--moss); border: 1px solid var(--moss-rule); }
.badge-off { background: var(--rust-wash); color: var(--rust); border: 1px solid var(--rust-rule); }

/* ---------------- status tiles ----------------
   min-height + flex centring keeps every tile in a columns() row aligned. */

.status-panel {
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    min-height: 66px;
    box-sizing: border-box;
    border-radius: var(--radius);
    padding: 14px 18px;
    font-size: .88rem;
    font-weight: 600;
    text-align: center;
    line-height: 1.35;
    box-shadow: var(--shadow);
}
.status-on  { background: var(--moss-wash); border: 1px solid var(--moss-rule); color: var(--moss); }
.status-off { background: var(--rust-wash); border: 1px solid var(--rust-rule); color: var(--rust); }

/* hexagon status marker, not a circle */
.status-panel::before {
    content: "";
    flex: none;
    width: 9px; height: 10px;
    background: currentColor;
    clip-path: polygon(50% 0, 100% 25%, 100% 75%, 50% 100%, 0 75%, 0 25%);
}

/* ---------------- info card ---------------- */

.info-card {
    display: flex;
    flex-direction: column;
    justify-content: center;
    min-height: 66px;
    box-sizing: border-box;
    background: var(--surface);
    border: 1px solid var(--rule);
    border-radius: var(--radius);
    padding: 13px 18px;
    box-shadow: var(--shadow);
}
.info-card .label {
    font-family: 'JetBrains Mono', monospace;
    font-size: .64rem;
    font-weight: 700;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: var(--ink-faint);
    margin-bottom: 4px;
}
.info-card .value {
    font-family: 'JetBrains Mono', monospace;
    font-size: .8rem;
    color: var(--ink);
    word-break: break-all;
    line-height: 1.4;
}

/* ---------------- pills ---------------- */

.redact-pill, .flag-pill {
    display: inline-block;
    border-radius: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: .69rem;
    font-weight: 500;
    padding: 3px 12px;
    margin: 7px 6px 0 0;
    letter-spacing: .02em;
}
.redact-pill { background: var(--rust-wash); color: var(--rust); border: 1px solid var(--rust-rule); }
.flag-pill   { background: var(--gold-wash); color: #8A6100;    border: 1px solid #E8CE8E; }

/* inline redaction inside answers */
.redacted {
    display: inline-block;
    background: repeating-linear-gradient(90deg, var(--ink) 0 6px, #3A322A 6px 12px);
    color: transparent;
    border-radius: 3px;
    padding: 0 6px;
    user-select: none;
}

/* ---------------- small helpers ---------------- */

.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: .66rem;
    font-weight: 700;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: var(--ink-faint);
    margin: 26px 0 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--rule);
}
.note {
    font-size: .84rem;
    color: var(--ink-soft);
    line-height: 1.6;
    border-left: 2px solid var(--gold);
    padding-left: 13px;
    margin: 12px 0;
}

/* ---------------- sidebar ---------------- */

section[data-testid="stSidebar"] {
    background: var(--paper-tint);
    border-right: 1px solid var(--rule);
}
section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4 {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: .68rem !important;
    font-weight: 700;
    letter-spacing: .13em;
    text-transform: uppercase;
    color: var(--ink-faint) !important;
    margin: 22px 0 9px;
}
section[data-testid="stSidebar"] [data-testid="stAlert"] {
    background: var(--moss-wash);
    border: 1px solid var(--moss-rule);
    border-radius: 8px;
    padding: 9px 12px;
    font-size: .77rem;
    line-height: 1.45;
}

/* ---------------- widgets ---------------- */

div.stButton > button {
    background: var(--surface);
    color: var(--ink);
    border: 1px solid var(--rule-strong);
    border-radius: 8px;
    font-size: .84rem;
    font-weight: 600;
    padding: 9px 16px;
    width: 100%;
    transition: background .16s, border-color .16s, color .16s;
}
div.stButton > button:hover {
    background: var(--gold-wash);
    border-color: var(--gold);
    color: #7A5A00;
}
div.stButton > button:focus-visible {
    outline: 2px solid var(--gold);
    outline-offset: 2px;
}

[data-testid="stFileUploaderDropzone"] {
    background: var(--surface);
    border: 1px dashed var(--rule-strong);
    border-radius: 8px;
}
div[data-baseweb="select"] > div {
    background: var(--surface);
    border-color: var(--rule-strong);
    border-radius: 8px;
    font-size: .84rem;
}
[data-testid="stChatInput"] {
    background: var(--surface);
    border: 1px solid var(--rule-strong);
    border-radius: 10px;
}
[data-testid="stChatInput"] textarea { font-size: .9rem; }

code {
    font-family: 'JetBrains Mono', monospace;
    font-size: .81rem;
    color: #8A6100;
    background: var(--gold-wash);
    padding: 1px 6px;
    border-radius: 4px;
}

@media (max-width: 640px) {
    .demo-header { padding: 24px 20px; }
    .demo-header h1 { font-size: 1.6rem; }
}
@media (prefers-reduced-motion: reduce) {
    * { transition: none !important; animation: none !important; }
}
</style>
"""

_HEADER = """
<div class="demo-header">
    <p class="eyebrow">AI Bees Academy</p>
    <h1>AI Bees RAG <span class="plus">+</span> PII Demo</h1>
    <p>Ask a question against the indexed customer records, then flip guardrails on
    and ask it again. Off, sensitive values come back in the clear. On, they never
    leave the pipeline.</p>
    <div class="redact-strip">
        <span class="field">name</span>
        <span class="bar">Xxxxxxxxxxx</span>
        <span class="field">ssn</span>
        <span class="bar">XXXXXXXXXXX</span>
        <span class="field">card</span>
        <span class="clear">4127&nbsp;9930&nbsp;1188&nbsp;2044</span>
        <span class="field">&larr; still exposed</span>
    </div>
</div>
"""


def inject_theme() -> None:
    """Inject the stylesheet. Call once, immediately after set_page_config()."""
    st.markdown(_CSS, unsafe_allow_html=True)


def header() -> None:
    st.markdown(_HEADER, unsafe_allow_html=True)


def badge(label: str, active: bool = True) -> None:
    css = "badge-on" if active else "badge-off"
    st.markdown(f'<span class="badge {css}">{label}</span>', unsafe_allow_html=True)


def status_panel(label: str, active: bool) -> None:
    """A state tile. Tiles in the same st.columns() row will line up."""
    css = "status-on" if active else "status-off"
    st.markdown(f'<div class="status-panel {css}">{label}</div>', unsafe_allow_html=True)


def pill(text: str, kind: str = "redact") -> None:
    st.markdown(f'<span class="{kind}-pill">{text}</span>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Additive helpers
# --------------------------------------------------------------------------

def info_card(label: str, value: str) -> None:
    """Neutral tile, same height as status_panel, so rows stay aligned."""
    st.markdown(
        f'<div class="info-card"><div class="label">{label}</div>'
        f'<div class="value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def section_label(text: str) -> None:
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def redacted(text: str) -> str:
    """Return an inline redaction bar. Use inside an st.markdown() string."""
    return f'<span class="redacted">{text}</span>'
