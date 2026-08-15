# theme.py — all AIBees branding (colors, SVG avatars, CSS) lives here.
# Students don't need to read this file; app.py just calls these functions.
import base64
import streamlit as st

# ── Brand colors ──────────────────────────────────────────────────────
ORANGE, DARK, YELLOW, LIGHT, ORANGE2 = (
    "#E8500A", "#3A3A3A", "#F5C518", "#FFF8F3", "#FF6B2B"
)

# ── SVG art ───────────────────────────────────────────────────────────
_USER_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <circle cx="50" cy="50" r="50" fill="#3A3A3A"/>
  <circle cx="50" cy="36" r="16" fill="#F5C518"/>
  <ellipse cx="50" cy="80" rx="26" ry="20" fill="#F5C518"/>
</svg>"""

_BEE_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <circle cx="50" cy="50" r="50" fill="#E8500A"/>
  <ellipse cx="50" cy="58" rx="14" ry="18" fill="#3A3A3A"/>
  <rect x="36" y="53" width="28" height="5" rx="2" fill="#F5C518"/>
  <rect x="36" y="62" width="28" height="5" rx="2" fill="#F5C518"/>
  <circle cx="50" cy="38" r="11" fill="#F5C518"/>
  <circle cx="46" cy="37" r="2.5" fill="#3A3A3A"/>
  <circle cx="54" cy="37" r="2.5" fill="#3A3A3A"/>
  <line x1="46" y1="28" x2="41" y2="20" stroke="#3A3A3A" stroke-width="2" stroke-linecap="round"/>
  <circle cx="41" cy="19" r="2.5" fill="#3A3A3A"/>
  <line x1="54" y1="28" x2="59" y2="20" stroke="#3A3A3A" stroke-width="2" stroke-linecap="round"/>
  <circle cx="59" cy="19" r="2.5" fill="#3A3A3A"/>
  <ellipse cx="34" cy="48" rx="11" ry="7" fill="white" fill-opacity="0.75" transform="rotate(-20 34 48)"/>
  <ellipse cx="66" cy="48" rx="11" ry="7" fill="white" fill-opacity="0.75" transform="rotate(20 66 48)"/>
</svg>"""

_LOGO_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 140 80">
  <rect width="140" height="80" rx="12" fill="#E8500A"/>
  <ellipse cx="42" cy="46" rx="10" ry="13" fill="#3A3A3A"/>
  <rect x="32" y="41" width="20" height="4" rx="2" fill="#F5C518"/>
  <rect x="32" y="48" width="20" height="4" rx="2" fill="#F5C518"/>
  <circle cx="42" cy="31" r="9" fill="#F5C518"/>
  <circle cx="39" cy="30" r="2" fill="#3A3A3A"/>
  <circle cx="45" cy="30" r="2" fill="#3A3A3A"/>
  <line x1="39" y1="23" x2="35" y2="16" stroke="#3A3A3A" stroke-width="1.8" stroke-linecap="round"/>
  <circle cx="34" cy="15" r="2" fill="#3A3A3A"/>
  <line x1="45" y1="23" x2="49" y2="16" stroke="#3A3A3A" stroke-width="1.8" stroke-linecap="round"/>
  <circle cx="50" cy="15" r="2" fill="#3A3A3A"/>
  <ellipse cx="30" cy="38" rx="9" ry="6" fill="white" fill-opacity="0.8" transform="rotate(-15 30 38)"/>
  <ellipse cx="54" cy="38" rx="9" ry="6" fill="white" fill-opacity="0.8" transform="rotate(15 54 38)"/>
  <text x="68" y="34" font-family="'Trebuchet MS', sans-serif" font-size="22" font-weight="800" fill="white">AI</text>
  <text x="68" y="60" font-family="'Trebuchet MS', sans-serif" font-size="22" font-weight="800" fill="white">Bees</text>
</svg>"""

def _uri(svg: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(svg.strip().encode()).decode()

# Avatars used by app.py
USER_AVATAR = _uri(_USER_SVG)
BOT_AVATAR  = _uri(_BEE_SVG)
_LOGO_B64   = base64.b64encode(_LOGO_SVG.strip().encode()).decode()

# ── CSS ───────────────────────────────────────────────────────────────
_CSS = f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');
  html, body, [data-testid="stAppViewContainer"] {{
      background-color: {LIGHT} !important; font-family: 'Nunito', sans-serif !important;
  }}
  [data-testid="stSidebar"] {{
      background: linear-gradient(160deg, {DARK} 0%, #1e1e1e 100%) !important;
      border-right: 3px solid {ORANGE} !important;
  }}
  [data-testid="stSidebar"] * {{ color: #f0f0f0 !important; font-family: 'Nunito', sans-serif !important; }}
  [data-testid="stSidebar"] h3 {{ color: {YELLOW} !important; font-weight: 800 !important; }}
  [data-testid="stSidebar"] .stButton > button {{
      background: {ORANGE} !important; color: white !important; border: none !important;
      border-radius: 8px !important; font-weight: 700 !important;
  }}
  [data-testid="stSidebar"] .stButton > button:hover {{ background: {ORANGE2} !important; }}
  [data-testid="stSidebar"] code {{ background: rgba(255,255,255,0.12) !important; color: {YELLOW} !important; }}

  /* ── Sidebar file uploader (make it readable on the dark sidebar) ── */
  [data-testid="stSidebar"] section[data-testid="stFileUploaderDropzone"],
  [data-testid="stSidebar"] [data-testid="stFileUploader"] section {{
      background: rgba(255,255,255,0.08) !important;
      border: 1px dashed rgba(245,197,24,0.6) !important;
      border-radius: 10px !important;
  }}
  [data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"],
  [data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] * {{
      color: #e6e6e6 !important;
  }}
  /* the "Browse files" button */
  [data-testid="stSidebar"] [data-testid="stFileUploader"] button {{
      background: {YELLOW} !important;
      color: {DARK} !important;
      border: none !important;
      font-weight: 700 !important;
      border-radius: 6px !important;
  }}
  /* Uploaded-file box: Streamlit paints it white. Force EVERY div inside the
     uploader to a dark tone so no white pill can show through, whatever the
     exact testid is in this Streamlit version. Buttons keep their own colors. */
  [data-testid="stSidebar"] [data-testid="stFileUploader"] div {{
      background-color: #2f2f2f !important;
  }}
  [data-testid="stSidebar"] [data-testid="stFileUploader"],
  [data-testid="stSidebar"] [data-testid="stFileUploader"] * {{
      color: #f0f0f0 !important;
  }}
  [data-testid="stSidebar"] [data-testid="stFileUploaderFile"] {{
      border: 1px solid rgba(245,197,24,0.35) !important;
      border-radius: 8px !important;
      padding: 4px 6px !important;
  }}
  /* Force the WHOLE Browse/Upload button (and everything inside it — divs,
     spans, the icon) to yellow with dark content. This overrides the blanket
     dark-div rule above, whatever tag the inner black box actually is. */
  [data-testid="stSidebar"] [data-testid="stFileUploader"] button,
  [data-testid="stSidebar"] [data-testid="stFileUploader"] button * {{
      background-color: {YELLOW} !important;
      color: {DARK} !important;
      border: none !important;
  }}
  /* Keep Material icons as GLYPHS. The sidebar font override above was turning
     the upload icon's ligature into the literal word "upload" (the overlap). */
  [data-testid="stSidebar"] [data-testid="stIconMaterial"],
  [data-testid="stSidebar"] span.material-symbols-rounded,
  [data-testid="stSidebar"] span[class*="material-icons"] {{
      font-family: 'Material Symbols Rounded','Material Symbols Outlined','Material Icons' !important;
  }}
  /* Browse button is icon-only here — keep the glyph dark & visible on yellow */
  [data-testid="stSidebar"] [data-testid="stFileUploader"] button [data-testid="stIconMaterial"] {{
      color: {DARK} !important;
  }}
  .aibees-header {{
      display: flex; align-items: center; gap: 18px; padding: 18px 24px;
      background: linear-gradient(135deg, {DARK} 0%, #2a2a2a 100%);
      border-radius: 16px; margin-bottom: 20px; border-left: 5px solid {ORANGE};
  }}
  .aibees-header-text h1 {{ margin: 0; font-size: 1.7rem; font-weight: 800; color: white; }}
  .aibees-header-text h1 span {{ color: {YELLOW}; }}
  .aibees-header-text p {{ margin: 4px 0 0 0; font-size: 0.82rem; color: #aaa; }}
  [data-testid="stChatMessage"] {{
      background: white !important; border-radius: 14px !important; padding: 14px 18px !important;
      box-shadow: 0 2px 10px rgba(0,0,0,0.07) !important; border: 1px solid #f0e8e0 !important;
  }}
</style>
"""

# ── Public helpers used by app.py ─────────────────────────────────────
def inject_css():
    st.markdown(_CSS, unsafe_allow_html=True)

def render_header():
    st.markdown(f"""
    <div class="aibees-header">
        <img src="data:image/svg+xml;base64,{_LOGO_B64}" width="100" alt="AIBees Logo"/>
        <div class="aibees-header-text">
            <h1>IT Helpdesk <span>Assistant</span></h1>
            <p>RAG · Gemini on Vertex AI · FAISS · AIBees Academy</p>
        </div>
    </div>""", unsafe_allow_html=True)

def render_sidebar_logo():
    st.markdown(
        f'<div style="text-align:center;padding:10px 0;">'
        f'<img src="data:image/svg+xml;base64,{_LOGO_B64}" width="160"/></div>',
        unsafe_allow_html=True,
    )