import base64
import streamlit as st

# ── HIVE Brand colors ─────────────────────────────────────────────────
HIVE_ORANGE = "#E85C1C"    # Vibrant corporate orange
HIVE_DARK = "#2B1E19"      # Deep chocolate brown (headings and texts)
HIVE_MUTED = "#5C4D46"     # Muted warm brown-gray for secondary text
HIVE_BG = "#F4EDE4"        # Warm corporate beige/sand
HIVE_LIGHT_ORANGE = "#FAF2EE" # Border and soft highlights

# ── SVG art ───────────────────────────────────────────────────────────
# Clean Hexagonal Logo (similar to HIVE brand)
_LOGO_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 140 80">
  <rect width="140" height="80" rx="14" fill="#E85C1C"/>
  <!-- Bold Hexagon Outline and Core -->
  <polygon points="40,22 55,30 55,48 40,56 25,48 25,30" fill="none" stroke="white" stroke-width="3"/>
  <polygon points="40,31 48,35 48,45 40,49 32,45 32,35" fill="white"/>
  
  <text x="68" y="38" font-family="'Trebuchet MS', sans-serif" font-size="20" font-weight="900" fill="white">HIVE</text>
  <text x="68" y="56" font-family="'Trebuchet MS', sans-serif" font-size="10" font-weight="800" fill="#FAF2EE" letter-spacing="0.5px">AI BEES</text>
</svg>"""

# Minimal User Icon
_USER_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <circle cx="50" cy="50" r="50" fill="#2B1E19"/>
  <circle cx="50" cy="36" r="16" fill="#FAF2EE"/>
  <ellipse cx="50" cy="80" rx="26" ry="20" fill="#FAF2EE"/>
</svg>"""

# Minimal Hex-derived Bee Icon
_BEE_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <polygon points="50,15 80,30 80,68 50,83 20,68 20,30" fill="#E85C1C"/>
  <polygon points="50,22 72,33 72,63 50,74 28,63 28,33" fill="#2B1E19"/>
  <rect x="36" y="44" width="28" height="5" rx="2" fill="#E85C1C"/>
  <rect x="36" y="53" width="28" height="5" rx="2" fill="#E85C1C"/>
  <ellipse cx="36" cy="33" rx="6" ry="4" fill="white" fill-opacity="0.85" transform="rotate(-15 36 33)"/>
  <ellipse cx="64" cy="33" rx="6" ry="4" fill="white" fill-opacity="0.85" transform="rotate(15 64 33)"/>
</svg>"""

# Repeating Honeycomb SVG background pattern
_HONEYCOMB_SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="60" height="104" viewBox="0 0 60 104">
  <path d="M30 0L60 17.32v34.64L30 69.28L0 51.96V17.32zm0 69.28L60 86.6v34.64L30 138.56L0 121.24V86.6z" fill="none" stroke="#FAF2EE" stroke-width="1.2"/>
</svg>"""

def _uri(svg: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(svg.strip().encode()).decode()

USER_AVATAR = _uri(_USER_SVG)
BOT_AVATAR  = _uri(_BEE_SVG)
_LOGO_B64   = base64.b64encode(_LOGO_SVG.strip().encode()).decode()
_HIVE_PATTERN_B64 = base64.b64encode(_HONEYCOMB_SVG.strip().encode()).decode()

# ── HIVE Corporate Theme CSS ──────────────────────────────────────────
_CSS = f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&display=swap');
  
  /* Main Canvas background & Repeating Honeycomb Pattern */
  html, body, [data-testid="stAppViewContainer"] {{
      background-color: {HIVE_BG} !important; 
      background-image: url("data:image/svg+xml;base64,{_HIVE_PATTERN_B64}") !important;
      background-repeat: repeat !important;
  }}
  
  /* Safe text selectors: Apply Nunito font only to actual text containers */
  p, label, h1, h2, h3, button, li, a, input, div.stMarkdown {{
      font-family: 'Nunito', sans-serif !important;
  }}
  
  /* Clear and clean, light-colored Sidebar matching the light enterprise template */
  [data-testid="stSidebar"] {{
      background-color: #FCFAF7 !important;
      border-right: 1px solid {HIVE_LIGHT_ORANGE} !important;
      box-shadow: 2px 0 15px rgba(43,30,25,0.03) !important;
  }}
  
  [data-testid="stSidebar"] .stMarkdown p,
  [data-testid="stSidebar"] .stMarkdown span,
  [data-testid="stSidebar"] h1,
  [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3,
  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] div {{ 
      color: {HIVE_DARK} !important; 
  }}
  
  [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h1 {{ 
      color: {HIVE_ORANGE} !important; 
      font-weight: 800 !important; 
  }}
  
  /* Clear, high-contrast, rounded text input field in sidebar */
  [data-testid="stSidebar"] input {{
      color: {HIVE_DARK} !important;
      background-color: white !important;
      border: 1px solid {HIVE_LIGHT_ORANGE} !important;
      border-radius: 8px !important;
      font-weight: 600 !important;
  }}
  
  /* Large, pill-shaped corporate orange buttons */
  [data-testid="stSidebar"] .stButton > button {{
      background: {HIVE_ORANGE} !important;
      color: white !important; 
      border: none !important;
      border-radius: 20px !important; /* pill-shaped */
      font-weight: 800 !important;
      box-shadow: 0 4px 10px rgba(232,92,28,0.2) !important;
      padding: 6px 20px !important;
      transition: all 0.2s ease !important;
  }}
  
  [data-testid="stSidebar"] .stButton > button:hover {{ 
      background: #D14F14 !important;
      box-shadow: 0 4px 15px rgba(232,92,28,0.4) !important;
      transform: translateY(-1px) !important;
  }}
  
  [data-testid="stSidebar"] code {{ 
      background: {HIVE_LIGHT_ORANGE} !important; 
      color: {HIVE_ORANGE} !important; 
      font-size: 0.8rem !important;
  }}

  /* Shrink sidebar fonts slightly for high-density, professional layout */
  [data-testid="stSidebar"] .stMarkdown p, 
  [data-testid="stSidebar"] .stMarkdown span,
  [data-testid="stSidebar"] section,
  [data-testid="stSidebar"] label {{
      font-size: 0.84rem !important;
      color: {HIVE_MUTED} !important;
  }}

  /* Honeycomb Top Header Card - Light, clean, bordered */
  .aibees-header {{
      display: flex; 
      align-items: center; 
      gap: 20px; 
      padding: 20px 28px;
      background-color: white;
      border: 1px solid {HIVE_LIGHT_ORANGE};
      border-radius: 16px; 
      margin-bottom: 24px; 
      border-left: 6px solid {HIVE_ORANGE};
      box-shadow: 0 4px 15px rgba(43,30,25,0.03);
  }}
  
  .aibees-header-text h1 {{ 
      margin: 0; 
      font-size: 1.8rem; 
      font-weight: 900; 
      color: {HIVE_DARK}; 
      letter-spacing: 0.5px;
  }}
  
  .aibees-header-text h1 span {{ 
      color: {HIVE_ORANGE}; 
  }}
  
  .aibees-header-text p {{ 
      margin: 4px 0 0 0; 
      font-size: 0.82rem; 
      color: {HIVE_MUTED}; 
  }}
  
  /* Chat Message balloons - crisp white cards with soft shadows */
  [data-testid="stChatMessage"] {{
      background-color: white !important; 
      border-radius: 14px !important; 
      padding: 16px 20px !important;
      box-shadow: 0 4px 15px rgba(43,30,25,0.03) !important; 
      border: 1px solid {HIVE_LIGHT_ORANGE} !important;
  }}
  
  /* Primary buttons in main canvas (pill-shaped) */
  .stApp .stButton > button {{
      background: {HIVE_ORANGE} !important;
      color: white !important; 
      border: none !important;
      border-radius: 20px !important;
      font-weight: 800 !important;
      box-shadow: 0 4px 10px rgba(232,92,28,0.2) !important;
      transition: all 0.2s ease !important;
  }}
  
  .stApp .stButton > button:hover {{
      background: #D14F14 !important;
      box-shadow: 0 4px 15px rgba(232,92,28,0.4) !important;
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
            <h1>Medi Bee Assist <span>by AIBees</span></h1>
            <p>Enterprise Medical RAG · Vertex AI Vector Search · Gemini 2.5 Pro</p>
        </div>
    </div>""", unsafe_allow_html=True)

def render_sidebar_logo():
    st.sidebar.markdown(
        f'<div style="text-align:center;padding:10px 0;">'
        f'<img src="data:image/svg+xml;base64,{_LOGO_B64}" width="160"/></div>',
        unsafe_allow_html=True,
    )
