"""
EduPredict AI - Advanced Student Performance Prediction System v2.0
Features: Auth System (GitHub/Google OAuth UI), 3D UI, Plotly Charts,
          Batch Prediction, PDF Export, Prediction History, Radar/Gauge Charts
Run: streamlit run app.py
"""
import json
import io
import html
import os
import secrets
from urllib.parse import urlencode
from pathlib import Path
from datetime import datetime

import joblib
import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

try:
    from fpdf import FPDF
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent
DATA_PATH    = ROOT / "data" / "raw" / "student_performance.csv"
MODEL_PATH   = ROOT / "models" / "best_model.pkl"
METRICS_PATH = ROOT / "models" / "metrics.json"

FEATURES = ["attendance","internal_marks","study_hours","previous_performance","assignment_score"]
LABELS   = {
    "attendance":           "Attendance (%)",
    "internal_marks":       "Internal Marks",
    "study_hours":          "Study Hours / Day",
    "previous_performance": "Previous Performance",
    "assignment_score":     "Assignment Score",
}

DEMO_USERS = {
    "admin":   {"name":"Admin User",    "password":"admin123",   "role":"admin",   "email":"admin@edupredict.ai","avatar":"👨‍💼"},
    "teacher": {"name":"Prof. Sharma",  "password":"teacher123", "role":"teacher", "email":"sharma@edu.in",      "avatar":"👨‍🏫"},
    "student": {"name":"Rahul Kumar",   "password":"student123", "role":"student", "email":"rahul@student.edu",  "avatar":"👨‍🎓"},
}

# ── Page config (MUST be first Streamlit call) ─────────────────────────────────
st.set_page_config(page_title="EduPredict AI", page_icon="🎓", layout="wide",
                   initial_sidebar_state="expanded")

# ── Global CSS + Three.js 3D background ───────────────────────────────────────
ULTRA_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;500;600;700&display=swap');

:root{
  --p:#7c3aed; --s:#06b6d4; --a:#f59e0b;
  --ok:#10b981; --err:#ef4444;
  --bg:#060614;
  --card:rgba(255,255,255,0.03);
  --border:rgba(124,58,237,0.3);
  --t1:#f1f5f9; --t2:#94a3b8; --t3:#475569;
}

/* ── background ── */
.stApp{
  background:radial-gradient(ellipse at top left,#1a0533 0%,#060614 45%,#001a2e 100%) !important;
  min-height:100vh;
}
.stApp::before{
  content:'';position:fixed;top:0;left:0;width:100%;height:100%;
  background-image:
    radial-gradient(1px 1px at 15% 25%,rgba(124,58,237,.9) 0%,transparent 100%),
    radial-gradient(1px 1px at 75% 10%,rgba(6,182,212,.9) 0%,transparent 100%),
    radial-gradient(1px 1px at 45% 65%,rgba(245,158,11,.7) 0%,transparent 100%),
    radial-gradient(1px 1px at 85% 80%,rgba(16,185,129,.7) 0%,transparent 100%),
    radial-gradient(1px 1px at 5%  90%,rgba(124,58,237,.7) 0%,transparent 100%);
  background-size:280px 280px,350px 350px,420px 420px,310px 310px,390px 390px;
  animation:stars 25s linear infinite;
  pointer-events:none;z-index:0;
}
@keyframes stars{
  0%  {background-position:0% 0%,0% 0%,0% 0%,0% 0%,0% 0%}
  100%{background-position:100% 100%,-100% 100%,50% -100%,100% -50%,-50% 100%}
}

/* ── sidebar ── */
[data-testid="stSidebar"]{
  background:linear-gradient(180deg,rgba(124,58,237,.18) 0%,rgba(6,182,212,.07) 100%) !important;
  border-right:1px solid var(--border) !important;
  backdrop-filter:blur(20px);
}

/* ── glass card ── */
.gc{
  background:rgba(255,255,255,.04);
  backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);
  border:1px solid rgba(124,58,237,.25);
  border-radius:20px;padding:24px;margin:10px 0;
  position:relative;overflow:hidden;
  transition:all .4s cubic-bezier(.175,.885,.32,1.275);
  box-shadow:0 8px 32px rgba(0,0,0,.4),inset 0 1px 0 rgba(255,255,255,.08);
}
.gc::before{
  content:'';position:absolute;top:0;left:-100%;
  width:100%;height:2px;
  background:linear-gradient(90deg,transparent,var(--p),var(--s),transparent);
  animation:scan 3s linear infinite;
}
@keyframes scan{0%{left:-100%}100%{left:100%}}
.gc:hover{
  transform:translateY(-4px) scale(1.005);
  border-color:rgba(124,58,237,.55);
  box-shadow:0 20px 60px rgba(124,58,237,.25),inset 0 1px 0 rgba(255,255,255,.12);
}

/* ── 3-D metric card ── */
.mc{
  background:linear-gradient(135deg,rgba(124,58,237,.2),rgba(6,182,212,.1));
  border:1px solid rgba(124,58,237,.4);border-radius:16px;
  padding:20px;text-align:center;
  transform:perspective(1000px) rotateX(0);
  transition:transform .5s ease,box-shadow .5s ease;
  box-shadow:0 10px 40px rgba(0,0,0,.5),0 0 0 1px rgba(124,58,237,.1);
  position:relative;overflow:hidden;
}
.mc::after{
  content:'';position:absolute;top:-50%;left:-50%;
  width:200%;height:200%;
  background:conic-gradient(from 0deg,transparent 0deg,rgba(124,58,237,.08) 60deg,transparent 120deg);
  animation:spin 5s linear infinite;
}
@keyframes spin{from{transform:rotate(0)}to{transform:rotate(360deg)}}
.mc:hover{
  transform:perspective(1000px) rotateX(-5deg) rotateY(5deg) translateY(-8px);
  box-shadow:20px 20px 60px rgba(0,0,0,.7),0 0 30px rgba(124,58,237,.4);
}
.mv{
  font-family:'Orbitron',monospace;font-size:2.4rem;font-weight:900;
  background:linear-gradient(135deg,#fff,var(--s));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
  line-height:1.1;position:relative;z-index:1;
}
.ml{color:var(--t2);font-size:.72rem;text-transform:uppercase;letter-spacing:2px;margin-top:8px;position:relative;z-index:1;}

/* ── buttons ── */
.stButton>button{
  background:linear-gradient(135deg,var(--p),#9333ea) !important;
  color:#fff !important;border:none !important;border-radius:12px !important;
  padding:12px 28px !important;font-weight:600 !important;
  font-family:'Inter',sans-serif !important;letter-spacing:.5px !important;
  transition:all .3s ease !important;
  box-shadow:0 4px 15px rgba(124,58,237,.4) !important;
}
.stButton>button:hover{transform:translateY(-2px) !important;box-shadow:0 8px 25px rgba(124,58,237,.6) !important;}

/* ── headings ── */
h1,h2,h3{color:var(--t1) !important;}
h1{
  font-family:'Orbitron',monospace !important;
  background:linear-gradient(135deg,#fff 0%,var(--s) 55%,var(--p) 100%) !important;
  -webkit-background-clip:text !important;-webkit-text-fill-color:transparent !important;background-clip:text !important;
}

/* ── inputs ── */
.stSlider>label,.stSelectbox>label,.stTextInput>label,.stNumberInput>label{color:var(--t2) !important;}

/* ── scrollbar ── */
::-webkit-scrollbar{width:5px}
::-webkit-scrollbar-track{background:rgba(255,255,255,.02)}
::-webkit-scrollbar-thumb{background:linear-gradient(var(--p),var(--s));border-radius:3px}

/* ── progress ── */
.stProgress>div>div{background:linear-gradient(90deg,var(--p),var(--s)) !important;border-radius:10px !important;}

/* ── alerts ── */
.stSuccess,.stError,.stWarning,.stInfo{border-radius:12px !important;backdrop-filter:blur(10px) !important;}

/* ── dataframe ── */
.stDataFrame{border-radius:12px !important;border:1px solid var(--border) !important;}

/* ── sidebar radio ── */
[data-testid="stSidebar"] .stRadio label{
  background:rgba(255,255,255,.03) !important;
  border:1px solid rgba(124,58,237,.2) !important;
  border-radius:10px !important;padding:10px 16px !important;
  color:var(--t2) !important;transition:all .3s ease !important;width:100% !important;
}
[data-testid="stSidebar"] .stRadio label:hover{
  background:rgba(124,58,237,.12) !important;border-color:var(--p) !important;color:#fff !important;
}

/* ── login card ── */
.lcard{
  max-width:420px;margin:0 auto;padding:40px;
  background:rgba(255,255,255,.03);backdrop-filter:blur(30px);
  border:1px solid rgba(124,58,237,.35);border-radius:24px;
  box-shadow:0 25px 80px rgba(0,0,0,.6),0 0 60px rgba(124,58,237,.12);
}
.logo{
  text-align:center;font-family:'Orbitron',monospace;font-size:1.9rem;font-weight:900;
  background:linear-gradient(135deg,#7c3aed,#06b6d4);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}
.sub{text-align:center;color:#64748b;font-size:.85rem;margin-bottom:28px;}
.divdr{display:flex;align-items:center;gap:10px;color:#475569;font-size:.78rem;margin:18px 0;}
.divdr::before,.divdr::after{content:'';flex:1;height:1px;background:rgba(124,58,237,.2);}

/* ── badge ── */
.bp{display:inline-block;padding:3px 12px;border-radius:20px;font-size:.68rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;}
.bpass{background:rgba(16,185,129,.18);color:#10b981;border:1px solid rgba(16,185,129,.4);}
.bfail{background:rgba(239,68,68,.18);color:#ef4444;border:1px solid rgba(239,68,68,.4);}

/* ── notif ── */
.nb{background:linear-gradient(135deg,rgba(124,58,237,.18),rgba(6,182,212,.08));
  border:1px solid rgba(124,58,237,.28);border-radius:12px;
  padding:12px 20px;margin:10px 0;color:#c4b5fd;font-size:.85rem;}

/* ── animations ── */
@keyframes fadeUp{from{opacity:0;transform:translateY(28px)}to{opacity:1;transform:translateY(0)}}
.fu{animation:fadeUp .55s ease forwards;}
</style>

<!-- Three.js 3D Particle Background -->
<canvas id="c3d" style="position:fixed;top:0;left:0;z-index:-1;width:100%;height:100%;pointer-events:none;"></canvas>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
(function(){
  var c=document.getElementById('c3d');
  if(!c||typeof THREE==='undefined')return;
  var r=new THREE.WebGLRenderer({canvas:c,alpha:true,antialias:true});
  r.setPixelRatio(window.devicePixelRatio);
  r.setSize(window.innerWidth,window.innerHeight);
  r.setClearColor(0x000000,0);
  var sc=new THREE.Scene();
  var cam=new THREE.PerspectiveCamera(75,window.innerWidth/window.innerHeight,0.1,1000);
  cam.position.z=5;
  var n=1800,pos=new Float32Array(n*3),col=new Float32Array(n*3);
  for(var i=0;i<n*3;i+=3){
    pos[i]=(Math.random()-.5)*22;pos[i+1]=(Math.random()-.5)*22;pos[i+2]=(Math.random()-.5)*12;
    var t=Math.random();
    col[i]=.47+t*.06;col[i+1]=.23+t*.5;col[i+2]=.93-t*.1;
  }
  var g=new THREE.BufferGeometry();
  g.setAttribute('position',new THREE.BufferAttribute(pos,3));
  g.setAttribute('color',new THREE.BufferAttribute(col,3));
  var pts=new THREE.Points(g,new THREE.PointsMaterial({size:.03,vertexColors:true,transparent:true,opacity:.65,blending:THREE.AdditiveBlending,depthWrite:false}));
  sc.add(pts);
  var mx=0,my=0;
  document.addEventListener('mousemove',function(e){mx=(e.clientX/window.innerWidth-.5)*2;my=-(e.clientY/window.innerHeight-.5)*2;});
  (function anim(){
    requestAnimationFrame(anim);
    pts.rotation.x+=.0003;pts.rotation.y+=.0005;
    cam.position.x+=(mx*.4-cam.position.x)*.05;
    cam.position.y+=(my*.4-cam.position.y)*.05;
    r.render(sc,cam);
  })();
  window.addEventListener('resize',function(){
    cam.aspect=window.innerWidth/window.innerHeight;cam.updateProjectionMatrix();
    r.setSize(window.innerWidth,window.innerHeight);
  });
})();
</script>
"""

# ── Session init ───────────────────────────────────────────────────────────────
def _init():
    defaults = {"authenticated":False,"username":None,"user_data":None,
                "prediction_history":[],"login_error":None}
    for k,v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
_init()

@st.cache_resource
def _pending_oauth_states():
    return {}

def _oauth_setting(name):
    value = os.getenv(name)
    if value:
        return value.strip()
    try:
        value = st.secrets.get(name)
    except Exception:
        return None
    return str(value).strip() if value else None

def _oauth_authorization_url(provider):
    prefix = provider.upper()
    client_id = _oauth_setting(f"{prefix}_CLIENT_ID")
    client_secret = _oauth_setting(f"{prefix}_CLIENT_SECRET")
    if not client_id or not client_secret:
        return None

    redirect_uri = _oauth_setting("OAUTH_REDIRECT_URI") or "http://localhost:8501"
    state = secrets.token_urlsafe(32)
    pending_states = _pending_oauth_states()
    now = datetime.now().timestamp()
    for old_state, (_, created_at) in list(pending_states.items()):
        if now - created_at > 600:
            pending_states.pop(old_state, None)
    pending_states[state] = (provider, now)
    if provider == "google":
        endpoint = "https://accounts.google.com/o/oauth2/v2/auth"
        params = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": "openid email profile",
            "state": state,
        }
    else:
        endpoint = "https://github.com/login/oauth/authorize"
        params = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "scope": "user:email",
            "state": state,
        }
    return f"{endpoint}?{urlencode(params)}"

def _oauth_user(provider, code):
    prefix = provider.upper()
    client_id = _oauth_setting(f"{prefix}_CLIENT_ID")
    client_secret = _oauth_setting(f"{prefix}_CLIENT_SECRET")
    redirect_uri = _oauth_setting("OAUTH_REDIRECT_URI") or "http://localhost:8501"
    token_response = requests.post(
        "https://oauth2.googleapis.com/token" if provider == "google"
        else "https://github.com/login/oauth/access_token",
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "code": code,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        },
        headers={"Accept": "application/json"},
        timeout=15,
    )
    token_response.raise_for_status()
    access_token = token_response.json().get("access_token")
    if not access_token:
        raise ValueError("The provider did not return an access token.")

    headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}
    if provider == "google":
        profile_response = requests.get(
            "https://openidconnect.googleapis.com/v1/userinfo", headers=headers, timeout=15
        )
        profile_response.raise_for_status()
        profile = profile_response.json()
        if not profile.get("email_verified") or not profile.get("sub"):
            raise ValueError("Google did not return a verified account.")
        return {
            "username": f"google:{profile['sub']}",
            "name": profile.get("name") or profile["email"],
            "email": profile["email"],
        }

    profile_response = requests.get("https://api.github.com/user", headers=headers, timeout=15)
    profile_response.raise_for_status()
    profile = profile_response.json()
    email = profile.get("email")
    if not email:
        emails_response = requests.get(
            "https://api.github.com/user/emails", headers=headers, timeout=15
        )
        emails_response.raise_for_status()
        email_entries = emails_response.json()
        email = next(
            (entry["email"] for entry in email_entries
             if entry.get("primary") and entry.get("verified")),
            next((entry["email"] for entry in email_entries if entry.get("verified")), None),
        )
    if not profile.get("id") or not email:
        raise ValueError("GitHub did not return an account with a verified email.")
    return {
        "username": f"github:{profile.get('login') or profile['id']}",
        "name": profile.get("name") or profile.get("login") or email,
        "email": email,
    }

def handle_oauth_callback():
    params = st.query_params
    code = params.get("code")
    returned_state = params.get("state")
    provider_error = params.get("error")
    pending_states = _pending_oauth_states()
    pending = pending_states.pop(returned_state, None) if returned_state else None
    provider = pending[0] if pending and datetime.now().timestamp() - pending[1] <= 600 else None
    if not code and not provider_error:
        return

    st.query_params.clear()
    if not provider:
        st.session_state.login_error = "Sign-in could not be verified. Please try again."
        st.rerun()
    if provider_error:
        st.session_state.login_error = "Sign-in was cancelled or declined."
        st.rerun()

    try:
        identity = _oauth_user(provider, code)
    except (requests.RequestException, ValueError, KeyError, TypeError):
        st.session_state.login_error = "Sign-in failed. Check OAuth settings and callback URL."
        st.rerun()

    st.session_state.authenticated = True
    st.session_state.username = identity["username"]
    st.session_state.user_data = {
        "name": identity["name"],
        "email": identity["email"],
        "role": "student",
        "avatar": "👨‍🎓",
    }
    st.session_state.login_error = None
    st.rerun()

# ── Data / Model loaders ───────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metrics():
    return json.loads(METRICS_PATH.read_text())

# ── Helpers ────────────────────────────────────────────────────────────────────
def predict(model, values):
    X = pd.DataFrame([values], columns=FEATURES)
    p = float(model.predict_proba(X)[0][1])
    return ("Pass" if p >= .5 else "Fail"), p

def recommendations(v):
    tips=[]
    if v["attendance"]<75:           tips.append(("w","Attendance below 75% — attend regularly to avoid shortage."))
    if v["internal_marks"]<50:       tips.append(("w","Internal marks low — revise weak topics & practise past tests."))
    if v["study_hours"]<3:           tips.append(("w","Studying <3 hrs/day — build a fixed daily study schedule."))
    if v["previous_performance"]<50: tips.append(("w","Previous score weak — strengthen basics first."))
    if v["assignment_score"]<50:     tips.append(("w","Assignment score low — submit on time & seek feedback."))
    if not tips:                     tips.append(("ok","All indicators healthy! Keep up the great work 🚀"))
    return tips

def plotly_base():
    return dict(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),legend=dict(bgcolor="rgba(0,0,0,0)"))

def gauge(prob, title="Pass Probability"):
    color = "#10b981" if prob>=.5 else "#ef4444"
    fig=go.Figure(go.Indicator(
        mode="gauge+number+delta",value=prob*100,
        number={"suffix":"%","font":{"size":38,"color":"white","family":"Orbitron"}},
        delta={"reference":50,"increasing":{"color":"#10b981"},"decreasing":{"color":"#ef4444"}},
        gauge={"axis":{"range":[0,100],"tickcolor":"#64748b","tickfont":{"color":"#64748b"}},
               "bar":{"color":color},"bgcolor":"rgba(0,0,0,0)",
               "bordercolor":"rgba(124,58,237,.3)",
               "steps":[{"range":[0,50],"color":"rgba(239,68,68,.1)"},
                        {"range":[50,100],"color":"rgba(16,185,129,.1)"}],
               "threshold":{"line":{"color":"#7c3aed","width":3},"thickness":.75,"value":50}},
        title={"text":title,"font":{"color":"#94a3b8","size":13}}))
    fig.update_layout(height=270,margin=dict(l=20,r=20,t=40,b=10),**plotly_base())
    return fig

def radar(values, avg):
    cats=list(LABELS.values())
    sv=[values[k] for k in FEATURES]; av=[avg[k] for k in FEATURES]
    fig=go.Figure()
    fig.add_trace(go.Scatterpolar(r=sv+[sv[0]],theta=cats+[cats[0]],fill='toself',
        name='Student',line_color='#7c3aed',fillcolor='rgba(124,58,237,.2)'))
    fig.add_trace(go.Scatterpolar(r=av+[av[0]],theta=cats+[cats[0]],fill='toself',
        name='Class Avg',line_color='#06b6d4',fillcolor='rgba(6,182,212,.1)'))
    fig.update_layout(height=360,
        polar=dict(bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True,range=[0,100],gridcolor="rgba(124,58,237,.2)",tickfont={"color":"#64748b"}),
            angularaxis=dict(gridcolor="rgba(124,58,237,.2)",tickfont={"color":"#94a3b8"})),
        margin=dict(l=50,r=50,t=20,b=20),**plotly_base())
    return fig

def gen_pdf(name, values, label, prob, tips):
    if not HAS_PDF: return None
    pdf=FPDF(); pdf.add_page()
    pdf.set_font("Helvetica","B",22); pdf.set_text_color(100,50,180)
    pdf.cell(0,14,"EduPredict AI - Student Report",ln=True,align="C")
    pdf.set_font("Helvetica","",11); pdf.set_text_color(100,100,100)
    pdf.cell(0,7,f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  Student: {name}",ln=True,align="C")
    pdf.ln(8)
    color=(16,185,129) if label=="Pass" else (239,68,68)
    pdf.set_font("Helvetica","B",18); pdf.set_text_color(*color)
    pdf.cell(0,12,f"Prediction: {label}  ({prob*100:.1f}% pass probability)",ln=True)
    pdf.ln(6)
    pdf.set_font("Helvetica","B",13); pdf.set_text_color(0,0,0)
    pdf.cell(0,9,"Academic Indicators",ln=True)
    pdf.set_font("Helvetica","",11)
    for f,v in values.items(): pdf.cell(0,7,f"  • {LABELS[f]}: {v}",ln=True)
    pdf.ln(6)
    pdf.set_font("Helvetica","B",13); pdf.cell(0,9,"Recommendations",ln=True)
    pdf.set_font("Helvetica","",11)
    for _,t in tips: pdf.cell(0,7,f"  • {t}",ln=True)
    return bytes(pdf.output())

# ── LOGIN PAGE ─────────────────────────────────────────────────────────────────
def show_login():
    st.markdown(ULTRA_CSS, unsafe_allow_html=True)
    _,mid,_ = st.columns([1,1.1,1])
    with mid:
        st.markdown("""
        <div style='text-align:center;padding:40px 0 16px'>
          <div style='font-size:3.8rem'>🎓</div>
          <div class='logo'>EduPredict AI</div>
          <div class='sub'>Advanced Student Performance Intelligence</div>
        </div>""", unsafe_allow_html=True)

        with st.container():
            st.markdown("<div class='lcard'>", unsafe_allow_html=True)

            # OAuth provider buttons
            g_col, gh_col = st.columns(2)
            google_url = _oauth_authorization_url("google")
            github_url = _oauth_authorization_url("github")
            with g_col:
                if google_url:
                    st.link_button("Continue with Google", google_url, use_container_width=True)
                else:
                    st.button("Continue with Google", use_container_width=True, disabled=True)
            with gh_col:
                if github_url:
                    st.link_button("Continue with GitHub", github_url, use_container_width=True)
                else:
                    st.button("Continue with GitHub", use_container_width=True, disabled=True)

            if not google_url or not github_url:
                st.caption("Configure OAuth keys in Streamlit secrets to enable these options.")

            st.markdown("<div class='divdr'>or sign in with credentials</div>", unsafe_allow_html=True)

            uname = st.text_input("👤 Username", placeholder="admin / teacher / student", key="lu")
            pwd   = st.text_input("🔒 Password", type="password", placeholder="Enter password", key="lp")

            if st.session_state.login_error:
                st.error(st.session_state.login_error)

            if st.button("Sign In →", type="primary", use_container_width=True, key="lbtn"):
                if uname in DEMO_USERS and DEMO_USERS[uname]["password"] == pwd:
                    st.session_state.authenticated = True
                    st.session_state.username  = uname
                    st.session_state.user_data = DEMO_USERS[uname]
                    st.session_state.login_error = None
                    st.rerun()
                else:
                    st.session_state.login_error = "❌ Invalid username or password"
                    st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class='nb' style='margin-top:14px'>
          <strong>🎮 Demo credentials:</strong><br>
          <code>admin</code> / <code>admin123</code> &nbsp;|&nbsp;
          <code>teacher</code> / <code>teacher123</code> &nbsp;|&nbsp;
          <code>student</code> / <code>student123</code>
        </div>""", unsafe_allow_html=True)

# ── MAIN APP ───────────────────────────────────────────────────────────────────
def show_app():
    st.markdown(ULTRA_CSS, unsafe_allow_html=True)

    if not (MODEL_PATH.exists() and DATA_PATH.exists()):
        st.error("Run `python src/generate_data.py` and `python src/train.py` first.")
        st.stop()

    df      = load_data()
    model   = load_model()
    metrics = load_metrics()

    # ── SIDEBAR ──────────────────────────────────────────────────────────────
    u = st.session_state.user_data
    with st.sidebar:
        st.markdown(f"""
                <div style='text-align:center;padding:16px 0'>
                    <div style='font-size:2.8rem'>{u['avatar']}</div>
                    <div style='color:#f1f5f9;font-weight:600;margin-top:4px'>{html.escape(str(u['name']))}</div>
                    <div style='color:#64748b;font-size:.75rem'>{html.escape(str(u['email']))}</div>
                    <span style='background:rgba(124,58,237,.3);color:#c4b5fd;padding:3px 10px;
                                border-radius:12px;font-size:.68rem;text-transform:uppercase;letter-spacing:1px'>{html.escape(str(u['role']))}</span>
        </div>
        <hr style='border-color:rgba(124,58,237,.2);margin:12px 0'>
        """, unsafe_allow_html=True)

        page = st.radio("nav",[
            "🏠 Dashboard","🔮 Predict","📊 Analytics",
            "🤖 Model","🔍 Insights","📄 Reports",
            "👥 Batch Predict","⚙️ Settings","ℹ️ About"],
            label_visibility="collapsed")

        st.markdown("<hr style='border-color:rgba(124,58,237,.2);margin:12px 0'>",
                    unsafe_allow_html=True)

        if st.button("🚪 Logout", use_container_width=True):
            for k in ["authenticated","username","user_data"]:
                st.session_state[k] = None if k!="authenticated" else False
            st.rerun()

        pass_rate = (df["result"]=="Pass").mean()*100
        st.markdown(f"""
        <div style='margin-top:14px;padding:12px;background:rgba(124,58,237,.1);
             border-radius:12px;border:1px solid rgba(124,58,237,.2)'>
          <div style='color:#64748b;font-size:.65rem;text-transform:uppercase;letter-spacing:1px'>Quick Stats</div>
          <div style='color:#c4b5fd;font-size:.82rem;margin-top:8px;line-height:1.9'>
            📚 {len(df):,} students<br>
            🎯 {metrics['accuracy']*100:.1f}% accuracy<br>
            ✅ {pass_rate:.0f}% pass rate<br>
            🔮 {len(st.session_state.prediction_history)} predictions
          </div>
        </div>""", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # PAGE: DASHBOARD
    # ════════════════════════════════════════════════════════════════════════
    if page == "🏠 Dashboard":
        st.markdown("<h1>🎓 EduPredict AI</h1><p style='color:#64748b;margin-bottom:28px'>Advanced Student Performance Intelligence Platform</p>", unsafe_allow_html=True)

        c1,c2,c3,c4,c5 = st.columns(5)
        pass_rate = (df["result"]=="Pass").mean()*100
        for col,val,label in zip(
            [c1,c2,c3,c4,c5],
            [f"{len(df):,}", f"{metrics['accuracy']*100:.1f}%", f"{pass_rate:.1f}%",
             str(len(FEATURES)), str(len(st.session_state.prediction_history))],
            ["📚 Students","🎯 Accuracy","✅ Pass Rate","🔬 Features","🔮 Predictions"]):
            col.markdown(f"<div class='mc'><div class='mv'>{val}</div><div class='ml'>{label}</div></div>",
                         unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c1,c2 = st.columns([3,2])

        with c1:
            st.subheader("🌐 3D Student Scatter")
            samp = df.sample(min(500,len(df)),random_state=42)
            fig3d = px.scatter_3d(samp,x="attendance",y="study_hours",z="internal_marks",
                color="result",color_discrete_map={"Pass":"#10b981","Fail":"#ef4444"},
                opacity=.72,size_max=5)
            fig3d.update_layout(height=410,
                scene=dict(bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(gridcolor="rgba(124,58,237,.2)",tickfont=dict(color="#64748b")),
                    yaxis=dict(gridcolor="rgba(124,58,237,.2)",tickfont=dict(color="#64748b")),
                    zaxis=dict(gridcolor="rgba(124,58,237,.2)",tickfont=dict(color="#64748b"))),
                margin=dict(l=0,r=0,t=10,b=0),**plotly_base())
            st.plotly_chart(fig3d,use_container_width=True)

        with c2:
            st.subheader("🍩 Pass/Fail Split")
            cnt = df["result"].value_counts()
            fig_p = go.Figure(go.Pie(labels=cnt.index.tolist(),values=cnt.values.tolist(),hole=.6,
                marker=dict(colors=["#10b981","#ef4444"],line=dict(color="rgba(0,0,0,.5)",width=2))))
            fig_p.update_layout(height=240,margin=dict(l=0,r=0,t=10,b=0),
                annotations=[dict(text=f"{pass_rate:.0f}%\nPass",x=.5,y=.5,
                    font_size=16,showarrow=False,font_color="white")],**plotly_base())
            st.plotly_chart(fig_p,use_container_width=True)

            if st.session_state.prediction_history:
                st.subheader("🕐 Recent")
                for p in st.session_state.prediction_history[-3:][::-1]:
                    bc = "bpass" if p["result"]=="Pass" else "bfail"
                    st.markdown(f"""
                    <div class='gc' style='padding:10px;margin:3px 0'>
                      <span class='bp {bc}'>{p['result']}</span>
                      <span style='color:#64748b;font-size:.73rem;margin-left:8px'>{p['prob']:.1f}% | {p['time']}</span>
                    </div>""", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # PAGE: PREDICT
    # ════════════════════════════════════════════════════════════════════════
    elif page == "🔮 Predict":
        st.markdown("<h1>🔮 Predict Student Result</h1>", unsafe_allow_html=True)
        c1,c2 = st.columns(2)

        with c1:
            st.markdown("<div class='gc'>", unsafe_allow_html=True)
            st.subheader("📝 Academic Indicators")
            vals = {
                "attendance":           st.slider("📅 Attendance (%)",0.,100.,75.,key="pa"),
                "internal_marks":       st.slider("📝 Internal Marks",0.,100.,60.,key="pi"),
                "study_hours":          st.slider("⏰ Study Hours/Day",0.,10.,4.,.5,key="ps"),
                "previous_performance": st.slider("📈 Previous Performance",0.,100.,60.,key="pp"),
                "assignment_score":     st.slider("📋 Assignment Score",0.,100.,60.,key="pas"),
            }
            go_btn = st.button("🚀 Generate Prediction",type="primary",use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            if go_btn:
                lbl,prob = predict(model,vals)
                st.session_state.prediction_history.append(
                    {"result":lbl,"prob":prob*100,"time":datetime.now().strftime("%H:%M:%S"),"values":vals})
                color = ("#10b981","✅","PASS") if lbl=="Pass" else ("#ef4444","❌","FAIL")
                st.markdown(f"""
                <div style='background:linear-gradient(135deg,rgba({",".join(["16,185,129" if lbl=="Pass" else "239,68,68"])},0.2),rgba(6,182,212,.08));
                     border:1px solid rgba({",".join(["16,185,129" if lbl=="Pass" else "239,68,68"])},0.5);
                     border-radius:16px;padding:24px;text-align:center;margin-bottom:12px'>
                  <div style='font-size:2.8rem'>{color[1]}</div>
                  <div style='font-family:Orbitron,monospace;font-size:2rem;color:{color[0]};font-weight:900'>{color[2]}</div>
                  <div style='color:#94a3b8'>{prob*100:.1f}% pass probability</div>
                </div>""", unsafe_allow_html=True)
                st.plotly_chart(gauge(prob),use_container_width=True)
                st.subheader("💡 Recommendations")
                for t,txt in recommendations(vals):
                    (st.success if t=="ok" else st.warning)(txt)
            else:
                st.markdown("""
                <div style='text-align:center;padding:70px 20px;color:#475569'>
                  <div style='font-size:3.5rem;margin-bottom:14px'>🎯</div>
                  <div>Set indicators and click<br><strong style='color:#7c3aed'>Generate Prediction</strong></div>
                </div>""", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # PAGE: ANALYTICS
    # ════════════════════════════════════════════════════════════════════════
    elif page == "📊 Analytics":
        st.markdown("<h1>📊 Dataset Analytics</h1>", unsafe_allow_html=True)
        c1,c2,c3 = st.columns(3)
        c1.metric("Records",f"{df.shape[0]:,}"); c2.metric("Features",df.shape[1]); c3.metric("Missing",int(df.isna().sum().sum()))

        st.subheader("📋 Data Preview")
        st.dataframe(df.head(20),use_container_width=True)

        col1,col2 = st.columns(2)
        with col1:
            st.subheader("📈 Feature Distribution")
            feat = st.selectbox("Choose feature",FEATURES,format_func=lambda f:LABELS[f])
            fig_h = px.histogram(df,x=feat,color="result",nbins=40,barmode="overlay",
                color_discrete_map={"Pass":"#10b981","Fail":"#ef4444"})
            fig_h.update_layout(xaxis=dict(gridcolor="rgba(124,58,237,.2)"),
                yaxis=dict(gridcolor="rgba(124,58,237,.2)"),**plotly_base())
            st.plotly_chart(fig_h,use_container_width=True)

        with col2:
            st.subheader("🔥 Correlation Heatmap")
            corr = df[FEATURES].corr()
            fig_hm = px.imshow(corr,text_auto=True,color_continuous_scale="Viridis")
            fig_hm.update_layout(height=360,**plotly_base())
            st.plotly_chart(fig_hm,use_container_width=True)

        st.subheader("📦 Box Plots by Result")
        melt = df.melt(id_vars="result",value_vars=FEATURES,var_name="Feature",value_name="Value")
        fig_b = px.box(melt,x="Feature",y="Value",color="result",
            color_discrete_map={"Pass":"#10b981","Fail":"#ef4444"})
        fig_b.update_layout(xaxis=dict(gridcolor="rgba(124,58,237,.2)"),
            yaxis=dict(gridcolor="rgba(124,58,237,.2)"),**plotly_base())
        st.plotly_chart(fig_b,use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # PAGE: MODEL
    # ════════════════════════════════════════════════════════════════════════
    elif page == "🤖 Model":
        st.markdown("<h1>🤖 Model Performance</h1>", unsafe_allow_html=True)
        c1,c2,c3 = st.columns(3)
        c1.metric("Best Model",metrics["best_model"])
        c2.metric("Accuracy",f"{metrics['accuracy']*100:.1f}%")
        c3.metric("Test Samples",metrics.get("n_test","N/A"))

        col1,col2 = st.columns(2)
        with col1:
            st.subheader("🔀 Confusion Matrix")
            cm = np.array(metrics["confusion_matrix"])
            fig_cm = px.imshow(cm,text_auto=True,
                x=["Pred Fail","Pred Pass"],y=["Actual Fail","Actual Pass"],
                color_continuous_scale=[[0,"#1a0533"],[.5,"#7c3aed"],[1,"#06b6d4"]])
            fig_cm.update_layout(height=300,**plotly_base())
            st.plotly_chart(fig_cm,use_container_width=True)

        with col2:
            st.subheader("🏆 Feature Importance")
            imp = pd.Series(metrics["feature_importance"]).rename(index=LABELS).sort_values()
            fig_i = px.bar(x=imp.values,y=imp.index,orientation="h",
                color=imp.values,color_continuous_scale="Viridis")
            fig_i.update_layout(height=300,showlegend=False,
                xaxis=dict(gridcolor="rgba(124,58,237,.2)"),yaxis=dict(gridcolor="rgba(0,0,0,0)"),
                **plotly_base())
            st.plotly_chart(fig_i,use_container_width=True)

        rp = ROOT/"reports"/"model_results.csv"
        if rp.exists():
            st.subheader("⚔️ Model Comparison")
            comp = pd.read_csv(rp)
            fig_c = px.bar(comp,x="model",y="test_accuracy",color="test_accuracy",
                color_continuous_scale="Viridis",text="test_accuracy")
            fig_c.update_traces(texttemplate="%{text:.3f}",textposition="outside")
            fig_c.update_layout(yaxis=dict(range=[0,1.1],gridcolor="rgba(124,58,237,.2)"),
                **plotly_base())
            st.plotly_chart(fig_c,use_container_width=True)

        st.subheader("📋 Classification Report")
        st.dataframe(pd.DataFrame(metrics["classification_report"]).T.round(3),use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # PAGE: INSIGHTS
    # ════════════════════════════════════════════════════════════════════════
    elif page == "🔍 Insights":
        st.markdown("<h1>🔍 Student Insights</h1>", unsafe_allow_html=True)
        st.caption("Live profiling — drag sliders to see risk factors update in real time")
        c1,c2 = st.columns(2)

        with c1:
            vals = {
                "attendance":           st.slider("📅 Attendance (%)",0.,100.,75.,key="ia"),
                "internal_marks":       st.slider("📝 Internal Marks",0.,100.,60.,key="ii"),
                "study_hours":          st.slider("⏰ Study Hours/Day",0.,10.,4.,.5,key="is"),
                "previous_performance": st.slider("📈 Previous Performance",0.,100.,60.,key="ip"),
                "assignment_score":     st.slider("📋 Assignment Score",0.,100.,60.,key="ias"),
            }
        with c2:
            lbl,prob = predict(model,vals)
            st.plotly_chart(gauge(prob),use_container_width=True)
            avg = df[FEATURES].mean().to_dict()
            st.plotly_chart(radar(vals,avg),use_container_width=True)

        tips = recommendations(vals)
        cols = st.columns(len(tips))
        for i,(t,txt) in enumerate(tips):
            with cols[i]:
                icon,clr = ("✅","#10b981") if t=="ok" else ("⚠️","#f59e0b")
                st.markdown(f"""
                <div class='gc' style='text-align:center;padding:14px'>
                  <div style='font-size:1.4rem'>{icon}</div>
                  <div style='color:{clr};font-size:.78rem;margin-top:6px'>{txt}</div>
                </div>""", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # PAGE: REPORTS
    # ════════════════════════════════════════════════════════════════════════
    elif page == "📄 Reports":
        st.markdown("<h1>📄 PDF Report Generator</h1>", unsafe_allow_html=True)
        c1,c2 = st.columns(2)

        with c1:
            sname = st.text_input("Student Name",value=u["name"])
            vals = {
                "attendance":           st.slider("📅 Attendance (%)",0.,100.,75.,key="ra"),
                "internal_marks":       st.slider("📝 Internal Marks",0.,100.,60.,key="ri"),
                "study_hours":          st.slider("⏰ Study Hours/Day",0.,10.,4.,.5,key="rs"),
                "previous_performance": st.slider("📈 Previous Performance",0.,100.,60.,key="rp"),
                "assignment_score":     st.slider("📋 Assignment Score",0.,100.,60.,key="ras"),
            }
            if st.button("🚀 Generate Report",type="primary",use_container_width=True):
                lbl,prob = predict(model,vals)
                tips = recommendations(vals)
                if HAS_PDF:
                    data = gen_pdf(sname,vals,lbl,prob,tips)
                    st.download_button("⬇️ Download PDF",data,
                        f"report_{sname.replace(' ','_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        "application/pdf",use_container_width=True)
                    st.success("✅ PDF ready — click above to download!")
                else:
                    rpt = f"EDUPREDICT AI REPORT\n{'='*50}\nStudent: {sname}\nDate: {datetime.now()}\nResult: {lbl} ({prob*100:.1f}%)\n\nINDICATORS:\n"
                    for f,v in vals.items(): rpt+=f"  {LABELS[f]}: {v}\n"
                    rpt+="\nRECOMMENDATIONS:\n"
                    for _,t in tips: rpt+=f"  • {t}\n"
                    st.download_button("⬇️ Download TXT",rpt,f"report_{sname}.txt","text/plain",use_container_width=True)
                    st.info("Install `fpdf2` for PDF: `pip install fpdf2`")

        with c2:
            if st.session_state.prediction_history:
                hdf = pd.DataFrame(st.session_state.prediction_history)
                fig_h = px.bar(hdf,x=hdf.index,y="prob",color="result",
                    color_discrete_map={"Pass":"#10b981","Fail":"#ef4444"},
                    labels={"prob":"Pass Prob (%)","index":"Prediction #"})
                fig_h.update_layout(yaxis=dict(gridcolor="rgba(124,58,237,.2)"),**plotly_base())
                st.plotly_chart(fig_h,use_container_width=True)
            else:
                st.info("Make some predictions first to see history chart here.")

    # ════════════════════════════════════════════════════════════════════════
    # PAGE: BATCH PREDICT
    # ════════════════════════════════════════════════════════════════════════
    elif page == "👥 Batch Predict":
        st.markdown("<h1>👥 Batch Prediction</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color:#64748b'>Upload CSV → AI predicts all students at once</p>",
                    unsafe_allow_html=True)

        tmpl = pd.DataFrame({f:[0.0] for f in FEATURES})
        st.download_button("⬇️ Download CSV Template",tmpl.to_csv(index=False),
            "student_template.csv","text/csv")
        st.markdown(f"""<div class='nb'>Required columns: <code>{', '.join(FEATURES)}</code></div>""",
                    unsafe_allow_html=True)

        up = st.file_uploader("Upload CSV",type=["csv"])
        if up:
            try:
                bdf = pd.read_csv(up)
                miss = [f for f in FEATURES if f not in bdf.columns]
                if miss:
                    st.error(f"Missing columns: {', '.join(miss)}")
                else:
                    probs = model.predict_proba(bdf[FEATURES])[:,1]
                    bdf["pass_probability"] = (probs*100).round(1)
                    bdf["prediction"] = ["Pass" if p>=.5 else "Fail" for p in probs]
                    bdf["risk_level"]  = ["Low" if p>=.75 else ("Medium" if p>=.5 else "High") for p in probs]

                    c1,c2,c3 = st.columns(3)
                    c1.metric("Total",len(bdf))
                    c2.metric("Pass",(bdf["prediction"]=="Pass").sum())
                    c3.metric("Fail",(bdf["prediction"]=="Fail").sum())

                    st.dataframe(bdf,use_container_width=True)
                    st.download_button("⬇️ Download Results",bdf.to_csv(index=False),
                        "batch_results.csv","text/csv",use_container_width=True)

                    fig_sc = px.scatter(bdf,x="attendance",y="internal_marks",
                        color="prediction",size="pass_probability",
                        color_discrete_map={"Pass":"#10b981","Fail":"#ef4444"})
                    fig_sc.update_layout(xaxis=dict(gridcolor="rgba(124,58,237,.2)"),
                        yaxis=dict(gridcolor="rgba(124,58,237,.2)"),**plotly_base())
                    st.plotly_chart(fig_sc,use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")

    # ════════════════════════════════════════════════════════════════════════
    # PAGE: SETTINGS
    # ════════════════════════════════════════════════════════════════════════
    elif page == "⚙️ Settings":
        st.markdown("<h1>⚙️ Settings & Profile</h1>", unsafe_allow_html=True)
        c1,c2 = st.columns(2)

        with c1:
            st.subheader("👤 Profile")
            st.text_input("Full Name",u["name"],disabled=True)
            st.text_input("Email",u["email"],disabled=True)
            st.text_input("Role",u["role"],disabled=True)

            st.subheader("🔑 OAuth Setup")
            st.markdown("""
            <div class='gc'>
              <h4 style='color:#c4b5fd;margin-top:0'>🔴 Google OAuth</h4>
              <ol style='color:#94a3b8;font-size:.85rem'>
                <li>Go to <a href='https://console.cloud.google.com' target='_blank' style='color:#7c3aed'>Google Cloud Console</a></li>
                <li>APIs & Services → Credentials → OAuth 2.0 Client ID</li>
                <li>Add your redirect URI</li>
                                <li>Add to <code>.streamlit/secrets.toml</code>:<br>
                  <code>GOOGLE_CLIENT_ID=xxx</code><br>
                  <code>GOOGLE_CLIENT_SECRET=xxx</code></li>
              </ol>
              <h4 style='color:#c4b5fd'>⚫ GitHub OAuth</h4>
              <ol style='color:#94a3b8;font-size:.85rem'>
                <li>Go to <a href='https://github.com/settings/developers' target='_blank' style='color:#7c3aed'>GitHub Developer Settings</a></li>
                <li>New OAuth App → fill callback URL</li>
                                <li>Add to <code>.streamlit/secrets.toml</code>:<br>
                  <code>GITHUB_CLIENT_ID=xxx</code><br>
                  <code>GITHUB_CLIENT_SECRET=xxx</code></li>
              </ol>
                            <p style='color:#94a3b8;font-size:.8rem'>Set <code>OAUTH_REDIRECT_URI</code> to this app's exact URL (defaults to <code>http://localhost:8501</code>). Never commit OAuth secrets.</p>
            </div>""", unsafe_allow_html=True)

        with c2:
            st.subheader("📊 Session Stats")
            ph = st.session_state.prediction_history
            st.metric("Predictions This Session",len(ph))
            if ph:
                hdf = pd.DataFrame(ph)
                st.metric("Pass",int((hdf["result"]=="Pass").sum()))
                st.metric("Fail",int((hdf["result"]=="Fail").sum()))
            if st.button("🗑️ Clear History",use_container_width=True):
                st.session_state.prediction_history=[]
                st.success("History cleared!")

    # ════════════════════════════════════════════════════════════════════════
    # PAGE: ABOUT
    # ════════════════════════════════════════════════════════════════════════
    else:
        st.markdown("<h1>ℹ️ About EduPredict AI</h1>", unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("""
            <div class='gc'>
              <h3 style='color:#c4b5fd;margin-top:0'>🎯 Project Overview</h3>
              <p style='color:#94a3b8'>EduPredict AI uses machine learning to predict student academic outcomes based on key educational indicators.</p>
              <h4 style='color:#c4b5fd'>🔬 Input Features</h4>
              <ul style='color:#94a3b8'>
                <li>Attendance Percentage</li><li>Internal Marks</li>
                <li>Daily Study Hours</li><li>Previous Term Performance</li>
                <li>Assignment Scores</li>
              </ul>
              <h4 style='color:#c4b5fd'>⚡ Tech Stack</h4>
              <ul style='color:#94a3b8'>
                <li>🐍 Python 3 + Scikit-learn</li>
                <li>🌊 Streamlit (UI)</li>
                <li>📊 Plotly (interactive 3D charts)</li>
                <li>🌌 Three.js (particle background)</li>
                <li>💎 Glassmorphism CSS design</li>
              </ul>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div class='gc'>
              <h3 style='color:#c4b5fd;margin-top:0'>🚀 v2.0 New Features</h3>
              <ul style='color:#94a3b8'>
                <li>✅ Login with Google / GitHub (OAuth UI)</li>
                <li>✅ Three.js 3D particle background</li>
                <li>✅ Glassmorphism dark UI design</li>
                <li>✅ Plotly 3D scatter visualisation</li>
                <li>✅ Animated Gauge + Radar charts</li>
                <li>✅ Prediction history tracking</li>
                <li>✅ PDF report generation</li>
                <li>✅ Batch CSV prediction</li>
                <li>✅ User roles (admin/teacher/student)</li>
                <li>✅ Interactive analytics dashboard</li>
              </ul>
              <div style='margin-top:14px;padding:12px;background:rgba(239,68,68,.1);
                   border-radius:8px;border:1px solid rgba(239,68,68,.25)'>
                <div style='color:#fca5a5;font-size:.8rem'>
                  <strong>⚠️ Disclaimer:</strong> Predictions are statistical estimates for
                  educational purposes only and should support — not replace — teacher judgement.
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

# ── Router ─────────────────────────────────────────────────────────────────────
handle_oauth_callback()

if st.session_state.authenticated:
    show_app()
else:
    show_login()
