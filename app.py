'''
STREAMLIT CHAT INTERFACE FOR THE DISCHARGE INSTRUCTIONS ASSISTANT

'''
import json
import streamlit as st
import time
import html as html_lib
from pipeline import DischargeAssistant, get_llm_client


st.set_page_config(page_title='Discharge Assistant',layout='centered')

# STYLING
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; max-width: 820px; }
 
    .app-header {
        position: relative; overflow: hidden;
        background: linear-gradient(135deg, #155e75 0%, #0e7490 55%, #0891b2 100%);
        padding: 1.5rem 1.8rem; border-radius: 18px; margin-bottom: 1.2rem;
        animation: glowPulse 3s ease-in-out infinite;
    }
    @keyframes glowPulse { 0%,100% { box-shadow: 0 8px 24px rgba(8,145,178,0.20); }
                           50% { box-shadow: 0 8px 34px rgba(34,211,238,0.50); } }
    .app-header .content { position: relative; z-index: 2; }
    .app-header h1 { color: #ffffff; margin: 0; font-size: 1.6rem; font-weight: 700; }
    .app-header p { color: #cffafe; margin: 0.5rem 0 0 0; font-size: 0.9rem; line-height: 1.45; }
    .ecg { position: absolute; left: 0; bottom: 6px; width: 100%; height: 48px;
           opacity: 0.4; z-index: 1; pointer-events: none; }
    .ecg path { filter: drop-shadow(0 0 5px #7dd3fc); }
 
    @keyframes fadeUp { from { opacity: 0; transform: translateY(10px); }
                        to   { opacity: 1; transform: translateY(0); } }
    .bubble-row { display: flex; margin: 0.45rem 0; animation: fadeUp 0.35s ease both; }
    .bubble { max-width: 78%; padding: 0.7rem 1rem; border-radius: 18px; color: #f0f9ff;
              line-height: 1.5; font-size: 0.95rem; box-shadow: 0 2px 8px rgba(0,0,0,0.25); }
    .bubble-user { background: linear-gradient(135deg, #0e7490, #0891b2); border-bottom-right-radius: 4px; }
    .bubble-bot  { background: #1f2937; border-bottom-left-radius: 4px; }
 
    @keyframes blink { 0%, 80%, 100% { opacity: 0.2; } 40% { opacity: 1; } }
    .typing span { animation: blink 1.4s infinite both; font-size: 0.7rem; }
    .typing span:nth-child(2) { animation-delay: 0.2s; }
    .typing span:nth-child(3) { animation-delay: 0.4s; }
 
    .stButton button { border-radius: 12px; border: 1px solid rgba(8,145,178,0.55);
        background: rgba(8,145,178,0.10); color: #e0f7ff; text-align: left;
        padding: 0.7rem 1rem; font-weight: 500; transition: all 0.15s ease; }
    .stButton button:hover { background: rgba(8,145,178,0.28); border-color: #22d3ee; color: #ffffff; }
    </style>
    """,
    unsafe_allow_html=True,
)
 
st.markdown(
    """
    <div class="app-header">
      <svg class="ecg" viewBox="0 0 600 50" preserveAspectRatio="none">
        <g>
          <path d="M -120 25 L -72 25 L -64 25 L -58 6 L -50 44 L -42 25 L 0 25 L 0 25 L 48 25 L 56 25 L 62 6 L 70 44 L 78 25 L 120 25 L 120 25 L 168 25 L 176 25 L 182 6 L 190 44 L 198 25 L 240 25 L 240 25 L 288 25 L 296 25 L 302 6 L 310 44 L 318 25 L 360 25 L 360 25 L 408 25 L 416 25 L 422 6 L 430 44 L 438 25 L 480 25 L 480 25 L 528 25 L 536 25 L 542 6 L 550 44 L 558 25 L 600 25 L 600 25 L 648 25 L 656 25 L 662 6 L 670 44 L 678 25 L 720 25" fill="none" stroke="#bae6fd" stroke-width="2.2"
                stroke-linecap="round" stroke-linejoin="round"/>
          <animateTransform attributeName="transform" type="translate"
                            from="0 0" to="-120 0" dur="1.05s" repeatCount="indefinite"/>
        </g>
      </svg>
      <div class="content">
        <h1>Discharge Instructions Assistant</h1>
        <p>Ask anything about your discharge summary. This explains the instructions you
        were given. It does not provide medical advice. For anything urgent, contact your
        care team or emergency services.</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
 
# Each entry maps section keywords to a question. A question is only offered if the
# selected note actually contains a matching section, so suggestions stay answerable.
QUESTION_MAP = [
    (("hospital course",), "What happened during my hospital stay?"),
    (("diagnosis",), "What was I diagnosed with?"),
    (("medication",), "What medications was I prescribed?"),
    (("follow-up", "follow up"), "When is my follow-up appointment?"),
    (("discharge instruction",), "What are my discharge instructions?"),
    (("condition at discharge", "discharge condition"), "What was my condition at discharge?"),
    (("treatment",), "What treatment did I receive?"),
    (("medical history",), "What is in my medical history?"),
    (("diet", "nutrition"), "What should I eat while recovering?"),
    (("activity", "restriction", "precaution"), "What activities should I avoid?"),
]
 
UNIVERSAL_QUESTIONS = [
    "What happened during my hospital stay?",
    "What was my condition at discharge?",
    "What should I do to recover at home?",
]
 
 
def note_sections(note_text):
    secs = set()
    for line in note_text.split("\n"):
        s = line.strip()
        if s.endswith(":") and 1 < len(s) <= 45:
            secs.add(s.rstrip(":").strip().lower())
    return secs
 
 
def suggested_questions(note_text, max_q=4):
    secs = note_sections(note_text)
    found = [q for keywords, q in QUESTION_MAP
             if any(k in sec for k in keywords for sec in secs)]
    found += UNIVERSAL_QUESTIONS
    seen, out = set(), []
    for q in found:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out[:max_q]
 
TYPING_HTML = ('<div class="bubble-row" style="justify-content:flex-start;">'
               '<div class="bubble bubble-bot typing"><span>&#9679;</span>'
               '<span>&#9679;</span><span>&#9679;</span></div></div>')
 
 
def bubble_html(role, text):
    safe = html_lib.escape(text).replace("\n", "<br>")
    side = "flex-end" if role == "user" else "flex-start"
    cls = "bubble-user" if role == "user" else "bubble-bot"
    return (f'<div class="bubble-row" style="justify-content:{side};">'
            f'<div class="bubble {cls}">{safe}</div></div>')
 
 
def render_bubble(role, text):
    st.markdown(bubble_html(role, text), unsafe_allow_html=True)
 
 
@st.cache_data
def load_notes():
    with open("data/notes.json") as f:
        return json.load(f)
 
 
notes = load_notes()
 
 
def note_label(note_text):
    lines = note_text.split("\n")
    for line in lines:
        s = line.strip()
        if s.lower().startswith("patient:"):
            label = s[len("patient:"):].strip()
            if label and "[" not in label:
                return label[:60]
    for line in lines:
        s = line.strip()
        if s and not s.endswith(":") and "[" not in s:
            return s[:60]
    return "discharge summary"
 
 
# SIDEBAR
st.sidebar.header("Patient record")
idx = st.sidebar.selectbox(
    "Choose a discharge summary",
    range(len(notes)),
    format_func=lambda i: f"{notes[i]['patient_id']}: {note_label(notes[i]['note'])}",
)
with st.sidebar.expander("View the full discharge summary"):
    st.text(notes[idx]["note"])
 
st.sidebar.markdown("**Questions you can ask this patient**")
for _q in suggested_questions(notes[idx]["note"]):
    st.sidebar.markdown(f"- {_q}")
 
if st.sidebar.button("Clear conversation"):
    st.session_state.messages = []
    st.rerun()
 
 
@st.cache_resource
def build_assistant(i):
    return DischargeAssistant(notes[i]["note"], llm_client=get_llm_client())
 
 
assistant = build_assistant(idx)
 
if st.session_state.get("note_idx") != idx:
    st.session_state.messages = []
    st.session_state.note_idx = idx
 
question = None
 
if not st.session_state.get("messages"):
    st.markdown("##### Questions you can ask this patient. Tap one, or type your own:")
    cols = st.columns(2)
    for i, q in enumerate(suggested_questions(notes[idx]["note"])):
        if cols[i % 2].button(q, use_container_width=True):
            question = q
 
for m in st.session_state.get("messages", []):
    render_bubble(m["role"], m["content"])
 
typed = st.chat_input("Ask about your discharge instructions...")
if typed:
    question = typed
 
if question:
    st.session_state.setdefault("messages", [])
    st.session_state.messages.append({"role": "user", "content": question})
    render_bubble("user", question)
 
    placeholder = st.empty()
    placeholder.markdown(TYPING_HTML, unsafe_allow_html=True)
   
    #TYPING ANIMATION
    start = time.time()
    reply = assistant.answer(question)
 
    pause = 0.8 - (time.time() - start)
    if pause > 0:
        time.sleep(pause)
 
    # reveal the reply word by word, like a real chatbot
    words = reply.split(" ")
    delay = 0.03 if len(words) < 60 else 0.012
    shown = ""
    for w in words:
        shown += w + " "
        placeholder.markdown(bubble_html("assistant", shown), unsafe_allow_html=True)
        time.sleep(delay)
    placeholder.markdown(bubble_html("assistant", reply), unsafe_allow_html=True)
 
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()
 