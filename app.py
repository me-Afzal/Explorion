"""
app.py
------
Explorion — Streamlit UI for the Agentic Research Pipeline.
Run with: streamlit run app.py
"""

import json
import re
import threading

from dotenv import load_dotenv
load_dotenv()   # load .env before any package import reads env vars

import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

from assistant.graph import build_graph, stream_pipeline, PipelineState
from assistant.agent_tool import (
    search_log as tavily_log,
    register_search_callback,
    clear_search_callback,
)
from langsmith import trace as ls_trace
from assistant.vectordb import initialize, create_session, store_interaction, retrieve_context

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Explorion",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS  — black & white professional theme
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
/* ── hide default Streamlit chrome (only footer/menu, NOT the header which holds sidebar toggle) ── */
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent !important; }

/* ── global font ── */
html, body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}

/* ── main content area ── */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: #000 !important;
    color: #fff !important;
}

/* ── top header bar ── */
.exp-header {
    background: #000;
    padding: 8px 2rem;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    margin: -4rem -4rem 1.5rem -4rem;
    border-bottom: 1px solid #fff;
}
.exp-logo  { font-size: 32px; font-weight: 800; color: #fff; letter-spacing: -0.5px; }
.exp-sep   { color: #666; font-size: 20px; }
.exp-tag   { font-size: 14px; color: #aaa; font-weight: 400; }

/* ── sidebar ── */
section[data-testid="stSidebar"] {
    display: flex !important;
    background-color: #000 !important;
    border-right: 1px solid #fff !important;
    color: #fff !important;
}
section[data-testid="stSidebar"] * { color: #fff !important; }

/* ── context cards ── */
.ctx-card {
    background: #000;
    border: 1px solid #fff;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
}
.ctx-label {
    font-size: 10px; font-weight: 700; color: #aaa;
    text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 3px;
}
.ctx-value { font-size: 13px; color: #fff; font-weight: 500; }

/* ── report card ── */
.report-card {
    border: 1px solid #fff;
    border-radius: 10px;
    padding: 18px 22px;
    margin: 0.5rem 0 1rem 0;
    background: #000;
}
.report-header {
    font-size: 13px; font-weight: 700; color: #fff;
    margin-bottom: 10px; display: flex; align-items: center; gap: 6px;
}

/* ── chat messages ── */
[data-testid="stChatMessage"] {
    background: #111 !important;
    border: 1px solid #fff !important;
    border-radius: 10px !important;
    color: #fff !important;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] td { color: #fff !important; }

/* ── buttons ── */
div[data-testid="stButton"] > button,
div[data-testid="stDownloadButton"] > button {
    background: #000 !important;
    color: #fff !important;
    border: 1px solid #fff !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 8px 18px !important;
}
div[data-testid="stButton"] > button:hover,
div[data-testid="stDownloadButton"] > button:hover {
    background: #222 !important;
    border-color: #fff !important;
}

/* ── chat input ── */
div[data-testid="stChatInput"] textarea {
    background: #111 !important;
    color: #fff !important;
    border: 1px solid #fff !important;
    border-radius: 8px !important;
    min-height: 80px !important;
    font-size: 15px !important;
    padding: 14px 16px !important;
    line-height: 1.5 !important;
}
div[data-testid="stChatInput"] textarea:focus {
    border-color: #fff !important;
    box-shadow: 0 0 0 1px #fff !important;
}
div[data-testid="stChatInput"] {
    background: #000 !important;
    border-top: 1px solid #333 !important;
    padding: 12px 0 !important;
}

/* ── expander ── */
div[data-testid="stExpander"] {
    background: #111 !important;
    border: 1px solid #fff !important;
    border-radius: 8px !important;
}
div[data-testid="stExpander"] summary,
div[data-testid="stExpander"] p,
div[data-testid="stExpander"] li { color: #fff !important; }

/* ── status box (progress) ── */
div[data-testid="stStatusWidget"],
div[data-testid="stStatus"] {
    background: #111 !important;
    border: 1px solid #fff !important;
    border-radius: 8px !important;
    color: #fff !important;
}
div[data-testid="stStatusWidget"] p,
div[data-testid="stStatus"] p,
div[data-testid="stStatus"] span { color: #fff !important; }

/* ── spinner ── */
div[data-testid="stSpinner"] p { color: #fff !important; }

/* ── divider ── */
hr { border-color: #333 !important; }

/* ── scrollbar ── */
::-webkit-scrollbar { width: 6px; background: #000; }
::-webkit-scrollbar-thumb { background: #444; border-radius: 3px; }

/* ── sidebar close button (inside sidebar) — transparent bg, white icon ── */
[data-testid="stSidebarCollapseButton"] button,
button[aria-label="Close sidebar"] {
    background: transparent !important;
    border: 1px solid #fff !important;
    border-radius: 6px !important;
    width: 34px !important;
    height: 34px !important;
}
[data-testid="stSidebarCollapseButton"] button:hover,
button[aria-label="Close sidebar"]:hover {
    background: #222 !important;
}

/* ── sidebar open button (main area, shown when sidebar is collapsed) — white bg, black icon ── */
[data-testid="collapsedControl"],
[data-testid="collapsedControl"] button,
button[aria-label="Open sidebar"] {
    background: #fff !important;
    border-radius: 6px !important;
    color: #000 !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PDF GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def _clean_for_pdf(text: str) -> str:
    """Replace common non-latin-1 characters so fpdf2 doesn't break."""
    replacements = {
        "’": "'",  "‘": "'",
        "“": '"',  "”": '"',
        "—": "--", "–": "-",
        "•": "*",  "→": "->",
        "·": "*",  "…": "...",
    }
    for ch, rep in replacements.items():
        text = text.replace(ch, rep)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def generate_pdf(report_text: str, topic: str) -> bytes:
    from fpdf import FPDF, XPos, YPos

    pdf = FPDF()
    pdf.set_margins(20, 20, 20)
    pdf.add_page()

    # ── header ──────────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 22)
    pdf.cell(0, 10, "Explorion", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6, "Research Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)
    pdf.set_draw_color(0, 0, 0)
    pdf.line(20, pdf.get_y(), 190, pdf.get_y())
    pdf.ln(6)

    # ── body — parse markdown ────────────────────────────────────────────────
    NX, NY = XPos.LMARGIN, YPos.NEXT   # always reset cursor to left margin
    for raw_line in report_text.split("\n"):
        line = _clean_for_pdf(raw_line.rstrip())
        pdf.set_x(pdf.l_margin)        # guard: ensure x is at left margin

        if line.startswith("#### "):
            pdf.set_font("Helvetica", "B", 10)
            pdf.multi_cell(0, 6, line[5:], new_x=NX, new_y=NY)
            pdf.ln(1)
        elif line.startswith("### "):
            pdf.set_font("Helvetica", "B", 11)
            pdf.multi_cell(0, 7, line[4:], new_x=NX, new_y=NY)
            pdf.ln(2)
        elif line.startswith("## "):
            pdf.set_font("Helvetica", "B", 13)
            pdf.multi_cell(0, 8, line[3:], new_x=NX, new_y=NY)
            pdf.ln(2)
        elif line.startswith("# "):
            pdf.set_font("Helvetica", "B", 16)
            pdf.multi_cell(0, 10, line[2:], new_x=NX, new_y=NY)
            pdf.ln(3)
        elif re.match(r"^(\-|\*|\+) ", line):
            body = re.sub(r"\*\*(.+?)\*\*", r"\1", line[2:])
            body = re.sub(r"\*(.+?)\*", r"\1", body)
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 5, f"  - {body}", new_x=NX, new_y=NY)
        elif re.match(r"^\d+\. ", line):
            body = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
            body = re.sub(r"\*(.+?)\*", r"\1", body)
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 5, f"  {body}", new_x=NX, new_y=NY)
        elif line.strip() == "" or line.strip().startswith("---"):
            pdf.ln(3)
        else:
            clean = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
            clean = re.sub(r"\*(.+?)\*", r"\1", clean)
            clean = re.sub(r"`(.+?)`", r"\1", clean)
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 5, clean, new_x=NX, new_y=NY)

    # ── footer ───────────────────────────────────────────────────────────────
    pdf.set_y(-18)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 6, f"Generated by Explorion  |  Page {pdf.page_no()}", align="C")

    return bytes(pdf.output())


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════════════════

BUFFER_MAX = 2

if "session_ready" not in st.session_state:
    st.session_state.messages           = []
    st.session_state.context            = {"topic": None, "goal": None, "location": None}
    st.session_state.buffer_memory      = []
    st.session_state.session_id         = create_session()
    st.session_state.graph              = build_graph()
    st.session_state.report_topic       = None
    st.session_state.interaction_count  = 0
    st.session_state.session_ready      = True

# One-time VectorDB + embedding model init (per Streamlit process, not per session)
if "db_ready" not in st.session_state:
    with st.spinner("Starting up Explorion — loading models…"):
        try:
            initialize()
            st.session_state.db_ready = True
        except Exception as e:
            st.error(f"Startup error: {e}")
            st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="exp-header">
    <span class="exp-logo">Explorion</span>
    <span class="exp-sep">|</span>
    <span class="exp-tag">Intelligent Research Platform</span>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — research context + controls
# ══════════════════════════════════════════════════════════════════════════════

CARD = (
    "background:#000;border:1px solid #fff;border-radius:8px;"
    "padding:10px 14px;margin-bottom:8px;"
)
LABEL = (
    "font-size:10px;font-weight:700;color:#aaa;"
    "text-transform:uppercase;letter-spacing:0.6px;margin-bottom:4px;"
)
VALUE = "font-size:13px;color:#fff;font-weight:500;word-break:break-word;"

with st.sidebar:
    st.markdown("### Research Context")
    st.markdown("<br>", unsafe_allow_html=True)

    ctx          = st.session_state.context
    topic_val    = ctx.get("topic")    or "—"
    goal_val     = ctx.get("goal")     or "—"
    loc_raw      = ctx.get("location")
    location_val = "—" if not loc_raw or loc_raw == "N/A" else loc_raw

    for label, value in [("Topic", topic_val), ("Goal", goal_val), ("Location", location_val)]:
        st.markdown(
            f'<div style="{CARD}">'
            f'  <div style="{LABEL}">{label}</div>'
            f'  <div style="{VALUE}">{value}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("New Research Session", use_container_width=True):
        keys_to_clear = [
            "messages", "context", "buffer_memory",
            "session_id", "graph", "report_topic", "session_ready",
            "interaction_count",
        ]
        for k in keys_to_clear:
            st.session_state.pop(k, None)
        st.rerun()

    st.divider()
    st.markdown("**How it works**")
    st.markdown(
        "1. Tell me what you want to research\n"
        "2. Share your goal\n"
        "3. Optionally specify a location\n"
        "4. Confirm to start deep research\n"
        "5. Download your report as PDF"
    )


# ══════════════════════════════════════════════════════════════════════════════
# CHAT — render existing messages
# ══════════════════════════════════════════════════════════════════════════════

for msg in st.session_state.messages:
    if msg["type"] == "report":
        with st.chat_message("assistant"):
            st.markdown("**Research complete!** Your report is ready.")
            st.markdown('<div class="report-card">', unsafe_allow_html=True)
            st.markdown('<div class="report-header">📄 Research Report</div>', unsafe_allow_html=True)
            with st.expander("Preview report", expanded=False):
                st.markdown(msg["content"])
            st.markdown("</div>", unsafe_allow_html=True)

            topic_slug = (st.session_state.report_topic or "research").lower().replace(" ", "_")[:40]
            st.download_button(
                label="⬇  Download PDF",
                data=generate_pdf(msg["content"], st.session_state.report_topic or "Research"),
                file_name=f"explorion_{topic_slug}.pdf",
                mime="application/pdf",
                key=f"dl_{id(msg)}",
            )
    else:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


# ══════════════════════════════════════════════════════════════════════════════
# CHAT INPUT — process new message
# ══════════════════════════════════════════════════════════════════════════════

if prompt := st.chat_input("What would you like to research?"):

    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt, "type": "text"})

    with st.chat_message("assistant"):
        try:
            retrieved_context = retrieve_context(
                st.session_state.session_id, prompt, top_k=5
            )
            initial_state: PipelineState = {
                "user_input":        prompt,
                "context":           st.session_state.context,
                "session_id":        st.session_state.session_id,
                "buffer_memory":     st.session_state.buffer_memory,
                "retrieved_context": retrieved_context,
            }

            # Clear previous run's search log and any leftover callback.
            tavily_log.clear()
            clear_search_callback()

            final_state = dict(initial_state)

            # Mutable ref so the nested callback can update the placeholder.
            _live_ph = [None]

            st.session_state.interaction_count += 1
            _trace_name = (
                f"Interaction_{st.session_state.interaction_count}"
                f"_{st.session_state.session_id}"
            )

            try:
                with ls_trace(
                    _trace_name,
                    run_type="chain",
                    inputs={"user_input": prompt},
                    metadata={"session_id": st.session_state.session_id},
                ):
                    with st.status("Thinking…", expanded=True) as status:
                        for node_name, update in stream_pipeline(
                            st.session_state.graph, initial_state
                        ):
                            final_state.update(update)

                            if node_name == "manager_node":
                                st.write("Request analyzed")
                                if update.get("ready_to_research"):
                                    st.write("Starting web research…")
                                    # Placeholder updated in real-time as each
                                    # web_search() fires (possibly from a thread).
                                    _live_ph[0] = st.empty()
                                    _st_ctx = get_script_run_ctx()

                                    def _on_search(
                                        _q,
                                        _ph=_live_ph,
                                        _log=tavily_log,
                                        _ctx=_st_ctx,
                                    ):
                                        # Inject Streamlit context so background
                                        # ThreadPoolExecutor threads can write to UI.
                                        if _ctx is not None:
                                            add_script_run_ctx(
                                                threading.current_thread(), _ctx
                                            )
                                        lines = "\n".join(
                                            f"   {i}. {q}"
                                            for i, q in enumerate(_log, 1)
                                        )
                                        _ph[0].markdown(lines)

                                    register_search_callback(_on_search)

                            elif node_name == "research_node":
                                # Replace the live placeholder with the final list.
                                if _live_ph[0] is not None:
                                    _live_ph[0].empty()
                                n = len(tavily_log)
                                st.write(
                                    f"✓ Web research complete — "
                                    f"{n} search{'es' if n != 1 else ''} performed"
                                )
                                for i, q in enumerate(tavily_log, 1):
                                    st.write(f"   {i}. {q}")
                                st.write("📝 Analyzing findings and generating report…")

                            elif node_name == "analyze_node":
                                st.write("✓ Report ready")

                        status.update(label="Done!", state="complete", expanded=False)
            finally:
                clear_search_callback()

        except Exception as exc:
            st.error(f"Pipeline error: {exc}")
            st.stop()

        # ── extract results ─────────────────────────────────────────────────
        context              = final_state.get("context", st.session_state.context)
        manager_response_str = final_state.get("manager_response", "")
        ready_to_research    = final_state.get("ready_to_research", False)
        final_report         = final_state.get("final_report")

        # ── update session context ───────────────────────────────────────────
        st.session_state.context = context

        # ── render response ──────────────────────────────────────────────────
        if final_report:
            st.markdown("**Research complete!** Your report is ready.")
            st.markdown('<div class="report-card">', unsafe_allow_html=True)
            st.markdown('<div class="report-header">📄 Research Report</div>', unsafe_allow_html=True)
            with st.expander("Preview report", expanded=True):
                st.markdown(final_report)
            st.markdown("</div>", unsafe_allow_html=True)

            topic_slug = (context.get("topic") or "research").lower().replace(" ", "_")[:40]
            st.session_state.report_topic = context.get("topic") or "Research"
            st.download_button(
                label="⬇  Download PDF",
                data=generate_pdf(final_report, st.session_state.report_topic),
                file_name=f"explorion_{topic_slug}.pdf",
                mime="application/pdf",
                key=f"dl_new_{len(st.session_state.messages)}",
            )

            st.session_state.messages.append({
                "role": "assistant", "content": final_report, "type": "report",
            })
        else:
            st.markdown(manager_response_str)
            st.session_state.messages.append({
                "role": "assistant", "content": manager_response_str, "type": "text",
            })

    # ── build buffer JSON ────────────────────────────────────────────────────
    assistant_json_str = json.dumps(
        {
            "topic":             context.get("topic"),
            "goal":              context.get("goal"),
            "location":          context.get("location"),
            "manager_response":  manager_response_str,
            "ready_to_research": ready_to_research,
        },
        ensure_ascii=False,
    )

    # ── persist to vector DB ─────────────────────────────────────────────────
    if final_report:
        store_interaction(
            st.session_state.session_id, prompt, final_report, is_report=True
        )
    else:
        store_interaction(
            st.session_state.session_id, prompt, manager_response_str, is_report=False
        )

    # ── update buffer memory ─────────────────────────────────────────────────
    st.session_state.buffer_memory.append({
        "user":      prompt,
        "assistant": assistant_json_str,
        "report":    final_report,
    })
    if len(st.session_state.buffer_memory) > BUFFER_MAX:
        st.session_state.buffer_memory = st.session_state.buffer_memory[-BUFFER_MAX:]

    # Force rerun so the sidebar context cards update
    st.rerun()
