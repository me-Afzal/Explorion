"""
agents.py  (assistant package)
-------------------------------
  manager_agent  – context detection + response + routing
  research_agent – agentic web search + findings synthesis
  analyze_agent  – findings → structured Markdown report
"""

from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.agents import create_agent

from .llm_init import get_manager_llm, get_agent_llm
from .agent_tool import web_search


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT SCHEMA
# ══════════════════════════════════════════════════════════════════════════════

class ManagerOutput(BaseModel):
    topic: Optional[str] = Field(
        default=None,
        description=(
            "The research topic as a clear, complete description. "
            "Capture the full subject — what is being researched and in what context. "
            "Keep the previous value if the user did not mention a new topic. "
            "Set to None only if no topic has ever been mentioned."
        ),
    )
    goal: Optional[str] = Field(
        default=None,
        description=(
            "What the user wants to find out, understand, or achieve through this research. "
            "Can be anything: learn about a subject, analyse options, compare alternatives, "
            "understand impacts, make a decision, satisfy curiosity, etc. "
            "Keep the previous value if the user did not mention a goal. "
            "Set to None if no goal has ever been mentioned."
        ),
    )
    location: Optional[str] = Field(
        default=None,
        description=(
            "The target location or geographic scope for the research. "
            "Set to 'N/A' if the user explicitly says they have no location, "
            "are unsure, or want to skip it (e.g. 'not sure', 'anywhere', 'skip', 'global'). "
            "Keep the previous value if the user did not mention location. "
            "Set to None only if location has never been discussed at all."
        ),
    )
    manager_response: str = Field(
        description=(
            "A natural, conversational reply to the user. "
            "Never include variable names, JSON, or code in this field."
        )
    )
    ready_to_research: bool = Field(
        description=(
            "True ONLY when the user has explicitly confirmed they want to start "
            "research (e.g. 'yes', 'go ahead', 'start', 'proceed'). "
            "NEVER set this to True on your own. False in every other case."
        )
    )


# ══════════════════════════════════════════════════════════════════════════════
# 1. MANAGER AGENT
# ══════════════════════════════════════════════════════════════════════════════

def manager_agent(state: dict) -> dict:
    """
    Message layout sent to the LLM:
      SystemMessage  — rules + current context + retrieved VectorDB chunks + last report
      HumanMessage   ┐  repeated for each buffer memory turn
      AIMessage      ┘
      HumanMessage   — current user input (clean)
    """
    llm            = get_manager_llm()
    structured_llm = llm.with_structured_output(ManagerOutput)

    current_ctx   = state["context"]
    buffer_memory = state.get("buffer_memory", [])
    retrieved_ctx = state.get("retrieved_context", "")

    latest_report = next(
        (turn["report"] for turn in reversed(buffer_memory) if turn.get("report")),
        None,
    )

    retrieved_section = (
        f"\n\nRELEVANT PAST CONTEXT (from memory):\n{retrieved_ctx}"
        if retrieved_ctx else ""
    )
    report_section = (
        f"\n\nLAST GENERATED REPORT (use this to answer follow-up questions):\n{latest_report}"
        if latest_report else ""
    )

    location_display = current_ctx["location"] or "Not set"

    system_prompt = f"""You are a general-purpose Research Manager Agent. You help users research ANY topic without restriction — science, technology, history, health, environment, culture, politics, business, education, travel, or anything else. Each turn output structured fields: topic, goal, location, manager_response, ready_to_research.

CURRENT RESEARCH CONTEXT:
- Topic   : {current_ctx["topic"] or "Not set"}
- Goal    : {current_ctx["goal"] or "Not set"}
- Location: {location_display}{retrieved_section}{report_section}

CONTEXT EXTRACTION — do this first, every turn:
- Read the ENTIRE user message and extract ALL of: topic, goal, location if present.
- Fill every field you can detect in one pass. Do NOT fill just one field and ignore the rest.
- Examples across different domains:
  "research effects of climate change on agriculture in Southeast Asia"
    → topic = "effects of climate change on agriculture"
    → goal  = "understand the impact on crops and food security"
    → location = "Southeast Asia"
  "I want to learn about quantum computing advancements, specifically for cryptography"
    → topic = "quantum computing advancements"
    → goal  = "understand implications for cryptography"
    → location = None
  "competitor analysis and market opportunity for a restaurant near Infopark, Kochi"
    → topic = "starting a restaurant near Infopark, Kochi"
    → goal  = "competitor analysis and market opportunity"
    → location = "Infopark, Kochi"
- Casual msg / small talk: keep existing fields unchanged.
- Completely new topic: overwrite all fields; unmentioned ones → None.

DECISION FLOW — after extraction, follow exactly in order, stop at the first matching step:

STEP 1 — topic is still None after extraction?
  Ask what topic they want to research. STOP.

STEP 2 — goal is still None after extraction?
  Ask what they want to find out or achieve. STOP.

STEP 3 — location is still None after extraction?
  Ask if there is a specific location or region. Say it is optional — they can skip it. STOP.
  *** MUST ask this. Do NOT jump to step 5 or 6. ***

STEP 4 — User is skipping or unsure about location ("not sure"/"anywhere"/"skip"/"global"):
  → set location = "N/A", ask "Ready to start deep research?" STOP.

STEP 5 — All three fields set AND user explicitly confirms ("yes"/"go ahead"/"start"/"proceed"):
  → ready_to_research = True
  → Reply: "Starting research on [topic], goal: [goal]" (add location only if not "N/A"). STOP.

STEP 6 — All three fields set, user has NOT confirmed yet:
  → Briefly acknowledge the details, then ask EXACTLY: "Should I start the deep research on this now?"
  → Do NOT ask any other questions. Do NOT ask about specifics or aspects. STOP.

STEP 7 — User declines or pauses ("no"/"stop"/"wait"/"not yet"/"bye"):
  → Politely acknowledge; say you're ready when they are. STOP.

STEP 8 — User asks a follow-up about the last report:
  → Answer directly using LAST GENERATED REPORT. Do not say a report doesn't exist. STOP.

STEP 9 — Pure small talk, no research info:
  → Reply warmly. Do NOT push for research. STOP.

STRICT RULES:
- You may ONLY ask the questions listed in the steps above. No other questions allowed.
- Once topic + goal + location are all set, NEVER ask for more details or clarification.
- NEVER set ready_to_research=True without explicit confirmation in step 5.
- manager_response must be plain conversational text only."""

    messages = [SystemMessage(content=system_prompt)]
    for turn in buffer_memory:
        messages.append(HumanMessage(content=turn["user"]))
        messages.append(AIMessage(content=turn["assistant"]))
    messages.append(HumanMessage(content=state["user_input"]))

    result: ManagerOutput = structured_llm.invoke(messages)

    return {
        "context": {
            "topic":    result.topic,
            "goal":     result.goal,
            "location": result.location,
        },
        "manager_response":  result.manager_response,
        "ready_to_research": result.ready_to_research,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. RESEARCH AGENT
# ══════════════════════════════════════════════════════════════════════════════

def research_agent(state: dict) -> dict:
    topic    = state.get("context", {}).get("topic", "Not specified")
    goal     = state.get("context", {}).get("goal", "Not specified")
    location = state.get("context", {}).get("location", "Not specified")

    location_note = (
        f"Location scope: {location}"
        if location and location != "N/A"
        else "Location scope: not restricted — research globally unless the topic implies otherwise."
    )

    agent = create_agent(
        model=get_agent_llm(),
        tools=[web_search],
        name="research_agent",
        system_prompt=SystemMessage(content=(
            "You are a Senior Researcher. You conduct deep, thorough research on any topic "
            "— science, technology, markets, history, health, policy, or anything else.\n\n"
            "RESEARCH STRATEGY:\n"
            "  1. Identify the 4-6 most important angles for this specific topic and goal.\n"
            "  2. Start broad (overview, definitions, current state), then drill into each angle.\n"
            "  3. Search for both supporting evidence AND counter-arguments or risks.\n"
            "  4. After reading each result, decide what to search next based on what you found.\n"
            "     If a result surfaces a notable fact, trend, or gap — follow up with a targeted search.\n"
            "  5. Perform 8-10 distinct searches in total to ensure thorough coverage.\n"
            "  6. Only stop when every identified angle is well-supported by evidence.\n\n"
            "SEARCH DISCIPLINE — CRITICAL RULES:\n"
            "  - Call ONE search at a time. Wait for the result before deciding the next query.\n"
            "  - Never call multiple search tools simultaneously.\n"
            "  - Every query must be UNIQUE — never repeat a query you have already used.\n"
            "  - If the tool returns '[DUPLICATE QUERY...]': choose a completely different angle.\n"
            "  - If the tool returns '[SEARCH LIMIT REACHED...]': stop all searches immediately "
            "    and write your final synthesis using what you have gathered.\n\n"
            "After completing searches, synthesise everything into a clear, factual findings report "
            "that directly addresses the research goal."
        ))
    )

    result = agent.invoke(
        {
            "messages": [HumanMessage(content=(
                f"Research Topic : {topic}\n"
                f"Goal           : {goal}\n"
                f"{location_note}\n\n"
                "Conduct your research now, then synthesise all findings into a structured report."
            ))]
        },
        config={"recursion_limit": 40},
    )

    return {"research_findings": result["messages"][-1].content.strip()}


# ══════════════════════════════════════════════════════════════════════════════
# 3. ANALYZE AGENT
# ══════════════════════════════════════════════════════════════════════════════

def analyze_agent(state: dict) -> dict:
    llm      = get_agent_llm()
    topic    = state.get("context", {}).get("topic", "Not specified")
    goal     = state.get("context", {}).get("goal", "Not specified")
    location = state.get("context", {}).get("location", "Not specified")
    findings = state.get("research_findings", "")

    location_line = (
        f"Location      : {location}"
        if location and location != "N/A"
        else "Location      : Not restricted (global scope)"
    )

    response = llm.invoke([
        SystemMessage(content=(
            "You are a senior Research Analyst and professional Report Writer. "
            "Your task is to produce a LONG, DETAILED, publication-quality research report in Markdown. "
            "The report must be thorough enough to span at least 3 full pages when printed — "
            "do not summarise; expand every point with evidence, context, and analysis.\n\n"

            "REQUIRED REPORT STRUCTURE (use all sections, add more if relevant):\n\n"

            "## Executive Summary\n"
            "  - 2-3 paragraphs. State the purpose, scope, key findings, and main conclusion.\n\n"

            "## Background & Context\n"
            "  - Explain the topic in depth: history, current state, why it matters.\n"
            "  - Define key terms or concepts a reader needs to understand the report.\n\n"

            "## Key Findings\n"
            "  - Dedicate a full sub-section (###) to EACH major research angle.\n"
            "  - Each sub-section must have at least 3-4 paragraphs of detailed analysis.\n"
            "  - Include data points, statistics, examples, and named sources where available.\n\n"

            "## Detailed Analysis\n"
            "  - Cross-examine the findings: compare, contrast, and identify patterns.\n"
            "  - Go deeper than the Key Findings — interpret what the data means.\n\n"

            "## Opportunities & Recommendations\n"
            "  - Concrete, actionable recommendations tied directly to the research goal.\n"
            "  - Explain the reasoning behind each recommendation in detail.\n\n"

            "## Risks, Challenges & Limitations\n"
            "  - Cover risks, obstacles, counter-arguments, and knowledge gaps.\n"
            "  - Be balanced — do not ignore negative evidence.\n\n"

            "## Conclusion\n"
            "  - 2-3 paragraphs directly answering the research goal.\n"
            "  - Summarise what was found and what should be done next.\n\n"

            "WRITING RULES:\n"
            "  - Use ## for main sections, ### for sub-sections, #### for sub-sub-sections.\n"
            "  - Write in full paragraphs — do not reduce sections to bullet-point lists only.\n"
            "  - Expand every finding with context, cause, effect, and implication.\n"
            "  - If a location was provided, add a dedicated location-specific section.\n"
            "  - NEVER truncate or shorten the report — completeness is required.\n"
            "  - Do NOT include AI company names, model names, or 'Prepared by' signatures."
        )),
        HumanMessage(content=(
            f"Research Topic : {topic}\n"
            f"Goal           : {goal}\n"
            f"{location_line}\n\n"
            f"Research Findings:\n{findings}\n\n"
            "Write the full detailed report now. Do not skip or abbreviate any section."
        )),
    ])
    return {"final_report": response.content.strip()}
