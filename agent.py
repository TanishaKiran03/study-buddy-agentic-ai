from __future__ import annotations

import ast
import datetime as dt
import math
import os
import re
import uuid
from typing import Any, TypedDict

import chromadb
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from sentence_transformers import SentenceTransformer


DOCUMENTS: list[dict[str, str]] = [
    {
        "id": "doc_001",
        "topic": "Newton's Laws of Motion",
        "text": (
            "Newton's First Law (Law of Inertia): An object at rest stays at rest, and an object in "
            "motion stays in motion with the same speed and direction unless acted upon by an "
            "unbalanced external force. Newton's Second Law: The net force acting on an object "
            "equals the product of its mass and acceleration: F = ma, where F is force in Newtons "
            "(N), m is mass in kilograms (kg), and a is acceleration in m/s^2. Newton's Third Law: "
            "For every action, there is an equal and opposite reaction. Example: When you push "
            "against a wall with 10 N, the wall pushes back on you with 10 N in the opposite "
            "direction. Applications: These laws explain why seatbelts are necessary (inertia), how "
            "rockets propel in space (action-reaction), and how brakes decelerate vehicles (net "
            "force changes velocity). Important: Newton's laws apply in inertial, non-accelerating "
            "reference frames."
        ),
    },
    {
        "id": "doc_002",
        "topic": "Work, Energy and Power",
        "text": (
            "Work is defined as the product of force and displacement in the direction of force: "
            "W = F * d * cos(theta), where theta is the angle between force and displacement. Unit: "
            "Joule (J). Kinetic Energy (KE) is the energy of motion: KE = 1/2 mv^2, where m = mass "
            "(kg) and v = velocity (m/s). Potential Energy (PE) is stored energy due to position: "
            "PE = mgh, where g = 9.8 m/s^2 and h = height (m). Law of Conservation of Energy: "
            "Energy cannot be created or destroyed; it only transforms. Total mechanical energy "
            "E = KE + PE = constant in the absence of friction. Power is the rate of doing work: "
            "P = W/t, Unit: Watt (W). Also P = F * v. Work-Energy Theorem: The net work done on an "
            "object equals its change in kinetic energy: W_net = change in KE."
        ),
    },
    {
        "id": "doc_003",
        "topic": "Gravitation",
        "text": (
            "Newton's Law of Universal Gravitation: Every two masses attract each other with a "
            "force: F = G * m1 * m2 / r^2, where G = 6.674 x 10^-11 N m^2/kg^2, m1 and m2 are the "
            "masses, and r is the distance between their centers. Acceleration due to gravity on "
            "Earth's surface: g = 9.8 m/s^2. Escape velocity: The minimum velocity to escape "
            "Earth's gravitational field is v_escape = sqrt(2GM/R), approximately 11.2 km/s. "
            "Orbital velocity: v_orbital = sqrt(GM/r). Kepler's Laws: Planets orbit in ellipses "
            "with the sun at one focus; equal areas are swept in equal times; T^2 is proportional "
            "to r^3."
        ),
    },
    {
        "id": "doc_004",
        "topic": "Thermodynamics",
        "text": (
            "Thermodynamics deals with heat, temperature, and energy transfer. Zeroth Law: If body "
            "A is in thermal equilibrium with B, and B with C, then A and C are also in equilibrium. "
            "First Law, or Conservation of Energy: delta U = Q - W, where delta U = change in "
            "internal energy, Q = heat added to system, W = work done by system. Second Law: Heat "
            "flows spontaneously from hot to cold. The entropy of an isolated system never "
            "decreases. Third Law: As temperature approaches absolute zero, 0 K, entropy approaches "
            "a minimum constant value. Specific Heat Capacity: Q = mc delta T. For water, c = 4200 "
            "J/kg K. Ideal Gas Law: PV = nRT, where R = 8.314 J/mol K. Isothermal: T constant. "
            "Adiabatic: Q = 0. Isochoric: V constant. Isobaric: P constant."
        ),
    },
    {
        "id": "doc_005",
        "topic": "Waves and Oscillations",
        "text": (
            "Simple Harmonic Motion: Restoring force is proportional to displacement: F = -kx. "
            "Period of a spring-mass system: T = 2*pi*sqrt(m/k). Period of a simple pendulum: "
            "T = 2*pi*sqrt(L/g). Waves are periodic disturbances that carry energy. Transverse "
            "waves have displacement perpendicular to propagation, such as light. Longitudinal "
            "waves have displacement parallel to propagation, such as sound. Wave equation: "
            "v = f lambda, where v = wave speed, f = frequency, and lambda = wavelength. Speed of "
            "sound in air at 0 C is approximately 331 m/s and increases about 0.6 m/s per C. "
            "Doppler Effect: Apparent frequency changes when source or observer moves. Resonance "
            "occurs when driving frequency equals natural frequency, giving maximum amplitude."
        ),
    },
    {
        "id": "doc_006",
        "topic": "Optics - Reflection and Refraction",
        "text": (
            "Law of Reflection: Angle of incidence equals angle of reflection, both measured from "
            "the normal. Refraction: When light passes from one medium to another, it bends. "
            "Snell's Law: n1 sin(theta1) = n2 sin(theta2). Refractive index n = c / v, where c is "
            "the speed of light in vacuum, 3 x 10^8 m/s, and v is speed in the medium. Total "
            "Internal Reflection occurs when light travels from denser to rarer medium and angle "
            "of incidence is greater than the critical angle. Critical angle theta_c = sin^-1(n2/n1). "
            "Applications: optical fibers and diamonds. Mirror formula: 1/f = 1/v + 1/u. Lens "
            "formula: 1/f = 1/v - 1/u. Magnification m = v/u."
        ),
    },
    {
        "id": "doc_007",
        "topic": "Electrostatics",
        "text": (
            "Coulomb's Law: Force between two point charges: F = k * q1 * q2 / r^2, where "
            "k = 9 x 10^9 N m^2/C^2, q1 and q2 are charges in Coulombs, and r is distance in "
            "meters. Electric Field E = F/q0 = kQ/r^2. Field lines go from positive to negative. "
            "Electric Potential V = kQ/r. Potential is a scalar. Relation: E = -dV/dr. Capacitance "
            "C = Q/V. Parallel plate capacitor: C = epsilon0 A/d. epsilon0 = 8.85 x 10^-12 F/m. "
            "Gauss's Law: Total electric flux through a closed surface = Q_enclosed / epsilon0. "
            "Energy stored in capacitor: U = 1/2 C V^2 = Q^2 / 2C."
        ),
    },
    {
        "id": "doc_008",
        "topic": "Current Electricity",
        "text": (
            "Electric current I = Q/t. Conventional current flows from positive to negative. Ohm's "
            "Law: V = IR, where V = voltage, I = current, and R = resistance. Resistance R = rho L/A, "
            "where rho = resistivity, L = length, and A = cross-section area. Series: R_total = "
            "R1 + R2 + ... Current is the same through all. Parallel: 1/R_total = 1/R1 + 1/R2 + ... "
            "Voltage is the same across all. Power: P = VI = I^2 R = V^2/R. Kirchhoff's Laws: KCL "
            "says the sum of currents at a junction is 0. KVL says the sum of voltages around a "
            "loop is 0. Terminal voltage V = emf - Ir."
        ),
    },
    {
        "id": "doc_009",
        "topic": "Magnetism and Electromagnetic Induction",
        "text": (
            "Magnetic force on a moving charge: F = qvB sin(theta), where B = magnetic field. Force "
            "on a current-carrying conductor: F = BIL sin(theta). Biot-Savart Law: dB = mu0/4pi * "
            "I dl x r_hat / r^2. Ampere's Law: integral B dot dl = mu0 I_enclosed. Magnetic field "
            "at center of circular loop: B = mu0 I / 2R. Faraday's Law of Induction: EMF = -dPhi/dt, "
            "where Phi = magnetic flux = B A cos(theta). Lenz's Law: Induced current opposes the "
            "change that caused it. Transformer: V1/V2 = N1/N2. Self-inductance: EMF = -L dI/dt. "
            "Energy stored: U = 1/2 L I^2."
        ),
    },
    {
        "id": "doc_010",
        "topic": "Modern Physics - Photoelectric Effect and Atomic Models",
        "text": (
            "Photoelectric Effect: Light strikes a metal surface and ejects electrons. Energy of "
            "photon: E = hf = hc/lambda, where h = 6.626 x 10^-34 J s. KE_max = hf - phi, where "
            "phi is the work function, the minimum energy to eject electron. Threshold frequency: "
            "f0 = phi/h. Below f0, no emission occurs regardless of intensity. Bohr's Atomic Model: "
            "Electrons orbit nucleus in fixed energy levels. Energy of nth level in hydrogen: "
            "E_n = -13.6/n^2 eV. Ground state: n=1, E = -13.6 eV. When electron transitions from "
            "n2 to n1, delta E = hf. de Broglie wavelength: lambda = h/mv. Heisenberg Uncertainty "
            "Principle: delta x * delta p >= h/4pi."
        ),
    },
    {
        "id": "doc_011",
        "topic": "Projectile Motion and Kinematics",
        "text": (
            "Kinematics equations for uniform acceleration: v = u + at; s = ut + 1/2 at^2; "
            "v^2 = u^2 + 2as; s = (u+v)/2 * t. Projectile Motion: Horizontal and vertical "
            "components are independent. Horizontal: x = u_x * t = u cos(theta) * t with constant "
            "velocity. Vertical: y = u sin(theta) * t - 1/2 gt^2 under gravity. Time of flight: "
            "T = 2u sin(theta)/g. Maximum height: H = u^2 sin^2(theta) / 2g. Horizontal range: "
            "R = u^2 sin(2theta)/g, maximum at theta = 45 degrees. Relative velocity: v_AB = "
            "v_A - v_B."
        ),
    },
    {
        "id": "doc_012",
        "topic": "Rotational Motion",
        "text": (
            "Angular displacement theta, angular velocity omega = dtheta/dt, and angular "
            "acceleration alpha = domega/dt. Kinematic analogies: theta = omega0 t + 1/2 alpha t^2; "
            "omega^2 = omega0^2 + 2 alpha theta. Torque: tau = r x F = I alpha, where I is moment "
            "of inertia. Moment of inertia depends on mass distribution: I = sum mr^2. For a solid "
            "disk: I = 1/2 MR^2. For a solid sphere: I = 2/5 MR^2. For a rod about center: "
            "I = ML^2/12. Parallel axis theorem: I = I_cm + Md^2. Angular momentum: L = I omega. "
            "If net torque is 0, angular momentum is constant. Rolling without slipping: v_cm = R omega."
        ),
    },
]


class CapstoneState(TypedDict):
    question: str
    messages: list[dict[str, str]]
    route: str
    retrieved: str
    sources: list[str]
    tool_result: str
    answer: str
    faithfulness: float
    eval_retries: int
    student_name: str | None


MAX_EVAL_RETRIES = 2

ROUTER_PROMPT = """You are a routing assistant for a Physics Study Buddy.
Given the student's question, decide which route to take:

- retrieve: physics concept, formula, law, definition, application, or textbook topic
- tool: numeric calculation using given values OR current time/date
- memory_only: greetings, small talk, or questions about the student's own name/previous context

Reply with ONE WORD ONLY: retrieve, tool, or memory_only

Student question: {question}"""

SYSTEM_PROMPT = """You are Study Buddy, an intelligent physics tutor for B.Tech students.

RULES:
1. Answer ONLY using the provided context or tool result.
2. If the context does not contain the answer, say exactly:
   "I don't have information on that topic in my knowledge base. Please check your textbook or ask your professor."
3. Never fabricate formulas, constants, values, or source topics.
4. If a student name is known, address them by name.
5. Keep answers structured: explanation, formula if any, and example if relevant.
6. For memory-only questions, use the conversation history and known student name.
7. If this is an eval retry, be even stricter about grounding."""

EVAL_PROMPT = """Rate how faithful the ANSWER is to the CONTEXT below.
Score 0.0 (not grounded) to 1.0 (fully grounded).
Reply with a single float only.

CONTEXT:
{context}

ANSWER:
{answer}

SCORE:"""


def _extract_name(question: str) -> str | None:
    match = re.search(r"\bmy name is\s+([A-Za-z][A-Za-z .'-]{0,40})", question, re.I)
    if not match:
        return None
    raw = re.split(r"[.?!,]", match.group(1).strip())[0]
    return raw.split()[0].capitalize() if raw else None


def _safe_number_after(question: str, *names: str) -> float | None:
    pattern = r"(?:^|\b)(" + "|".join(re.escape(name) for name in names) + r")\s*=?\s*(-?\d+(?:\.\d+)?)"
    match = re.search(pattern, question, re.I)
    return float(match.group(2)) if match else None


def _quick_formula_expression(question: str) -> str | None:
    lower = question.lower()
    mass = _safe_number_after(lower, "m", "mass")
    acceleration = _safe_number_after(lower, "a", "acceleration")
    if mass is not None and acceleration is not None and ("force" in lower or "f=" in lower):
        return f"{mass} * {acceleration}"

    velocity = _safe_number_after(lower, "v", "velocity")
    if mass is not None and velocity is not None and ("kinetic" in lower or "ke" in lower):
        return f"0.5 * {mass} * {velocity} ** 2"

    height = _safe_number_after(lower, "h", "height")
    gravity = _safe_number_after(lower, "g", "gravity") or 9.8
    if mass is not None and height is not None and ("potential" in lower or "pe" in lower):
        return f"{mass} * {gravity} * {height}"
    return None


class SafeCalculator(ast.NodeVisitor):
    allowed_names = {"pi": math.pi, "e": math.e}
    allowed_funcs = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "radians": math.radians,
    }

    def visit_Expression(self, node: ast.Expression) -> float:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> float:
        if isinstance(node.value, int | float):
            return float(node.value)
        raise ValueError("Only numeric constants are allowed.")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> float:
        value = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value
        raise ValueError("Unsupported unary operator.")

    def visit_BinOp(self, node: ast.BinOp) -> float:
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left**right
        raise ValueError("Unsupported binary operator.")

    def visit_Name(self, node: ast.Name) -> float:
        if node.id in self.allowed_names:
            return float(self.allowed_names[node.id])
        raise ValueError(f"Unknown name: {node.id}")

    def visit_Call(self, node: ast.Call) -> float:
        if not isinstance(node.func, ast.Name) or node.func.id not in self.allowed_funcs:
            raise ValueError("Unsupported function.")
        return float(self.allowed_funcs[node.func.id](*(self.visit(arg) for arg in node.args)))

    def generic_visit(self, node: ast.AST) -> float:
        raise ValueError(f"Unsupported expression element: {type(node).__name__}")


def safe_eval(expression: str) -> float:
    parsed = ast.parse(expression.replace("^", "**"), mode="eval")
    return SafeCalculator().visit(parsed)


class StudyBuddyAgent:
    """Course-style Study Buddy: LangGraph + ChromaDB + Groq + MemorySaver."""

    def __init__(
        self,
        groq_api_key: str | None = None,
        model: str = "llama-3.3-70b-versatile",
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is required. Enter it at runtime; do not hardcode it.")

        self.llm = ChatGroq(model=model, temperature=0.2, api_key=api_key)
        self.embedder = SentenceTransformer(embedding_model)
        self.collection = self._build_collection()
        self.app = self._build_graph()

    def _build_collection(self):
        client = chromadb.Client()
        collection = client.get_or_create_collection(name=f"physics_kb_{uuid.uuid4().hex}")
        doc_texts = [doc["text"] for doc in DOCUMENTS]
        doc_ids = [doc["id"] for doc in DOCUMENTS]
        doc_metas = [{"topic": doc["topic"]} for doc in DOCUMENTS]
        embeddings = self.embedder.encode(doc_texts).tolist()
        collection.add(documents=doc_texts, embeddings=embeddings, ids=doc_ids, metadatas=doc_metas)
        return collection

    def memory_node(self, state: CapstoneState) -> dict[str, Any]:
        messages = list(state.get("messages", []))
        question = state["question"]
        messages.append({"role": "user", "content": question})
        student_name = _extract_name(question) or state.get("student_name")
        return {"messages": messages[-6:], "student_name": student_name}

    def router_node(self, state: CapstoneState) -> dict[str, Any]:
        response = self.llm.invoke(ROUTER_PROMPT.format(question=state["question"]))
        route = response.content.strip().lower().split()[0]
        if route not in {"retrieve", "tool", "memory_only"}:
            route = "retrieve"
        return {"route": route, "eval_retries": state.get("eval_retries", 0)}

    def retrieval_node(self, state: CapstoneState) -> dict[str, Any]:
        question_embedding = self.embedder.encode([state["question"]]).tolist()
        results = self.collection.query(query_embeddings=question_embedding, n_results=3)
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results.get("distances", [[0, 0, 0]])[0]
        filtered = [
            (doc, meta)
            for doc, meta, distance in zip(docs, metas, distances, strict=False)
            if distance < 1.6
        ]
        context = "\n\n".join(f"[{meta['topic']}]\n{doc}" for doc, meta in filtered)
        return {"retrieved": context, "sources": [meta["topic"] for _, meta in filtered]}

    def skip_retrieval_node(self, state: CapstoneState) -> dict[str, Any]:
        return {"retrieved": "", "sources": []}

    def tool_node(self, state: CapstoneState) -> dict[str, Any]:
        question = state["question"].lower()
        if any(word in question for word in ["time", "date", "today", "day"]):
            now = dt.datetime.now()
            return {
                "tool_result": f"Current local date and time: {now.strftime('%A, %d %B %Y, %I:%M %p')}.",
                "retrieved": "",
                "sources": ["datetime tool"],
            }

        expression = _quick_formula_expression(question)
        if not expression:
            prompt = (
                "Extract a single-line Python math expression from this physics question. "
                "Use only numbers, +, -, *, /, **, parentheses, sqrt(), sin(), cos(), tan(), radians(), pi. "
                "Return ONLY the expression.\n\n"
                f"Question: {state['question']}"
            )
            try:
                expression = self.llm.invoke(prompt).content.strip().replace("```", "").strip()
                expression = expression.removeprefix("python").strip()
            except Exception as exc:
                return {"tool_result": f"Calculator error: could not extract expression. {exc}", "sources": ["calculator tool"]}

        try:
            value = safe_eval(expression)
            result = f"Calculation result: {expression} = {value:.4g}"
        except Exception as exc:
            result = f"Calculator error: could not evaluate expression. {exc}"
        return {"tool_result": result, "retrieved": "", "sources": ["calculator tool"]}

    def answer_node(self, state: CapstoneState) -> dict[str, Any]:
        name_str = f"Student name: {state['student_name']}." if state.get("student_name") else ""
        history = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in state.get("messages", [])[:-1])
        context = ""
        if state.get("retrieved"):
            context += f"\n\n--- RETRIEVED CONTEXT ---\n{state['retrieved']}"
        if state.get("tool_result"):
            context += f"\n\n--- TOOL RESULT ---\n{state['tool_result']}"
        retry = f"\nRetry number: {state.get('eval_retries', 0)}" if state.get("eval_retries", 0) else ""
        prompt = (
            f"{SYSTEM_PROMPT}\n{name_str}\n"
            f"--- CONVERSATION HISTORY ---\n{history}\n"
            f"{context}{retry}\n\n"
            f"--- STUDENT QUESTION ---\n{state['question']}"
        )
        response = self.llm.invoke(prompt)
        return {"answer": response.content.strip()}

    def eval_node(self, state: CapstoneState) -> dict[str, Any]:
        if state.get("route") == "memory_only":
            return {"faithfulness": 1.0, "eval_retries": state.get("eval_retries", 0)}
        context = state.get("retrieved") or state.get("tool_result") or ""
        if not context:
            return {"faithfulness": 0.0, "eval_retries": state.get("eval_retries", 0)}
        try:
            response = self.llm.invoke(EVAL_PROMPT.format(context=context[:2500], answer=state["answer"]))
            score = max(0.0, min(1.0, float(response.content.strip().split()[0])))
        except Exception:
            score = 0.5
        retries = state.get("eval_retries", 0) + (1 if score < 0.7 else 0)
        return {"faithfulness": score, "eval_retries": retries}

    def save_node(self, state: CapstoneState) -> dict[str, Any]:
        messages = list(state.get("messages", []))
        messages.append({"role": "assistant", "content": state["answer"]})
        return {"messages": messages[-6:]}

    def route_decision(self, state: CapstoneState) -> str:
        if state.get("route") == "tool":
            return "tool"
        if state.get("route") == "memory_only":
            return "skip"
        return "retrieve"

    def eval_decision(self, state: CapstoneState) -> str:
        if state.get("faithfulness", 1.0) < 0.7 and state.get("eval_retries", 0) <= MAX_EVAL_RETRIES:
            return "answer"
        return "save"

    def _build_graph(self):
        graph = StateGraph(CapstoneState)
        graph.add_node("memory", self.memory_node)
        graph.add_node("router", self.router_node)
        graph.add_node("retrieve", self.retrieval_node)
        graph.add_node("skip", self.skip_retrieval_node)
        graph.add_node("tool", self.tool_node)
        graph.add_node("answer", self.answer_node)
        graph.add_node("eval", self.eval_node)
        graph.add_node("save", self.save_node)

        graph.set_entry_point("memory")
        graph.add_edge("memory", "router")
        graph.add_conditional_edges("router", self.route_decision, {"retrieve": "retrieve", "skip": "skip", "tool": "tool"})
        graph.add_edge("retrieve", "answer")
        graph.add_edge("skip", "answer")
        graph.add_edge("tool", "answer")
        graph.add_edge("answer", "eval")
        graph.add_conditional_edges("eval", self.eval_decision, {"answer": "answer", "save": "save"})
        graph.add_edge("save", END)
        return graph.compile(checkpointer=MemorySaver())

    def ask(self, question: str, thread_id: str = "default") -> CapstoneState:
        config = {"configurable": {"thread_id": thread_id}}
        initial_state: CapstoneState = {
            "question": question,
            "messages": [],
            "route": "",
            "retrieved": "",
            "sources": [],
            "tool_result": "",
            "answer": "",
            "faithfulness": 0.0,
            "eval_retries": 0,
            "student_name": None,
        }
        return self.app.invoke(initial_state, config=config)


def ask(question: str, thread_id: str = "default", groq_api_key: str | None = None) -> CapstoneState:
    return StudyBuddyAgent(groq_api_key=groq_api_key).ask(question, thread_id)
