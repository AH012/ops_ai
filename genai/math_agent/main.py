import os
import operator
from typing import TypedDict, Dict, List

from langchain.tools import tool
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph


if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "XXXXXXXXXXXXX"
# --------------------------------------------------------------------------- #
# 1) État partagé
# --------------------------------------------------------------------------- #
class State(TypedDict):
    input: str
    messages: List[HumanMessage | AIMessage | ToolMessage]
    output: str


# --------------------------------------------------------------------------- #
# 2) Tool = simple fonction
# --------------------------------------------------------------------------- #
@tool
def calculator(operation: str, a: float, b: float) -> str:
    """Effectue add, sub, mul ou div sur a et b."""
    ops = {
        "add": operator.add,
        "sub": operator.sub,
        "mul": operator.mul,
        "div": operator.truediv,
    }
    if operation not in ops:
        return f"Erreur : operation doit être l'une de {', '.join(ops)}"
    try:
        return str(ops[operation](float(a), float(b)))
    except ZeroDivisionError:
        return "Erreur : division par zéro."


# --------------------------------------------------------------------------- #
# 3) Modèle + binding des tools (fonction-calling OpenAI)
# --------------------------------------------------------------------------- #
llm_base = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm = llm_base.bind_tools([calculator])        # << inclut le schema JSON du tool

SYSTEM_PROMPT = (
    "Tu es un assistant IA spécialisé en raisonnement mathématique. "
    "Lorsque c'est pertinent, appelle la fonction calculator pour "
    "obtenir un résultat numérique exact."
)


# --------------------------------------------------------------------------- #
# 4) Nœud « agent » : boucle tool-calling manuelle
# --------------------------------------------------------------------------- #
def agent_node(state: State) -> State:
    """Un tour de dialogue : LLM → (tool) → réponse finale."""
    msgs: List = [SystemMessage(content=SYSTEM_PROMPT)]
    msgs += state.get("messages", [])           # historique éventuel
    msgs.append(HumanMessage(content=state["input"]))

    while True:
        ai_msg: AIMessage = llm.invoke(msgs)    # appel modèle
        msgs.append(ai_msg)

        # Si le LLM demande un tool, on l’exécute puis on boucle
        if ai_msg.tool_calls:
            for tc in ai_msg.tool_calls:
                if tc["name"] == "calculator":
                    # args est déjà un dict prêt à l’emploi
                    result = calculator.invoke(input=tc["args"])
                    msgs.append(
                        ToolMessage(content=result, tool_call_id=tc["id"])
                    )
            # reprise du dialogue pour que le LLM utilise le résultat
            continue

        # Pas de tool : réponse finale
        state["messages"] = msgs
        state["output"] = ai_msg.content
        return state


# --------------------------------------------------------------------------- #
# 5) Graphe LangGraph
# --------------------------------------------------------------------------- #
graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.set_entry_point("agent")
graph.set_finish_point("agent")

math_agent = graph.compile()


# --------------------------------------------------------------------------- #
# 6) Exemple d’exécution
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    question = "Que vaut (25 * 8) - 17 ?"
    result: Dict[str, str] = math_agent.invoke({"input": question})
    print("Question :", question)
    print("Réponse  :", result["output"])
