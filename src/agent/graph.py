"""LangGraph agent with tools and memory."""
from typing import TypedDict, Annotated, Sequence, Literal, Dict, Any
import operator
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.state import CompiledStateGraph

from src.config.logging_config import get_logger
from src.config.settings import settings
from src.services.llm import LLMService
from src.agent.tools import AgentTools
from src.agent.prompts import SYSTEM_PROMPT, QUERY_CLASSIFICATION_PROMPT
from src.agent.memory import ConversationMemory

logger = get_logger(__name__)


class AgentState(TypedDict):
    """State for the agent graph."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_query: str
    conversation_history: str
    tool_calls: list[Dict[str, Any]]
    final_response: str
    iteration: int


class FileAssistantAgent:
    """LangGraph-based agent for file assistance."""

    def __init__(
        self,
        tools: AgentTools,
        memory: ConversationMemory,
        llm_service: LLMService,
    ):
        self.tools = tools
        self.memory = memory
        self.llm = llm_service

        # Build graph
        self.graph = self._build_graph()

        logger.info("agent_initialized")

    def _build_graph(self) -> CompiledStateGraph:
        """Build the agent graph."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("classify_query", self.classify_query)
        workflow.add_node("execute_tool", self.execute_tool)
        workflow.add_node("generate_response", self.generate_response)

        # Add edges
        workflow.set_entry_point("classify_query")

        workflow.add_conditional_edges(
            "classify_query",
            self.should_use_tool,
            {
                "tool": "execute_tool",
                "direct": "generate_response",
            }
        )

        workflow.add_edge("execute_tool", "generate_response")
        workflow.add_edge("generate_response", END)

        return workflow.compile()

    async def classify_query(self, state: AgentState) -> AgentState:
        """Classify the user query."""
        logger.info("classifying_query", query=state["user_query"])

        prompt = QUERY_CLASSIFICATION_PROMPT.format(query=state["user_query"])

        classification = await self.llm.generate(
            prompt=prompt,
            temperature=0.0,
        )

        classification = classification.strip().lower()
        logger.info("query_classified", classification=classification)

        state["tool_calls"] = []

        # Determine which tool to use
        query_lower = state["user_query"].lower()
        if "search" in classification or "find" in query_lower or "about" in query_lower:
            state["tool_calls"].append({
                "tool": "search_files",
                "query": state["user_query"],
            })
        elif "list" in classification or "show files" in query_lower:
            state["tool_calls"].append({
                "tool": "list_files",
            })
        elif "stats" in classification or "statistics" in query_lower or "how many" in query_lower:
            state["tool_calls"].append({
                "tool": "get_file_stats",
            })

        return state

    def should_use_tool(self, state: AgentState) -> Literal["tool", "direct"]:
        """Decide if tool usage is needed."""
        if state["tool_calls"]:
            return "tool"
        return "direct"

    async def execute_tool(self, state: AgentState) -> AgentState:
        """Execute the selected tool."""
        logger.info("executing_tools", count=len(state["tool_calls"]))

        tool_results = []

        for tool_call in state["tool_calls"]:
            tool_name = tool_call["tool"]
            logger.info("executing_tool", tool=tool_name)

            try:
                if tool_name == "search_files":
                    result = await self.tools.search_files(
                        query=tool_call.get("query", state["user_query"]),
                        top_k=5,
                    )
                elif tool_name == "list_files":
                    result = await self.tools.list_files(
                        pattern=tool_call.get("pattern"),
                        limit=20,
                    )
                elif tool_name == "get_file_stats":
                    result = await self.tools.get_file_stats()
                else:
                    result = f"Unknown tool: {tool_name}"

                tool_results.append({
                    "tool": tool_name,
                    "result": result,
                })

            except Exception as e:
                logger.error("tool_execution_failed", tool=tool_name, error=str(e))
                tool_results.append({
                    "tool": tool_name,
                    "result": f"Error: {str(e)}",
                })

        # Add tool results to messages
        tool_message = "\n\n".join([
            f"Tool: {tr['tool']}\nResult:\n{tr['result']}"
            for tr in tool_results
        ])

        state["messages"] = list(state["messages"]) + [AIMessage(content=tool_message)]

        return state

    async def generate_response(self, state: AgentState) -> AgentState:
        """Generate final response."""
        logger.info("generating_response")

        # Build context
        context_parts = [SYSTEM_PROMPT]

        if state["conversation_history"]:
            context_parts.append(f"\nPrevious conversation:\n{state['conversation_history']}")

        # Add tool results if any
        tool_context = ""
        for msg in state["messages"]:
            if isinstance(msg, AIMessage) and "Tool:" in msg.content:
                tool_context += f"\n{msg.content}\n"

        if tool_context:
            context_parts.append(f"\nInformation from tools:{tool_context}")

        context_parts.append(f"\nUser query: {state['user_query']}")
        context_parts.append("\nProvide a helpful response based on the above information.")

        full_prompt = "\n".join(context_parts)

        # Generate response
        response = await self.llm.generate(
            prompt=full_prompt,
            temperature=settings.agent_temperature,
        )

        state["final_response"] = response
        logger.info("response_generated", length=len(response))

        return state

    async def run(
        self,
        user_query: str,
        user_id: str,
        chat_id: str,
    ) -> str:
        """Run the agent on a user query."""
        logger.info("agent_run_started", query=user_query, user_id=user_id)

        # Get conversation history
        recent_messages = await self.memory.get_recent_messages(
            user_id=user_id,
            chat_id=chat_id,
            limit=5,
        )

        conversation_history = "\n".join([
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in recent_messages
        ])

        # Initial state
        initial_state: AgentState = {
            "messages": [HumanMessage(content=user_query)],
            "user_query": user_query,
            "conversation_history": conversation_history,
            "tool_calls": [],
            "final_response": "",
            "iteration": 0,
        }

        # Run graph
        try:
            final_state = await self.graph.ainvoke(initial_state)
            response = final_state["final_response"]

            # Save to memory
            await self.memory.add_message(
                user_id=user_id,
                chat_id=chat_id,
                role="user",
                content=user_query,
            )

            await self.memory.add_message(
                user_id=user_id,
                chat_id=chat_id,
                role="assistant",
                content=response,
            )

            logger.info("agent_run_complete", response_length=len(response))
            return response

        except Exception as e:
            logger.error("agent_run_failed", error=str(e), exc_info=True)
            return f"Sorry, I encountered an error: {str(e)}"