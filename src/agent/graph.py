"""LangGraph agent - STRICT: No hallucinations, exact tool results only."""
from typing import TypedDict, Annotated, Sequence, Literal, Dict, Any
import operator
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.state import CompiledStateGraph

from src.config.logging_config import get_logger
from src.services.llm import LLMService
from src.agent.tools import AgentTools
from src.agent.prompts import SYSTEM_PROMPT_ENHANCED, QUERY_CLASSIFICATION_PROMPT_ENHANCED
from src.agent.memory import ConversationMemory

logger = get_logger(__name__)


class AgentState(TypedDict):
    """State for the agent graph."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_query: str
    conversation_history: str
    tool_calls: list[Dict[str, Any]]
    tool_results: list[Dict[str, Any]]
    files_to_send: list[Dict[str, Any]]
    final_response: str
    iteration: int


class FileAssistantAgent:
    """LangGraph agent - STRICT tool usage only."""

    def __init__(
        self,
        tools: AgentTools,
        memory: ConversationMemory,
        llm_service: LLMService,
    ):
        self.tools = tools
        self.memory = memory
        self.llm = llm_service

        self.graph = self._build_graph()

        logger.info("strict_agent_initialized")

    def _build_graph(self) -> CompiledStateGraph:
        """Build the agent graph."""
        workflow = StateGraph(AgentState)

        workflow.add_node("classify_query", self.classify_query)
        workflow.add_node("execute_tool", self.execute_tool)
        workflow.add_node("generate_response", self.generate_response)

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

        prompt = QUERY_CLASSIFICATION_PROMPT_ENHANCED.format(query=state["user_query"])

        classification = await self.llm.generate(
            prompt=prompt,
            temperature=0.0,
        )

        classification = classification.strip().lower()
        logger.info("query_classified", classification=classification)

        state["tool_calls"] = []
        state["files_to_send"] = []

        query_lower = state["user_query"].lower()

        # STRICT classification with specific keywords

        # 1. User wants specific content (paragraphs, sections, questions)
        if any(word in query_lower for word in ["paragraph", "section", "question", "answer", "line", "page", "chapter"]):
            state["tool_calls"].append({
                "tool": "find_specific_content",
                "query": state["user_query"],
            })
            logger.info("classified_as_find_specific")

        # 2. User wants to send/download file
        elif any(word in query_lower for word in ["send", "give me", "share", "download", "send me the file"]):
            state["tool_calls"].append({
                "tool": "get_file_for_sending",
                "query": state["user_query"],
            })
            logger.info("classified_as_send_file")

        # 3. User wants to search
        elif any(word in query_lower for word in ["find", "search", "what is", "about", "documents about"]):
            state["tool_calls"].append({
                "tool": "search_files",
                "query": state["user_query"],
            })
            logger.info("classified_as_search")

        # 4. User wants to list files
        elif any(word in query_lower for word in ["list", "show files", "what files", "all files"]):
            state["tool_calls"].append({
                "tool": "list_files",
            })
            logger.info("classified_as_list")

        # 5. User wants statistics
        elif any(word in query_lower for word in ["stats", "statistics", "how many", "storage"]):
            state["tool_calls"].append({
                "tool": "get_file_stats",
            })
            logger.info("classified_as_stats")

        # Default: search
        else:
            state["tool_calls"].append({
                "tool": "search_files",
                "query": state["user_query"],
            })
            logger.info("classified_as_default_search")

        return state

    def should_use_tool(self, state: AgentState) -> Literal["tool", "direct"]:
        """Decide if tool usage is needed."""
        return "tool" if state["tool_calls"] else "direct"

    async def execute_tool(self, state: AgentState) -> AgentState:
        """Execute tools and return EXACT results."""
        logger.info("executing_tools", count=len(state["tool_calls"]))

        tool_results = []

        for tool_call in state["tool_calls"]:
            tool_name = tool_call["tool"]
            logger.info("executing_tool", tool=tool_name)

            try:
                if tool_name == "search_files":
                    result_dict = await self.tools.search_files(
                        query=tool_call.get("query", state["user_query"]),
                        top_k=5,
                    )
                    result = result_dict["text"]
                    # Never auto-add files from search

                elif tool_name == "find_specific_content":
                    result_dict = await self.tools.find_specific_content(
                        query=tool_call.get("query", state["user_query"]),
                    )
                    result = result_dict["text"]
                    # Return exact content, no files

                elif tool_name == "get_file_for_sending":
                    query = tool_call.get("query", state["user_query"])
                    file_info = await self.tools.get_file_for_sending(query)

                    if file_info and file_info.get("exists"):
                        if file_info.get("too_large"):
                            result = f"❌ File is too large ({file_info['size_mb']:.1f}MB). Limit: 50MB."
                        else:
                            state["files_to_send"].append(file_info)
                            result = f"📤 Sending {file_info['name']} ({file_info['size_mb']:.1f}MB)..."  # ✅ Clear message
                    else:
                        result = f"❌ No file found matching your request."

                elif tool_name == "list_files":
                    result = await self.tools.list_files(limit=20)

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

        state["tool_results"] = tool_results

        # Add to messages
        tool_message = "\n\n".join([
            f"Tool: {tr['tool']}\nResult:\n{tr['result']}"
            for tr in tool_results
        ])

        state["messages"] = list(state["messages"]) + [AIMessage(content=tool_message)]

        # NEW - Count ALL tool content:
        tool_results = state.get("tool_results", [])
        content_tools = [tr for tr in tool_results if tr["tool"] in ["search_files", "find_specific_content"]]
        files_count = len(state["files_to_send"])
        has_content = len(content_tools) > 0
        files_queued = files_count + (1 if has_content else 0)

        logger.info("tools_executed", files_queued=files_queued, files_count=files_count, has_content=has_content,
                    tools=len(tool_results))

        return state

    async def generate_response(self, state: AgentState) -> AgentState:
        """RAG-optimized response - FORCE use of file content."""
        logger.info("generating_strict_response")

        tool_content = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and "✅ Found" in msg.content:
                tool_content = msg.content
                logger.info("full_tool_context_captured", length=len(tool_content))
                break

        prompt = f"""You are a file content assistant.

        USER QUERY: {state['user_query']}

        FILE SEARCH RESULTS (USE ONLY THIS CONTENT):
        {tool_content}

        INSTRUCTIONS:
        1. Extract ALL relevant content about "{state['user_query']}" from above
        2. If MobTechExam.txt appears, show its CONTENT (questions, text, etc.)
        3. Quote exact text from the search results
        4. If no exact match, show closest matching files + their content

        ANSWER:"""

        logger.info("rag_prompt_length", length=len(prompt))

        response = await self.llm.generate(
            prompt=prompt,
            temperature=0.0,
        )

        state["final_response"] = response
        logger.info("strict_response_generated", length=len(response))
        return state

    async def run(
        self,
        user_query: str,
        user_id: str,
        chat_id: str,
    ) -> Dict[str, Any]:
        """Run the agent with strict tool usage."""
        logger.info("strict_agent_run", query=user_query)

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
            "tool_results": [],
            "files_to_send": [],
            "final_response": "",
            "iteration": 0,
        }

        # Run graph
        try:
            final_state = await self.graph.ainvoke(initial_state)
            response = final_state["final_response"]
            files_to_send = final_state.get("files_to_send", [])

            # Deduplicate files
            unique_files = []
            seen_paths = set()
            for file_info in files_to_send:
                path = file_info.get("path")
                if path and path not in seen_paths:
                    unique_files.append(file_info)
                    seen_paths.add(path)

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

            logger.info(
                "strict_agent_complete",
                files=len(unique_files),
                response_len=len(response)
            )

            return {
                "response": response,
                "files_to_send": unique_files
            }

        except Exception as e:
            logger.error("strict_agent_failed", error=str(e), exc_info=True)
            return {
                "response": f"Sorry, I encountered an error: {str(e)}",
                "files_to_send": []
            }