"""
rag_chat.py
-----------
Day 9: Conversational RAG with Chat History & Session Management.

Adds stateful multi-turn conversation on top of the Day 8 RAG + LLM pipeline.
Users can ask follow-up questions that reference prior answers, with full chat
history injected into each LLM prompt for grounded, context-aware replies.

Flow:
    User Message (Turn N)
        ↓
    Query Embedding (all-MiniLM-L6-v2)
        ↓
    Multi-Doc FAISS Retrieval + Adaptive Threshold Filtering
        ↓
    Chat History Assembly (last N turns)
        ↓
    Conversational Grounded Prompt (history + context + question)
        ↓
    Gemini LLM Generation
        ↓
    Answer + Source Citations → stored in ChatSession
        ↓
    Next Turn (follow-up aware)

Usage:
    python rag_chat.py test_files/sample.pdf test_files/ai_architecture.pdf
    python rag_chat.py test_files/sample.pdf --history-window 5 --min-score 0.35
    python rag_chat.py test_files/sample.pdf --save-session chat_sessions/session1.json

Environment:
    GEMINI_API_KEY must be set in your environment to call the Gemini API.
"""

import os
import sys
import json
import argparse
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

from rag_llm import (
    RAGLLMPipeline,
    retrieve_context,
    prepare_context,
    generate_answer,
    build_store,
    create_gemini_client,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_TOP_K,
    DEFAULT_MIN_SCORE,
    DEFAULT_MAX_SCORE_DROP,
    NO_CONTEXT_MESSAGE,
)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

DEFAULT_HISTORY_WINDOW = 6   # Maximum number of prior turns to include in prompt
MAX_HISTORY_TURNS = 20       # Hard limit for in-memory history storage


# ------------------------------------------------------------
# ChatMessage Data Model
# ------------------------------------------------------------

@dataclass
class ChatMessage:
    """
    Represents a single turn in the conversation.

    Attributes:
        role:      'user' or 'assistant'.
        content:   Text of the message.
        timestamp: ISO 8601 UTC timestamp when the message was created.
        sources:   List of source citation dicts (only for assistant messages).
    """

    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    sources: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatMessage":
        """Deserialize from a dictionary."""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            sources=data.get("sources", []),
        )


# ------------------------------------------------------------
# ChatSession — History Manager
# ------------------------------------------------------------

class ChatSession:
    """
    Manages the conversation history for a single chat session.

    Stores all messages in memory (up to MAX_HISTORY_TURNS) and
    exposes a sliding window of recent turns for prompt construction.
    Supports JSON serialization for persistence across runs.
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        history_window: int = DEFAULT_HISTORY_WINDOW,
    ):
        """
        Args:
            session_id:     Optional identifier for this session.
            history_window: Max number of prior turns to inject into each prompt.
        """
        self.session_id: str = session_id or datetime.now(timezone.utc).strftime(
            "session_%Y%m%d_%H%M%S"
        )
        self.history_window: int = max(1, history_window)
        self.messages: List[ChatMessage] = []
        self.created_at: str = datetime.now(timezone.utc).isoformat()

    # ----------------------------------------------------------
    # Add Messages
    # ----------------------------------------------------------

    def add_message(self, role: str, content: str, sources: Optional[List[Dict]] = None) -> ChatMessage:
        """
        Append a message to the session history.

        Args:
            role:    'user' or 'assistant'.
            content: Text content.
            sources: Optional source citations (for assistant messages).

        Returns:
            The created ChatMessage.
        """
        if role not in ("user", "assistant"):
            raise ValueError(f"role must be 'user' or 'assistant', got: {role!r}")
        if not content or not content.strip():
            raise ValueError("Message content cannot be empty.")

        msg = ChatMessage(role=role, content=content.strip(), sources=sources or [])
        self.messages.append(msg)

        # Trim in-memory history to prevent unbounded growth
        if len(self.messages) > MAX_HISTORY_TURNS * 2:
            self.messages = self.messages[-(MAX_HISTORY_TURNS * 2):]

        return msg

    # ----------------------------------------------------------
    # History for Prompt Construction
    # ----------------------------------------------------------

    def get_window(self) -> List[ChatMessage]:
        """
        Return the last `history_window` complete Q&A turns (user + assistant pairs).

        A 'turn' is one user message plus the following assistant reply.
        This returns the most recent `history_window` such pairs.
        """
        # Collect complete turns (user → assistant pairs)
        turns: List[tuple] = []
        i = 0
        while i < len(self.messages):
            if (
                self.messages[i].role == "user"
                and i + 1 < len(self.messages)
                and self.messages[i + 1].role == "assistant"
            ):
                turns.append((self.messages[i], self.messages[i + 1]))
                i += 2
            else:
                i += 1

        # Keep only the most recent window turns
        recent_turns = turns[-self.history_window:]

        # Flatten back to message list
        windowed: List[ChatMessage] = []
        for user_msg, assistant_msg in recent_turns:
            windowed.append(user_msg)
            windowed.append(assistant_msg)
        return windowed

    def get_history_prompt(self) -> str:
        """
        Format the windowed history into a human-readable block for the LLM prompt.

        Returns:
            Multi-line string of prior Q&A turns, or empty string if no history.
        """
        window = self.get_window()
        if not window:
            return ""

        lines = ["PRIOR CONVERSATION HISTORY (most recent first):"]
        turn_number = len(window) // 2

        i = 0
        while i < len(window):
            if (
                window[i].role == "user"
                and i + 1 < len(window)
                and window[i + 1].role == "assistant"
            ):
                lines.append(f"\n[Turn {turn_number}]")
                lines.append(f"  User     : {window[i].content}")
                lines.append(f"  Assistant: {window[i + 1].content}")
                turn_number -= 1
                i += 2
            else:
                i += 1

        return "\n".join(lines)

    # ----------------------------------------------------------
    # Session Utilities
    # ----------------------------------------------------------

    def clear(self) -> None:
        """Clear all messages from the current session."""
        self.messages.clear()

    def turn_count(self) -> int:
        """Return the number of complete Q&A turns in the session."""
        count = 0
        i = 0
        while i < len(self.messages):
            if (
                self.messages[i].role == "user"
                and i + 1 < len(self.messages)
                and self.messages[i + 1].role == "assistant"
            ):
                count += 1
                i += 2
            else:
                i += 1
        return count

    # ----------------------------------------------------------
    # Persistence (JSON)
    # ----------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Persist the session to a JSON file.

        Args:
            path: File path to write the session JSON.
        """
        parent = Path(path).parent
        parent.mkdir(parents=True, exist_ok=True)

        data = {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "history_window": self.history_window,
            "messages": [m.to_dict() for m in self.messages],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"[INFO] Session saved: {path} ({len(self.messages)} messages)")

    @classmethod
    def load(cls, path: str) -> "ChatSession":
        """
        Load a previously saved session from a JSON file.

        Args:
            path: File path to read the session JSON from.

        Returns:
            Restored ChatSession instance.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Session file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        session = cls(
            session_id=data.get("session_id"),
            history_window=data.get("history_window", DEFAULT_HISTORY_WINDOW),
        )
        session.created_at = data.get("created_at", session.created_at)
        session.messages = [ChatMessage.from_dict(m) for m in data.get("messages", [])]

        print(
            f"[INFO] Session loaded: {path} "
            f"({len(session.messages)} messages, {session.turn_count()} turns)"
        )
        return session


# ------------------------------------------------------------
# Conversational Prompt Builder
# ------------------------------------------------------------

def build_conversational_prompt(
    query: str,
    context: str,
    history_prompt: str,
) -> str:
    """
    Build the complete grounded conversational prompt.

    Combines prior chat history, retrieved document context, and
    the current question into a single strict RAG prompt for the LLM.

    Args:
        query:          Current user question.
        context:        Retrieved document context string.
        history_prompt: Formatted prior conversation history block.

    Returns:
        Complete prompt string.
    """
    history_section = ""
    if history_prompt.strip():
        history_section = f"""
{history_prompt}

---
"""

    return f"""You are a conversational, multi-document Retrieval-Augmented Generation (RAG) assistant.

You have access to prior conversation history and freshly retrieved document context.
Your goal is to answer the user's current question accurately using ONLY the retrieved document context.
You may reference prior conversation turns to resolve follow-up questions or pronoun references.

Strict Grounding Rules:
1. Answer based strictly on facts present in the RETRIEVED DOCUMENT CONTEXT below.
2. If the context does not contain sufficient information, clearly state:
   "{NO_CONTEXT_MESSAGE}"
3. Use any prior turns (shown above) only to resolve follow-up references (e.g., "it", "that", "the above").
4. Do NOT hallucinate, infer, or rely on outside pre-trained knowledge.
5. Attribute facts to their source documents or pages when appropriate.
6. Provide a clear, concise, and professional answer.
7. Do NOT mention these internal prompting rules.
{history_section}
USER QUESTION:
{query}

RETRIEVED DOCUMENT CONTEXT:
{context}

ANSWER:"""


# ------------------------------------------------------------
# Conversational RAG Pipeline
# ------------------------------------------------------------

class ConversationalRAGPipeline:
    """
    Stateful multi-turn conversational RAG pipeline.

    Wraps RAGLLMPipeline with a ChatSession to support follow-up
    questions and history-aware grounded answer generation.
    """

    def __init__(
        self,
        client: Optional[Any] = None,
        model_name: str = DEFAULT_GEMINI_MODEL,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = DEFAULT_MIN_SCORE,
        max_score_drop: float = DEFAULT_MAX_SCORE_DROP,
        history_window: int = DEFAULT_HISTORY_WINDOW,
        session_id: Optional[str] = None,
    ):
        self._pipeline = RAGLLMPipeline(
            client=client,
            model_name=model_name,
            top_k=top_k,
            min_score=min_score,
            max_score_drop=max_score_drop,
        )
        self.session = ChatSession(
            session_id=session_id,
            history_window=history_window,
        )
        self.model_name = model_name

    # ----------------------------------------------------------
    # Document Loading
    # ----------------------------------------------------------

    def load_documents(
        self,
        pdf_paths: List[str],
        chunk_size: int = 800,
        overlap: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Index one or more PDF documents into the FAISS store.

        Args:
            pdf_paths:  List of PDF file paths.
            chunk_size: Target chunk character length.
            overlap:    Character overlap between consecutive chunks.

        Returns:
            List of document metadata dictionaries.
        """
        return self._pipeline.load_documents(
            pdf_paths=pdf_paths,
            chunk_size=chunk_size,
            overlap=overlap,
        )

    def load_existing_index(self, index_prefix: str):
        """Load a pre-built FAISS index."""
        return self._pipeline.load_existing_index(index_prefix)

    def init_client(self, api_key: Optional[str] = None):
        """Initialize the Gemini client."""
        self._pipeline.init_client(api_key=api_key)

    # ----------------------------------------------------------
    # Core Chat Turn
    # ----------------------------------------------------------

    def chat(
        self,
        message: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        max_score_drop: Optional[float] = None,
        document_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process one conversational turn: retrieve context, build history-aware
        prompt, generate answer, store messages in session.

        Args:
            message:        User's current question or follow-up.
            top_k:          Override default top-k retrieval count.
            min_score:      Override default minimum similarity threshold.
            max_score_drop: Override default max score drop threshold.
            document_id:    Constrain retrieval to one specific document.

        Returns:
            Dict with keys: query, answer, found, sources, retrieved_chunks,
                            model, turn, session_id.
        """
        message = (message or "").strip()

        if not message:
            return {
                "query": "",
                "answer": "Message cannot be empty.",
                "found": False,
                "sources": [],
                "retrieved_chunks": 0,
                "model": self.model_name,
                "turn": self.session.turn_count() + 1,
                "session_id": self.session.session_id,
            }

        if self._pipeline.store is None:
            raise RuntimeError(
                "No documents loaded. Call load_documents() or load_existing_index() first."
            )

        # Resolve effective parameters
        effective_top_k = top_k if top_k is not None else self._pipeline.top_k
        effective_min_score = min_score if min_score is not None else self._pipeline.min_score
        effective_max_drop = max_score_drop if max_score_drop is not None else self._pipeline.max_score_drop

        # Step 1: Record user turn
        self.session.add_message(role="user", content=message)

        # Step 2: FAISS retrieval + adaptive filtering
        results = retrieve_context(
            store=self._pipeline.store,
            query=message,
            top_k=effective_top_k,
            min_score=effective_min_score,
            max_score_drop=effective_max_drop,
            document_id=document_id,
        )

        # Step 3: Assemble document context
        context = prepare_context(results) if results else ""

        # Step 4: Build history block
        history_prompt = self.session.get_history_prompt()

        # Step 5: Generate LLM answer
        if not context.strip():
            answer = NO_CONTEXT_MESSAGE
        else:
            # Build the full history-aware conversational prompt
            prompt = build_conversational_prompt(
                query=message,
                context=context,
                history_prompt=history_prompt,
            )

            client = self._pipeline.client
            if client is None:
                self._pipeline.init_client()
                client = self._pipeline.client

            # Always call with the conversational prompt so mock clients and
            # real Gemini clients both receive history-injected context.
            if callable(client):
                answer = client(prompt)
            else:
                try:
                    response = client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                    )
                    raw = getattr(response, "text", None)
                    answer = raw.strip() if raw else "The LLM returned an empty response."
                except Exception as exc:
                    raise RuntimeError(f"Gemini LLM generation failed: {exc}") from exc

        # Step 6: Build source attribution
        sources = []
        for result in results:
            sources.append(
                {
                    "source": result.get("source", "unknown"),
                    "filename": result.get("filename", result.get("source", "unknown")),
                    "document_id": result.get("document_id", ""),
                    "chunk_id": result.get("chunk_id", -1),
                    "page": result.get("page", 1),
                    "score": round(float(result.get("score", 0.0)), 4),
                    "confidence": result.get("confidence", "MEDIUM"),
                    "text_preview": result.get("text", "")[:150].replace("\n", " "),
                }
            )

        # Step 7: Record assistant turn
        self.session.add_message(role="assistant", content=answer, sources=sources)

        return {
            "query": message,
            "answer": answer,
            "found": bool(results),
            "sources": sources,
            "retrieved_chunks": len(results),
            "model": self.model_name,
            "turn": self.session.turn_count(),
            "session_id": self.session.session_id,
        }

    # ----------------------------------------------------------
    # Session Utilities
    # ----------------------------------------------------------

    def reset(self) -> None:
        """Clear all conversation history from the current session."""
        self.session.clear()
        print("[INFO] Chat session history cleared.")

    def save_session(self, path: str) -> None:
        """Persist the current chat session to a JSON file."""
        self.session.save(path)

    def load_session(self, path: str) -> None:
        """Restore a previously saved chat session from a JSON file."""
        self.session = ChatSession.load(path)


# ------------------------------------------------------------
# Formatted Output Helpers
# ------------------------------------------------------------

def print_turn_response(response: Dict[str, Any]) -> None:
    """Display a single conversational turn response."""
    print(f"\n[ANSWER] (Turn {response['turn']})")
    print(response["answer"])

    if response.get("sources"):
        print("\n[SOURCES]")
        for i, src in enumerate(response["sources"], start=1):
            print(
                f"  [{i}] {src['filename']} "
                f"(Page {src['page']}) | "
                f"Score: {src['score']:.4f} ({src['confidence']})"
            )

    if not response["found"]:
        print("\n[INFO] No relevant context found in indexed documents.")


def print_session_summary(session: ChatSession) -> None:
    """Display a brief summary of the current session state."""
    print(f"\n{'─' * 60}")
    print(f"  Session : {session.session_id}")
    print(f"  Turns   : {session.turn_count()}")
    print(f"  Messages: {len(session.messages)}")
    print(f"{'─' * 60}\n")


# ------------------------------------------------------------
# Interactive REPL CLI
# ------------------------------------------------------------

def run_repl(
    pipeline: "ConversationalRAGPipeline",
    save_path: Optional[str] = None,
) -> None:
    """
    Run an interactive conversational session in the terminal.

    Commands:
        /reset   — Clear chat history
        /save    — Save session to file
        /history — Show conversation so far
        /quit    — Exit the session
    """
    print("\n" + "=" * 60)
    print("  CONVERSATIONAL RAG — INTERACTIVE SESSION")
    print("=" * 60)
    print("  Type your question and press Enter.")
    print("  Commands: /reset  /save  /history  /quit")
    print(f"  Session : {pipeline.session.session_id}")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n[INFO] Session ended.")
            if save_path:
                pipeline.save_session(save_path)
            break

        if not user_input:
            continue

        # Built-in commands
        if user_input.lower() == "/quit":
            print("[INFO] Ending session.")
            if save_path:
                pipeline.save_session(save_path)
            break

        if user_input.lower() == "/reset":
            pipeline.reset()
            continue

        if user_input.lower() == "/save":
            path = save_path or f"{pipeline.session.session_id}.json"
            pipeline.save_session(path)
            continue

        if user_input.lower() == "/history":
            history = pipeline.session.get_history_prompt()
            if history:
                print("\n" + history + "\n")
            else:
                print("[INFO] No history yet.\n")
            continue

        # Process chat turn
        response = pipeline.chat(message=user_input)
        print_turn_response(response)
        print()


# ------------------------------------------------------------
# CLI Entry Point
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Day 9: Conversational Multi-Document RAG with Chat History"
    )
    parser.add_argument(
        "pdfs",
        nargs="+",
        help="Path(s) to PDF documents to index and query",
    )
    parser.add_argument(
        "--history-window",
        type=int,
        default=DEFAULT_HISTORY_WINDOW,
        help=f"Number of prior Q&A turns to include in each prompt (default: {DEFAULT_HISTORY_WINDOW})",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Max chunks to retrieve per query (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--min-score",
        "-s",
        type=float,
        default=DEFAULT_MIN_SCORE,
        help=f"Minimum cosine similarity threshold (default: {DEFAULT_MIN_SCORE})",
    )
    parser.add_argument(
        "--max-drop",
        type=float,
        default=DEFAULT_MAX_SCORE_DROP,
        help=f"Max score drop from top candidate (default: {DEFAULT_MAX_SCORE_DROP})",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=DEFAULT_GEMINI_MODEL,
        help=f"Gemini model name (default: {DEFAULT_GEMINI_MODEL})",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
        help="Chunk character size (default: 800)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=100,
        help="Chunk character overlap (default: 100)",
    )
    parser.add_argument(
        "--save-session",
        type=str,
        default=None,
        help="Path to save conversation session JSON on exit",
    )
    parser.add_argument(
        "--load-session",
        type=str,
        default=None,
        help="Path to restore a previously saved session JSON",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("DAY 9: CONVERSATIONAL MULTI-DOCUMENT RAG PIPELINE")
    print("=" * 60)

    try:
        # Initialize pipeline
        pipeline = ConversationalRAGPipeline(
            model_name=args.model,
            top_k=args.top_k,
            min_score=args.min_score,
            max_score_drop=args.max_drop,
            history_window=args.history_window,
        )

        # Load previous session if requested
        if args.load_session:
            pipeline.load_session(args.load_session)

        # Initialize Gemini client
        print("\n[STEP 1] Initializing Gemini LLM Client...")
        pipeline.init_client()
        print(f"[PASS] Gemini client ready (Model: {args.model})")

        # Index documents
        print("\n[STEP 2] Ingesting and Indexing PDF Document(s)...")
        metadata = pipeline.load_documents(
            pdf_paths=args.pdfs,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )
        print(f"[PASS] {len(metadata)} document(s) indexed")

        # Start interactive session
        run_repl(pipeline=pipeline, save_path=args.save_session)

    except KeyboardInterrupt:
        print("\n\n[INFO] Session interrupted by user.")
        sys.exit(0)
    except Exception as exc:
        print(f"\n[ERROR] Pipeline failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
