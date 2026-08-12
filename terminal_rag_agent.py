"""
Terminal RAG Agent
==================

What this script does:
1. Fetches a YouTube transcript once at startup.
2. Splits the transcript into chunks.
3. Creates a FAISS vector index with Ollama embeddings.
4. Retrieves the most relevant chunks for each user question.
5. Sends the retrieved context to an Ollama LLM.
6. Keeps asking for questions until the user types: exit

IMPORTANT BACKGROUND:
- Ollama must be installed and running locally.
- The embedding model and chat model must already exist in Ollama.
- FAISS is built once when the program starts, not on every question.
- The transcript is fetched once when the program starts, not on every question.
"""

# IMPORTS
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter


# CONFIGURATION
VIDEO_ID = "Gfr50f6ZBvo"
TRANSCRIPT_LANGUAGES = ["en"]
TOP_K = 5
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "embeddinggemma:300m"
LLM_MODEL = "gemma4:e4b"
TEMPERATURE = 0.2


# HELPER FUNCTIONS

def get_transcript(video_id: str) -> str:
    """
    Fetch the transcript for a YouTube video.

    Parameters
    ----------
    video_id:
        The YouTube video ID, for example "Gfr50f6ZBvo".

    Returns
    -------
    str
        The complete transcript as one string.

    Raises
    ------
    RuntimeError
        If the transcript cannot be fetched.

    Why this function exists:
    -------------------------
    Keeping YouTube-related code inside one function makes the rest of the
    application independent from the transcript API.
    """

    ytt_api = YouTubeTranscriptApi()

    try:
        transcript_list = ytt_api.fetch(
            video_id=video_id,
            languages=TRANSCRIPT_LANGUAGES,
        )
        transcript = " ".join(item.text for item in transcript_list)

        if not transcript.strip():
            raise RuntimeError("YouTube returned an empty transcript.")

        print("[INFO] Transcript fetched successfully.")
        return transcript

    except TranscriptsDisabled as exc:
        raise RuntimeError(
            f"Transcripts are disabled for video: {video_id}"
        ) from exc

    except Exception as exc:
        raise RuntimeError(
            f"Could not fetch transcript for video '{video_id}': {exc}"
        ) from exc


def split_text(text: str):
    """
    Split long transcript text into overlapping document chunks.

    Why chunking is needed:
    -----------------------
    Embedding the entire transcript as one document would make retrieval much
    less useful. Smaller chunks let FAISS find the specific parts related to a
    user's question.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    return splitter.create_documents([text])


def build_retriever(documents):
    """
    Build the FAISS vector store and return a retriever.

    Parameters
    ----------
    documents:
        Chunked transcript documents.

    Returns
    -------
    BaseRetriever
        A retriever that can search the vector store for relevant chunks.

    IMPORTANT:
    ----------
    This function is called only once during startup. Do not rebuild FAISS
    inside the terminal loop or every question will become unnecessarily slow.
    """

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
    )

    vector_store = FAISS.from_documents(
        documents,
        embeddings,
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )

    print("[INFO] FAISS vector store created.")
    return retriever


def format_docs(retrieved_docs) -> str:
    """
    Convert retrieved LangChain documents into plain text context.

    Why this function exists:
    -------------------------
    The LLM does not need the Document objects themselves. It needs the text
    from those documents placed into the prompt.
    """

    return "\n\n".join(
        document.page_content
        for document in retrieved_docs
    )


def build_prompt() -> PromptTemplate:
    """
    Create the instruction template given to the LLM.

    The important rule is that the model should answer using only the retrieved
    transcript context. If the context does not contain the answer, it should
    explicitly say that it does not know.
    """

    return PromptTemplate(
        template="""
        You are a helpful question-answering assistant.

        Rules:
        1. Answer ONLY from the provided context.
        2. Do not invent facts that are not present in the context.
        3. If the context is insufficient, say: "I don't know based on the provided transcript."
        4. Keep the answer clear and directly relevant to the user's question.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """.strip(),
        input_variables=["context", "question"],
    )


def build_llm() -> OllamaLLM:
    """
    Initialize the local Ollama language model.

    Keeping model creation in a function makes it easy to swap models later.
    """

    return OllamaLLM(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
    )


def answer_question(
    question: str,
    retriever,
    prompt: PromptTemplate,
    model: OllamaLLM,
) -> str:
    """
    Retrieve relevant transcript chunks and generate an answer.

    Processing pipeline:
        user question
              ↓
        vector similarity search
              ↓
        relevant transcript chunks
              ↓
        prompt construction
              ↓
        Ollama LLM
              ↓
        final answer
    """

    retrieved_docs = retriever.invoke(question)

    context = format_docs(retrieved_docs)

    if not context.strip():
        return "I don't know based on the provided transcript."

    final_prompt = prompt.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    answer = model.invoke(final_prompt)

    return str(answer).strip()


def create_agent():
    """
    Perform all one-time startup work.

    This is intentionally separate from main() so the entire initialization
    pipeline is easy to understand and reuse.

    Startup pipeline:
        YouTube → transcript → chunks → embeddings → FAISS → retriever
                                           +
                                      Ollama LLM
    """

    print("\n[1/4] Fetching transcript...")
    transcript = get_transcript(VIDEO_ID)

    print("[2/4] Splitting transcript into chunks...")
    documents = split_text(transcript)
    print(f"[INFO] Created {len(documents)} chunks.")

    print("[3/4] Building FAISS retriever...")
    retriever = build_retriever(documents)

    print("[4/4] Loading Ollama model...")
    model = build_llm()
    prompt = build_prompt()

    print("[INFO] Agent initialization completed.\n")

    return retriever, prompt, model


# TERMINAL LOOP

def run_terminal_agent(
    retriever,
    prompt: PromptTemplate,
    model: OllamaLLM,
) -> None:
    """
    Run the interactive terminal agent.

    The loop continues until the user types:
        exit
    or:
        quit

    This replaces the original one-shot:
        input(...) -> answer -> end

    with a persistent terminal session.
    """

    print("=" * 70)
    print("YouTube RAG Terminal Agent")
    print("=" * 70)
    print("Ask questions about the transcript.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            question = input("You: ").strip()

        except (KeyboardInterrupt, EOFError):
            # Ctrl+C / Ctrl+D also exits cleanly.
            print("\n\n[INFO] Exiting agent.")
            break

        # Ignore empty input instead of sending useless requests to the model.
        if not question:
            continue

        # Exit commands are checked before performing retrieval or LLM work.
        if question.lower() in {"exit", "quit"}:
            print("[INFO] Exiting agent.")
            break

        try:
            print("\nAssistant: ", end="", flush=True)

            answer = answer_question(
                question=question,
                retriever=retriever,
                prompt=prompt,
                model=model,
            )

            print(answer)
            print()

        except Exception as exc:
            # A single failed request should not kill the entire terminal agent.
            print(f"\n[ERROR] Could not answer the question: {exc}\n")


# PROGRAM ENTRY POINT

def main() -> None:
    """
    Main application entry point.

    The if __name__ == "__main__" block below calls this function when the file
    is executed directly.

    Keeping main() small makes the application's high-level behavior obvious:
        1. Build agent once.
        2. Run terminal loop.
    """

    try:
        retriever, prompt, model = create_agent()

    except Exception as exc:
        print("\n[FATAL ERROR] Agent initialization failed.")
        print(f"Reason: {exc}")
        print("\nCheck:")
        print("  - Ollama is running.")
        print("  - The embedding model exists in Ollama.")
        print("  - The LLM model exists in Ollama.")
        print("  - The YouTube video has an accessible transcript.")
        return

    run_terminal_agent(
        retriever=retriever,
        prompt=prompt,
        model=model,
    )


if __name__ == "__main__":
    main()
