from __future__ import annotations

from research_brief.models import SourceDocument

MOCK_CORPUS: dict[str, list[SourceDocument]] = {
    "rag": [
        SourceDocument(
            id="rag-1",
            title="Retrieval-Augmented Generation for Knowledge-Intensive NLP",
            url="https://arxiv.org/abs/2005.11401",
            snippet="RAG combines parametric and non-parametric memory for knowledge-intensive tasks.",
            excerpt=(
                "We introduce RAG models where the parametric memory is a pre-trained seq2seq model "
                "and the non-parametric memory is a dense vector index of Wikipedia. "
                "RAG reduces hallucination on open-domain QA by grounding generation in retrieved passages."
            ),
        ),
        SourceDocument(
            id="rag-2",
            title="Precise Zero-Shot Dense Retrieval (RePlug)",
            url="https://arxiv.org/abs/2301.12652",
            snippet="Retrieval quality strongly affects downstream generation quality.",
            excerpt=(
                "Retrieval-augmented LMs benefit when retrieved documents are relevant and diverse. "
                "Poor retrieval increases noise and can hurt answer faithfulness compared to smaller, "
                "curated context windows."
            ),
        ),
        SourceDocument(
            id="rag-3",
            title="Fine-Tuning vs Retrieval in Production LLM Systems",
            url="https://example.com/rag-vs-finetune",
            snippet="Fine-tuning adapts style and task behavior; retrieval adapts facts.",
            excerpt=(
                "Fine-tuning is often preferred for stable task formats, tone, and domain jargon. "
                "Retrieval is preferred for changing facts, citations, and lower retraining cost. "
                "Many production systems combine both."
            ),
        ),
    ],
    "agents": [
        SourceDocument(
            id="agent-1",
            title="ReAct: Synergizing Reasoning and Acting in Language Models",
            url="https://arxiv.org/abs/2210.03629",
            snippet="LLM agents interleave reasoning traces with tool actions.",
            excerpt=(
                "ReAct prompts LLMs to generate reasoning traces and task-specific actions in an interleaved manner. "
                "This improves interpretability and reduces error propagation in multi-step tasks."
            ),
        ),
        SourceDocument(
            id="agent-2",
            title="Toolformer: Language Models Can Teach Themselves to Use Tools",
            url="https://arxiv.org/abs/2302.04761",
            snippet="Models learn when and how to call APIs.",
            excerpt=(
                "Tool use can be learned with self-supervision. Models decide which APIs to call, "
                "with what arguments, and how to incorporate results into future token prediction."
            ),
        ),
    ],
}


def get_mock_sources(topic: str, max_sources: int) -> list[SourceDocument]:
    topic_lower = topic.lower()
    if any(k in topic_lower for k in ("rag", "retrieval", "fine-tun", "finetun")):
        pool = MOCK_CORPUS["rag"]
    elif any(k in topic_lower for k in ("agent", "tool", "langchain", "orchestr")):
        pool = MOCK_CORPUS["agents"]
    else:
        pool = MOCK_CORPUS["rag"] + MOCK_CORPUS["agents"]

    selected = pool[:max_sources]
    return [s.model_copy(deep=True) for s in selected]
