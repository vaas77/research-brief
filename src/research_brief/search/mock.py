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
    "mlops": [
        SourceDocument(
            id="mlops-1",
            title="MLOps: Continuous delivery and automation pipelines in ML",
            url="https://example.com/mlops-cicd",
            snippet="CI/CD for ML includes data validation, training, and deployment gates.",
            excerpt=(
                "MLOps extends DevOps practices to machine learning by versioning data, code, and models. "
                "Automated evaluation gates prevent low-quality models from reaching production."
            ),
        ),
        SourceDocument(
            id="mlops-2",
            title="Model monitoring and drift detection",
            url="https://example.com/model-monitoring",
            snippet="Production models require drift and performance monitoring.",
            excerpt=(
                "Serving a model is not the end state. Teams monitor feature drift, prediction drift, "
                "and business KPIs to trigger retraining or rollback decisions."
            ),
        ),
    ],
}


def get_mock_sources(topic: str, max_sources: int) -> list[SourceDocument]:
    topic_lower = topic.lower()
    if any(k in topic_lower for k in ("fine-tun", "finetun")):
        pool = [MOCK_CORPUS["rag"][2], *MOCK_CORPUS["rag"][:2]]
    elif any(k in topic_lower for k in ("rag", "retrieval")):
        pool = MOCK_CORPUS["rag"]
    elif any(k in topic_lower for k in ("agent", "tool", "langchain", "orchestr")):
        pool = MOCK_CORPUS["agents"]
    elif any(k in topic_lower for k in ("mlops", "cicd", "deploy", "monitor")):
        pool = MOCK_CORPUS["mlops"]
    else:
        pool = MOCK_CORPUS["rag"] + MOCK_CORPUS["agents"]

    selected = pool[:max_sources]
    return [s.model_copy(deep=True) for s in selected]
