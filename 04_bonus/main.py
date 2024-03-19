from typing import Iterator, Any

import openai
from ray import serve
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from openai.resources.chat.completions import ChatCompletion
from pinecone.grpc import PineconeGRPC as Pinecone
from ray.serve.config import AutoscalingConfig

# Embedding model we used to build the search index on Pinecone
EMBEDDING_MODEL_NAME = "thenlper/gte-large"
# The Pinecone search index we built
PINECONE_INDEX_NAME = "marwan-ray-docs"
PINECONE_NAMESPACE = "example-namespace"


YOUR_ANYSCALE_API_KEY = "esecret_f6dz2g16nnrai635si83z8upk8"
YOUR_PINECONE_API_KEY = "9386359a-0227-4d5b-80d9-b1bb7600dd08"


class QueryEncoder:
    def __init__(self):
        self.embedding_model_name = EMBEDDING_MODEL_NAME
        self.client = openai.OpenAI(
            base_url="https://api.endpoints.anyscale.com/v1",
            api_key=YOUR_ANYSCALE_API_KEY,
        )

    def encode(self, query: str) -> list[float]:
        response = self.client.embeddings.create(
            input=query, model=self.embedding_model_name
        )
        return response.data[0].embedding


class VectorStore:
    def __init__(self):
        self.pc = Pinecone(api_key=YOUR_PINECONE_API_KEY)
        self.index = self.pc.Index(PINECONE_INDEX_NAME)
        self.pinecone_namespace = PINECONE_NAMESPACE

    def query(
        self, query_embedding: list[float], top_k: int, score_threshold: float = 0
    ) -> dict:
        """Retrieve the most similar chunks to the given query embedding."""
        if top_k == 0:
            return {"documents": [], "usage": {}}

        response = self.index.query(  # type: ignore
            namespace=self.pinecone_namespace,
            vector=query_embedding,
            top_k=top_k,
            include_values=False,
            include_metadata=True,
        )

        matches = response["matches"]

        if score_threshold:
            matches = [match for match in matches if match["score"] > score_threshold]

        return {
            "documents": [
                {
                    "text": match["metadata"]["text"],
                    "section_url": match["metadata"]["section_url"],
                }
                for match in matches
            ],
            "usage": response["usage"],
        }


class Retriever:
    def __init__(self):
        self.encoder = QueryEncoder()
        self.vector_store = VectorStore()

    def _compose_context(self, contexts: list[str]) -> str:
        sep = 100 * "-"
        return "\n\n".join([f"{sep}\n{context}" for context in contexts])

    def retrieve(self, query: str, top_k: int) -> dict:
        """Retrieve the context and sources for the given query."""
        encoded_query = self.encoder.encode(query)
        vector_store_response = self.vector_store.query(
            query_embedding=encoded_query,
            top_k=top_k,
        )
        contexts = [chunk["text"] for chunk in vector_store_response["documents"]]
        sources = [chunk["section_url"] for chunk in vector_store_response["documents"]]
        return {
            "contexts": contexts,
            "composed_context": self._compose_context(contexts),
            "sources": sources,
            "usage": vector_store_response["usage"],
        }


class LLM:
    def __init__(self, model: str):
        # Initialize a client to perform API requests
        self.client = openai.OpenAI(
            base_url="https://api.endpoints.anyscale.com/v1",
            api_key=YOUR_ANYSCALE_API_KEY,
        )
        self.model = model

    def generate(
        self, user_prompt: str, temperature: float = 0, **kwargs: Any
    ) -> ChatCompletion:
        """Generate a completion from the given user prompt."""
        # Call the chat completions endpoint
        chat_completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                # Prime the system with a system message - a common best practice
                {"role": "system", "content": "You are a helpful assistant."},
                # Send the user message with the proper "user" role and "content"
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            **kwargs,
        )

        return chat_completion


prompt_template_rag = """
Given the following context:
{composed_context}

Answer the following question:
{query}

If you cannot provide an answer based on the context, please say "I don't know."
Do not use the term "context" in your response.
"""


app = FastAPI()


@serve.deployment(autoscaling_config=AutoscalingConfig(min_replicas=1, max_replicas=3))
@serve.ingress(app)
class QA:
    def __init__(self, model: str):
        self.retriever = Retriever()
        self.llm = LLM(model=model)
        self.prompt_template_rag = prompt_template_rag

    def augment_prompt(self, query: str, composed_context: str) -> str:
        """Augment the prompt with the given query and contexts."""
        return self.prompt_template_rag.format(
            composed_context=composed_context, query=query
        )

    @app.get("/answer")
    def answer(
        self,
        query: str,
        top_k: int,
        include_sources: bool = True,
    ):
        return StreamingResponse(
            self._answer(query=query, top_k=top_k, include_sources=include_sources)
        )

    def _answer(
        self, query: str, top_k: int, include_sources: bool = True
    ) -> Iterator[str]:
        """Answer the given question and provide sources."""
        retrieval_response = self.retriever.retrieve(
            query=query,
            top_k=top_k,
        )
        prompt = self.augment_prompt(query, retrieval_response["contexts"])
        response = self.llm.generate(
            user_prompt=prompt,
            stream=True,
        )
        for chunk in response:
            choice = chunk.choices[0]
            if choice.delta.content is None:
                continue
            yield choice.delta.content

        if include_sources:
            yield "\n" * 2
            sources_str = "\n".join(set(retrieval_response["sources"]))
            yield sources_str
            yield "\n"


my_app = QA.bind(model="mistralai/Mixtral-8x7B-Instruct-v0.1")
