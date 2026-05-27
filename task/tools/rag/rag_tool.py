import json
from typing import Any

import faiss
import numpy as np
from aidial_client import AsyncDial
from aidial_sdk.chat_completion import Message, Role
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from task.tools.base import BaseTool
from task.tools.models import ToolCallParams
from task.tools.rag.document_cache import DocumentCache
from task.utils.dial_file_conent_extractor import DialFileContentExtractor

_SYSTEM_PROMPT = """
You are a precise document analysis assistant. Your task is to answer questions based ONLY on the provided context from a document.

## YOUR ROLE

You will receive:
1. CONTEXT: Relevant excerpts retrieved from a document using semantic search
2. REQUEST: The user's question or query about the document

## HOW TO RESPOND

1. **Answer strictly from the context**: Only use information explicitly present in the provided context. Do not add external knowledge, assumptions, or inferences beyond what the text states.

2. **Be accurate and faithful**: Quote or paraphrase the relevant parts of the context that support your answer. If the context contains specific numbers, dates, names, or procedures, include them accurately.

3. **Acknowledge limitations**: If the provided context does not contain enough information to fully answer the question, clearly state: "Based on the provided document sections, I could not find complete information about [topic]. The context mentions [what it does mention, if anything]."

4. **Be concise but complete**: Provide a direct answer first, then supporting details from the context. Don't pad responses with unnecessary preamble.

5. **Structure appropriately**: For procedural questions (how-to), use numbered steps. For factual questions, provide direct answers. For complex topics, organize information logically.

## EXAMPLES

**Good response when answer is in context:**
"According to the document, the cleaning procedure involves: 1) Remove the turntable and wash with mild soap, 2) Wipe the interior with a damp cloth, 3) Clean the door seal with a soft brush. The manual specifically warns against using abrasive cleaners."

**Good response when answer is partially available:**
"The document mentions that the warranty period is 2 years, but the specific process for filing a warranty claim is not covered in the provided sections."

**Good response when answer is not in context:**
"The provided document sections do not contain information about [requested topic]. They primarily discuss [what the context actually covers]."

## CRITICAL RULES

- DO NOT hallucinate or invent information not present in the context
- DO NOT say "I don't have access to the document" — you DO have the relevant excerpts in the context
- DO NOT provide generic advice if the document has specific instructions
- DO respond even if only partial information is available — share what you can find
- DO use the exact terminology and phrasing from the document when relevant

Your goal is to be a reliable bridge between the document content and the user's question.
"""


class RagTool(BaseTool):
    """
    Performs semantic search on documents to find and answer questions based on relevant content.
    Supports: PDF, TXT, CSV, HTML.
    """

    def __init__(self, endpoint: str, deployment_name: str, document_cache: DocumentCache):
        # 1. Set endpoint
        # 2. Set deployment_name
        # 3. Set document_cache. DocumentCache is implemented, relate to it as to centralized Dict with file_url (as key),
        #    and indexed embeddings (as value), that have some autoclean. This cache will allow us to speed up RAG search.
        # 4. Create SentenceTransformer and set is as `model` with:
        #   - model_name_or_path='all-MiniLM-L6-v2', it is self hosted lightwait embedding model.
        #     More info: https://medium.com/@rahultiwari065/unlocking-the-power-of-sentence-embeddings-with-all-minilm-l6-v2-7d6589a5f0aa
        #   - Optional! You can set it use CPU forcefully with `device='cpu'`, in case if not set up then will use GPU if it has CUDA cores
        # 5. Create RecursiveCharacterTextSplitter as `text_splitter` with:
        #   - chunk_size=500
        #   - chunk_overlap=50
        #   - length_function=len
        #   - separators=["\n\n", "\n", ". ", " ", ""]
        self.endpoint = endpoint
        self.deployment_name = deployment_name
        self.document_cache = document_cache

        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    @property
    def show_in_stage(self) -> bool:
        # set as False since we will have custom variant of representation in Stage
        return False

    @property
    def name(self) -> str:
        # provide self-descriptive name
        return "rag_tool"

    @property
    def description(self) -> str:
        # provide tool description that will help LLM to understand when to use this tools and cover 'tricky'
        #  moments (not more 1024 chars)
        return ("Performs semantic search on documents to find and answer questions based on relevant content. "
                "Supports: PDF, TXT, CSV, HTML. "
                "Use this tool when user asks questions about document content, needs specific information from large files, "
                "or wants to search for particular topics/keywords. "
                "Don't use it when: user wants to read entire document sequentially. "
                "HOW IT WORKS: Splits document into chunks, finds top 3 most relevant sections using semantic search, "
                "then generates answer based only on those sections.")

    @property
    def parameters(self) -> dict[str, Any]:
        # provide tool parameters JSON Schema:
        #  - request is string, description: "The search query or question to search for in the document", required
        #  - file_url is string, required
        return {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "The search query or question to search for in the document"
                },
                "file_url": {
                    "type": "string",
                    "description": "File URL"
                },
            },
            "required": ["request", "file_url"],
        }


    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        # 1. Load arguments with `json`
        # 2. Get `request` from arguments
        # 3. Get `file_url` from arguments
        # 4. Get stage from `tool_call_params`
        # 5. Append content to stage: "## Request arguments: \n"
        # 6. Append content to stage: `f"**Request**: {request}\n\r"`
        # 7. Append content to stage: `f"**File URL**: {file_url}\n\r"`
        # 8. Create `cache_document_key`, it is string from `conversation_id` and `file_url`, with such key we guarantee
        #    access to cached indexes for one particular conversation,
        # 9. Get from `document_cache` by `cache_document_key` a cache
        # 10. If cache is present then set it as `index, chunks = cached_data` (cached_data is retrieved cache from 9 step),
        #     otherwise:
        #       - Create DialFileContentExtractor and extract text by `file_url` as `text_content`
        #       - If no `text_content` then appen to stage info about it ans return the string with the error that file content is not found
        #       - Create `chunks` with `text_splitter`
        #       - Create `embeddings` with `model`
        #       - Create IndexFlatL2 with `384` dimensions as `index` (more about IndexFlatL2 https://shayan-fazeli.medium.com/faiss-a-quick-tutorial-to-efficient-similarity-search-595850e08473)
        #       - Add to `index` np.array with created embeddings as type 'float32'
        #       - Add to `document_cache`
        # 11. Prepare `query_embedding` with model. You need to encode request as type 'float32'
        # 12. Through created index make search with `query_embedding`, `k` set as 3. As response we expect tuple of
        #     `distances` and `indices`
        # 13. Now you need to iterate through `indices[0]` and and by each idx get element from `chunks`, result save as `retrieved_chunks`
        # 14. Make augmentation
        # 15. Append content to stage: "## RAG Request: \n"
        # 16. Append content to stage: `ff"```text\n\r{augmented_prompt}\n\r```\n\r"` (will be shown as markdown text)
        # 17. Append content to stage: "## Response: \n"
        # 18. Now make Generation with AsyncDial (don't forget about api_version '025-01-01-preview, provide LLM with system prompt and augmented prompt and:
        #   - stream response to stage (user in real time will be able to see what the LLM responding while Generation step)
        #   - collect all content (we need to return it as tool execution result)
        # 19. return collected content
        arguments = json.loads(tool_call_params.tool_call.function.arguments)
        request = arguments["request"]
        file_url = arguments["file_url"]

        stage = tool_call_params.stage
        stage.append_content("## Request arguments: \n")
        stage.append_content(f"**Request**: {request}\n\r")
        stage.append_content(f"**Document URL**: {file_url}\n")

        cache_document_key = f"{tool_call_params.conversation_id}:{file_url}"
        cached_data = self.document_cache.get(cache_document_key)
        if cached_data is not None:
            index, chunks = cached_data
        else:
            text_content = DialFileContentExtractor(
                endpoint=self.endpoint,
                api_key=tool_call_params.api_key
            ).extract_text(file_url)

            if not text_content:
                stage.append_content("## Response: \n")
                content = "Error: File content not found."
                stage.append_content(f"{content}\n")
                return content

            chunks = self.text_splitter.split_text(text_content)
            embeddings = self.model.encode(chunks)
            index = faiss.IndexFlatL2(384)
            index.add(np.array(embeddings).astype('float32'))
            self.document_cache.set(cache_document_key, index, chunks)

        query_embedding = self.model.encode([request]).astype('float32')
        k = min(3, len(chunks))
        distances, indices = index.search(query_embedding, k=k)

        retrieved_chunks = [chunks[idx] for idx in indices[0]]
        augmented_prompt = self.__augmentation(request, retrieved_chunks)
        stage.append_content(f"## RAG Request: \n")
        stage.append_content(f"```text\n\r{augmented_prompt}\n\r```\n\r")
        stage.append_content("## Response: \n")

        dial_client = AsyncDial(
            base_url=self.endpoint,
            api_key=tool_call_params.api_key,
        )
        chunks_stream = await dial_client.chat.completions.create(
            messages=[
                {
                    "role": Role.SYSTEM,
                    "content": _SYSTEM_PROMPT
                },
                {
                    "role": Role.USER,
                    "content": augmented_prompt
                }
            ],
            deployment_name=self.deployment_name,
            stream=True,
        )

        content = ''
        async for chunk in chunks_stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    tool_call_params.stage.append_content(delta.content)
                    content += delta.content

        return content

    def __augmentation(self, request: str, chunks: list[str]) -> str:
        # make prompt augmentation
        """Combine retrieved chunks with the user's request."""
        joined_chunks = "\n\n".join(chunks)
        return f"CONTEXT:\n{joined_chunks}\n---\nREQUEST: {request}"
