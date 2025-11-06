"""
Azure OpenAI service for embeddings, text summaries, and image descriptions.

Handles:
- Text embeddings generation (text-embedding-3-small)
- Short text summaries via chat completions (LLM)
- Image descriptions using GPT-4 Vision
- Batch processing with rate limiting
"""

import base64
import logging
import time
from typing import Any

from openai import AzureOpenAI

from src.config import AzureOpenAIConfig, AzureOpenAIVisionConfig

logger = logging.getLogger(__name__)


class OpenAIService:
    """
    Service for Azure OpenAI operations.

    Provides embeddings generation and GPT-4 Vision image descriptions.
    """

    def __init__(
        self,
        embedding_config: AzureOpenAIConfig,
        vision_config: AzureOpenAIVisionConfig | None = None,
        offline: bool = False,
    ):
        """
        Initialize OpenAI service.

        Args:
            embedding_config: Configuration for embeddings
            vision_config: Optional configuration for GPT-4 Vision
        """
        self.embedding_config = embedding_config
        self.vision_config = vision_config
        # Offline is controlled explicitly by runtime flag only
        self.offline = bool(offline)

        # Initialize embedding client (only if embedding config present)
        self.embedding_client = None
        if (
            not self.offline
            and bool(getattr(embedding_config, "endpoint", None))
            and bool(getattr(embedding_config, "key", None))
            and bool(getattr(embedding_config, "deployment_name", None))
        ):
            self.embedding_client = AzureOpenAI(
                api_key=embedding_config.key,
                api_version=embedding_config.api_version,
                azure_endpoint=embedding_config.endpoint,
            )

        # Initialize chat client using chat-specific config if available
        self.chat_client = None
        self.chat_deployment = None
        chat_endpoint = getattr(embedding_config, "chat_endpoint", "") or getattr(
            embedding_config, "endpoint", ""
        )
        chat_key = getattr(embedding_config, "chat_key", "") or getattr(
            embedding_config, "key", ""
        )
        # Prefer explicit chat deployment name if provided
        chat_deployment_alt = getattr(embedding_config, "deployment_alt4", "")
        self.chat_deployment = chat_deployment_alt or getattr(
            embedding_config, "deployment_name", ""
        )
        if not self.offline and chat_endpoint and chat_key and self.chat_deployment:
            try:
                self.chat_client = AzureOpenAI(
                    api_key=chat_key,
                    api_version=embedding_config.api_version,
                    azure_endpoint=chat_endpoint,
                )
            except Exception as e:
                logger.warning(f"Failed to initialize chat client: {str(e)}")

        # Initialize vision client if config provided
        self.vision_client = None
        if vision_config and not self.offline:
            self.vision_client = AzureOpenAI(
                api_key=vision_config.key,
                api_version=vision_config.api_version,
                azure_endpoint=vision_config.endpoint,
            )

        logger.info(
            f"Initialized OpenAIService with embedding model: {embedding_config.deployment_name}"
        )
        if self.chat_client:
            logger.info(f"Chat client initialized with deployment: {self.chat_deployment}")
        if vision_config and not self.offline:
            logger.info(f"Vision model: {vision_config.deployment}")
        if self.offline:
            logger.info("OpenAIService running in OFFLINE mode (stubbed embeddings/summaries)")

    def generate_embedding(self, text: str, max_retries: int = 3) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed
            max_retries: Maximum number of retry attempts

        Returns:
            list[float]: Embedding vector (1536 dimensions)

        Raises:
            Exception: If embedding generation fails after retries

        Example:
            >>> openai_service = OpenAIService(config)
            >>> embedding = openai_service.generate_embedding("Hello world")
            >>> print(len(embedding))  # 1536
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding, returning zero vector")
            return [0.0] * self.embedding_config.embedding_dimensions

        if self.offline:
            # Deterministic stub vector based on text hash
            import hashlib
            h = hashlib.sha256(text.encode("utf-8")).digest()
            vec = [b / 255.0 for b in h[: self.embedding_config.embedding_dimensions]]
            # Pad/truncate to expected dimension
            if len(vec) < self.embedding_config.embedding_dimensions:
                vec = vec + [0.0] * (self.embedding_config.embedding_dimensions - len(vec))
            else:
                vec = vec[: self.embedding_config.embedding_dimensions]
            return vec

        for attempt in range(max_retries):
            try:
                response = self.embedding_client.embeddings.create(
                    input=text, model=self.embedding_config.deployment_name
                )

                embedding = response.data[0].embedding

                # Validate dimension
                if len(embedding) != self.embedding_config.embedding_dimensions:
                    logger.warning(
                        f"Unexpected embedding dimension: {len(embedding)} "
                        f"(expected {self.embedding_config.embedding_dimensions})"
                    )

                return embedding

            except Exception as e:
                logger.warning(
                    f"Embedding generation attempt {attempt + 1}/{max_retries} failed: {str(e)}"
                )
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)  # Exponential backoff
                else:
                    logger.error(f"Failed to generate embedding after {max_retries} attempts")
                    raise

    def generate_embeddings_batch(
        self, texts: list[str], batch_size: int = 16, max_retries: int = 3
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts in batches.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts per batch (default: 16)
            max_retries: Maximum retry attempts per batch

        Returns:
            list[list[float]]: List of embedding vectors

        Example:
            >>> openai_service = OpenAIService(config)
            >>> texts = ["chunk 1", "chunk 2", "chunk 3"]
            >>> embeddings = openai_service.generate_embeddings_batch(texts)
            >>> print(len(embeddings))  # 3
        """
        if not texts:
            return []

        logger.info(f"Generating embeddings for {len(texts)} texts (batch_size={batch_size})")

        all_embeddings: list[list[float]] = []

        # Offline: return stub embeddings for each text
        if self.offline:
            return [self.generate_embedding(t or " ") for t in texts]

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            logger.debug(f"Processing batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}")

            for attempt in range(max_retries):
                try:
                    # Filter out empty texts
                    valid_texts = [t if t and t.strip() else " " for t in batch]

                    response = self.embedding_client.embeddings.create(
                        input=valid_texts, model=self.embedding_config.deployment_name
                    )

                    batch_embeddings = [data.embedding for data in response.data]
                    all_embeddings.extend(batch_embeddings)
                    break  # Success, move to next batch

                except Exception as e:
                    logger.warning(
                        f"Batch embedding attempt {attempt + 1}/{max_retries} failed: {str(e)}"
                    )
                    if attempt < max_retries - 1:
                        time.sleep(2**attempt)
                    else:
                        logger.error(f"Failed to generate batch embeddings after {max_retries} attempts")
                        # Return zero vectors for failed batch
                        zero_vector = [0.0] * self.embedding_config.embedding_dimensions
                        all_embeddings.extend([zero_vector] * len(batch))

        logger.info(f"Successfully generated {len(all_embeddings)} embeddings")
        return all_embeddings

    def summarize_text(self, text: str, max_chars: int = 300, max_retries: int = 3) -> str:
        """
        Generate a concise, single-paragraph summary for a page of text.

        Args:
            text: Source text to summarize
            max_chars: Target maximum character length for the summary
            max_retries: Max retries for LLM calls

        Returns:
            str: Short descriptive summary (plain text)
        """
        if not text or not text.strip():
            return ""

        # Trim input to a reasonable size to control token usage
        trimmed = text.strip()
        if len(trimmed) > 8000:
            trimmed = trimmed[:8000]

        prompt = (
            f"Summarize the following page into a single-sentence page description. "
            f"Keep it factual, concise, and under {max_chars} characters. "
            f"Avoid markdown and tables; use plain text only."
        )

        # Offline: simple extractive summary
        if self.offline:
            import re
            sentences = re.split(r"(?<=[.!?])\s+", trimmed)
            summary = " ".join(sentences[:2])
            return summary[:max_chars]

        # Prefer vision client for chat if available, else use dedicated chat client
        for attempt in range(max_retries):
            try:
                if self.vision_client:
                    response = self.vision_client.chat.completions.create(
                        model=self.vision_config.deployment,
                        messages=[
                            {"role": "system", "content": "You write concise page descriptions."},
                            {"role": "user", "content": prompt + "\n\n" + trimmed},
                        ],
                        max_tokens=200,
                        temperature=0.2,
                    )
                else:
                    if not self.chat_client:
                        raise RuntimeError("Chat client not configured")
                    response = self.chat_client.chat.completions.create(
                        model=self.chat_deployment,
                        messages=[
                            {"role": "system", "content": "You write concise page descriptions."},
                            {"role": "user", "content": prompt + "\n\n" + trimmed},
                        ],
                        max_tokens=200,
                        temperature=0.2,
                    )

                summary = (response.choices[0].message.content or "").strip()

                if len(summary) > max_chars:
                    summary = summary[:max_chars] + "..."
                return summary

            except Exception as e:
                logger.warning(
                    f"Text summary attempt {attempt + 1}/{max_retries} failed: {str(e)}"
                )
                time.sleep(0.5)

        # Fallback: simple extractive summary (first sentence or first N chars)
        try:
            first_period = trimmed.find(".")
            if 0 < first_period < max_chars:
                return trimmed[: first_period + 1]
        except Exception:
            pass
        return trimmed[:max_chars]

    def generate_image_description(
        self,
        image_bytes: bytes,
        ocr_text: str = "",
        max_length: int = 200,
        max_retries: int = 3,
    ) -> str:
        """
        Generate description for an image using GPT-4 Vision.

        Args:
            image_bytes: Image content as bytes
            ocr_text: Optional OCR text extracted from the image
            max_length: Maximum length of description (chars)
            max_retries: Maximum retry attempts

        Returns:
            str: Image description

        Raises:
            Exception: If vision client not initialized or generation fails

        Example:
            >>> openai_service = OpenAIService(embed_config, vision_config)
            >>> with open("chart.png", "rb") as f:
            >>>     image_bytes = f.read()
            >>> description = openai_service.generate_image_description(
            >>>     image_bytes, ocr_text="Q1: $1.2M"
            >>> )
            >>> print(description)
        """
        if not self.vision_client:
            raise RuntimeError("Vision client not initialized. Provide vision_config to constructor.")

        if not image_bytes:
            logger.warning("Empty image provided, returning default description")
            return "Image not available"

        logger.debug("Generating image description using GPT-4 Vision")

        # Encode image to base64
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # Construct prompt
        prompt = f"""Describe this image concisely in under {max_length} characters.

Context: This image is from an insurance document (policy, guide, or form).

{"OCR extracted text: " + ocr_text if ocr_text else ""}

Provide:
1. Image type (chart, diagram, photo, table, screenshot, flowchart, etc.)
2. Main content/subject
3. Key visual elements or data points

Be concise and factual."""

        if self.offline:
            # Minimal placeholder using OCR text or a default
            base = ocr_text.strip() if ocr_text.strip() else "Image description not available"
            return base[:max_length]

        for attempt in range(max_retries):
            try:
                response = self.vision_client.chat.completions.create(
                    model=self.vision_config.deployment,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_base64}"
                                    },
                                },
                            ],
                        }
                    ],
                    max_tokens=150,
                    temperature=0.3,  # Lower temperature for more factual descriptions
                )

                description = response.choices[0].message.content.strip()

                # Truncate if needed
                if len(description) > max_length:
                    description = description[:max_length] + "..."

                logger.debug(f"Generated image description: {description[:50]}...")
                return description

            except Exception as e:
                logger.warning(
                    f"Image description attempt {attempt + 1}/{max_retries} failed: {str(e)}"
                )
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                else:
                    logger.error("Failed to generate image description, using fallback")
                    return "Image description unavailable"

    def generate_image_descriptions_batch(
        self, images: list[tuple[bytes, str]], max_length: int = 200
    ) -> list[str]:
        """
        Generate descriptions for multiple images.

        Note: Processes sequentially to avoid rate limits.
        For production, consider implementing proper rate limiting.

        Args:
            images: List of (image_bytes, ocr_text) tuples
            max_length: Maximum description length

        Returns:
            list[str]: List of descriptions

        Example:
            >>> images = [(img1_bytes, "OCR text 1"), (img2_bytes, "OCR text 2")]
            >>> descriptions = openai_service.generate_image_descriptions_batch(images)
        """
        if not images:
            return []

        logger.info(f"Generating descriptions for {len(images)} images")

        descriptions: list[str] = []

        for idx, (image_bytes, ocr_text) in enumerate(images):
            logger.debug(f"Processing image {idx + 1}/{len(images)}")

            try:
                description = self.generate_image_description(
                    image_bytes, ocr_text, max_length
                )
                descriptions.append(description)

                # Add delay to avoid rate limits
                if idx < len(images) - 1:
                    time.sleep(0.5)

            except Exception as e:
                logger.error(f"Failed to generate description for image {idx}: {str(e)}")
                descriptions.append("Image description unavailable")

        logger.info(f"Generated {len(descriptions)} image descriptions")
        return descriptions

    def test_connection(self) -> bool:
        """
        Test connection to Azure OpenAI service.

        Returns:
            bool: True if connection successful
        """
        try:
            logger.info("Testing OpenAI service connection...")
            test_embedding = self.generate_embedding("test")
            assert len(test_embedding) == self.embedding_config.embedding_dimensions
            logger.info("✅ Embedding service connection successful")

            if self.vision_client:
                # Create a minimal 1x1 test image
                test_image = base64.b64decode(
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
                )
                test_description = self.generate_image_description(test_image, max_length=50)
                assert len(test_description) > 0
                logger.info("✅ Vision service connection successful")

            return True

        except Exception as e:
            logger.error(f"❌ OpenAI service connection test failed: {str(e)}")
            return False


# Example usage
if __name__ == "__main__":
    import sys

    from src.config import get_config

    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    try:
        config = get_config()
        openai_service = OpenAIService(
            config.azure_openai,
            config.azure_openai_vision
        )

        # Test connection
        print("Testing OpenAI service...\n")
        if openai_service.test_connection():
            print("\n✅ All tests passed!")

            # Test embedding
            print("\nGenerating sample embedding...")
            text = "This is a sample insurance policy document."
            embedding = openai_service.generate_embedding(text)
            print(f"Embedding dimension: {len(embedding)}")
            print(f"First 5 values: {embedding[:5]}")

            # Test batch embeddings
            print("\nGenerating batch embeddings...")
            texts = ["chunk 1", "chunk 2", "chunk 3"]
            embeddings = openai_service.generate_embeddings_batch(texts)
            print(f"Generated {len(embeddings)} embeddings")

        else:
            print("\n❌ Connection test failed")
            sys.exit(1)

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
