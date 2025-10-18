"""Vision processing for images and vision models.

This module provides image processing, vision model analysis, and multimodal embeddings.
"""

import base64
import io
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

from ..types import (EmbeddingModelMultimodal, ImageFormat, ImageInfo,
                     VisionAnalysis, VisionModel)
from ..utilities import get_mime_type

if TYPE_CHECKING:
    from kerb.core.enums import Device


# ============================================================================
# Image Processing Functions
# ============================================================================


def load_image(file_path: str) -> Any:
    """Load an image from file.

    Args:
        file_path: Path to the image file

    Returns:
        PIL.Image: Loaded image object

    Raises:
        ImportError: If PIL is not installed
        FileNotFoundError: If file doesn't exist

    Examples:
        >>> img = load_image("photo.jpg")
        >>> img.size
        (1920, 1080)
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError(
            "PIL (Pillow) is required for image processing. Install with: pip install Pillow"
        )

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image file not found: {file_path}")

    return Image.open(file_path)


def get_image_info(file_path: str) -> ImageInfo:
    """Get detailed information about an image.

    Args:
        file_path: Path to the image file

    Returns:
        ImageInfo: Image information object

    Examples:
        >>> info = get_image_info("photo.jpg")
        >>> print(f"{info.width}x{info.height}")
        1920x1080
    """
    img = load_image(file_path)
    size_bytes = os.path.getsize(file_path)

    format_map = {
        "JPEG": ImageFormat.JPEG,
        "PNG": ImageFormat.PNG,
        "WEBP": ImageFormat.WEBP,
        "GIF": ImageFormat.GIF,
        "BMP": ImageFormat.BMP,
        "TIFF": ImageFormat.TIFF,
        "SVG": ImageFormat.SVG,
    }

    img_format = format_map.get(img.format, ImageFormat.JPEG)
    aspect_ratio = img.width / img.height if img.height > 0 else 0.0

    metadata = {}
    if hasattr(img, "info"):
        metadata = dict(img.info)

    return ImageInfo(
        width=img.width,
        height=img.height,
        format=img_format,
        mode=img.mode,
        size_bytes=size_bytes,
        aspect_ratio=aspect_ratio,
        metadata=metadata,
    )


def convert_image_format(
    file_path: str,
    target_format: Union[str, ImageFormat],
    output_path: Optional[str] = None,
    quality: int = 85,
) -> str:
    """Convert image to a different format.

    Args:
        file_path: Path to the input image
        target_format: Target format (e.g., "PNG", "JPEG")
        output_path: Output path (auto-generated if None)
        quality: Quality for lossy formats

    Returns:
        str: Path to the converted image

    Examples:
        >>> convert_image_format("photo.png", "JPEG")
        'photo.jpg'
    """
    img = load_image(file_path)

    if isinstance(target_format, ImageFormat):
        target_format = target_format.value.upper()
    else:
        target_format = target_format.upper()

    if output_path is None:
        base = os.path.splitext(file_path)[0]
        ext_map = {
            "JPEG": ".jpg",
            "PNG": ".png",
            "WEBP": ".webp",
            "GIF": ".gif",
            "BMP": ".bmp",
            "TIFF": ".tiff",
        }
        output_path = base + ext_map.get(target_format, ".jpg")

    # Convert RGBA to RGB for JPEG
    if target_format == "JPEG" and img.mode in ("RGBA", "LA", "P"):
        from PIL import Image

        background = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "P":
            img = img.convert("RGBA")
        background.paste(
            img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None
        )
        img = background

    save_kwargs = {}
    if target_format in ("JPEG", "WEBP"):
        save_kwargs["quality"] = quality
        save_kwargs["optimize"] = True

    img.save(output_path, format=target_format, **save_kwargs)
    return output_path


def image_to_base64(file_path: str, include_prefix: bool = True) -> str:
    """Convert image to base64 string.

    Args:
        file_path: Path to the image file
        include_prefix: Whether to include data URI prefix

    Returns:
        str: Base64-encoded image string

    Examples:
        >>> b64 = image_to_base64("photo.jpg")
        >>> b64[:30]
        'data:image/jpeg;base64,/9j/4A'
    """
    with open(file_path, "rb") as f:
        image_data = f.read()

    b64_string = base64.b64encode(image_data).decode("utf-8")

    if include_prefix:
        mime_type = get_mime_type(file_path)
        return f"data:{mime_type};base64,{b64_string}"

    return b64_string


def base64_to_image(b64_string: str, output_path: str) -> str:
    """Convert base64 string to image file.

    Args:
        b64_string: Base64-encoded image (with or without prefix)
        output_path: Path to save the image

    Returns:
        str: Path to the saved image

    Examples:
        >>> base64_to_image(b64_data, "output.jpg")
        'output.jpg'
    """
    # Remove data URI prefix if present
    if "," in b64_string and b64_string.startswith("data:"):
        b64_string = b64_string.split(",", 1)[1]

    image_data = base64.b64decode(b64_string)

    with open(output_path, "wb") as f:
        f.write(image_data)

    return output_path


def extract_dominant_colors(
    file_path: str, num_colors: int = 5
) -> List[Tuple[int, int, int]]:
    """Extract dominant colors from an image.

    Args:
        file_path: Path to the image file
        num_colors: Number of dominant colors to extract

    Returns:
        List of RGB tuples representing dominant colors

    Examples:
        >>> colors = extract_dominant_colors("photo.jpg", 3)
        >>> colors
        [(45, 67, 89), (120, 130, 140), (200, 210, 220)]
    """
    img = load_image(file_path)

    # Convert to RGB if needed
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Resize for faster processing
    img.thumbnail((150, 150))

    # Get pixel data
    pixels = list(img.getdata())

    # Simple k-means clustering approach
    # For production, consider using sklearn or similar
    from collections import Counter

    # Group similar colors
    def round_color(rgb, step=32):
        return tuple((c // step) * step for c in rgb)

    rounded = [round_color(p) for p in pixels]
    color_counts = Counter(rounded)

    # Get top colors
    top_colors = [color for color, _ in color_counts.most_common(num_colors)]

    return top_colors


def calculate_image_hash(file_path: str, hash_size: int = 8) -> str:
    """Calculate perceptual hash of an image for similarity comparison.

    Args:
        file_path: Path to the image file
        hash_size: Size of hash (default 8 gives 64-bit hash)

    Returns:
        str: Hexadecimal hash string

    Examples:
        >>> hash1 = calculate_image_hash("photo1.jpg")
        >>> hash2 = calculate_image_hash("photo2.jpg")
        >>> hash1 == hash2  # Similar images have same hash
        True
    """
    img = load_image(file_path)

    # Convert to grayscale and resize
    img = img.convert("L")
    img = img.resize((hash_size + 1, hash_size), resample=1)

    # Calculate difference hash (dHash)
    pixels = list(img.getdata())
    difference = []

    for row in range(hash_size):
        for col in range(hash_size):
            pixel_left = pixels[row * (hash_size + 1) + col]
            pixel_right = pixels[row * (hash_size + 1) + col + 1]
            difference.append(pixel_left > pixel_right)

    # Convert to hex
    hex_string = ""
    for i in range(0, len(difference), 4):
        chunk = difference[i : i + 4]
        hex_value = sum([2**j for j, b in enumerate(chunk) if b])
        hex_string += format(hex_value, "x")

    return hex_string


# ============================================================================
# Vision Model Integration
# ============================================================================


def analyze_image_with_vision_model(
    image_path: str,
    prompt: str,
    model: Union[str, VisionModel] = VisionModel.GPT4O,
    api_key: Optional[str] = None,
    max_tokens: int = 300,
) -> VisionAnalysis:
    """Analyze an image using a vision model.

    Args:
        image_path: Path to the image file
        prompt: Text prompt/question about the image
        model: Vision model to use
        api_key: API key for the model provider
        max_tokens: Maximum tokens in response

    Returns:
        VisionAnalysis: Analysis result with description and metadata

    Examples:
        >>> analysis = analyze_image_with_vision_model(
        ...     "photo.jpg",
        ...     "What objects are in this image?"
        ... )
        >>> print(analysis.description)
        'The image contains a cat, a book, and a coffee mug on a table.'
    """
    model_str = model.value if isinstance(model, VisionModel) else model

    # Determine provider
    if model_str.startswith("gpt-4"):
        return _analyze_openai_vision(
            image_path, prompt, model_str, api_key, max_tokens
        )
    elif model_str.startswith("claude-3"):
        return _analyze_anthropic_vision(
            image_path, prompt, model_str, api_key, max_tokens
        )
    elif model_str.startswith("gemini"):
        return _analyze_google_vision(
            image_path, prompt, model_str, api_key, max_tokens
        )
    else:
        raise ValueError(f"Unsupported vision model: {model_str}")


def _analyze_openai_vision(
    image_path: str, prompt: str, model: str, api_key: Optional[str], max_tokens: int
) -> VisionAnalysis:
    """Analyze image using OpenAI vision model."""
    try:
        import openai
    except ImportError:
        raise ImportError("openai is required. Install with: pip install openai")

    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key required")

    client = openai.OpenAI(api_key=api_key)

    # Convert image to base64
    image_b64 = image_to_base64(image_path, include_prefix=True)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_b64}},
                ],
            }
        ],
        max_tokens=max_tokens,
    )

    description = response.choices[0].message.content

    return VisionAnalysis(
        description=description,
        metadata={
            "model": model,
            "usage": response.usage.model_dump() if response.usage else {},
        },
    )


def _analyze_anthropic_vision(
    image_path: str, prompt: str, model: str, api_key: Optional[str], max_tokens: int
) -> VisionAnalysis:
    """Analyze image using Anthropic Claude vision model."""
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic is required. Install with: pip install anthropic")

    api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Anthropic API key required")

    client = anthropic.Anthropic(api_key=api_key)

    # Read image and encode
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    mime_type = get_mime_type(image_path)

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime_type,
                            "data": image_data,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )

    description = response.content[0].text

    return VisionAnalysis(
        description=description,
        metadata={
            "model": model,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        },
    )


def _analyze_google_vision(
    image_path: str, prompt: str, model: str, api_key: Optional[str], max_tokens: int
) -> VisionAnalysis:
    """Analyze image using Google Gemini vision model."""
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError(
            "google-generativeai is required. Install with: pip install google-generativeai"
        )

    api_key = api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Google API key required")

    genai.configure(api_key=api_key)

    model_obj = genai.GenerativeModel(model)

    # Load image
    from PIL import Image

    img = Image.open(image_path)

    response = model_obj.generate_content([prompt, img])

    return VisionAnalysis(
        description=response.text,
        metadata={
            "model": model,
            "finish_reason": (
                response.candidates[0].finish_reason if response.candidates else None
            ),
        },
    )


# ============================================================================
# Multi-Modal Embeddings
# ============================================================================


def embed_multimodal(
    content: Union[str, bytes],
    content_type: str,
    model: Union[
        str, EmbeddingModelMultimodal
    ] = EmbeddingModelMultimodal.CLIP_VIT_B_32,
    device: Union["Device", str] = "cpu",
) -> List[float]:
    """Generate multi-modal embeddings for images, audio, or text.

    Args:
        content: Content to embed (file path for images/audio, text string for text)
        content_type: Type of content ("image", "audio", "text")
        model: Embedding model to use
        device: Device to run model on (Device enum or string: "cpu", "cuda", "cuda:0", "cuda:1", "mps")

    Returns:
        List of embedding values

    Examples:
        >>> # Using enum (recommended)
        >>> from kerb.core.enums import Device
        >>> embedding = embed_multimodal("photo.jpg", "image", device=Device.CUDA)

        >>> # Using string (for backward compatibility)
        >>> embedding = embed_multimodal("photo.jpg", "image")
        >>> len(embedding)
        512
    """
    from kerb.core.enums import Device, validate_enum_or_string

    model_str = model.value if isinstance(model, EmbeddingModelMultimodal) else model

    # Validate and normalize device
    device_val = validate_enum_or_string(device, Device, "device")
    if isinstance(device_val, Device):
        device_str = device_val.value
    else:
        device_str = device_val

    if model_str.startswith("clip"):
        return _embed_clip(content, content_type, model_str, device_str)
    elif model_str == "imagebind":
        return _embed_imagebind(content, content_type, device_str)
    else:
        raise ValueError(f"Unsupported embedding model: {model_str}")


def _embed_clip(
    content: Union[str, bytes], content_type: str, model: str, device: str
) -> List[float]:
    """Generate CLIP embeddings."""
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
    except ImportError:
        raise ImportError(
            "transformers and torch required. Install with: pip install transformers torch"
        )

    # Load model
    model_obj = CLIPModel.from_pretrained(model).to(device)
    processor = CLIPProcessor.from_pretrained(model)

    if content_type == "image":
        from PIL import Image

        image = Image.open(content)
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            image_features = model_obj.get_image_features(**inputs)

        embedding = image_features[0].cpu().numpy().tolist()

    elif content_type == "text":
        inputs = processor(text=content, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            text_features = model_obj.get_text_features(**inputs)

        embedding = text_features[0].cpu().numpy().tolist()

    else:
        raise ValueError(f"Content type {content_type} not supported for CLIP")

    return embedding


def _embed_imagebind(
    content: Union[str, bytes], content_type: str, device: str
) -> List[float]:
    """Generate ImageBind embeddings (supports image, audio, text)."""
    try:
        import torch

        # ImageBind would be imported here
        # from imagebind import data
        # from imagebind.models import imagebind_model
    except ImportError:
        raise ImportError("imagebind required for ImageBind embeddings")

    # Placeholder - actual implementation would use ImageBind
    raise NotImplementedError("ImageBind implementation requires the imagebind package")


def compute_multimodal_similarity(
    embedding1: List[float], embedding2: List[float]
) -> float:
    """Compute cosine similarity between two multi-modal embeddings.

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector

    Returns:
        float: Cosine similarity score (-1 to 1)

    Examples:
        >>> emb1 = embed_multimodal("photo1.jpg", "image")
        >>> emb2 = embed_multimodal("photo2.jpg", "image")
        >>> similarity = compute_multimodal_similarity(emb1, emb2)
        >>> print(f"Similarity: {similarity:.3f}")
        Similarity: 0.892
    """
    # Compute dot product
    dot_product = sum(a * b for a, b in zip(embedding1, embedding2))

    # Compute magnitudes
    mag1 = math.sqrt(sum(a * a for a in embedding1))
    mag2 = math.sqrt(sum(b * b for b in embedding2))

    # Avoid division by zero
    if mag1 == 0 or mag2 == 0:
        return 0.0

    return dot_product / (mag1 * mag2)
