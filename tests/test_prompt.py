"""Comprehensive tests for the prompt management module."""

import pytest

from kerb.prompt import (  # Template engine; Versioning; Few-shot examples; Compression
    ExampleSelector, FewShotExample, PromptRegistry, PromptVersion,
    analyze_prompt, compare_versions, compress_prompt, create_example,
    create_version, extract_template_variables, format_examples, get_prompt,
    list_versions, optimize_whitespace, register_prompt, render_template,
    render_template_safe, select_examples, select_version, validate_template)

# ============================================================================
# Template Engine Tests
# ============================================================================


def test_render_template_basic():
    """Test basic template rendering with single variable."""
    template = "Hello {{name}}!"
    variables = {"name": "Alice"}
    result = render_template(template, variables)
    assert result == "Hello Alice!"


def test_render_template_multiple_variables():
    """Test template with multiple variables."""
    template = "{{greeting}} {{name}}, you are {{age}} years old!"
    variables = {"greeting": "Hi", "name": "Bob", "age": 25}
    result = render_template(template, variables)
    assert result == "Hi Bob, you are 25 years old!"


def test_render_template_nested_variables():
    """Test template with nested object access."""
    template = "User: {{user.name}}, Email: {{user.email}}"
    variables = {"user": {"name": "Charlie", "email": "charlie@example.com"}}
    result = render_template(template, variables)
    assert result == "User: Charlie, Email: charlie@example.com"


def test_render_template_custom_delimiters():
    """Test template with custom delimiters."""
    template = "Value: {value}, Count: {count}"
    variables = {"value": 42, "count": 10}
    result = render_template(template, variables, delimiters=("{", "}"))
    assert result == "Value: 42, Count: 10"


def test_render_template_missing_variable_raises_error():
    """Test that missing variables raise KeyError by default."""
    template = "Hello {{name}}!"
    variables = {}
    with pytest.raises(KeyError):
        render_template(template, variables)


def test_render_template_allow_missing():
    """Test template rendering with allow_missing=True."""
    template = "Hello {{name}}, status: {{status}}"
    variables = {"name": "Alice"}
    result = render_template(template, variables, allow_missing=True)
    assert result == "Hello Alice, status: {{status}}"


def test_render_template_safe_basic():
    """Test safe rendering with missing variables replaced by default."""
    template = "Hello {{name}}!"
    variables = {}
    result = render_template_safe(template, variables)
    assert result == "Hello !"


def test_render_template_safe_custom_default():
    """Test safe rendering with custom default value."""
    template = "Hello {{name}}!"
    variables = {}
    result = render_template_safe(template, variables, default="[unknown]")
    assert result == "Hello [unknown]!"


def test_validate_template_valid():
    """Test template validation with all variables present."""
    template = "Hello {{name}}, you are {{age}} years old!"
    variables = {"name": "Alice", "age": 25}
    is_valid, missing = validate_template(template, variables)
    assert is_valid is True
    assert missing == []


def test_validate_template_missing_variables():
    """Test template validation with missing variables."""
    template = "Hello {{name}}, you are {{age}} years old!"
    variables = {"name": "Alice"}
    is_valid, missing = validate_template(template, variables)
    assert is_valid is False
    assert "age" in missing


def test_extract_template_variables_single():
    """Test extracting variables from template with one variable."""
    template = "Hello {{name}}!"
    variables = extract_template_variables(template)
    assert variables == ["name"]


def test_extract_template_variables_multiple():
    """Test extracting multiple variables from template."""
    template = "{{greeting}} {{name}}, you are {{age}} years old!"
    variables = extract_template_variables(template)
    assert variables == ["greeting", "name", "age"]


def test_extract_template_variables_nested():
    """Test extracting nested variables."""
    template = "User: {{user.name}}, City: {{user.address.city}}"
    variables = extract_template_variables(template)
    assert "user.name" in variables
    assert "user.address.city" in variables


def test_extract_template_variables_custom_delimiters():
    """Test extracting variables with custom delimiters."""
    template = "Value: {value}, Count: {count}"
    variables = extract_template_variables(template, delimiters=("{", "}"))
    assert variables == ["value", "count"]


# ============================================================================
# Prompt Versioning Tests
# ============================================================================


def test_create_prompt_version():
    """Test creating a basic prompt version."""
    version = create_version(
        name="greeting",
        version="1.0",
        template="Hello {{name}}!",
        description="Simple greeting",
    )
    assert version.name == "greeting"
    assert version.version == "1.0"
    assert version.template == "Hello {{name}}!"
    assert version.description == "Simple greeting"


def test_prompt_version_render():
    """Test rendering a prompt version."""
    version = create_version(
        name="greeting",
        version="1.0",
        template="Hello {{name}}!",
    )
    result = version.render({"name": "Alice"})
    assert result == "Hello Alice!"


def test_prompt_version_render_with_defaults():
    """Test rendering with default variables."""
    version = create_version(
        name="greeting",
        version="1.0",
        template="Hello {{name}}!",
        variables={"name": "World"},
    )
    result = version.render()
    assert result == "Hello World!"

    # Override default
    result = version.render({"name": "Alice"})
    assert result == "Hello Alice!"


def test_prompt_registry_register_and_get():
    """Test registering and retrieving prompts."""
    registry = PromptRegistry()

    version = create_version(name="greeting", version="1.0", template="Hello {{name}}!")
    registry.register(version)

    retrieved = registry.get("greeting", "1.0")
    assert retrieved is not None
    assert retrieved.version == "1.0"


def test_prompt_registry_get_latest():
    """Test getting latest version when version not specified."""
    registry = PromptRegistry()

    v1 = create_version(name="greeting", version="1.0", template="Hello {{name}}!")
    v2 = create_version(name="greeting", version="2.0", template="Hi {{name}}!")

    registry.register(v1)
    registry.register(v2)

    latest = registry.get("greeting")
    assert latest is not None
    # Latest should be v2 (most recent)
    assert latest.version == "2.0"


def test_prompt_registry_list_versions():
    """Test listing all versions of a prompt."""
    registry = PromptRegistry()

    v1 = create_version(name="greeting", version="1.0", template="Hello!")
    v2 = create_version(name="greeting", version="2.0", template="Hi!")
    v3 = create_version(name="greeting", version="3.0", template="Hey!")

    registry.register(v1)
    registry.register(v2)
    registry.register(v3)

    versions = registry.list_versions("greeting")
    assert len(versions) == 3
    assert "1.0" in versions
    assert "2.0" in versions
    assert "3.0" in versions


def test_prompt_registry_compare_versions():
    """Test comparing different prompt versions."""
    registry = PromptRegistry()

    v1 = create_version(
        name="greeting",
        version="1.0",
        template="Hello {{name}}!",
        description="First version",
    )
    v2 = create_version(
        name="greeting",
        version="2.0",
        template="Hi {{name}}, welcome!",
        description="Second version",
    )

    registry.register(v1)
    registry.register(v2)

    comparison = registry.compare("greeting")
    assert "versions" in comparison
    assert "1.0" in comparison["versions"]
    assert "2.0" in comparison["versions"]
    assert (
        comparison["versions"]["1.0"]["length"]
        < comparison["versions"]["2.0"]["length"]
    )


def test_prompt_registry_select_ab_random():
    """Test random A/B version selection."""
    registry = PromptRegistry()

    v1 = create_version(name="greeting", version="1.0", template="Hello!")
    v2 = create_version(name="greeting", version="2.0", template="Hi!")

    registry.register(v1)
    registry.register(v2)

    selected = registry.select_ab_version("greeting", strategy="random")
    assert selected is not None
    assert selected.version in ["1.0", "2.0"]


def test_prompt_registry_select_ab_newest():
    """Test selecting newest version."""
    registry = PromptRegistry()

    v1 = create_version(name="greeting", version="1.0", template="Hello!")
    v2 = create_version(name="greeting", version="2.0", template="Hi!")

    registry.register(v1)
    registry.register(v2)

    selected = registry.select_ab_version("greeting", strategy="newest")
    assert selected is not None
    assert selected.version == "2.0"


def test_global_registry_functions():
    """Test global registry convenience functions."""
    version = create_version(
        name="test_global", version="1.0", template="Test {{value}}"
    )
    register_prompt(version)

    retrieved = get_prompt("test_global", "1.0")
    assert retrieved is not None
    assert retrieved.template == "Test {{value}}"

    versions = list_versions("test_global")
    assert "1.0" in versions


# ============================================================================
# Few-Shot Example Tests
# ============================================================================


def test_create_example():
    """Test creating a few-shot example."""
    example = create_example(
        input_text="What is 2+2?", output_text="4", metadata={"difficulty": "easy"}
    )
    assert example.input == "What is 2+2?"
    assert example.output == "4"
    assert example.metadata["difficulty"] == "easy"


def test_example_format_default():
    """Test formatting example with default template."""
    example = create_example("What is 2+2?", "4")
    formatted = example.format()
    assert "Input: What is 2+2?" in formatted
    assert "Output: 4" in formatted


def test_example_format_custom():
    """Test formatting example with custom template."""
    example = create_example("What is 2+2?", "4")
    formatted = example.format(template="Q: {input}\nA: {output}")
    assert formatted == "Q: What is 2+2?\nA: 4"


def test_example_selector_add():
    """Test adding examples to selector."""
    selector = ExampleSelector()

    ex1 = create_example("What is 2+2?", "4")
    ex2 = create_example("What is 3+3?", "6")

    selector.add(ex1)
    selector.add(ex2)

    assert len(selector.examples) == 2


def test_example_selector_random():
    """Test random selection strategy."""
    examples = [create_example(f"Question {i}", f"Answer {i}") for i in range(10)]

    selector = ExampleSelector(examples)
    selected = selector.select(k=3, strategy="random")

    assert len(selected) == 3
    assert all(isinstance(ex, FewShotExample) for ex in selected)


def test_example_selector_first():
    """Test first selection strategy."""
    examples = [create_example(f"Question {i}", f"Answer {i}") for i in range(10)]

    selector = ExampleSelector(examples)
    selected = selector.select(k=3, strategy="first")

    assert len(selected) == 3
    assert selected[0].input == "Question 0"
    assert selected[1].input == "Question 1"
    assert selected[2].input == "Question 2"


def test_example_selector_last():
    """Test last selection strategy."""
    examples = [create_example(f"Question {i}", f"Answer {i}") for i in range(10)]

    selector = ExampleSelector(examples)
    selected = selector.select(k=3, strategy="last")

    assert len(selected) == 3
    assert selected[0].input == "Question 7"
    assert selected[1].input == "Question 8"
    assert selected[2].input == "Question 9"


def test_example_selector_diverse():
    """Test diverse selection strategy."""
    examples = [create_example(f"Question {i}", f"Answer {i}") for i in range(10)]

    selector = ExampleSelector(examples)
    selected = selector.select(k=3, strategy="diverse")

    assert len(selected) == 3
    # Diverse strategy should space out selections


def test_example_selector_with_filter():
    """Test selection with filter function."""
    examples = [
        create_example(
            f"Question {i}",
            f"Answer {i}",
            metadata={"difficulty": "easy" if i < 5 else "hard"},
        )
        for i in range(10)
    ]

    selector = ExampleSelector(examples)
    selected = selector.select(
        k=3, strategy="first", filter_fn=lambda ex: ex.metadata["difficulty"] == "easy"
    )

    assert len(selected) == 3
    assert all(ex.metadata["difficulty"] == "easy" for ex in selected)


def test_format_examples_function():
    """Test formatting multiple examples."""
    examples = [
        create_example("What is 2+2?", "4"),
        create_example("What is 3+3?", "6"),
    ]

    formatted = format_examples(examples)
    assert "Input: What is 2+2?" in formatted
    assert "Output: 4" in formatted
    assert "Input: What is 3+3?" in formatted
    assert "Output: 6" in formatted


def test_format_examples_custom_template():
    """Test formatting with custom template and separator."""
    examples = [
        create_example("What is 2+2?", "4"),
        create_example("What is 3+3?", "6"),
    ]

    formatted = format_examples(
        examples, template="Q: {input} | A: {output}", separator=" | "
    )
    assert "Q: What is 2+2? | A: 4 | Q: What is 3+3? | A: 6" == formatted


# ============================================================================
# Compression and Optimization Tests
# ============================================================================


def test_optimize_whitespace_multiple_spaces():
    """Test removing multiple spaces."""
    prompt = "Hello    world!  This   is   a    test."
    optimized = optimize_whitespace(prompt)
    assert optimized == "Hello world! This is a test."


def test_optimize_whitespace_multiple_newlines():
    """Test normalizing multiple newlines."""
    prompt = "Hello world!\n\n\n\nThis is a test."
    optimized = optimize_whitespace(prompt)
    assert optimized == "Hello world!\n\nThis is a test."


def test_optimize_whitespace_trailing():
    """Test removing trailing whitespace."""
    prompt = "Hello world!   \nThis is a test.   "
    optimized = optimize_whitespace(prompt)
    assert "   " not in optimized
    assert optimized == "Hello world!\nThis is a test."


def test_compress_prompt_whitespace():
    """Test compression with whitespace strategy."""
    prompt = "Hello    world!  This   is   a    test."
    compressed = compress_prompt(prompt, strategies=["whitespace"])
    assert compressed == "Hello world! This is a test."


def test_compress_prompt_all_strategies():
    """Test compression with all strategies."""
    prompt = "Please    help me.   This   is   important."
    compressed = compress_prompt(prompt)
    assert "  " not in compressed


def test_compress_prompt_with_max_length():
    """Test compression with length constraint."""
    prompt = "This is a very long prompt that needs to be compressed to fit within the specified maximum length."
    compressed = compress_prompt(prompt, max_length=50)
    assert len(compressed) <= 50


def test_truncate_prompt_end():
    """Test truncating prompt at the end - now uses preprocessing module."""
    from kerb.preprocessing import truncate_text

    prompt = "This is a long prompt that needs truncation"
    truncated = truncate_text(prompt, max_length=20, strategy="end", suffix="...")
    assert len(truncated) == 20
    assert truncated.endswith("...")
    assert truncated.startswith("This is")


def test_truncate_prompt_middle():
    """Test truncating prompt in the middle - now uses preprocessing module."""
    from kerb.preprocessing import truncate_text

    prompt = "This is a long prompt that needs truncation"
    truncated = truncate_text(prompt, max_length=20, strategy="middle", suffix="...")
    assert len(truncated) == 20
    assert "..." in truncated
    assert truncated.startswith("This")
    assert truncated.endswith("tion")


def test_truncate_prompt_smart():
    """Test smart truncation at sentence boundary - now uses preprocessing module."""
    from kerb.preprocessing import truncate_text

    prompt = "First sentence. Second sentence. Third sentence. Fourth sentence."
    truncated = truncate_text(prompt, max_length=35, strategy="smart", suffix="...")
    assert len(truncated) <= 35
    # Should truncate at sentence boundary
    assert "First sentence." in truncated


def test_truncate_prompt_no_truncation_needed():
    """Test that short prompts are not truncated - now uses preprocessing module."""
    from kerb.preprocessing import truncate_text

    prompt = "Short prompt"
    truncated = truncate_text(prompt, max_length=50, suffix="...")
    assert truncated == prompt


def test_analyze_prompt_basic():
    """Test basic prompt analysis."""
    prompt = "Hello world! This is a test."
    analysis = analyze_prompt(prompt)

    assert analysis["length"] == len(prompt)
    assert analysis["words"] == 6
    assert analysis["lines"] == 1
    assert analysis["sentences"] == 2


def test_analyze_prompt_with_variables():
    """Test analyzing prompt with template variables."""
    prompt = "Hello {{name}}! You are {{age}} years old."
    analysis = analyze_prompt(prompt)

    assert "name" in analysis["variables"]
    assert "age" in analysis["variables"]


def test_analyze_prompt_empty():
    """Test analyzing empty prompt."""
    analysis = analyze_prompt("")
    assert analysis["length"] == 0
    assert analysis["words"] == 0


# ============================================================================
# Integration Tests
# ============================================================================


def test_complete_workflow_template_to_prompt():
    """Test complete workflow from template creation to rendering."""
    # Create template
    template = "Hello {{user.name}}! Your balance is ${{user.balance}}."

    # Validate template
    variables = {"user": {"name": "Alice", "balance": 100}}
    is_valid, missing = validate_template(template, variables)
    assert is_valid is True

    # Render template
    result = render_template(template, variables)
    assert result == "Hello Alice! Your balance is $100."


def test_complete_workflow_versioning_and_rendering():
    """Test complete versioning workflow."""
    # Create version
    v1 = create_version(
        name="welcome",
        version="1.0",
        template="Welcome {{name}}!",
        variables={"name": "Guest"},
    )

    # Register
    register_prompt(v1)

    # Retrieve and render
    prompt = get_prompt("welcome", "1.0")
    assert prompt is not None

    result = prompt.render({"name": "Alice"})
    assert result == "Welcome Alice!"


def test_complete_workflow_few_shot_examples():
    """Test complete few-shot workflow."""
    # Create examples
    examples = [
        create_example("What is 2+2?", "4"),
        create_example("What is 3+3?", "6"),
        create_example("What is 5+5?", "10"),
    ]

    # Select examples
    selected = select_examples(examples, k=2, strategy="first")
    assert len(selected) == 2

    # Format examples
    formatted = format_examples(selected, template="Q: {input}\nA: {output}")
    assert "Q: What is 2+2?" in formatted
    assert "A: 4" in formatted


def test_complete_workflow_prompt_optimization():
    """Test complete prompt optimization workflow."""
    # Create a messy prompt
    prompt = """
    Please    help   me   understand this   concept.
    
    
    
    This is very important for my work.
    """

    # Analyze before optimization
    analysis_before = analyze_prompt(prompt)

    # Compress
    compressed = compress_prompt(prompt)

    # Analyze after optimization
    analysis_after = analyze_prompt(compressed)

    # Should be shorter and cleaner
    assert analysis_after["length"] < analysis_before["length"]
    assert "  " not in compressed
