"""Chat Message Token Counting Example

This example demonstrates how to count tokens in chat-formatted messages,
which is critical for LLM applications using conversational models like
GPT-4, GPT-3.5-turbo, and other chat-based APIs.

Main concepts:
- Counting tokens in chat message format
- Understanding message formatting overhead
- Managing conversation history within token limits
- Optimizing multi-turn conversations
"""

from kerb.tokenizer import (
    count_tokens,
    count_tokens_for_messages,
    Tokenizer
)
from typing import List, Dict


def main():
    """Run chat message token counting examples."""
    
    print("="*80)
    print("CHAT MESSAGE TOKEN COUNTING EXAMPLE")
    print("="*80)
    
    # Example 1: Basic message token counting
    print("\n" + "-"*80)
    print("EXAMPLE 1: Basic Message Token Counting")
    print("-"*80)
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! How are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you! How can I help you today?"},
    ]
    
    print("Messages:")
    for msg in messages:
        print(f"  {msg['role']}: {msg['content']}")
    
    # Count tokens with message formatting
    total_tokens = count_tokens_for_messages(messages, tokenizer=Tokenizer.CL100K_BASE)
    
    # Compare to raw content tokens
    raw_content = " ".join(msg["content"] for msg in messages)
    raw_tokens = count_tokens(raw_content, tokenizer=Tokenizer.CL100K_BASE)
    
    print(f"\nToken counts:")
    print(f"  With message formatting: {total_tokens} tokens")
    print(f"  Raw content only: {raw_tokens} tokens")
    print(f"  Message overhead: {total_tokens - raw_tokens} tokens")
    print(f"  Overhead percentage: {(total_tokens - raw_tokens) / total_tokens * 100:.1f}%")
    
    # Example 2: Multi-turn conversation
    print("\n" + "-"*80)
    print("EXAMPLE 2: Multi-turn Conversation Token Tracking")
    print("-"*80)
    
    conversation = [
        {"role": "system", "content": "You are a Python programming tutor."},
    ]
    
    # Simulate a conversation
    turns = [
        ("user", "What is a list in Python?"),
        ("assistant", "A list is an ordered, mutable collection of items in Python. You create one using square brackets: my_list = [1, 2, 3]"),
        ("user", "How do I add items to a list?"),
        ("assistant", "You can use the append() method: my_list.append(4). Or use insert() to add at a specific position."),
        ("user", "What's the difference between append and extend?"),
        ("assistant", "append() adds a single element, while extend() adds all elements from another iterable. Example: list.append([1,2]) adds the list as one element, but list.extend([1,2]) adds 1 and 2 separately."),
    ]
    
    print("Conversation progression:\n")
    
    for role, content in turns:
        conversation.append({"role": role, "content": content})
        token_count = count_tokens_for_messages(conversation, tokenizer=Tokenizer.CL100K_BASE)
        print(f"After {len(conversation)-1} exchange(s): {token_count} tokens")
        print(f"  {role}: {content[:60]}...")
    
    print(f"\nFinal conversation: {len(conversation)} messages, {token_count} tokens")
    
    # Example 3: Managing conversation history
    print("\n" + "-"*80)
    print("EXAMPLE 3: Managing Conversation History Within Token Limits")
    print("-"*80)
    
    max_tokens = 200
    print(f"Token limit: {max_tokens}\n")
    
    long_conversation = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Tell me about machine learning."},
        {"role": "assistant", "content": "Machine learning is a branch of AI that enables computers to learn from data."},
        {"role": "user", "content": "What are the types?"},
        {"role": "assistant", "content": "The main types are supervised, unsupervised, and reinforcement learning."},
        {"role": "user", "content": "Explain supervised learning."},
        {"role": "assistant", "content": "Supervised learning uses labeled data to train models to make predictions."},
        {"role": "user", "content": "Give me an example."},
        {"role": "assistant", "content": "Email spam detection is a classic example of supervised learning."},
    ]
    
    current_tokens = count_tokens_for_messages(long_conversation, tokenizer=Tokenizer.CL100K_BASE)
    print(f"Full conversation: {len(long_conversation)} messages, {current_tokens} tokens")
    
    if current_tokens > max_tokens:
        print(f"Conversation exceeds limit by {current_tokens - max_tokens} tokens")
        print("\nTrimming strategy: Keep system message + most recent messages\n")
        
        # Keep system message and trim from the oldest user/assistant pairs
        system_msg = long_conversation[0]
        recent_messages = long_conversation[1:]
        
        trimmed_conversation = [system_msg]
        
        # Add messages from most recent, working backwards
        for i in range(len(recent_messages) - 1, -1, -1):
            test_conversation = [system_msg] + recent_messages[i:]
            test_tokens = count_tokens_for_messages(test_conversation, tokenizer=Tokenizer.CL100K_BASE)
            
            if test_tokens <= max_tokens:
                trimmed_conversation = test_conversation
            else:
                break
        
        trimmed_tokens = count_tokens_for_messages(trimmed_conversation, tokenizer=Tokenizer.CL100K_BASE)
        messages_removed = len(long_conversation) - len(trimmed_conversation)
        
        print(f"Trimmed conversation:")
        print(f"  Messages kept: {len(trimmed_conversation)}/{len(long_conversation)}")
        print(f"  Messages removed: {messages_removed}")
        print(f"  Tokens: {trimmed_tokens}/{max_tokens}")
        print(f"\nKept messages:")
        for msg in trimmed_conversation:
            preview = msg['content'][:50] + "..." if len(msg['content']) > 50 else msg['content']
            print(f"  {msg['role']}: {preview}")
    
    # Example 4: Token overhead per message type
    print("\n" + "-"*80)
    print("EXAMPLE 4: Message Type Token Overhead Analysis")
    print("-"*80)
    
    content = "This is test content."
    content_tokens = count_tokens(content, tokenizer=Tokenizer.CL100K_BASE)
    
    print(f"Content: '{content}'")
    print(f"Raw content tokens: {content_tokens}\n")
    
    message_types = [
        [{"role": "system", "content": content}],
        [{"role": "user", "content": content}],
        [{"role": "assistant", "content": content}],
    ]
    
    print("Message type overhead:")
    for msg_list in message_types:
        total = count_tokens_for_messages(msg_list, tokenizer=Tokenizer.CL100K_BASE)
        overhead = total - content_tokens
        print(f"  {msg_list[0]['role']}: {total} tokens (overhead: {overhead})")
    
    # Example 5: Real-world chatbot conversation management
    print("\n" + "-"*80)
    print("EXAMPLE 5: Chatbot Conversation Management")
    print("-"*80)
    
    # Simulate a chatbot managing context window
    context_limit = 4096
    max_completion_tokens = 500
    max_input_tokens = context_limit - max_completion_tokens
    
    print(f"Model: GPT-3.5-turbo")
    print(f"Context window: {context_limit} tokens")
    print(f"Reserved for completion: {max_completion_tokens} tokens")
    print(f"Available for input: {max_input_tokens} tokens\n")
    
    # Build a conversation
    chatbot_conversation = [
        {"role": "system", "content": "You are a customer support assistant for a tech company. Be helpful and professional."},
    ]
    
    # Simulate user queries
    user_queries = [
        "I can't log into my account.",
        "I've tried resetting my password but didn't receive the email.",
        "The email address on file is john.doe@example.com",
        "Yes, I checked spam folder too.",
    ]
    
    assistant_responses = [
        "I'm sorry to hear you're having trouble logging in. Have you tried resetting your password?",
        "Let me check on that for you. What email address is associated with your account?",
        "Thank you. I can see your account. Let me resend the password reset email. Please check your inbox in a few minutes.",
        "I've just sent a new reset email. If you don't receive it in 5 minutes, please let me know and we'll try an alternative method.",
    ]
    
    print("Conversation simulation:\n")
    
    for i, (user_msg, assistant_msg) in enumerate(zip(user_queries, assistant_responses), 1):
        # Add user message
        chatbot_conversation.append({"role": "user", "content": user_msg})
        tokens_after_user = count_tokens_for_messages(chatbot_conversation, tokenizer=Tokenizer.CL100K_BASE)
        
        # Add assistant response
        chatbot_conversation.append({"role": "assistant", "content": assistant_msg})
        tokens_after_assistant = count_tokens_for_messages(chatbot_conversation, tokenizer=Tokenizer.CL100K_BASE)
        
        print(f"Turn {i}:")
        print(f"  User: {user_msg[:60]}...")
        print(f"  After user message: {tokens_after_user} tokens")
        print(f"  Assistant: {assistant_msg[:60]}...")
        print(f"  After assistant response: {tokens_after_assistant} tokens")
        print(f"  Remaining capacity: {max_input_tokens - tokens_after_assistant} tokens")
        print()
    
    final_tokens = count_tokens_for_messages(chatbot_conversation, tokenizer=Tokenizer.CL100K_BASE)
    usage_percent = (final_tokens / max_input_tokens) * 100
    
    print(f"Final conversation state:")
    print(f"  Total messages: {len(chatbot_conversation)}")
    print(f"  Total tokens: {final_tokens}")
    print(f"  Input capacity used: {usage_percent:.1f}%")
    print(f"  Tokens remaining: {max_input_tokens - final_tokens}")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
