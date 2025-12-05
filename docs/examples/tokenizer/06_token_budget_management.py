"""
Token Budget Management Example
===============================

This example demonstrates how to manage token budgets in LLM applications,
ensuring that API calls stay within limits, tracking usage, and implementing
budget enforcement strategies.

Main concepts:
- Setting and enforcing token budgets
- Tracking cumulative token usage
- Implementing budget alerts
- Managing per-user or per-session limits
"""

from kerb.tokenizer import (
    count_tokens,
    count_tokens_for_messages,
    truncate_to_token_limit,
    Tokenizer
)
from typing import List, Dict, Optional
from datetime import datetime


class TokenBudgetManager:
    """Manages token budgets for LLM applications."""
    
    def __init__(self, budget_limit: int):
        """Initialize budget manager.

# %%
# Setup and Imports
# -----------------
        
        Args:
            budget_limit: Maximum tokens allowed
        """
        self.budget_limit = budget_limit
        self.tokens_used = 0
        self.requests = []
        
    def check_budget(self, tokens_needed: int) -> bool:
        """Check if tokens are available within budget.
        
        Args:
            tokens_needed: Number of tokens needed
            
        Returns:
            True if tokens available, False otherwise
        """
        return self.tokens_used + tokens_needed <= self.budget_limit
    
    def use_tokens(self, tokens: int, metadata: Optional[Dict] = None) -> bool:
        """Use tokens from budget.
        
        Args:
            tokens: Number of tokens to use
            metadata: Optional metadata about the request
            
        Returns:
            True if tokens were used, False if budget exceeded
        """
        if not self.check_budget(tokens):
            return False
        
        self.tokens_used += tokens
        self.requests.append({
            "timestamp": datetime.now(),
            "tokens": tokens,
            "metadata": metadata or {}
        })
        return True
    
    def get_remaining_budget(self) -> int:
        """Get remaining token budget."""
        return self.budget_limit - self.tokens_used
    

# %%
# Get Usage Stats
# ---------------

    def get_usage_stats(self) -> Dict:
        """Get usage statistics."""
        return {
            "total_budget": self.budget_limit,
            "tokens_used": self.tokens_used,
            "tokens_remaining": self.get_remaining_budget(),
            "usage_percent": (self.tokens_used / self.budget_limit * 100) if self.budget_limit > 0 else 0,
            "request_count": len(self.requests)
        }
    
    def reset(self):
        """Reset budget tracking."""
        self.tokens_used = 0
        self.requests = []



# %%
# Main
# ----

def main():
    """Run token budget management examples."""
    
    print("="*80)
    print("TOKEN BUDGET MANAGEMENT EXAMPLE")
    print("="*80)
    
    # Example 1: Basic budget management
    print("\n" + "-"*80)
    print("EXAMPLE 1: Basic Budget Management")
    print("-"*80)
    
    budget = TokenBudgetManager(budget_limit=1000)
    
    print(f"Budget limit: {budget.budget_limit} tokens")
    print(f"Initial usage: {budget.tokens_used} tokens\n")
    
    # Simulate API requests
    requests = [
        "Explain machine learning",
        "What is neural network?",
        "How does gradient descent work?",
        "Describe backpropagation",
    ]
    
    for i, request_text in enumerate(requests, 1):
        tokens_needed = count_tokens(request_text, tokenizer=Tokenizer.CL100K_BASE)
        # Assume response is 5x the request size
        tokens_with_response = tokens_needed * 6
        
        print(f"Request {i}: {request_text}")
        print(f"  Tokens needed: {tokens_with_response} (input: {tokens_needed}, output: ~{tokens_needed * 5})")
        
        if budget.check_budget(tokens_with_response):
            success = budget.use_tokens(tokens_with_response, metadata={"request": request_text})
            if success:
                stats = budget.get_usage_stats()
                print(f"  Status: APPROVED")
                print(f"  Budget used: {stats['usage_percent']:.1f}%")
                print(f"  Remaining: {stats['tokens_remaining']} tokens")
        else:
            print(f"  Status: REJECTED - Exceeds budget")
            print(f"  Remaining budget: {budget.get_remaining_budget()} tokens")
            print(f"  Shortfall: {tokens_with_response - budget.get_remaining_budget()} tokens")
        print()
    
    # Final stats
    final_stats = budget.get_usage_stats()
    print(f"Final statistics:")
    print(f"  Total requests: {final_stats['request_count']}")
    print(f"  Tokens used: {final_stats['tokens_used']}/{final_stats['total_budget']}")
    print(f"  Usage: {final_stats['usage_percent']:.1f}%")
    
    # Example 2: Per-user budget tracking
    print("\n" + "-"*80)
    print("EXAMPLE 2: Per-User Budget Tracking")
    print("-"*80)
    
    # Simulate multi-user system
    user_budgets = {
        "user_free": TokenBudgetManager(budget_limit=10000),
        "user_pro": TokenBudgetManager(budget_limit=100000),
        "user_enterprise": TokenBudgetManager(budget_limit=1000000),
    }
    
    tier_names = {
        "user_free": "Free Tier",
        "user_pro": "Pro Tier",
        "user_enterprise": "Enterprise Tier",
    }
    
    print("User budget limits:")
    for user_id, budget_mgr in user_budgets.items():
        print(f"  {tier_names[user_id]}: {budget_mgr.budget_limit:,} tokens/month")
    print()
    
    # Simulate usage
    usage_scenarios = [
        ("user_free", "Simple question", 500),
        ("user_pro", "Code generation task", 5000),
        ("user_enterprise", "Document analysis", 50000),
        ("user_free", "Another question", 600),
        ("user_free", "Large request", 12000),  # Should exceed budget
    ]
    
    for user_id, task, tokens in usage_scenarios:
        budget_mgr = user_budgets[user_id]
        tier = tier_names[user_id]
        
        print(f"{tier} ({user_id}):")
        print(f"  Task: {task}")
        print(f"  Tokens needed: {tokens:,}")
        
        if budget_mgr.use_tokens(tokens, metadata={"task": task}):
            stats = budget_mgr.get_usage_stats()
            print(f"  Status: APPROVED")
            print(f"  Usage: {stats['tokens_used']:,}/{stats['total_budget']:,} ({stats['usage_percent']:.1f}%)")
        else:
            stats = budget_mgr.get_usage_stats()
            print(f"  Status: REJECTED - Budget exceeded")
            print(f"  Current usage: {stats['tokens_used']:,}/{stats['total_budget']:,}")
            print(f"  Upgrade needed for this request")
        print()
    
    # Example 3: Budget alerts and warnings
    print("\n" + "-"*80)
    print("EXAMPLE 3: Budget Alerts and Warnings")
    print("-"*80)
    
    def check_budget_alerts(budget_mgr: TokenBudgetManager) -> List[str]:
        """Check for budget alerts."""
        stats = budget_mgr.get_usage_stats()
        alerts = []
        
        if stats['usage_percent'] >= 90:
            alerts.append("CRITICAL: 90% budget used")
        elif stats['usage_percent'] >= 75:
            alerts.append("WARNING: 75% budget used")
        elif stats['usage_percent'] >= 50:
            alerts.append("INFO: 50% budget used")
        
        return alerts
    
    budget = TokenBudgetManager(budget_limit=10000)
    
    print(f"Budget: {budget.budget_limit:,} tokens")
    print(f"Alert thresholds: 50%, 75%, 90%\n")
    
    # Simulate gradual usage
    usage_steps = [2000, 3000, 2000, 2000, 1500]
    
    for i, tokens in enumerate(usage_steps, 1):
        budget.use_tokens(tokens, metadata={"step": i})
        stats = budget.get_usage_stats()
        alerts = check_budget_alerts(budget)
        
        print(f"Step {i}: Used {tokens:,} tokens")
        print(f"  Total: {stats['tokens_used']:,}/{stats['total_budget']:,} ({stats['usage_percent']:.1f}%)")
        
        if alerts:
            for alert in alerts:
                print(f"  ALERT: {alert}")
        print()
    
    # Example 4: Dynamic budget adjustment
    print("\n" + "-"*80)
    print("EXAMPLE 4: Dynamic Budget Adjustment")
    print("-"*80)
    

# %%
# Adjust Request To Budget
# ------------------------

    def adjust_request_to_budget(
        text: str,
        available_tokens: int,
        input_output_ratio: float = 0.2
    ) -> tuple:
        """Adjust request to fit within available budget.
        
        Args:
            text: Request text
            available_tokens: Available token budget
            input_output_ratio: Ratio of input to total tokens (e.g., 0.2 means 20% input, 80% output)
            
        Returns:
            Tuple of (adjusted_text, estimated_total_tokens)
        """
        # Calculate max input tokens
        max_input_tokens = int(available_tokens * input_output_ratio)
        
        # Truncate if necessary
        actual_tokens = count_tokens(text, tokenizer=Tokenizer.CL100K_BASE)
        
        if actual_tokens > max_input_tokens:
            adjusted_text = truncate_to_token_limit(
                text,
                max_tokens=max_input_tokens,
                tokenizer=Tokenizer.CL100K_BASE
            )
            estimated_total = available_tokens
        else:
            adjusted_text = text
            estimated_total = int(actual_tokens / input_output_ratio)
        
        return adjusted_text, estimated_total
    
    budget = TokenBudgetManager(budget_limit=500)
    
    long_request = (
        "Please provide a comprehensive analysis of the following topic, "
        "including historical context, current state, future trends, "
        "practical applications, and detailed examples. " * 10
    )
    
    print(f"Budget: {budget.budget_limit} tokens")
    print(f"Original request: {len(long_request)} characters")
    print(f"Original tokens: {count_tokens(long_request, tokenizer=Tokenizer.CL100K_BASE)}")
    
    available = budget.get_remaining_budget()
    adjusted_text, estimated_tokens = adjust_request_to_budget(long_request, available)
    
    print(f"\nAdjusted request:")
    print(f"  Text: {adjusted_text[:100]}...")
    print(f"  Estimated tokens: {estimated_tokens}")
    print(f"  Fits in budget: {estimated_tokens <= available}")
    
    # Example 5: Session-based budget management
    print("\n" + "-"*80)
    print("EXAMPLE 5: Session-Based Budget Management")
    print("-"*80)
    
    class SessionBudget:
        """Manage budget for a conversation session."""
        
        def __init__(self, session_id: str, budget_per_session: int):
            self.session_id = session_id
            self.budget = TokenBudgetManager(budget_per_session)
            self.messages = []
        
        def add_message(self, role: str, content: str) -> bool:
            """Add message to session if budget allows."""
            self.messages.append({"role": role, "content": content})
            
            # Count tokens for entire conversation
            total_tokens = count_tokens_for_messages(
                self.messages,
                tokenizer=Tokenizer.CL100K_BASE
            )
            
            # Check if within budget
            if total_tokens <= self.budget.budget_limit:
                self.budget.tokens_used = total_tokens
                return True
            else:
                # Remove the message we just added
                self.messages.pop()
                return False
        

# %%
# Get Stats
# ---------

        def get_stats(self) -> Dict:
            """Get session statistics."""
            stats = self.budget.get_usage_stats()
            stats['message_count'] = len(self.messages)
            return stats
    
    # Create sessions
    session = SessionBudget("session_123", budget_per_session=300)
    
    print(f"Session: {session.session_id}")
    print(f"Budget: {session.budget.budget_limit} tokens\n")
    
    conversation_turns = [
        ("system", "You are a helpful assistant."),
        ("user", "What is Python?"),
        ("assistant", "Python is a high-level programming language known for simplicity."),
        ("user", "What are its main uses?"),
        ("assistant", "Python is used for web development, data science, automation, and AI."),
        ("user", "How do I get started?"),
        ("assistant", "Start with python.org, install Python, and try basic tutorials."),
    ]
    
    for role, content in conversation_turns:
        success = session.add_message(role, content)
        stats = session.get_stats()
        
        print(f"Add {role} message:")
        print(f"  Content: {content[:50]}...")
        print(f"  Success: {success}")
        print(f"  Session tokens: {stats['tokens_used']}/{stats['total_budget']}")
        print(f"  Messages: {stats['message_count']}")
        
        if not success:
            print(f"  REJECTED: Would exceed session budget")
            print(f"  Consider starting new session or trimming history")
        print()
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
