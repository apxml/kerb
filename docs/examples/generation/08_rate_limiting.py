"""Rate Limiting Example

This example demonstrates API rate limiting to stay within provider quotas.

Main concepts:
- Using RateLimiter to control request frequency
- Preventing API quota exhaustion
- Managing token-per-minute limits
- Implementing custom rate limiting strategies
- Monitoring request rates
"""

import time
from kerb.generation import generate, generate_batch, ModelName
from kerb.generation.utils import RateLimiter


def example_basic_rate_limiting():
    """Demonstrate basic rate limiting."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Rate Limiting")
    print("="*80)
    
    # Create rate limiter: max 10 requests per minute (faster for demo)
    rate_limiter = RateLimiter(requests_per_minute=10)
    
    prompts = [
        "What is Python?",
        "What is JavaScript?",
        "What is Go?",
    ]
    
    print(f"\nRate limit: 10 requests per minute")
    print(f"Sending {len(prompts)} requests...\n")
    
    start_time = time.time()
    
    for i, prompt in enumerate(prompts, 1):
        request_start = time.time()
        
        # Wait if needed to respect rate limit
        rate_limiter.wait_if_needed()
        
        try:
            response = generate(
                prompt,
                model=ModelName.GPT_4O_MINI,
                max_tokens=20
            )
            
            elapsed = time.time() - request_start
            print(f"[{i}/{len(prompts)}] {prompt}")
            print(f"         Wait time: {elapsed:.2f}s - {response.content[:40]}...")
        except Exception as e:
            print(f"[{i}] Error: {e}")
    
    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.2f}s")
    print(f"Average: {total_time/len(prompts):.2f}s per request")


def example_token_rate_limiting():
    """Demonstrate rate limiting based on token usage."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Token-Based Rate Limiting")
    print("="*80)
    
    # Rate limiter with token limit
    rate_limiter = RateLimiter(
        requests_per_minute=20,
        tokens_per_minute=1000
    )
    
    prompts = [
        "Define API",
        "Define REST API",
        "Define GraphQL API",
        "Define WebSocket",
    ]
    
    print(f"\nRate limits:")
    print(f"  Requests: 20 per minute")
    print(f"  Tokens: 1000 per minute\n")
    
    for i, prompt in enumerate(prompts, 1):
        # Estimate tokens (rough approximation: ~4 chars per token)
        estimated_tokens = len(prompt) // 4 + 100  # Prompt + expected response
        
        print(f"[{i}] {prompt} (est. ~{estimated_tokens} tokens)")
        
        rate_limiter.wait_if_needed(estimated_tokens)
        
        try:
            response = generate(
                prompt,
                model=ModelName.GPT_4O_MINI,
                max_tokens=50
            )
            actual_tokens = response.usage.total_tokens
            print(f"    Actual tokens: {actual_tokens}")
        except Exception as e:
            print(f"    Error: {e}")


def example_rate_limit_monitoring():
    """Monitor rate limit usage over time."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Rate Limit Monitoring")
    print("="*80)
    
    rate_limiter = RateLimiter(requests_per_minute=20)
    
    print(f"\nRate limit: 20 requests per minute")
    print("\nSending bursts of requests...\n")
    
    # Send requests in bursts (reduced for speed)
    for burst in range(2):
        print(f"Burst {burst + 1}:")
        burst_start = time.time()
        
        for i in range(2):
            rate_limiter.wait_if_needed()
            
            try:
                response = generate(
                    "Hi",
                    model=ModelName.GPT_4O_MINI,
                    max_tokens=5
                )
                print(f"  Request {i+1}: OK")
            except Exception as e:
                print(f"  Request {i+1}: Error")
        
        burst_time = time.time() - burst_start
        print(f"  Burst completed in {burst_time:.2f}s\n")
        
        # Wait between bursts
        if burst < 1:
            time.sleep(1)


def example_custom_rate_strategy():
    """Implement custom rate limiting strategy."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Custom Rate Strategy")
    print("="*80)
    
    class AdaptiveRateLimiter:
        """Rate limiter that adapts based on errors."""
        
        def __init__(self, initial_rpm: int = 10):
            self.requests_per_minute = initial_rpm
            self.rate_limiter = RateLimiter(requests_per_minute=initial_rpm)
            self.error_count = 0
            self.success_count = 0
        
        def wait_and_request(self, prompt: str):
            """Make request with adaptive rate limiting."""
            self.rate_limiter.wait_if_needed()
            
            try:
                response = generate(
                    prompt,
                    model=ModelName.GPT_4O_MINI,
                    max_tokens=20
                )
                self.success_count += 1
                
                # Increase rate after successes
                if self.success_count % 5 == 0:
                    self.requests_per_minute = min(
                        self.requests_per_minute + 1,
                        20
                    )
                    self.rate_limiter = RateLimiter(
                        requests_per_minute=self.requests_per_minute
                    )
                
                return response
            
            except Exception as e:
                self.error_count += 1
                
                # Decrease rate on errors
                if "rate" in str(e).lower():
                    self.requests_per_minute = max(
                        self.requests_per_minute - 2,
                        3
                    )
                    self.rate_limiter = RateLimiter(
                        requests_per_minute=self.requests_per_minute
                    )
                
                raise
    
    limiter = AdaptiveRateLimiter(initial_rpm=10)
    
    print(f"\nInitial rate: {limiter.requests_per_minute} req/min")
    print("\nSending requests...\n")
    
    prompts = ["Test prompt"] * 5  # Reduced from 8
    
    for i, prompt in enumerate(prompts, 1):
        try:
            response = limiter.wait_and_request(prompt)
            print(f"[{i}] Success (rate: {limiter.requests_per_minute} req/min)")
        except Exception as e:
            print(f"[{i}] Error (rate: {limiter.requests_per_minute} req/min)")
    
    print(f"\nFinal rate: {limiter.requests_per_minute} req/min")
    print(f"Success: {limiter.success_count}, Errors: {limiter.error_count}")


def example_burst_protection():
    """Prevent request bursts that exceed limits."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Burst Protection")
    print("="*80)
    
    rate_limiter = RateLimiter(requests_per_minute=12)
    
    print(f"\nRate limit: 12 requests per minute")
    print("Attempting to send 6 requests rapidly...\n")
    
    start_time = time.time()
    
    for i in range(6):  # Reduced from 10
        request_start = time.time()
        rate_limiter.wait_if_needed()
        request_time = time.time() - request_start
        
        if request_time > 0.1:
            print(f"Request {i+1}: Rate limited (waited {request_time:.2f}s)")
        else:
            print(f"Request {i+1}: Sent immediately")
        
        # Simulate actual request (without calling API for demo)
        time.sleep(0.05)
    
    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.2f}s")
    print(f"Rate limiting prevented burst congestion")


def example_concurrent_rate_limiting():
    """Rate limiting with concurrent requests."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Rate Limiting with Batch Requests")
    print("="*80)
    
    rate_limiter = RateLimiter(requests_per_minute=20)
    
    prompts = [f"Question {i}" for i in range(4)]  # Reduced from 8
    
    print(f"\nRate limit: 20 requests per minute")
    print(f"Processing {len(prompts)} prompts in batch...\n")
    
    start_time = time.time()
    
    # In a real scenario, you'd integrate rate limiter with batch_generate
    # For this demo, we'll show the concept
    
    responses = []
    for i, prompt in enumerate(prompts, 1):
        rate_limiter.wait_if_needed()
        
        try:
            response = generate(
                "Respond with OK",
                model=ModelName.GPT_4O_MINI,
                max_tokens=3
            )
            responses.append(response)
            print(f"[{i}/{len(prompts)}] Processed")
        except Exception as e:
            print(f"[{i}/{len(prompts)}] Error: {e}")
    
    elapsed = time.time() - start_time
    
    print(f"\nCompleted {len(responses)} requests in {elapsed:.2f}s")
    print(f"Average rate: {len(responses) / (elapsed / 60):.1f} req/min")


def main():
    """Run all rate limiting examples."""
    print("\n" + "#"*80)
    print("# RATE LIMITING EXAMPLES")
    print("#"*80)
    
    try:
        example_basic_rate_limiting()
    except Exception as e:
        print(f"\nExample 1 Error: {e}")
    
    try:
        example_token_rate_limiting()
    except Exception as e:
        print(f"\nExample 2 Error: {e}")
    
    try:
        example_rate_limit_monitoring()
    except Exception as e:
        print(f"\nExample 3 Error: {e}")
    
    try:
        example_custom_rate_strategy()
    except Exception as e:
        print(f"\nExample 4 Error: {e}")
    
    try:
        example_burst_protection()
    except Exception as e:
        print(f"\nExample 5 Error: {e}")
    
    try:
        example_concurrent_rate_limiting()
    except Exception as e:
        print(f"\nExample 6 Error: {e}")
    
    print("\n" + "#"*80)
    print("# Examples completed")
    print("#"*80 + "\n")


if __name__ == "__main__":
    main()
