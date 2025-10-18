"""Multi-Session Memory Management Example

This example demonstrates how to manage memory across multiple conversation sessions,
maintain user context over time, and handle session transitions in LLM applications.

Main concepts:
- Managing multiple concurrent sessions
- Session persistence and restoration
- Cross-session context sharing
- User profile building from sessions
- Session lifecycle management
"""

from pathlib import Path
import json
from datetime import datetime, timedelta
from kerb.memory import ConversationBuffer
from kerb.memory.utils import merge_conversations
from kerb.core.types import Message


class SessionManager:
    """Manages multiple conversation sessions."""
    
    def __init__(self, storage_dir: str = "sessions"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.active_sessions = {}
    
    def create_session(self, user_id: str, session_id: str = None) -> ConversationBuffer:
        """Create a new session for a user."""
        if session_id is None:
            session_id = f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        buffer = ConversationBuffer(
            max_messages=100,
            enable_summaries=True,
            enable_entity_tracking=True
        )
        buffer.metadata["user_id"] = user_id
        buffer.metadata["session_id"] = session_id
        buffer.metadata["created_at"] = datetime.now().isoformat()
        
        self.active_sessions[session_id] = buffer
        return buffer
    
    def get_session(self, session_id: str) -> ConversationBuffer:
        """Get an active or stored session."""
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]
        
        # Try to load from disk
        session_path = self.storage_dir / f"{session_id}.json"
        if session_path.exists():
            buffer = ConversationBuffer.load(str(session_path))
            self.active_sessions[session_id] = buffer
            return buffer
        
        return None
    
    def save_session(self, session_id: str):
        """Save session to disk."""
        if session_id in self.active_sessions:
            buffer = self.active_sessions[session_id]
            session_path = self.storage_dir / f"{session_id}.json"
            buffer.save(str(session_path))
    
    def get_user_sessions(self, user_id: str):
        """Get all sessions for a user."""
        sessions = []
        for session_file in self.storage_dir.glob(f"{user_id}_*.json"):
            sessions.append(session_file.stem)
        return sessions


def main():
    """Run multi-session memory management example."""
    
    print("="*80)
    print("MULTI-SESSION MEMORY MANAGEMENT EXAMPLE")
    print("="*80)
    
    # Initialize session manager
    session_mgr = SessionManager(storage_dir="temp_sessions")
    
    # Scenario 1: Single user, multiple sessions over time
    print("\n" + "-"*80)
    print("SCENARIO 1: MULTI-DAY USER SESSIONS")
    print("-"*80)
    
    user_id = "user_alice"
    
    # Day 1: Morning session
    print("\nDay 1 - Morning Session:")
    session1 = session_mgr.create_session(user_id, f"{user_id}_day1_morning")
    session1.add_message("user", "I'm starting a new Django project for e-commerce.")
    session1.add_message("assistant", "Great! Let's start with setting up Django and creating the basic structure.")
    session1.add_message("user", "What models should I create for products?")
    session1.add_message("assistant", "Create Product, Category, and Inventory models. Add fields for name, price, description, and stock.")
    
    print(f"  Messages: {len(session1.messages)}")
    session_mgr.save_session(session1.metadata["session_id"])
    print(f"  Saved: {session1.metadata['session_id']}")
    
    # Day 1: Evening session
    print("\nDay 1 - Evening Session:")
    session2 = session_mgr.create_session(user_id, f"{user_id}_day1_evening")
    session2.add_message("user", "I've created the models. Now I need to add user authentication.")
    session2.add_message("assistant", "Use Django's built-in authentication. Extend the User model with a Profile model for additional user data.")
    session2.add_message("user", "How do I implement shopping cart functionality?")
    session2.add_message("assistant", "Create Cart and CartItem models linked to User. Use sessions for anonymous users.")
    
    print(f"  Messages: {len(session2.messages)}")
    session_mgr.save_session(session2.metadata["session_id"])
    print(f"  Saved: {session2.metadata['session_id']}")
    
    # Day 2: New session with context from previous days
    print("\nDay 2 - Morning Session (with historical context):")
    session3 = session_mgr.create_session(user_id, f"{user_id}_day2_morning")
    
    # Load previous sessions to build context
    prev_sessions = session_mgr.get_user_sessions(user_id)
    print(f"  Found {len(prev_sessions)} previous sessions")
    
    # Build summary from previous sessions
    previous_context = []
    for session_id in prev_sessions[:2]:  # Last 2 sessions
        prev_buffer = session_mgr.get_session(session_id)
        if prev_buffer and prev_buffer.summaries:
            previous_context.append(prev_buffer.summaries[0].summary)
    
    if previous_context:
        context_summary = " ".join(previous_context)
        print(f"  Previous context: {context_summary[:100]}...")
    
    session3.add_message("user", "I'm back! Let's continue with the e-commerce project.")
    session3.add_message("assistant", f"Welcome back! Based on our previous sessions, you've set up Django models for products and user authentication. What would you like to work on today?")
    session3.add_message("user", "I want to add payment processing with Stripe.")
    session3.add_message("assistant", "Great choice! Install stripe package, create a Payment model, and implement webhook handlers for payment confirmation.")
    
    print(f"  Messages: {len(session3.messages)}")
    session_mgr.save_session(session3.metadata["session_id"])
    
    # Scenario 2: Concurrent sessions (same user, different topics)
    print("\n" + "-"*80)
    print("SCENARIO 2: CONCURRENT SESSIONS")
    print("-"*80)
    
    # Session A: Work project
    work_session = session_mgr.create_session(user_id, f"{user_id}_work")
    work_session.metadata["topic"] = "work"
    work_session.add_message("user", "Help me optimize our data pipeline at work.")
    work_session.add_message("assistant", "I'll help with your data pipeline. What technology stack are you using?")
    
    # Session B: Personal learning
    learning_session = session_mgr.create_session(user_id, f"{user_id}_learning")
    learning_session.metadata["topic"] = "learning"
    learning_session.add_message("user", "I want to learn about neural networks for personal projects.")
    learning_session.add_message("assistant", "Great! Let's start with the basics of neural networks and backpropagation.")
    
    print(f"\nWork session: {len(work_session.messages)} messages")
    print(f"Learning session: {len(learning_session.messages)} messages")
    
    # Continue work session
    work_session.add_message("user", "We use Apache Airflow and Spark.")
    work_session.add_message("assistant", "For Airflow optimization, ensure proper task dependencies and use sensor operators wisely.")
    
    print(f"\nUpdated work session: {len(work_session.messages)} messages")
    
    # Scenario 3: Cross-session knowledge transfer
    print("\n" + "-"*80)
    print("SCENARIO 3: CROSS-SESSION KNOWLEDGE")
    print("-"*80)
    
    # Build user knowledge graph from all sessions
    all_user_sessions = [session1, session2, session3, work_session, learning_session]
    
    # Extract entities across all sessions
    all_entities = {}
    for session in all_user_sessions:
        for entity_key, entity in session.entities.items():
            if entity_key in all_entities:
                all_entities[entity_key].mentions += entity.mentions
            else:
                all_entities[entity_key] = entity
    
    print(f"\nUser knowledge graph:")
    print(f"  Total sessions: {len(all_user_sessions)}")
    print(f"  Total entities tracked: {len(all_entities)}")
    
    # Show top entities
    top_entities = sorted(all_entities.values(), key=lambda e: e.mentions, reverse=True)[:5]
    print(f"\n  Top mentioned entities:")
    for entity in top_entities:
        print(f"    - {entity.name} ({entity.type}): {entity.mentions} mentions")
    
    # Scenario 4: Session merging for comprehensive history
    print("\n" + "-"*80)
    print("SCENARIO 4: SESSION MERGING")
    print("-"*80)
    
    # Merge all e-commerce related sessions
    ecommerce_sessions = [session1, session2, session3]
    merged_ecommerce = merge_conversations(*ecommerce_sessions, sort_by_time=True)
    
    print(f"\nMerged e-commerce sessions:")
    print(f"  Total messages: {len(merged_ecommerce.messages)}")
    print(f"  Total entities: {len(merged_ecommerce.entities)}")
    print(f"  Total summaries: {len(merged_ecommerce.summaries)}")
    
    # Create comprehensive summary
    if merged_ecommerce.messages:
        from kerb.memory import summarize_conversation
        comprehensive_summary = summarize_conversation(merged_ecommerce.messages)
        print(f"\n  Comprehensive summary:")
        print(f"    {comprehensive_summary.summary[:150]}...")
        print(f"    Key points: {len(comprehensive_summary.key_points)}")
    
    # Scenario 5: Session lifecycle management
    print("\n" + "-"*80)
    print("SCENARIO 5: SESSION LIFECYCLE")
    print("-"*80)
    
    class SessionLifecycle:
        """Manage session lifecycle with automatic cleanup."""
        
        @staticmethod
        def is_session_expired(session: ConversationBuffer, days: int = 30) -> bool:
            """Check if session is older than specified days."""
            created_at = datetime.fromisoformat(session.metadata.get("created_at"))
            age = datetime.now() - created_at
            return age.days > days
        
        @staticmethod
        def archive_old_sessions(session_mgr: SessionManager, user_id: str, days: int = 30):
            """Archive sessions older than specified days."""
            archived = []
            sessions = session_mgr.get_user_sessions(user_id)
            
            for session_id in sessions:
                session = session_mgr.get_session(session_id)
                if SessionLifecycle.is_session_expired(session, days):
                    # Create archive
                    archive_dir = session_mgr.storage_dir / "archive"
                    archive_dir.mkdir(exist_ok=True)
                    
                    # Move to archive
                    current_path = session_mgr.storage_dir / f"{session_id}.json"
                    archive_path = archive_dir / f"{session_id}.json"
                    
                    if current_path.exists():
                        current_path.rename(archive_path)
                        archived.append(session_id)
            
            return archived
    
    # Demo lifecycle management
    print("\nSession lifecycle management:")
    print(f"  Active sessions for {user_id}: {len(session_mgr.get_user_sessions(user_id))}")
    
    # Simulate archiving (with 0 days for demo)
    archived = SessionLifecycle.archive_old_sessions(session_mgr, user_id, days=0)
    print(f"  Archived {len(archived)} old sessions")
    
    # Scenario 6: User profile from sessions
    print("\n" + "-"*80)
    print("SCENARIO 6: USER PROFILE BUILDING")
    print("-"*80)
    
    def build_user_profile(sessions):
        """Build user profile from session history."""
        profile = {
            "interests": set(),
            "skills": set(),
            "projects": set(),
            "total_interactions": 0,
            "preferred_topics": {}
        }
        
        keywords = {
            "interests": ["learn", "interested", "want to", "curious"],
            "skills": ["Django", "Python", "Stripe", "Airflow", "Spark", "neural network"],
            "projects": ["project", "building", "developing", "implementing"]
        }
        
        for session in sessions:
            profile["total_interactions"] += len(session.messages)
            
            # Extract interests and skills
            for msg in session.messages:
                if msg.role == "user":
                    content_lower = msg.content.lower()
                    
                    for interest_kw in keywords["interests"]:
                        if interest_kw in content_lower:
                            # Extract following words
                            words = msg.content.split()
                            for i, word in enumerate(words):
                                if interest_kw in word.lower() and i + 1 < len(words):
                                    profile["interests"].add(words[i + 1])
                    
                    for skill in keywords["skills"]:
                        if skill.lower() in content_lower:
                            profile["skills"].add(skill)
                    
                    for proj_kw in keywords["projects"]:
                        if proj_kw in content_lower:
                            if "e-commerce" in content_lower or "django" in content_lower:
                                profile["projects"].add("Django E-commerce")
                            if "pipeline" in content_lower:
                                profile["projects"].add("Data Pipeline")
        
        return profile
    
    user_profile = build_user_profile(all_user_sessions)
    
    print(f"\nUser Profile for {user_id}:")
    print(f"  Total interactions: {user_profile['total_interactions']}")
    print(f"  Skills: {', '.join(list(user_profile['skills'])[:5])}")
    print(f"  Projects: {', '.join(user_profile['projects'])}")
    print(f"  Interests: {', '.join(list(user_profile['interests'])[:5])}")
    
    # Save user profile
    profile_path = session_mgr.storage_dir / f"{user_id}_profile.json"
    profile_path.write_text(json.dumps({
        "total_interactions": user_profile["total_interactions"],
        "skills": list(user_profile["skills"]),
        "projects": list(user_profile["projects"]),
        "interests": list(user_profile["interests"])
    }, indent=2))
    print(f"\n  Saved profile to: {profile_path}")
    
    # Cleanup
    print("\n" + "-"*80)
    print("CLEANUP")
    print("-"*80)
    
    import shutil
    if session_mgr.storage_dir.exists():
        shutil.rmtree(session_mgr.storage_dir)
        print(f"Cleaned up: {session_mgr.storage_dir}")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
