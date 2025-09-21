#!/usr/bin/env python3
"""
Demo script for the Custom AI System
Shows various capabilities of the AI
"""

from ai import CustomAI
import time

def run_demo():
    print("🤖 Custom AI System Demonstration")
    print("=" * 50)
    print("Building the smartest AI ever - 100% independent from OpenAI!")
    print()
    
    # Initialize AI
    print("🔧 Initializing AI system...")
    ai = CustomAI()
    
    # Enhanced training data
    training_texts = [
        "Hello, how are you today?",
        "What is artificial intelligence?",
        "Python is a programming language used for AI development.",
        "Machine learning helps computers learn from data.",
        "Neural networks are inspired by the human brain.",
        "Deep learning uses multiple layers of neural networks.",
        "Natural language processing enables AI to understand text.",
        "Computer vision allows AI to analyze images.",
        "Reinforcement learning teaches AI through rewards.",
        "I want to learn about artificial intelligence and machine learning.",
        "Can you help me understand neural networks?",
        "This is a custom AI system built from scratch.",
        "Learning from interactions makes AI smarter.",
        "Knowledge representation enables logical reasoning.",
        "Memory systems help AI remember past conversations.",
        "Reasoning engines allow AI to make inferences.",
        "The goal is to create the smartest AI possible.",
        "Python provides excellent libraries for AI development.",
        "Independent AI systems don't rely on external APIs.",
        "Custom implementations give full control over AI behavior."
    ]
    
    print("📚 Training language model with enhanced dataset...")
    ai.train_language_model(training_texts, epochs=75)
    
    print("\n🧠 AI System Features:")
    print("✅ Custom Neural Network (built from scratch)")
    print("✅ Natural Language Processing engine")
    print("✅ Knowledge graph with reasoning")
    print("✅ Memory management system")
    print("✅ Learning and adaptation capabilities")
    print("✅ No external AI API dependencies")
    
    print("\n🎯 Demo Interactions:")
    print("-" * 30)
    
    # Test various capabilities
    test_inputs = [
        "Hello!",
        "What is machine learning?",
        "How do neural networks work?",
        "Can you learn from our conversation?",
        "What makes you intelligent?",
        "Tell me about Python programming"
    ]
    
    for i, test_input in enumerate(test_inputs, 1):
        print(f"\n{i}. User: {test_input}")
        response = ai.process_input(test_input)
        print(f"   AI: {response}")
        
        # Learn from interaction
        ai.learn_from_interaction(test_input)
        time.sleep(0.5)  # Brief pause for readability
    
    print(f"\n📊 System Stats:")
    print(f"   • Knowledge base concepts: {len(ai.knowledge_base.nodes)}")
    print(f"   • Short-term memories: {len(ai.memory.short_term)}")
    print(f"   • Long-term memories: {len(ai.memory.long_term)}")
    print(f"   • Conversation history: {len(ai.conversation_history)}")
    print(f"   • Vocabulary size: {ai.text_processor.vocab_size}")
    
    print(f"\n💾 Saving AI state...")
    ai.save_state('demo_ai_state.pkl')
    
    print("\n🎉 Demo completed! The AI is ready for interactive use.")
    print("   Run 'python3 ai.py' to start chatting with the AI.")

if __name__ == "__main__":
    run_demo()