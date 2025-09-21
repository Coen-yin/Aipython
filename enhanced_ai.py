#!/usr/bin/env python3
"""
Enhanced AI System with Advanced Features
Additional capabilities for making the AI even smarter
"""

from ai import CustomAI, KnowledgeBase, ReasoningEngine
import re
import math
from typing import List, Dict, Any


class EnhancedAI(CustomAI):
    """Enhanced version of the AI with additional smart features"""
    
    def __init__(self):
        super().__init__()
        self.conversation_context = []
        self.user_preferences = {}
        self.learning_rate = 0.1
        self.setup_advanced_reasoning()
        self.load_extended_knowledge()
    
    def setup_advanced_reasoning(self):
        """Setup more sophisticated reasoning rules"""
        
        def math_rule(query, context, knowledge):
            # Handle basic math operations
            math_pattern = r'(\d+)\s*([\+\-\*\/])\s*(\d+)'
            match = re.search(math_pattern, query)
            if match:
                num1, operator, num2 = match.groups()
                num1, num2 = int(num1), int(num2)
                
                if operator == '+':
                    result = num1 + num2
                elif operator == '-':
                    result = num1 - num2
                elif operator == '*':
                    result = num1 * num2
                elif operator == '/':
                    result = num1 / num2 if num2 != 0 else "Cannot divide by zero"
                
                return f"The answer is {result}"
            return None
        
        def definition_rule(query, context, knowledge):
            # Handle "what is" questions
            if re.search(r'what is|define|meaning of', query.lower()):
                # Extract the term being asked about
                patterns = [
                    r'what is (\w+)',
                    r'define (\w+)',
                    r'meaning of (\w+)'
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, query.lower())
                    if match:
                        term = match.group(1)
                        node = self.knowledge_base.query(term)
                        if node:
                            definition = node.properties.get('definition', 'No definition available')
                            return f"{term.capitalize()}: {definition}"
                        else:
                            return f"I don't have a definition for '{term}' yet. Could you teach me about it?"
            return None
        
        def learning_rule(query, context, knowledge):
            # Handle teaching/learning interactions
            teaching_patterns = [
                r'(.+) is (.+)',
                r'(.+) means (.+)',
                r'let me teach you about (.+)'
            ]
            
            for pattern in teaching_patterns:
                match = re.search(pattern, query.lower())
                if match:
                    if len(match.groups()) == 2:
                        subject, description = match.groups()
                        subject = subject.strip()
                        description = description.strip()
                        
                        # Add to knowledge base
                        self.knowledge_base.add_concept(subject, {
                            'definition': description,
                            'learned_from': 'user_teaching',
                            'type': 'user_defined'
                        })
                        
                        return f"Thank you! I've learned that {subject} is {description}. I'll remember this."
            return None
        
        def context_rule(query, context, knowledge):
            # Use conversation context for better responses
            if context and len(context) > 1:
                last_messages = context[-3:]  # Look at last 3 messages
                
                # Check for follow-up questions
                if any(word in query.lower() for word in ['it', 'that', 'this', 'them']):
                    # Find the subject from previous messages
                    for msg in reversed(last_messages[:-1]):
                        if any(word in msg.lower() for word in ['ai', 'python', 'learning', 'neural']):
                            return f"Based on our previous discussion about {msg.lower()}, I can provide more details."
            return None
        
        # Add advanced rules
        self.reasoning_engine.add_rule("math", math_rule)
        self.reasoning_engine.add_rule("definition", definition_rule)
        self.reasoning_engine.add_rule("learning", learning_rule)
        self.reasoning_engine.add_rule("context", context_rule)
    
    def load_extended_knowledge(self):
        """Load extended knowledge base"""
        extended_concepts = {
            'machine': {
                'definition': 'A device that performs work using mechanical power',
                'type': 'object',
                'related': ['computer', 'automation']
            },
            'learning': {
                'definition': 'The process of acquiring knowledge or skills',
                'type': 'process',
                'related': ['education', 'training', 'experience']
            },
            'intelligence': {
                'definition': 'The ability to learn, understand, and apply knowledge',
                'type': 'concept',
                'related': ['thinking', 'reasoning', 'problem-solving']
            },
            'neural': {
                'definition': 'Related to neurons or neural networks',
                'type': 'adjective',
                'related': ['brain', 'network', 'artificial']
            },
            'network': {
                'definition': 'A system of interconnected elements',
                'type': 'concept',
                'related': ['connection', 'system', 'structure']
            },
            'data': {
                'definition': 'Information processed or stored by a computer',
                'type': 'concept',
                'related': ['information', 'processing', 'analysis']
            },
            'algorithm': {
                'definition': 'A set of rules for solving problems',
                'type': 'concept',
                'related': ['logic', 'computation', 'procedure']
            },
            'programming': {
                'definition': 'The process of creating computer programs',
                'type': 'activity',
                'related': ['coding', 'development', 'software']
            }
        }
        
        # Add concepts to knowledge base
        for concept, properties in extended_concepts.items():
            if concept not in self.knowledge_base.nodes:
                self.knowledge_base.add_concept(concept, properties)
                
                # Add relationships
                for related_concept in properties.get('related', []):
                    if related_concept in extended_concepts:
                        self.knowledge_base.add_relationship(concept, 'related_to', related_concept)
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Simple sentiment analysis"""
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like', 'enjoy', 'happy']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'sad', 'angry', 'frustrated', 'disappointed']
        
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        if positive_count > negative_count:
            sentiment = 'positive'
            confidence = positive_count / len(words)
        elif negative_count > positive_count:
            sentiment = 'negative'
            confidence = negative_count / len(words)
        else:
            sentiment = 'neutral'
            confidence = 0.5
        
        return {
            'sentiment': sentiment,
            'confidence': min(confidence, 1.0),
            'positive_score': positive_count,
            'negative_score': negative_count
        }
    
    def process_input(self, user_input: str) -> str:
        """Enhanced input processing with sentiment analysis and context"""
        # Analyze sentiment
        sentiment_analysis = self.analyze_sentiment(user_input)
        
        # Update conversation context
        self.conversation_context.append(user_input)
        if len(self.conversation_context) > 10:  # Keep last 10 messages
            self.conversation_context.pop(0)
        
        # Store sentiment in memory
        self.memory.add_to_short_term({
            'type': 'sentiment_analysis',
            'content': user_input,
            'sentiment': sentiment_analysis,
            'importance': 0.6
        })
        
        # Call parent processing
        response = super().process_input(user_input)
        
        # Enhance response based on sentiment
        if sentiment_analysis['sentiment'] == 'negative' and sentiment_analysis['confidence'] > 0.3:
            response += " I sense you might be frustrated. How can I help you better?"
        elif sentiment_analysis['sentiment'] == 'positive' and sentiment_analysis['confidence'] > 0.3:
            response += " I'm glad you're having a positive experience!"
        
        return response
    
    def get_smart_suggestions(self, user_input: str) -> List[str]:
        """Generate smart suggestions based on input"""
        suggestions = []
        
        # Analyze input for keywords
        words = user_input.lower().split()
        
        # Suggest related topics
        for word in words:
            node = self.knowledge_base.query(word)
            if node and 'related' in node.properties:
                for related in node.properties['related'][:2]:  # Limit to 2 suggestions
                    suggestions.append(f"Would you like to know about {related}?")
        
        # Suggest based on conversation history
        if len(self.conversation_history) > 2:
            recent_topics = []
            for msg in self.conversation_history[-3:]:
                if msg['role'] == 'user':
                    recent_topics.extend(msg['content'].lower().split())
            
            if 'ai' in recent_topics and 'learn' not in recent_topics:
                suggestions.append("Would you like to know how I learn?")
            
            if 'python' in recent_topics and 'programming' not in recent_topics:
                suggestions.append("Are you interested in Python programming concepts?")
        
        return suggestions[:3]  # Return top 3 suggestions
    
    def explain_reasoning(self, query: str) -> Dict[str, Any]:
        """Explain the AI's reasoning process"""
        reasoning_result = self.reasoning_engine.reason(query, self.conversation_context)
        
        explanation = {
            'query': query,
            'concepts_identified': reasoning_result.get('concepts', []),
            'knowledge_used': reasoning_result.get('relevant_knowledge', []),
            'rules_applied': len(reasoning_result.get('conclusions', [])),
            'confidence': reasoning_result.get('confidence', 0),
            'reasoning_steps': [
                "1. Analyzed input for key concepts",
                "2. Retrieved relevant knowledge",
                "3. Applied inference rules",
                "4. Generated response based on findings"
            ]
        }
        
        return explanation


def interactive_enhanced_ai():
    """Run the enhanced AI in interactive mode"""
    print("🚀 Enhanced Custom AI System")
    print("=" * 40)
    print("Features: Math solving, Definition learning, Sentiment analysis, Smart suggestions")
    print("Commands: 'quit' to exit, 'explain' to see reasoning, 'suggest' for suggestions")
    print()
    
    ai = EnhancedAI()
    
    # Quick training
    training_data = [
        "Mathematics involves numbers and calculations",
        "Python is used for artificial intelligence development",
        "Learning happens through practice and repetition",
        "Neural networks mimic brain structures",
        "Data analysis reveals patterns and insights"
    ]
    
    ai.train_language_model(training_data, epochs=30)
    
    print("Enhanced AI is ready! Try these examples:")
    print("- 'What is machine learning?'")
    print("- '10 + 5' (math calculation)")
    print("- 'Neural networks are computational models' (teach the AI)")
    print("-" * 40)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("AI: Thank you for using the Enhanced AI system!")
                break
            
            if user_input.lower() == 'explain':
                print("AI: Please ask me something and I'll explain my reasoning.")
                continue
            
            if user_input.lower() == 'suggest':
                suggestions = ai.get_smart_suggestions(ai.conversation_context[-1] if ai.conversation_context else "")
                if suggestions:
                    print("AI: Here are some suggestions:")
                    for i, suggestion in enumerate(suggestions, 1):
                        print(f"   {i}. {suggestion}")
                else:
                    print("AI: I don't have specific suggestions right now. Ask me anything!")
                continue
            
            if not user_input:
                continue
            
            # Get response
            response = ai.process_input(user_input)
            print(f"AI: {response}")
            
            # Show reasoning if requested
            if 'explain' in user_input.lower():
                explanation = ai.explain_reasoning(user_input)
                print(f"\n🧠 Reasoning Process:")
                print(f"   Concepts: {', '.join(explanation['concepts_identified'])}")
                print(f"   Confidence: {explanation['confidence']:.2f}")
                print(f"   Rules applied: {explanation['rules_applied']}")
            
        except KeyboardInterrupt:
            print("\n\nAI: Goodbye! Thank you for testing the Enhanced AI.")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue


if __name__ == "__main__":
    interactive_enhanced_ai()