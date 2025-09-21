#!/usr/bin/env python3
"""
Custom AI System - A fully independent AI implementation
Built from scratch without external AI APIs like OpenAI
"""

import numpy as np
import json
import re
import math
import random
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import pickle
import os


class NeuralNetwork:
    """Custom neural network implementation from scratch"""
    
    def __init__(self, layers: List[int], learning_rate: float = 0.01):
        self.layers = layers
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases
        for i in range(len(layers) - 1):
            # Xavier initialization
            w = np.random.randn(layers[i], layers[i + 1]) * np.sqrt(2.0 / layers[i])
            b = np.zeros((1, layers[i + 1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function"""
        return x * (1 - x)
    
    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU function"""
        return (x > 0).astype(float)
    
    def softmax(self, x):
        """Softmax activation function"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        """Forward propagation"""
        self.activations = [X]
        self.z_values = []
        
        for i in range(len(self.weights)):
            z = np.dot(self.activations[i], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            
            if i == len(self.weights) - 1:  # Output layer
                activation = self.softmax(z)
            else:  # Hidden layers
                activation = self.relu(z)
            
            self.activations.append(activation)
        
        return self.activations[-1]
    
    def backward(self, X, y, output):
        """Backward propagation"""
        m = X.shape[0]
        
        # Calculate gradients
        dw = []
        db = []
        
        # Output layer
        dz = output - y
        dw_curr = np.dot(self.activations[-2].T, dz) / m
        db_curr = np.sum(dz, axis=0, keepdims=True) / m
        dw.insert(0, dw_curr)
        db.insert(0, db_curr)
        
        # Hidden layers
        for i in range(len(self.weights) - 2, -1, -1):
            dz = np.dot(dz, self.weights[i + 1].T) * self.relu_derivative(self.activations[i + 1])
            dw_curr = np.dot(self.activations[i].T, dz) / m
            db_curr = np.sum(dz, axis=0, keepdims=True) / m
            dw.insert(0, dw_curr)
            db.insert(0, db_curr)
        
        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dw[i]
            self.biases[i] -= self.learning_rate * db[i]
    
    def train(self, X, y, epochs: int = 1000, verbose: bool = True):
        """Train the neural network"""
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            
            if verbose and epoch % 100 == 0:
                loss = self.calculate_loss(y, output)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    def calculate_loss(self, y_true, y_pred):
        """Calculate cross-entropy loss"""
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))
    
    def predict(self, X):
        """Make predictions"""
        return self.forward(X)


class TextProcessor:
    """Natural Language Processing engine"""
    
    def __init__(self):
        self.vocabulary = {}
        self.word_to_index = {}
        self.index_to_word = {}
        self.vocab_size = 0
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        # Simple tokenization - can be enhanced
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text.split()
    
    def build_vocabulary(self, texts: List[str]):
        """Build vocabulary from text corpus"""
        word_counts = {}
        
        for text in texts:
            tokens = self.tokenize(text)
            for token in tokens:
                word_counts[token] = word_counts.get(token, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Add special tokens
        self.word_to_index['<UNK>'] = 0
        self.word_to_index['<PAD>'] = 1
        self.word_to_index['<START>'] = 2
        self.word_to_index['<END>'] = 3
        
        index = 4
        for word, count in sorted_words:
            if count >= 2:  # Only include words that appear at least twice
                self.word_to_index[word] = index
                index += 1
        
        self.index_to_word = {v: k for k, v in self.word_to_index.items()}
        self.vocab_size = len(self.word_to_index)
    
    def text_to_sequence(self, text: str) -> List[int]:
        """Convert text to sequence of indices"""
        tokens = self.tokenize(text)
        return [self.word_to_index.get(token, 0) for token in tokens]  # 0 is <UNK>
    
    def sequence_to_text(self, sequence: List[int]) -> str:
        """Convert sequence of indices to text"""
        words = [self.index_to_word.get(idx, '<UNK>') for idx in sequence]
        return ' '.join(words)


@dataclass
class KnowledgeNode:
    """Node in the knowledge graph"""
    concept: str
    properties: Dict[str, Any]
    connections: List[str]
    confidence: float = 1.0


class KnowledgeBase:
    """Graph-based knowledge representation system"""
    
    def __init__(self):
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.relationships: Dict[str, List[Tuple[str, str, float]]] = {}
    
    def add_concept(self, concept: str, properties: Dict[str, Any] = None):
        """Add a new concept to the knowledge base"""
        if properties is None:
            properties = {}
        
        self.nodes[concept] = KnowledgeNode(
            concept=concept,
            properties=properties,
            connections=[]
        )
    
    def add_relationship(self, subject: str, predicate: str, object_: str, strength: float = 1.0):
        """Add a relationship between concepts"""
        if predicate not in self.relationships:
            self.relationships[predicate] = []
        
        self.relationships[predicate].append((subject, object_, strength))
        
        # Update connections
        if subject in self.nodes:
            self.nodes[subject].connections.append(object_)
        if object_ in self.nodes:
            self.nodes[object_].connections.append(subject)
    
    def query(self, concept: str) -> Optional[KnowledgeNode]:
        """Query the knowledge base for a concept"""
        return self.nodes.get(concept)
    
    def find_related(self, concept: str, max_depth: int = 2) -> List[str]:
        """Find related concepts within max_depth"""
        if concept not in self.nodes:
            return []
        
        visited = set()
        queue = [(concept, 0)]
        related = []
        
        while queue:
            current_concept, depth = queue.pop(0)
            
            if current_concept in visited or depth > max_depth:
                continue
            
            visited.add(current_concept)
            if depth > 0:
                related.append(current_concept)
            
            # Add connected concepts
            if current_concept in self.nodes:
                for connected in self.nodes[current_concept].connections:
                    if connected not in visited:
                        queue.append((connected, depth + 1))
        
        return related


class Memory:
    """Memory management system"""
    
    def __init__(self, max_short_term: int = 100, max_long_term: int = 1000):
        self.short_term: List[Dict[str, Any]] = []
        self.long_term: List[Dict[str, Any]] = []
        self.max_short_term = max_short_term
        self.max_long_term = max_long_term
    
    def add_to_short_term(self, memory: Dict[str, Any]):
        """Add memory to short-term storage"""
        memory['timestamp'] = datetime.now()
        self.short_term.append(memory)
        
        # Maintain size limit
        if len(self.short_term) > self.max_short_term:
            # Move oldest to long-term if important
            oldest = self.short_term.pop(0)
            if oldest.get('importance', 0) > 0.5:
                self.add_to_long_term(oldest)
    
    def add_to_long_term(self, memory: Dict[str, Any]):
        """Add memory to long-term storage"""
        if 'timestamp' not in memory:
            memory['timestamp'] = datetime.now()
        
        self.long_term.append(memory)
        
        # Maintain size limit
        if len(self.long_term) > self.max_long_term:
            # Remove least important memories
            self.long_term.sort(key=lambda x: x.get('importance', 0))
            self.long_term = self.long_term[self.max_long_term//10:]
    
    def recall(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Recall memories related to query"""
        relevant_memories = []
        
        # Search both short-term and long-term
        all_memories = self.short_term + self.long_term
        
        for memory in all_memories:
            relevance = self._calculate_relevance(query, memory)
            if relevance > 0.1:  # Threshold for relevance
                memory_copy = memory.copy()
                memory_copy['relevance'] = relevance
                relevant_memories.append(memory_copy)
        
        # Sort by relevance and return top results
        relevant_memories.sort(key=lambda x: x['relevance'], reverse=True)
        return relevant_memories[:max_results]
    
    def _calculate_relevance(self, query: str, memory: Dict[str, Any]) -> float:
        """Calculate relevance score between query and memory"""
        query_words = set(query.lower().split())
        memory_text = str(memory.get('content', '')).lower()
        memory_words = set(memory_text.split())
        
        # Simple Jaccard similarity
        intersection = len(query_words.intersection(memory_words))
        union = len(query_words.union(memory_words))
        
        if union == 0:
            return 0.0
        
        return intersection / union


class ReasoningEngine:
    """Logic-based reasoning and inference system"""
    
    def __init__(self, knowledge_base: KnowledgeBase):
        self.knowledge_base = knowledge_base
        self.inference_rules = []
    
    def add_rule(self, rule: str, action: callable):
        """Add an inference rule"""
        self.inference_rules.append((rule, action))
    
    def reason(self, query: str, context: List[str] = None) -> Dict[str, Any]:
        """Perform reasoning on a query"""
        if context is None:
            context = []
        
        # Extract key concepts from query
        concepts = self._extract_concepts(query)
        
        # Gather relevant knowledge
        relevant_knowledge = []
        for concept in concepts:
            related = self.knowledge_base.find_related(concept)
            relevant_knowledge.extend(related)
        
        # Apply inference rules
        conclusions = []
        for rule, action in self.inference_rules:
            if self._matches_rule(query, rule):
                conclusion = action(query, context, relevant_knowledge)
                if conclusion:
                    conclusions.append(conclusion)
        
        return {
            'concepts': concepts,
            'relevant_knowledge': relevant_knowledge,
            'conclusions': conclusions,
            'confidence': self._calculate_confidence(conclusions)
        }
    
    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text"""
        # Simple keyword extraction - can be enhanced
        words = text.lower().split()
        concepts = []
        
        for word in words:
            if word in self.knowledge_base.nodes:
                concepts.append(word)
        
        return concepts
    
    def _matches_rule(self, query: str, rule: str) -> bool:
        """Check if query matches a rule pattern"""
        # Simple pattern matching - can be enhanced with regex
        return rule.lower() in query.lower()
    
    def _calculate_confidence(self, conclusions: List[Any]) -> float:
        """Calculate confidence in the reasoning result"""
        if not conclusions:
            return 0.0
        
        # Simple confidence based on number of conclusions
        return min(1.0, len(conclusions) * 0.3)


class CustomAI:
    """Main AI system that integrates all components"""
    
    def __init__(self):
        self.text_processor = TextProcessor()
        self.knowledge_base = KnowledgeBase()
        self.memory = Memory()
        self.reasoning_engine = ReasoningEngine(self.knowledge_base)
        self.neural_network = None
        self.conversation_history = []
        
        # Initialize with basic knowledge
        self._initialize_basic_knowledge()
        self._setup_inference_rules()
    
    def _initialize_basic_knowledge(self):
        """Initialize the AI with basic knowledge"""
        # Add basic concepts
        self.knowledge_base.add_concept("AI", {
            "definition": "Artificial Intelligence",
            "type": "technology"
        })
        
        self.knowledge_base.add_concept("python", {
            "definition": "Programming language",
            "type": "language"
        })
        
        self.knowledge_base.add_concept("learning", {
            "definition": "Process of acquiring knowledge",
            "type": "process"
        })
        
        # Add relationships
        self.knowledge_base.add_relationship("AI", "implemented_in", "python")
        self.knowledge_base.add_relationship("AI", "capable_of", "learning")
    
    def _setup_inference_rules(self):
        """Setup basic inference rules"""
        def greeting_rule(query, context, knowledge):
            greetings = ["hello", "hi", "hey", "greetings"]
            if any(greeting in query.lower() for greeting in greetings):
                return "Hello! I'm a custom AI system built from scratch in Python. How can I help you today?"
            return None
        
        def question_rule(query, context, knowledge):
            question_words = ["what", "how", "why", "when", "where", "who"]
            if any(word in query.lower() for word in question_words):
                return f"I'm analyzing your question about: {', '.join(knowledge)}"
            return None
        
        self.reasoning_engine.add_rule("greeting", greeting_rule)
        self.reasoning_engine.add_rule("question", question_rule)
    
    def train_language_model(self, texts: List[str], epochs: int = 100):
        """Train the neural network for language understanding"""
        print("Building vocabulary...")
        self.text_processor.build_vocabulary(texts)
        
        print(f"Vocabulary size: {self.text_processor.vocab_size}")
        
        # Prepare training data
        X = []
        y = []
        
        for text in texts:
            sequence = self.text_processor.text_to_sequence(text)
            for i in range(len(sequence) - 1):
                # Use sliding window approach
                context = sequence[max(0, i-3):i+1]  # Context window of 4
                target = sequence[i + 1]
                
                # Pad context to fixed size
                while len(context) < 4:
                    context.insert(0, 1)  # <PAD> token
                
                X.append(context)
                # One-hot encode target
                target_one_hot = [0] * self.text_processor.vocab_size
                if target < self.text_processor.vocab_size:
                    target_one_hot[target] = 1
                y.append(target_one_hot)
        
        if X and y:
            X = np.array(X)
            y = np.array(y)
            
            # Create neural network
            input_size = X.shape[1]
            hidden_size = min(128, self.text_processor.vocab_size // 2)
            output_size = self.text_processor.vocab_size
            
            self.neural_network = NeuralNetwork([input_size, hidden_size, hidden_size, output_size])
            
            print("Training neural network...")
            self.neural_network.train(X, y, epochs=epochs)
            print("Training completed!")
    
    def process_input(self, user_input: str) -> str:
        """Process user input and generate response"""
        # Add to conversation history
        self.conversation_history.append({
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now()
        })
        
        # Store in memory
        self.memory.add_to_short_term({
            'type': 'user_input',
            'content': user_input,
            'importance': 0.7
        })
        
        # Perform reasoning
        reasoning_result = self.reasoning_engine.reason(
            user_input, 
            context=[msg['content'] for msg in self.conversation_history[-5:]]
        )
        
        # Generate response
        response = self._generate_response(user_input, reasoning_result)
        
        # Add response to conversation history
        self.conversation_history.append({
            'role': 'assistant',
            'content': response,
            'timestamp': datetime.now()
        })
        
        # Store response in memory
        self.memory.add_to_short_term({
            'type': 'ai_response',
            'content': response,
            'importance': 0.8
        })
        
        return response
    
    def _generate_response(self, user_input: str, reasoning_result: Dict[str, Any]) -> str:
        """Generate response based on reasoning results"""
        conclusions = reasoning_result.get('conclusions', [])
        
        # If we have conclusions from reasoning rules
        if conclusions:
            return conclusions[0]  # Return first conclusion
        
        # Fallback responses based on patterns
        if any(word in user_input.lower() for word in ['hello', 'hi', 'hey']):
            return "Hello! I'm a custom AI system. How can I assist you?"
        
        if any(word in user_input.lower() for word in ['what', 'how', 'why']):
            concepts = reasoning_result.get('concepts', [])
            if concepts:
                return f"I understand you're asking about {', '.join(concepts)}. Let me analyze this based on my knowledge."
            else:
                return "That's an interesting question. Let me think about it."
        
        if 'learn' in user_input.lower():
            return "I'm always learning! You can teach me by providing information, and I'll store it in my knowledge base."
        
        if 'smart' in user_input.lower() or 'intelligent' in user_input.lower():
            return "I aim to be as intelligent as possible by combining neural networks, reasoning, and knowledge representation!"
        
        # Default response
        return "I'm processing your input and learning from it. Could you provide more context or ask me something specific?"
    
    def learn_from_interaction(self, user_input: str, expected_response: str = None):
        """Learn from user interactions"""
        # Extract concepts and add to knowledge base
        concepts = user_input.lower().split()
        for concept in concepts:
            if len(concept) > 2 and concept.isalpha():
                if concept not in self.knowledge_base.nodes:
                    self.knowledge_base.add_concept(concept, {
                        'learned_from': 'user_interaction',
                        'frequency': 1
                    })
                else:
                    # Update frequency
                    node = self.knowledge_base.nodes[concept]
                    node.properties['frequency'] = node.properties.get('frequency', 0) + 1
        
        # Store interaction in long-term memory
        self.memory.add_to_long_term({
            'type': 'learning_interaction',
            'input': user_input,
            'expected_response': expected_response,
            'importance': 0.9
        })
    
    def save_state(self, filepath: str):
        """Save AI state to file"""
        state = {
            'knowledge_base': {
                'nodes': {k: {
                    'concept': v.concept,
                    'properties': v.properties,
                    'connections': v.connections,
                    'confidence': v.confidence
                } for k, v in self.knowledge_base.nodes.items()},
                'relationships': self.knowledge_base.relationships
            },
            'memory': {
                'short_term': self.memory.short_term,
                'long_term': self.memory.long_term
            },
            'vocabulary': self.text_processor.word_to_index,
            'conversation_history': self.conversation_history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"AI state saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load AI state from file"""
        if not os.path.exists(filepath):
            print(f"State file {filepath} not found.")
            return
        
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        # Restore knowledge base
        self.knowledge_base.nodes = {}
        for k, v in state['knowledge_base']['nodes'].items():
            self.knowledge_base.nodes[k] = KnowledgeNode(
                concept=v['concept'],
                properties=v['properties'],
                connections=v['connections'],
                confidence=v['confidence']
            )
        
        self.knowledge_base.relationships = state['knowledge_base']['relationships']
        
        # Restore memory
        self.memory.short_term = state['memory']['short_term']
        self.memory.long_term = state['memory']['long_term']
        
        # Restore vocabulary
        self.text_processor.word_to_index = state['vocabulary']
        self.text_processor.index_to_word = {v: k for k, v in state['vocabulary'].items()}
        self.text_processor.vocab_size = len(state['vocabulary'])
        
        # Restore conversation history
        self.conversation_history = state['conversation_history']
        
        print(f"AI state loaded from {filepath}")


def main():
    """Main function to run the AI system"""
    print("=" * 60)
    print("Custom AI System - Built from Scratch in Python")
    print("No external AI APIs - Fully independent implementation")
    print("=" * 60)
    
    # Initialize AI
    ai = CustomAI()
    
    # Training data for language model
    training_texts = [
        "Hello, how are you today?",
        "What is artificial intelligence?",
        "Python is a programming language.",
        "Machine learning helps computers learn.",
        "Neural networks process information.",
        "I want to learn about AI.",
        "Can you help me understand?",
        "This is a custom AI system.",
        "Learning from interactions is important.",
        "Knowledge representation enables reasoning."
    ]
    
    print("Training language model...")
    ai.train_language_model(training_texts, epochs=50)
    
    print("\nAI System is ready! Type 'quit' to exit, 'save' to save state, 'load' to load state.")
    print("Start chatting with the AI:")
    print("-" * 40)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("AI: Goodbye! Thank you for chatting with me.")
                break
            
            if user_input.lower() == 'save':
                ai.save_state('ai_state.pkl')
                continue
            
            if user_input.lower() == 'load':
                ai.load_state('ai_state.pkl')
                continue
            
            if not user_input:
                continue
            
            # Process input and get response
            response = ai.process_input(user_input)
            print(f"AI: {response}")
            
            # Learn from interaction
            ai.learn_from_interaction(user_input)
            
        except KeyboardInterrupt:
            print("\n\nAI: Goodbye! Thank you for chatting with me.")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue


if __name__ == "__main__":
    main()