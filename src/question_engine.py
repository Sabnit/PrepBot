"""
Question Engine Module
Orchestrates the complete question-answer workflow with adaptive learning.
"""

from typing import List, Dict, Optional, Tuple
import random

from document_processor import DocumentProcessor
from vector_store import get_vector_store
from session_manager import get_session_manager
from chains import (
    create_topic_extraction_chain,
    create_question_generation_chain,
    create_answer_evaluation_chain,
    create_explanation_chain,
    GeneratedQuestion,
    AnswerEvaluation
)

class QuestionEngine:
    """
    Orchestrates the question-answer workflow with adaptive learning.
    
    Features:
    - Processes and stores documents
    - Extracts topics automatically
    - Generates questions adaptively
    - Evaluates answers with detailed feedback
    - Tracks progress and identifies weak areas
    """
    
    def __init__(self, session_name: str, collection_name: Optional[str] = None):
        """
        Initialize the question engine for a study session.
        
        Args:
            session_name: Human-readable session name
            collection_name: Vector store collection name (auto-generated if None)
        """
        self.session_name = session_name
        self.collection_name = collection_name or self._generate_collection_name(session_name)
        
        # Initialize components
        self.session_manager = get_session_manager()
        self.vector_store = get_vector_store(self.collection_name)
        self.document_processor = DocumentProcessor()
        
        # Initialize chains
        self.topic_chain = create_topic_extraction_chain()
        self.question_chain = create_question_generation_chain(self.vector_store)
        self.evaluation_chain = create_answer_evaluation_chain()
        self.explanation_chain = create_explanation_chain(self.vector_store)
        
        # Get or create session
        self.session = self.session_manager.get_study_session_by_collection(
            self.collection_name
        )
        
        if not self.session:
            self.session = self.session_manager.create_study_session(
                name=session_name,
                collection_name=self.collection_name
            )
            print(f"✓ Created new session: {session_name}")
        else:
            print(f"✓ Loaded existing session: {session_name}")
            self.session_manager.update_session_activity(self.session.id)
    
    def _generate_collection_name(self, session_name: str) -> str:
        """Generate a valid collection name from session name."""
        # Replace spaces and special chars with underscores
        name = "".join(c if c.isalnum() else "_" for c in session_name.lower())
        return name[:50]  # Limit length
    
    # ==================== Document Management ====================
    
    def add_document_from_file(self, file_path: str) -> Dict:
        """
        Add a document from file (PDF or text).
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary with document info and extracted topics
        """
        print(f"\n📄 Processing document: {file_path}")
        
        # Process document
        result = self.document_processor.process_document(file_path=file_path)
        
        # Add to vector store
        print("🔄 Creating embeddings and storing in vector database...")
        self.vector_store.add_documents(result['chunks'])
        
        # Save document metadata to session
        doc_record = self.session_manager.add_document(
            session_id=self.session.id,
            doc_id=result['doc_id'],
            source_name=result['metadata']['source_name'],
            total_chunks=result['metadata']['total_chunks'],
            metadata=result['metadata']
        )
        
        # Extract topics
        print("🧠 Extracting topics for question generation...")
        combined_text = "\n".join([chunk.page_content for chunk in result['chunks'][:5]])
        topics = self.topic_chain.extract_topics(combined_text)
        
        # Save topics to database - with validation
        if topics and isinstance(topics, list):
            self.session_manager.add_topics(doc_record.id, topics)
        
        print(f"✓ Document processed: {len(result['chunks'])} chunks, {len(topics) if topics else 0} topics")
        
        return {
            'doc_id': result['doc_id'],
            'source_name': result['metadata']['source_name'],
            'chunks': len(result['chunks']),
            'topics': topics if topics else []
        }
    
    def add_document_from_text(self, text: str, source_name: str = "Direct Input") -> Dict:
        """
        Add a document from direct text input.
        
        Args:
            text: The text content
            source_name: Name to identify this content
            
        Returns:
            Dictionary with document info and extracted topics
        """
        print(f"\n📝 Processing text input: {source_name}")
        
        # Process text
        result = self.document_processor.process_document(
            text=text,
            source_name=source_name
        )
        
        # Add to vector store
        print("🔄 Creating embeddings and storing in vector database...")
        self.vector_store.add_documents(result['chunks'])
        
        # Save document metadata
        doc_record = self.session_manager.add_document(
            session_id=self.session.id,
            doc_id=result['doc_id'],
            source_name=source_name,
            total_chunks=result['metadata']['total_chunks'],
            metadata=result['metadata']
        )
        
        # Extract topics
        print("🧠 Extracting topics for question generation...")
        topics = self.topic_chain.extract_topics(text)
        
        # Save topics to database - doc_record.id is the database ID
        if topics and isinstance(topics, list):
            self.session_manager.add_topics(doc_record.id, topics)
        
        print(f"✓ Text processed: {len(result['chunks'])} chunks, {len(topics) if topics else 0} topics")
        
        return {
            'doc_id': result['doc_id'],
            'source_name': source_name,
            'chunks': len(result['chunks']),
            'topics': topics if topics else []
        }
    
    # ==================== Question Generation ====================
    
    def generate_next_question(
        self,
        difficulty: Optional[str] = None,
        specific_topic: Optional[str] = None
    ) -> Optional[GeneratedQuestion]:
        """
        Generate the next question adaptively.
        
        Args:
            difficulty: Optional difficulty level (easy/medium/hard)
            specific_topic: Optional specific topic to ask about
            
        Returns:
            GeneratedQuestion or None if generation fails
        """
        # Select topic adaptively
        if specific_topic:
            topic = specific_topic
        else:
            topic = self._select_adaptive_topic()
        
        # Select difficulty adaptively if not specified
        if not difficulty:
            difficulty = self._select_adaptive_difficulty()
        
        print(f"\n🎯 Generating {difficulty} question on: {topic}")
        
        # Generate question
        question = self.question_chain.generate_question(
            topic=topic,
            difficulty=difficulty
        )
        
        if question:
            # Save to database
            question_record = self.session_manager.add_question(
                session_id=self.session.id,
                question_text=question.question,
                topic=question.topic,
                difficulty=question.difficulty,
                context=question.expected_concepts
            )
            
            # Store question ID for later use
            question.id = question_record.id
            
            print("✓ Question generated successfully")
        
        return question
    
    def _select_adaptive_topic(self) -> str:
        """
        Select topic adaptively based on coverage and performance.
        
        Strategy:
        1. Prioritize least covered topics (70% weight)
        2. Include weak topics (30% weight)
        3. Random selection if no data
        """
        # Get least covered topics
        least_covered = self.session_manager.get_least_covered_topics(
            self.session.id,
            limit=5
        )
        
        # Get weak topics (low performance)
        weak_topics = self.session_manager.get_weak_topics(
            self.session.id,
            min_questions=2
        )
        weak_topic_names = [t[0] for t in weak_topics[:3]]
        
        # Combine with weighting
        candidates = []
        
        # 70% chance: least covered topics
        if least_covered and random.random() < 0.7:
            candidates = least_covered
        
        # 30% chance: weak topics
        elif weak_topic_names:
            candidates = weak_topic_names
        
        # Fallback: all topics
        if not candidates:
            all_topics = self.session_manager.get_session_topics(self.session.id)
            candidates = [t[0] for t in all_topics]
        
        # Random selection from candidates
        if candidates:
            return random.choice(candidates)
        else:
            return "General concepts from the material"
    
    def _select_adaptive_difficulty(self) -> str:
        """
        Select difficulty based on recent performance.
        
        Strategy:
        - < 50% accuracy: easy questions
        - 50-80% accuracy: medium questions
        - > 80% accuracy: hard questions
        """
        stats = self.session_manager.get_session_stats(self.session.id)
        
        if stats.get('answered_questions', 0) < 3:
            # Start with medium for first few questions
            return "medium"
        
        accuracy = stats.get('accuracy', 0)
        
        if accuracy < 50:
            return "easy"
        elif accuracy < 80:
            return "medium"
        else:
            return "hard"
    
    # ==================== Answer Evaluation ====================
    
    def evaluate_answer(
        self,
        question: GeneratedQuestion,
        user_answer: str,
        provide_explanation: bool = True
    ) -> Tuple[AnswerEvaluation, Optional[str]]:
        """
        Evaluate a user's answer and optionally provide explanation.
        
        Args:
            question: The GeneratedQuestion object
            user_answer: User's answer text
            provide_explanation: Whether to generate explanation for incorrect answers
            
        Returns:
            Tuple of (AnswerEvaluation, explanation or None)
        """
        print("\n📊 Evaluating answer...")
        
        # Get context for evaluation
        context_docs = self.vector_store.similarity_search(question.topic, k=2)
        context = "\n".join([doc.page_content for doc in context_docs])
        
        # Evaluate answer
        evaluation = self.evaluation_chain.evaluate_answer(
            question=question.question,
            student_answer=user_answer,
            expected_concepts=question.expected_concepts,
            context=context
        )
        
        explanation = None
        
        if evaluation:
            # Save evaluation to database
            self.session_manager.update_question_answer(
                question_id=question.id,
                user_answer=user_answer,
                is_correct=evaluation.is_correct,
                score=evaluation.score,
                evaluation_feedback=evaluation.feedback,
                correct_answer=None 
            )
            
            print(f"✓ Evaluation complete: Score {evaluation.score*100:.1f}%")
            
            # Generate explanation if answer is incorrect
            if provide_explanation and not evaluation.is_correct:
                print("💡 Generating explanation...")
                explanation = self.explanation_chain.generate_explanation(
                    question=question.question,
                    student_answer=user_answer,
                    evaluation_feedback=evaluation.feedback,
                    topic=question.topic
                )
        
        return evaluation, explanation
    
    # ==================== Progress & Analytics ====================
    
    def get_session_stats(self) -> Dict:
        """Get comprehensive session statistics."""
        return self.session_manager.get_session_stats(self.session.id)
    
    def get_weak_topics(self) -> List[Tuple[str, float]]:
        """Get topics where user is struggling."""
        return self.session_manager.get_weak_topics(self.session.id)
    
    def get_question_history(self) -> List:
        """Get all questions asked in this session."""
        return self.session_manager.get_session_questions(self.session.id)
    
    def get_topic_coverage(self) -> List[Tuple[str, int]]:
        """Get topic coverage statistics."""
        return self.session_manager.get_session_topics(self.session.id)
    
    # ==================== Practice Session Management ====================
    
    def start_practice_session(
        self,
        num_questions: int = 10,
        difficulty: Optional[str] = None
    ) -> Dict:
        """
        Start an interactive practice session.
        
        Args:
            num_questions: Number of questions to ask
            difficulty: Optional fixed difficulty, or None for adaptive
            
        Returns:
            Dictionary with session summary
        """
        print(f"\n🎓 Starting practice session: {num_questions} questions")
        print("=" * 60)
        
        questions_asked = []
        
        for i in range(num_questions):
            print(f"\n📝 Question {i+1}/{num_questions}")
            
            # Generate question
            question = self.generate_next_question(difficulty=difficulty)
            
            if not question:
                print("⚠️  Failed to generate question, skipping...")
                continue
            
            questions_asked.append(question)
        
        print("\n" + "=" * 60)
        print(f"✓ Practice session prepared: {len(questions_asked)} questions ready")
        
        return {
            'questions': questions_asked,
            'total_prepared': len(questions_asked)
        }
    
    def get_next_question_recommendation(self) -> Dict:
        """
        Get a recommendation for what to study next.
        
        Returns:
            Dictionary with recommended topic and difficulty
        """
        topic = self._select_adaptive_topic()
        difficulty = self._select_adaptive_difficulty()
        
        # Get coverage info
        weak_topics = self.get_weak_topics()
        
        return {
            'recommended_topic': topic,
            'recommended_difficulty': difficulty,
            'weak_topics': weak_topics[:3] if weak_topics else [],
            'reason': self._get_recommendation_reason(topic, difficulty, weak_topics)
        }
    
    def _get_recommendation_reason(
        self,
        topic: str,
        difficulty: str,
        weak_topics: List
    ) -> str:
        """Generate a human-readable reason for the recommendation."""
        stats = self.get_session_stats()
        
        if stats.get('answered_questions', 0) == 0:
            return f"Starting with {difficulty} questions on {topic} to assess your baseline."
        
        accuracy = stats.get('accuracy', 0)
        
        if weak_topics and any(t[0] == topic for t in weak_topics):
            return f"Focusing on {topic} where you scored {weak_topics[0][1]*100:.0f}% to improve understanding."
        elif accuracy < 50:
            return f"Using {difficulty} questions to build confidence. Current accuracy: {accuracy:.0f}%"
        elif accuracy > 80:
            return f"Challenging you with {difficulty} questions on {topic}. You're doing great!"
        else:
            return f"Continuing with {difficulty} questions on {topic} to maintain progress."


# ==================== Convenience Functions ====================

def create_question_engine(session_name: str) -> QuestionEngine:
    """
    Create a question engine instance.
    
    Args:
        session_name: Name for the study session
        
    Returns:
        QuestionEngine instance
    """
    return QuestionEngine(session_name)


def load_existing_session(session_name: str) -> Optional[QuestionEngine]:
    """
    Load an existing session by name.
    
    Args:
        session_name: Name of the session to load
        
    Returns:
        QuestionEngine instance or None if not found
    """
    engine = QuestionEngine(session_name)
    if engine.session:
        return engine
    return None