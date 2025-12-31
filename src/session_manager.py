"""
Session Management Module
Handles database operations for tracking sessions, questions, answers, and progress.
"""

from typing import List, Dict, Optional, Tuple
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.sql import func
import json

from config import settings

Base = declarative_base()


class StudySession(Base):
    """Represents a study session with a collection of documents."""
    __tablename__ = 'study_sessions'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False)
    collection_name = Column(String(200), nullable=False, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    documents = relationship("SessionDocument", back_populates="session", cascade="all, delete-orphan")
    questions = relationship("Question", back_populates="session", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<StudySession(id={self.id}, name='{self.name}')>"


class SessionDocument(Base):
    """Represents a document uploaded in a session."""
    __tablename__ = 'session_documents'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('study_sessions.id'), nullable=False)
    doc_id = Column(String(50), nullable=False)
    source_name = Column(String(200), nullable=False)
    total_chunks = Column(Integer, default=0)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    
    # Metadata as JSON
    metadata_json = Column(Text, nullable=True)
    
    # Relationships
    session = relationship("StudySession", back_populates="documents")
    topics = relationship("DocumentTopic", back_populates="document", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<SessionDocument(id={self.id}, source='{self.source_name}')>"


class DocumentTopic(Base):
    """Topics extracted from documents for targeted question generation."""
    __tablename__ = 'document_topics'
    
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey('session_documents.id'), nullable=False)
    topic = Column(String(500), nullable=False)
    times_covered = Column(Integer, default=0)
    
    # Relationships
    document = relationship("SessionDocument", back_populates="topics")
    
    def __repr__(self):
        return f"<DocumentTopic(id={self.id}, topic='{self.topic[:30]}...')>"


class Question(Base):
    """Represents a generated question and user's answer."""
    __tablename__ = 'questions'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('study_sessions.id'), nullable=False)
    
    # Question details
    question_text = Column(Text, nullable=False)
    topic = Column(String(500), nullable=True)
    difficulty = Column(String(50), default="medium")
    
    # User answer
    user_answer = Column(Text, nullable=True)
    is_correct = Column(Boolean, nullable=True)
    score = Column(Float, nullable=True) 
    
    # Evaluation details
    evaluation_feedback = Column(Text, nullable=True)
    correct_answer = Column(Text, nullable=True)
    
    # Timestamps
    asked_at = Column(DateTime, default=datetime.utcnow)
    answered_at = Column(DateTime, nullable=True)
    
    # Context used (for debugging/analysis)
    context_json = Column(Text, nullable=True)
    
    # Relationships
    session = relationship("StudySession", back_populates="questions")
    
    def __repr__(self):
        return f"<Question(id={self.id}, topic='{self.topic}')>"


class SessionManager:
    """
    Manages database operations for study sessions.
    Provides high-level interface for session and question management.
    """
    
    def __init__(self):
        """Initialize the session manager and create tables."""
        self.engine = create_engine(settings.database_url, echo=False)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        print("✓ Session database initialized")
    
    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()
    
    # ==================== Study Session Operations ====================
    
    def create_study_session(self, name: str, collection_name: str) -> StudySession:
        """
        Create a new study session.
        
        Args:
            name: Human-readable session name
            collection_name: Corresponding vector store collection name
            
        Returns:
            Created StudySession object
        """
        db = self.get_session()
        try:
            session = StudySession(
                name=name,
                collection_name=collection_name
            )
            db.add(session)
            db.commit()
            db.refresh(session)
            return session
        finally:
            db.close()
    
    def get_study_session(self, session_id: int) -> Optional[StudySession]:
        """Get a study session by ID."""
        db = self.get_session()
        try:
            return db.query(StudySession).filter(StudySession.id == session_id).first()
        finally:
            db.close()
    
    def get_study_session_by_collection(self, collection_name: str) -> Optional[StudySession]:
        """Get a study session by collection name."""
        db = self.get_session()
        try:
            return db.query(StudySession).filter(
                StudySession.collection_name == collection_name
            ).first()
        finally:
            db.close()
    
    def list_study_sessions(self) -> List[StudySession]:
        """List all study sessions, ordered by most recent."""
        db = self.get_session()
        try:
            return db.query(StudySession).order_by(
                StudySession.last_active.desc()
            ).all()
        finally:
            db.close()
    
    def delete_study_session(self, session_id: int):
        """Delete a study session and all related data."""
        db = self.get_session()
        try:
            session = db.query(StudySession).filter(StudySession.id == session_id).first()
            if session:
                db.delete(session)
                db.commit()
        finally:
            db.close()
    
    def update_session_activity(self, session_id: int):
        """Update the last_active timestamp for a session."""
        db = self.get_session()
        try:
            session = db.query(StudySession).filter(StudySession.id == session_id).first()
            if session:
                session.last_active = datetime.utcnow()
                db.commit()
        finally:
            db.close()
    
    # ==================== Document Operations ====================
    
    def add_document(
        self,
        session_id: int,
        doc_id: str,
        source_name: str,
        total_chunks: int,
        metadata: Optional[Dict] = None
    ) -> SessionDocument:
        """
        Add a document to a study session.
        
        Args:
            session_id: Study session ID
            doc_id: Document ID from document processor
            source_name: Original filename or source name
            total_chunks: Number of chunks created
            metadata: Optional metadata dictionary
            
        Returns:
            Created SessionDocument object
        """
        db = self.get_session()
        try:
            doc = SessionDocument(
                session_id=session_id,
                doc_id=doc_id,
                source_name=source_name,
                total_chunks=total_chunks,
                metadata_json=json.dumps(metadata) if metadata else None
            )
            db.add(doc)
            db.commit()
            db.refresh(doc)
            return doc
        finally:
            db.close()
    
    def get_session_documents(self, session_id: int) -> List[SessionDocument]:
        """Get all documents in a session."""
        db = self.get_session()
        try:
            return db.query(SessionDocument).filter(
                SessionDocument.session_id == session_id
            ).all()
        finally:
            db.close()
    
    # ==================== Topic Operations ====================
    
    def add_topics(self, document_id: int, topics: List[str]) -> List[DocumentTopic]:
        """
        Add topics for a document.
        
        Args:
            document_id: SessionDocument ID
            topics: List of topic strings
            
        Returns:
            List of created DocumentTopic objects
        """
        db = self.get_session()
        try:
            topic_objs = []
            for topic in topics:
                topic_obj = DocumentTopic(
                    document_id=document_id,
                    topic=topic,
                    times_covered=0
                )
                db.add(topic_obj)
                topic_objs.append(topic_obj)
            
            db.commit()
            return topic_objs
        finally:
            db.close()
    
    def get_session_topics(self, session_id: int) -> List[Tuple[str, int]]:
        """
        Get all topics for a session with coverage count.
        
        Args:
            session_id: Study session ID
            
        Returns:
            List of tuples (topic, times_covered)
        """
        db = self.get_session()
        try:
            topics = db.query(DocumentTopic.topic, DocumentTopic.times_covered).join(
                SessionDocument
            ).filter(
                SessionDocument.session_id == session_id
            ).all()
            return topics
        finally:
            db.close()
    
    def increment_topic_coverage(self, session_id: int, topic: str):
        """Increment the coverage count for a topic."""
        db = self.get_session()
        try:
            topic_obj = db.query(DocumentTopic).join(SessionDocument).filter(
                SessionDocument.session_id == session_id,
                DocumentTopic.topic == topic
            ).first()
            
            if topic_obj:
                topic_obj.times_covered += 1
                db.commit()
        finally:
            db.close()
    
    def get_least_covered_topics(self, session_id: int, limit: int = 5) -> List[str]:
        """Get topics that have been covered the least."""
        db = self.get_session()
        try:
            topics = db.query(DocumentTopic.topic).join(SessionDocument).filter(
                SessionDocument.session_id == session_id
            ).order_by(
                DocumentTopic.times_covered.asc()
            ).limit(limit).all()
            
            return [t[0] for t in topics]
        finally:
            db.close()
    
    # ==================== Question Operations ====================
    
    def add_question(
        self,
        session_id: int,
        question_text: str,
        topic: Optional[str] = None,
        difficulty: str = "medium",
        context: Optional[List[str]] = None
    ) -> Question:
        """
        Add a generated question to the session.
        
        Args:
            session_id: Study session ID
            question_text: The question text
            topic: Related topic
            difficulty: Question difficulty level
            context: Context chunks used for generation
            
        Returns:
            Created Question object
        """
        db = self.get_session()
        try:
            question = Question(
                session_id=session_id,
                question_text=question_text,
                topic=topic,
                difficulty=difficulty,
                context_json=json.dumps(context) if context else None
            )
            db.add(question)
            db.commit()
            db.refresh(question)
            
            # Update topic coverage
            if topic:
                self.increment_topic_coverage(session_id, topic)
            
            return question
        finally:
            db.close()
    
    def update_question_answer(
        self,
        question_id: int,
        user_answer: str,
        is_correct: bool,
        score: float,
        evaluation_feedback: str,
        correct_answer: Optional[str] = None
    ):
        """
        Update a question with user's answer and evaluation.
        
        Args:
            question_id: Question ID
            user_answer: User's answer text
            is_correct: Whether answer is correct
            score: Numerical score (0.0 to 1.0)
            evaluation_feedback: Feedback from evaluation
            correct_answer: The correct/expected answer
        """
        db = self.get_session()
        try:
            question = db.query(Question).filter(Question.id == question_id).first()
            if question:
                question.user_answer = user_answer
                question.is_correct = is_correct
                question.score = score
                question.evaluation_feedback = evaluation_feedback
                question.correct_answer = correct_answer
                question.answered_at = datetime.utcnow()
                db.commit()
        finally:
            db.close()
    
    def get_session_questions(self, session_id: int) -> List[Question]:
        """Get all questions for a session."""
        db = self.get_session()
        try:
            return db.query(Question).filter(
                Question.session_id == session_id
            ).order_by(Question.asked_at.desc()).all()
        finally:
            db.close()
    
    def get_unanswered_questions(self, session_id: int) -> List[Question]:
        """Get questions that haven't been answered yet."""
        db = self.get_session()
        try:
            return db.query(Question).filter(
                Question.session_id == session_id,
                Question.user_answer == None
            ).all()
        finally:
            db.close()
    
    # ==================== Analytics ====================
    
    def get_session_stats(self, session_id: int) -> Dict:
        """
        Get comprehensive statistics for a session.
        
        Args:
            session_id: Study session ID
            
        Returns:
            Dictionary with session statistics
        """
        db = self.get_session()
        try:
            session = db.query(StudySession).filter(StudySession.id == session_id).first()
            if not session:
                return {}
            
            # Question stats
            total_questions = db.query(func.count(Question.id)).filter(
                Question.session_id == session_id
            ).scalar()
            
            answered_questions = db.query(func.count(Question.id)).filter(
                Question.session_id == session_id,
                Question.user_answer != None
            ).scalar()
            
            correct_answers = db.query(func.count(Question.id)).filter(
                Question.session_id == session_id,
                Question.is_correct == True
            ).scalar()
            
            avg_score = db.query(func.avg(Question.score)).filter(
                Question.session_id == session_id,
                Question.score != None
            ).scalar() or 0.0
            
            # Document stats
            total_docs = db.query(func.count(SessionDocument.id)).filter(
                SessionDocument.session_id == session_id
            ).scalar()
            
            total_topics = db.query(func.count(DocumentTopic.id)).join(
                SessionDocument
            ).filter(
                SessionDocument.session_id == session_id
            ).scalar()
            
            return {
                'session_name': session.name,
                'created_at': session.created_at,
                'last_active': session.last_active,
                'total_documents': total_docs,
                'total_topics': total_topics,
                'total_questions': total_questions,
                'answered_questions': answered_questions,
                'correct_answers': correct_answers,
                'accuracy': (correct_answers / answered_questions * 100) if answered_questions > 0 else 0,
                'average_score': float(avg_score),
            }
        finally:
            db.close()
    
    def get_weak_topics(self, session_id: int, min_questions: int = 2) -> List[Tuple[str, float]]:
        """
        Identify topics where user is struggling.
        
        Args:
            session_id: Study session ID
            min_questions: Minimum questions per topic to consider
            
        Returns:
            List of tuples (topic, average_score) sorted by score
        """
        db = self.get_session()
        try:
            # Calculate average score per topic
            topic_scores = db.query(
                Question.topic,
                func.avg(Question.score).label('avg_score'),
                func.count(Question.id).label('question_count')
            ).filter(
                Question.session_id == session_id,
                Question.topic != None,
                Question.score != None
            ).group_by(
                Question.topic
            ).having(
                func.count(Question.id) >= min_questions
            ).order_by(
                func.avg(Question.score).asc()
            ).all()
            
            return [(t.topic, float(t.avg_score)) for t in topic_scores]
        finally:
            db.close()


# Global session manager instance
session_manager = SessionManager()


def get_session_manager() -> SessionManager:
    """Get the global session manager instance."""
    return session_manager