"""
LangChain Chains Module
Implements PDO-based prompting for question generation, answer evaluation, and explanations.
"""

from typing import List, Optional
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import json
import re

from bedrock_models import get_llm
from vector_store import VectorStore


# ==================== Output Models ====================

class TopicList(BaseModel):
    """Model for extracted topics."""
    topics: List[str] = Field(description="List of key topics or concepts from the text")


class GeneratedQuestion(BaseModel):
    """Model for a generated question."""
    question: str = Field(description="The generated question text")
    topic: str = Field(description="The main topic this question relates to")
    difficulty: str = Field(description="Difficulty level: easy, medium, or hard")
    expected_concepts: List[str] = Field(description="Key concepts that should be in a good answer")
    id: Optional[int] = Field(default=None, description="Database ID after saving")
    
    class Config:
        # Allow setting fields after creation
        validate_assignment = True


class AnswerEvaluation(BaseModel):
    """Model for answer evaluation results."""
    is_correct: bool = Field(description="Whether the answer is fundamentally correct")
    score: float = Field(description="Score from 0.0 to 1.0")
    feedback: str = Field(description="Detailed feedback on the answer")
    missing_concepts: List[str] = Field(description="Important concepts that were missed")
    strengths: List[str] = Field(description="What the student did well")


# ==================== Topic Extraction Chain ====================

class TopicExtractionChain:
    """
    Extracts key topics and concepts from document text.
    Uses PDO prompting for consistent, high-quality extraction.
    """
    
    def __init__(self):
        """Initialize the topic extraction chain."""
        self.llm = get_llm(temperature=0.3)  # Low temperature for consistency
        
        # PDO Prompt for Topic Extraction
        self.prompt = ChatPromptTemplate.from_template("""
You are an expert educator and curriculum designer with deep knowledge across multiple domains.

Your task is to analyze the provided text and extract 5-10 key topics or concepts that would be important for a student to understand and be tested on.

Guidelines:
- Focus on main concepts, theories, algorithms, or principles
- Topics should be specific enough to generate meaningful questions
- Avoid overly broad topics like "Introduction" or "Overview"
- Include both fundamental and advanced concepts if present
- Topics should be suitable for creating study questions

Text to analyze:
{text}

Output your response as a JSON object with a single key "topics" containing a list of topic strings.

Example format:
{{
    "topics": [
        "Gradient Descent Optimization",
        "Backpropagation Algorithm",
        "Neural Network Architecture"
    ]
}}

Your response (JSON only, no other text):
""")
        
        self.chain = self.prompt | self.llm
    
    def extract_topics(self, text: str, max_topics: int = 10) -> List[str]:
        """
        Extract topics from text.
        
        Args:
            text: Text to analyze (can be multiple chunks combined)
            max_topics: Maximum number of topics to extract
            
        Returns:
            List of extracted topics
        """
        try:
            # Truncate text if too long (keep first ~3000 chars for context)
            truncated_text = text[:3000] if len(text) > 3000 else text
            
            response = self.chain.invoke({"text": truncated_text})
            
            # Parse JSON response
            content = response.content
            
            # Clean response (remove markdown code blocks if present)
            content = re.sub(r'```json\s*|\s*```', '', content).strip()
            
            # Parse JSON
            result = json.loads(content)
            topics = result.get("topics", [])
            
            return topics[:max_topics]
            
        except Exception as e:
            print(f"Error extracting topics: {e}")
            # Fallback: return generic topic
            return ["General concepts from the material"]


# ==================== Question Generation Chain ====================

class QuestionGenerationChain:
    """
    Generates questions using RAG and PDO prompting.
    Retrieves relevant context and creates targeted questions.
    """
    
    def __init__(self, vector_store: VectorStore):
        """
        Initialize the question generation chain.
        
        Args:
            vector_store: VectorStore instance for RAG retrieval
        """
        self.vector_store = vector_store
        self.llm = get_llm(temperature=0.7)  # Balanced creativity
        
        # PDO Prompt for Question Generation
        self.prompt = ChatPromptTemplate.from_template("""
You are an expert educator creating study questions to help students learn effectively.

Your task is to generate ONE thoughtful, clear question based on the provided context and topic.

Topic to focus on: {topic}
Difficulty level: {difficulty}

Context from study materials:
{context}

Guidelines for question creation:
- Question should test understanding, not just memorization
- Be specific and clear in wording
- For "easy": test basic definitions or simple concepts
- For "medium": test application or relationships between concepts
- For "hard": test analysis, evaluation, or synthesis of multiple concepts
- Question should be answerable based on the provided context
- Avoid yes/no questions; prefer open-ended questions

Output your response as a JSON object with these fields:
- question: The question text
- topic: The main topic (use the provided topic: {topic})
- difficulty: The difficulty level (use: {difficulty})
- expected_concepts: List of 2-4 key concepts that should appear in a good answer

Example format:
{{
    "question": "How does gradient descent minimize the loss function in neural network training?",
    "topic": "Gradient Descent Optimization",
    "difficulty": "medium",
    "expected_concepts": ["iterative optimization", "partial derivatives", "learning rate", "convergence"]
}}

Your response (JSON only, no other text):
""")
        
        self.chain = self.prompt | self.llm
    
    def generate_question(
        self,
        topic: str,
        difficulty: str = "medium",
        k_context: int = 3
    ) -> Optional[GeneratedQuestion]:
        """
        Generate a question on a specific topic.
        
        Args:
            topic: Topic to generate question about
            difficulty: Question difficulty (easy, medium, hard)
            k_context: Number of context chunks to retrieve
            
        Returns:
            GeneratedQuestion object or None if generation fails
        """
        try:
            # Retrieve relevant context using RAG
            context_docs = self.vector_store.similarity_search(topic, k=k_context)
            
            if not context_docs:
                print(f"Warning: No context found for topic '{topic}'")
                return None
            
            # Combine context
            context = "\n\n".join([doc.page_content for doc in context_docs])
            
            # Generate question
            response = self.chain.invoke({
                "topic": topic,
                "difficulty": difficulty,
                "context": context[:2000]  # Limit context length
            })
            
            # Parse JSON response
            content = response.content
            content = re.sub(r'```json\s*|\s*```', '', content).strip()
            
            result = json.loads(content)
            
            return GeneratedQuestion(
                question=result["question"],
                topic=result["topic"],
                difficulty=result["difficulty"],
                expected_concepts=result["expected_concepts"]
            )
            
        except Exception as e:
            print(f"Error generating question: {e}")
            return None


# ==================== Answer Evaluation Chain ====================

class AnswerEvaluationChain:
    """
    Evaluates user answers with detailed feedback.
    Uses semantic understanding, not just keyword matching.
    """
    
    def __init__(self):
        """Initialize the answer evaluation chain."""
        self.llm = get_llm(temperature=0.1)  # Very low temperature for consistency
        
        # PDO Prompt for Answer Evaluation
        self.prompt = ChatPromptTemplate.from_template("""
You are an expert educator evaluating a student's answer with fairness but also with HIGH STANDARDS.

Question asked: {question}
Expected concepts in a good answer: {expected_concepts}

Student's answer: {student_answer}

Reference context (what the study material says):
{context}

Your task is to evaluate the student's answer comprehensively:

CRITICAL EVALUATION RULES:
1. If the student provides a non-answer (like "I don't know", "I won't answer", blank response, or gibberish), assign score 0.0
2. If the student just repeats words from the question without adding substance, assign score 0.0-0.1
3. If the student gives vague statements like "I would use CSS" or "It involves HTML" without ANY specific details, assign score 0.0-0.1
4. If the student shows they attempted but provides no technical details, examples, or explanations: 0.1-0.2
5. Only give 0.3+ if the answer contains at least ONE specific concept, detail, or example that actually answers the question

A real answer must include SPECIFICS - not just vague statements. For example:
- BAD (0.0): "I would use CSS to style it"
- BAD (0.1): "You use form tags and input tags"
- OK (0.4): "Use <form> tag with <input type='text'> for name field"
- GOOD (0.7): "Create <form> element, add <input type='text' name='username'> for name, <input type='email'> for email, style with CSS properties like border and padding"

Evaluation criteria:
1. Correctness: Is the core understanding correct?
2. Completeness: Are key concepts covered?
3. Specificity: Does it include actual details, not just vague statements?
4. Accuracy: Are there any misconceptions or errors?
5. Clarity: Is the explanation clear and well-structured?

Scoring guidelines (BE VERY STRICT):
- 0.0: No attempt, non-answer, just repeating question, or completely incorrect
- 0.1: Vague statement with no specifics (like "use CSS" or "create a form")
- 0.2: Shows minimal effort but demonstrates no real understanding or details
- 0.3-0.4: Mentions 1-2 specific concepts but incomplete explanation
- 0.5-0.6: Partial understanding with some details, missing key concepts
- 0.7-0.8: Good answer with specific details and minor gaps
- 0.9-1.0: Excellent answer with comprehensive specifics and examples

Mark is_correct as true ONLY if score is 0.6 or higher AND the student demonstrates genuine understanding with specific details.

Output your evaluation as a JSON object with these fields:
- is_correct: true if fundamentally correct (score >= 0.6), false otherwise
- score: numerical score from 0.0 to 1.0 (BE STRICT - most answers should be 0.0-0.5)
- feedback: 2-3 sentence constructive feedback (for vague answers, tell them to be more specific)
- missing_concepts: list of important concepts that were not mentioned
- strengths: list of 1-2 things the student did well (empty list if no real attempt or just vague statements)

Your evaluation (JSON only, no other text):
""")
        
        self.chain = self.prompt | self.llm
    
    def evaluate_answer(
        self,
        question: str,
        student_answer: str,
        expected_concepts: List[str],
        context: str
    ) -> Optional[AnswerEvaluation]:
        """
        Evaluate a student's answer.
        
        Args:
            question: The question that was asked
            student_answer: Student's answer text
            expected_concepts: Concepts expected in a good answer
            context: Reference context from study materials
            
        Returns:
            AnswerEvaluation object or None if evaluation fails
        """
        try:
            # Pre-check for obvious non-answers
            answer_lower = student_answer.lower().strip()
            answer_words = set(answer_lower.split())
            question_lower = question.lower()
            
            # Check 1: Too short
            if len(student_answer.strip()) < 10:
                return AnswerEvaluation(
                    is_correct=False,
                    score=0.0,
                    feedback="Your answer is too brief. Please provide a more detailed explanation with specific information.",
                    missing_concepts=expected_concepts,
                    strengths=[]
                )
            
            # Check 2: Non-answer phrases
            non_answer_phrases = [
                "i don't know", "idk", "no idea", "don't answer", "won't answer",
                "i wont", "skip", "pass", "next", "nothing", "no answer"
            ]
            if any(phrase in answer_lower for phrase in non_answer_phrases):
                return AnswerEvaluation(
                    is_correct=False,
                    score=0.0,
                    feedback="No valid answer provided. Please try to answer the question based on what you know. Even a partial answer helps you learn!",
                    missing_concepts=expected_concepts,
                    strengths=[]
                )
            
            # Check 3: Just repeating the question
            # Check if answer is suspiciously similar to question
            question_words = set(w for w in question_lower.split() if len(w) > 3)
            
            if question_words and len(answer_words) > 0:
                overlap = len(answer_words.intersection(question_words))
                overlap_ratio = overlap / len(answer_words) if len(answer_words) > 0 else 0
                
                # Calculate how much of the answer is just question words
                # High overlap + similar length = just copying the question
                answer_length = len(student_answer.split())
                question_length = len(question.split())
                
                # If answer is almost identical to question (very high overlap)
                if overlap_ratio > 0.85:
                    return AnswerEvaluation(
                        is_correct=False,
                        score=0.0,
                        feedback="Your answer appears to just repeat the question without providing an actual answer. Please explain the concept or provide the information requested.",
                        missing_concepts=expected_concepts,
                        strengths=[]
                    )
                
                # If answer and question are similar length AND high overlap
                # This catches when someone copies the entire question as answer
                length_ratio = answer_length / question_length if question_length > 0 else 0
                if 0.8 <= length_ratio <= 1.2 and overlap_ratio > 0.75:
                    return AnswerEvaluation(
                        is_correct=False,
                        score=0.0,
                        feedback="Your answer is nearly identical to the question itself. Please provide an actual answer with your own explanation or information.",
                        missing_concepts=expected_concepts,
                        strengths=[]
                    )
            
            # Check 4: Vague/generic answers that add no real information
            # Only apply this check to SHORT answers
            if len(student_answer.split()) < 20:
                vague_patterns = [
                    "i would", "you could", "it would", "they would",
                    "something about", "related to", "has to do with"
                ]
                # If answer is short AND contains vague patterns, likely not substantial
                if any(pattern in answer_lower for pattern in vague_patterns):
                    # Check if there are any specific technical terms or details
                    has_specifics = any(concept.lower() in answer_lower for concept in expected_concepts)
                    if not has_specifics:
                        return AnswerEvaluation(
                            is_correct=False,
                            score=0.05,  # Small credit for attempting
                            feedback="Your answer is too vague. Please provide specific details, examples, or technical information that actually answers the question.",
                            missing_concepts=expected_concepts,
                            strengths=[]
                        )
            
            response = self.chain.invoke({
                "question": question,
                "expected_concepts": ", ".join(expected_concepts),
                "student_answer": student_answer,
                "context": context[:2000]  # Limit context length
            })
            
            # Parse JSON response
            content = response.content
            content = re.sub(r'```json\s*|\s*```', '', content).strip()
            
            result = json.loads(content)
            
            return AnswerEvaluation(
                is_correct=result["is_correct"],
                score=float(result["score"]),
                feedback=result["feedback"],
                missing_concepts=result.get("missing_concepts", []),
                strengths=result.get("strengths", [])
            )
            
        except Exception as e:
            print(f"Error evaluating answer: {e}")
            return None


# ==================== Explanation Generation Chain ====================

class ExplanationChain:
    """
    Generates helpful explanations for incorrect answers or follow-up questions.
    """
    
    def __init__(self, vector_store: VectorStore):
        """
        Initialize the explanation chain.
        
        Args:
            vector_store: VectorStore instance for context retrieval
        """
        self.vector_store = vector_store
        self.llm = get_llm(temperature=0.5)  # Moderate creativity
        
        self.prompt = ChatPromptTemplate.from_template("""
You are a patience, encouraging tutor helping a student understand a concept better.

Question: {question}
Student's answer: {student_answer}
What went wrong: {evaluation_feedback}

Context from study materials:
{context}

Your task is to provide a clear, encouraging explanation that:
1. Acknowledges what the student got right
2. Clarifies the correct understanding
3. Explains why this is important
4. Gives a practical example if relevant

Keep your explanation concise (3-4 sentences) and encouraging.

Your explanation:
""")
        
        self.chain = self.prompt | self.llm
    
    def generate_explanation(
        self,
        question: str,
        student_answer: str,
        evaluation_feedback: str,
        topic: str
    ) -> str:
        """
        Generate an explanation for a student.
        
        Args:
            question: The original question
            student_answer: Student's answer
            evaluation_feedback: Feedback from evaluation
            topic: Topic for context retrieval
            
        Returns:
            Explanation text
        """
        try:
            # Retrieve relevant context
            context_docs = self.vector_store.similarity_search(topic, k=2)
            context = "\n\n".join([doc.page_content for doc in context_docs])
            
            response = self.chain.invoke({
                "question": question,
                "student_answer": student_answer,
                "evaluation_feedback": evaluation_feedback,
                "context": context[:1500]
            })
            
            return response.content
            
        except Exception as e:
            print(f"Error generating explanation: {e}")
            return "I encountered an error generating the explanation. Please review the study materials."


# ==================== Convenience Functions ====================

def create_topic_extraction_chain() -> TopicExtractionChain:
    """Create a topic extraction chain instance."""
    return TopicExtractionChain()


def create_question_generation_chain(vector_store: VectorStore) -> QuestionGenerationChain:
    """Create a question generation chain instance."""
    return QuestionGenerationChain(vector_store)


def create_answer_evaluation_chain() -> AnswerEvaluationChain:
    """Create an answer evaluation chain instance."""
    return AnswerEvaluationChain()


def create_explanation_chain(vector_store: VectorStore) -> ExplanationChain:
    """Create an explanation chain instance."""
    return ExplanationChain(vector_store)