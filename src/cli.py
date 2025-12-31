"""
Command Line Interface for Question Asker Bot
Provides interactive menu-driven interface for studying.
"""

import sys
from pathlib import Path
from typing import Optional
import os

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.progress import Progress

from question_engine import QuestionEngine, create_question_engine
from session_manager import get_session_manager

console = Console()


def to_nepal_time(utc_datetime):
    """
    Convert UTC datetime to Nepal time (UTC+5:45).
    
    Args:
        utc_datetime: datetime object in UTC
        
    Returns:
        Formatted string in Nepal time
    """
    if utc_datetime is None:
        return "N/A"
    from datetime import timedelta
    nepal_offset = timedelta(hours=5, minutes=45)
    nepal_time = utc_datetime + nepal_offset
    return nepal_time.strftime('%Y-%m-%d %H:%M NPT')


class QuestionAskerCLI:
    """Command-line interface for the Question Asker Bot."""
    
    def __init__(self):
        """Initialize the CLI."""
        self.engine: Optional[QuestionEngine] = None
        self.session_manager = get_session_manager()
        self.current_question = None
    
    def clear_screen(self):
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def show_banner(self):
        """Display welcome banner."""
        banner = """
[bold cyan]╔═══════════════════════════════════════════════╗
║                                               ║
║        📚 QUESTION ASKER BOT 📚               ║
║                                               ║
║     AI-Powered Study Assistant                ║
║     Using RAG + LangChain + AWS Bedrock      ║
║                                               ║
╚═══════════════════════════════════════════════╝[/bold cyan]
        """
        console.print(banner)
    
    def main_menu(self):
        """Display and handle main menu."""
        while True:
            console.print("\n[bold yellow]MAIN MENU[/bold yellow]")
            console.print("─" * 50)
            
            if self.engine:
                console.print(f"[green]✓ Active Session:[/green] {self.engine.session_name}")
            else:
                console.print("[dim]No active session[/dim]")
            
            console.print("\n[cyan]1.[/cyan] Create New Study Session")
            console.print("[cyan]2.[/cyan] Load Existing Session")
            console.print("[cyan]3.[/cyan] Upload Document")
            console.print("[cyan]4.[/cyan] Start Practice Session")
            console.print("[cyan]5.[/cyan] View Progress & Analytics")
            console.print("[cyan]6.[/cyan] View Study Recommendations")
            console.print("[cyan]7.[/cyan] Exit") 
            
            choice = Prompt.ask("\n[bold]Choose an option[/bold]", 
                          choices=["1", "2", "3", "4", "5", "6", "7"])
        
            if choice == "1":
                self.create_session()
            elif choice == "2":
                self.load_session()
            elif choice == "3":
                self.upload_document()
            elif choice == "4":
                self.start_practice()
            elif choice == "5":
                self.view_progress()
            elif choice == "6":
                self.view_recommendations()
            elif choice == "7":
                if Confirm.ask("\n[yellow]Are you sure you want to exit?[/yellow]"):
                    console.print("\n[green]Thanks for using Question Asker Bot! Happy studying! 📚[/green]\n")
                    sys.exit(0)
    
    def create_session(self):
        """Create a new study session."""
        console.print("\n[bold blue]CREATE NEW STUDY SESSION[/bold blue]")
        console.print("─" * 50)
        
        session_name = Prompt.ask("[cyan]Enter session name[/cyan]")
        
        if not session_name.strip():
            console.print("[red]✗ Session name cannot be empty[/red]")
            return
        
        # Check if session with same name exists
        existing_sessions = self.session_manager.list_study_sessions()
        existing_names = [s.name.lower() for s in existing_sessions]
        
        if session_name.lower() in existing_names:
            console.print(f"\n[yellow]⚠️  A session named '{session_name}' already exists![/yellow]")
            console.print("\n[cyan]What would you like to do?[/cyan]")
            console.print("1. Load the existing session")
            console.print("2. Create with a different name")
            console.print("3. Cancel")
            
            choice = Prompt.ask(
                "\n[bold]Choose option[/bold]",
                choices=["1", "2", "3"],
                default="1"
            )
            
            if choice == "1":
                # Load the existing session instead
                existing_session = next(s for s in existing_sessions if s.name.lower() == session_name.lower())
                self.engine = create_question_engine(existing_session.name)
                console.print(f"\n[green]✓ Loaded existing session: {existing_session.name}[/green]")
                
                # Show quick stats
                stats = self.session_manager.get_session_stats(existing_session.id)
                console.print(f"\n[dim]Documents: {stats.get('total_documents', 0)} | "
                            f"Questions: {stats.get('total_questions', 0)} | "
                            f"Accuracy: {stats.get('accuracy', 0):.1f}%[/dim]")
                return
            elif choice == "2":
                # Ask for new name
                session_name = Prompt.ask("\n[cyan]Enter a different session name[/cyan]")
                if not session_name.strip() or session_name.lower() in existing_names:
                    console.print("[red]✗ Invalid name or still exists. Cancelled.[/red]")
                    return
            else:
                console.print("[red]Session creation cancelled.[/red]")
                return
        
        try:
            console.print("\n[yellow]Creating session...[/yellow]")
            self.engine = create_question_engine(session_name)
            
            console.print(Panel(
                f"[green]✓ Session created successfully![/green]\n\n"
                f"Session: [bold]{session_name}[/bold]\n"
                f"Next step: Upload study materials",
                title="✅ Session Created",
                border_style="green"
            ))
            
            if Confirm.ask("\n[cyan]Would you like to upload a document now?[/cyan]"):
                self.upload_document()
                
        except Exception as e:
            console.print(f"[red]✗ Error creating session: {e}[/red]")
    
    def load_session(self):
        """Load an existing session."""
        console.print("\n[bold blue]LOAD EXISTING SESSION[/bold blue]")
        console.print("─" * 50)
        
        # List available sessions
        sessions = self.session_manager.list_study_sessions()
        
        if not sessions:
            console.print("[yellow]No existing sessions found.[/yellow]")
            if Confirm.ask("\n[cyan]Would you like to create a new session?[/cyan]"):
                self.create_session()
            return
        
        # Display sessions in a table
        table = Table(title="Available Sessions")
        table.add_column("#", style="cyan", width=5)
        table.add_column("Session Name", style="green")
        table.add_column("Documents", style="yellow")
        table.add_column("Questions", style="blue")
        table.add_column("Last Active", style="dim")
        
        for i, session in enumerate(sessions, 1):
            stats = self.session_manager.get_session_stats(session.id)
            table.add_row(
                str(i),
                session.name,
                str(stats.get('total_documents', 0)),
                str(stats.get('total_questions', 0)),
                to_nepal_time(session.last_active)
            )
        
        console.print(table)
        
        # Get user choice
        choice = Prompt.ask(
            "\n[cyan]Enter session number (or 'c' to cancel)[/cyan]",
            default="c"
        )
        
        if choice.lower() == 'c':
            return
        
        try:
            session_idx = int(choice) - 1
            if 0 <= session_idx < len(sessions):
                selected_session = sessions[session_idx]
                self.engine = create_question_engine(selected_session.name)
                
                console.print(f"\n[green]✓ Loaded session: {selected_session.name}[/green]")
                
                # Show quick stats
                stats = self.session_manager.get_session_stats(selected_session.id)
                console.print(f"\n[dim]Documents: {stats.get('total_documents', 0)} | "
                            f"Questions: {stats.get('total_questions', 0)} | "
                            f"Accuracy: {stats.get('accuracy', 0):.1f}%[/dim]")
                console.print(f"[dim]Last active: {to_nepal_time(selected_session.last_active)}[/dim]")
            else:
                console.print("[red]✗ Invalid selection[/red]")
        except ValueError:
            console.print("[red]✗ Invalid input[/red]")
    
    def upload_document(self):
        """Upload a document to the current session."""
        if not self.engine:
            console.print("[red]✗ No active session. Please create or load a session first.[/red]")
            return
        
        console.print("\n[bold blue]UPLOAD DOCUMENT[/bold blue]")
        console.print("─" * 50)
        
        console.print("\n[cyan]Upload Options:[/cyan]")
        console.print("1. Upload PDF or Text file")
        console.print("2. Paste text directly")
        
        choice = Prompt.ask("[bold]Choose option[/bold]", choices=["1", "2"])
        
        if choice == "1":
            self._upload_file()
        else:
            self._upload_text()
    
    def _upload_file(self):
        """Upload a file (PDF or TXT)."""
        console.print("\n[yellow]Enter the full path to your file[/yellow]")
        console.print("[dim]Example: C:\\Users\\ACER\\Desktop\\notes.pdf[/dim]")
        console.print("[dim]or: /home/user/documents/study.txt[/dim]")
        
        file_path = Prompt.ask("\n[cyan]File path[/cyan]")
        
        # Expand user path and check if file exists
        file_path = os.path.expanduser(file_path)
        
        if not os.path.exists(file_path):
            console.print(f"[red]✗ File not found: {file_path}[/red]")
            console.print("[yellow]Tip: Make sure to use the full path[/yellow]")
            return
        
        # Check file extension
        path = Path(file_path)
        if path.suffix.lower() not in ['.pdf', '.txt', '.md']:
            console.print(f"[red]✗ Unsupported file type: {path.suffix}[/red]")
            console.print("[yellow]Supported types: .pdf, .txt, .md[/yellow]")
            return
        
        try:
            console.print("\n[yellow]Processing document...[/yellow]")
            console.print("[dim]This may take a moment for large files[/dim]")
            
            with Progress() as progress:
                task = progress.add_task("[cyan]Uploading and processing...", total=100)
                
                # Process document
                progress.update(task, advance=30)
                doc_info = self.engine.add_document_from_file(file_path)
                progress.update(task, advance=70)
            
            # Display success
            console.print(Panel(
                f"[green]✓ File uploaded successfully![/green]\n\n"
                f"📄 File: [bold]{doc_info['source_name']}[/bold]\n"
                f"📊 Split into: {doc_info['chunks']} chunks (for better AI retrieval)\n"
                f"🏷️ Topics identified: {len(doc_info['topics'])}\n\n"
                f"[cyan]Key Topics Found:[/cyan]\n" +
                "\n".join([f"  • {topic}" for topic in doc_info['topics'][:5]]) +
                (f"\n  • ... and {len(doc_info['topics']) - 5} more" if len(doc_info['topics']) > 5 else ""),
                title="✅ Upload Complete",
                border_style="green"
            ))
            
        except Exception as e:
            console.print(f"[red]✗ Error uploading document: {e}[/red]")
    
    def _upload_text(self):
        """Upload text directly."""
        console.print("\n[bold yellow]📝 Paste Your Text[/bold yellow]")
        console.print("─" * 50)
        console.print("\n[cyan]Instructions:[/cyan]")
        console.print("• Paste or type your text (can be multiple lines)")
        console.print("• When finished, type [bold yellow]END[/bold yellow] on a new line")
        console.print("• Press Enter after typing END\n")
        
        console.print("[dim]Example:[/dim]")
        console.print("[dim]Line 1 of your content[Enter][/dim]")
        console.print("[dim]Line 2 of your content[Enter][/dim]")
        console.print("[dim]END[Enter][/dim]\n")
        
        lines = []
        console.print("[green]>>> Start entering text:[/green]\n")
        
        line_count = 0
        while True:
            try:
                line = input()
                if line.strip().upper() == 'END':
                    if lines:  # Only break if we have content
                        console.print(f"\n[green]✓ Captured {line_count} lines ({sum(len(l) for l in lines)} characters)[/green]")
                        break
                    else:
                        console.print("[yellow]⚠️  No text entered yet.[/yellow]")
                        if Confirm.ask("Cancel upload?", default=False):
                            console.print("[red]Upload cancelled.[/red]")
                            return
                        else:
                            console.print("[cyan]Continue entering text...[/cyan]")
                lines.append(line)
                line_count += 1
            except EOFError:
                break
        
        text = "\n".join(lines)
        self._process_uploaded_text(text)
        return
    
    def _process_uploaded_text(self, text: str):
        """Process the uploaded text."""
        if not text.strip():
            console.print("[red]✗ No text entered[/red]")
            return
        
        # Get a name for this content
        source_name = Prompt.ask(
            "\n[cyan]Enter a name for this content[/cyan]",
            default="Pasted Content"
        )
        
        try:
            console.print("\n[yellow]Processing text...[/yellow]")
            
            doc_info = self.engine.add_document_from_text(text, source_name)
            
            console.print(Panel(
                f"[green]✓ Text uploaded successfully![/green]\n\n"
                f"📝 Name: [bold]{source_name}[/bold]\n"
                f"📏 Length: {len(text)} characters\n"
                f"📊 Split into: {doc_info.get('chunks', 0)} chunks\n"
                f"🏷️ Topics identified: {len(doc_info.get('topics', []))}\n\n"
                f"[cyan]Key Topics Found:[/cyan]\n" +
                "\n".join([f"  • {topic}" for topic in doc_info.get('topics', [])[:5]]),
                title="✅ Upload Complete",
                border_style="green"
            ))
            
        except Exception as e:
            console.print(f"[red]✗ Error processing text: {e}[/red]")
            console.print("[yellow]Debug info:[/yellow]")
            console.print(f"  Source name: {source_name}")
            console.print(f"  Text length: {len(text)}")
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        
        return
    
    def start_practice(self):
        """Start an interactive practice session."""
        if not self.engine:
            console.print("[red]✗ No active session. Please create or load a session first.[/red]")
            return
        
        # Check if documents are uploaded
        stats = self.engine.get_session_stats()
        if stats.get('total_documents', 0) == 0:
            console.print("[yellow]⚠️  No documents uploaded yet![/yellow]")
            if Confirm.ask("\n[cyan]Would you like to upload a document now?[/cyan]"):
                self.upload_document()
            return
        
        console.print("\n[bold blue]PRACTICE SESSION[/bold blue]")
        console.print("─" * 50)
        
        # Ask for number of questions
        num_questions = Prompt.ask(
            "\n[cyan]How many questions would you like to practice?[/cyan]",
            default="5"
        )
        
        try:
            num_questions = int(num_questions)
        except ValueError:
            console.print("[red]✗ Invalid number[/red]")
            return
        
        console.print(f"\n[green]Starting practice with {num_questions} questions![/green]")
        console.print("[dim]Answer each question to the best of your ability[/dim]\n")
        
        for i in range(num_questions):
            console.print("\n" + "=" * 60)
            console.print(f"[bold yellow]Question {i+1}/{num_questions}[/bold yellow]")
            console.print("=" * 60)
            
            # Generate question
            try:
                question = self.engine.generate_next_question()
                
                if not question:
                    console.print("[red]Failed to generate question. Skipping...[/red]")
                    continue
                
                # Display question
                console.print(Panel(
                    f"[bold cyan]{question.question}[/bold cyan]\n\n"
                    f"[dim]Topic: {question.topic}[/dim]\n"
                    f"[dim]Difficulty: {question.difficulty}[/dim]",
                    border_style="cyan",
                    title=f"📝 Question {i+1}"
                ))
                
                # Get user answer
                console.print("\n[yellow]Your answer:[/yellow]")
                console.print("[dim]Type your answer and press Enter[/dim]")
                user_answer = input("> ")
                
                if not user_answer.strip():
                    console.print("[yellow]⚠️  Empty answer. Skipping evaluation.[/yellow]")
                    continue
                
                # Evaluate answer
                console.print("\n[yellow]⏳ Evaluating your answer...[/yellow]")
                evaluation, explanation = self.engine.evaluate_answer(
                    question=question,
                    user_answer=user_answer,
                    provide_explanation=True
                )
                
                if evaluation:
                    # Display evaluation
                    status = "✓ Correct!" if evaluation.is_correct else "✗ Needs Improvement"
                    color = "green" if evaluation.is_correct else "yellow"
                    
                    feedback_text = f"[{color}][bold]{status}[/bold][/{color}]\n\n"
                    feedback_text += f"[cyan]Score:[/cyan] {evaluation.score * 100:.1f}%\n\n"
                    feedback_text += f"[cyan]Feedback:[/cyan]\n{evaluation.feedback}\n\n"
                    
                    if evaluation.strengths:
                        feedback_text += "[green]Strengths:[/green]\n"
                        for strength in evaluation.strengths:
                            feedback_text += f"  ✓ {strength}\n"
                        feedback_text += "\n"
                    
                    if evaluation.missing_concepts:
                        feedback_text += "[yellow]Missing Concepts:[/yellow]\n"
                        for concept in evaluation.missing_concepts:
                            feedback_text += f"  • {concept}\n"
                        feedback_text += "\n"
                    
                    if explanation:
                        feedback_text += f"[blue]💡 Explanation:[/blue]\n{explanation}"
                    
                    console.print(Panel(
                        feedback_text,
                        title="📊 Evaluation",
                        border_style=color
                    ))
                
                # Ask if user wants to continue
                if i < num_questions - 1:
                    if not Confirm.ask("\n[cyan]Continue to next question?[/cyan]", default=True):
                        break
                
            except Exception as e:
                console.print(f"[red]✗ Error during practice: {e}[/red]")
                if not Confirm.ask("\n[cyan]Continue anyway?[/cyan]"):
                    break
        
        console.print("\n" + "=" * 60)
        console.print("[bold green]Practice Session Complete! 🎉[/bold green]")
        console.print("=" * 60)

        stats = self.engine.get_session_stats()
        console.print(f"\n[cyan]Session Statistics:[/cyan]")
        console.print(f"  Questions answered: {stats.get('answered_questions', 0)}")
        console.print(f"  Accuracy: {stats.get('accuracy', 0):.1f}%")
        console.print(f"  Average score: {stats.get('average_score', 0) * 100:.1f}%")
        
    
    def view_progress(self):
        """View progress and analytics."""
        if not self.engine:
            console.print("[red]✗ No active session.[/red]")
            return
        
        console.print("\n[bold blue]PROGRESS & ANALYTICS[/bold blue]")
        console.print("─" * 50)
        
        stats = self.engine.get_session_stats()
        
        # Main statistics table
        table = Table(title="Session Statistics", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Session Name", stats['session_name'])
        table.add_row("Created", to_nepal_time(stats.get('created_at')))
        table.add_row("Last Active", to_nepal_time(stats.get('last_active')))
        table.add_row("Total Documents", str(stats['total_documents']))
        table.add_row("Topics Identified", str(stats['total_topics']))
        table.add_row("Questions Asked", str(stats['total_questions']))
        table.add_row("Questions Answered", str(stats['answered_questions']))
        table.add_row("Correct Answers", str(stats['correct_answers']))
        table.add_row("Accuracy", f"{stats['accuracy']:.1f}%")
        table.add_row("Average Score", f"{stats['average_score'] * 100:.1f}%")
        
        console.print(table)
        
        # Topic coverage
        coverage = self.engine.get_topic_coverage()
        if coverage:
            console.print("\n[cyan]📊 Topic Coverage (Questions Asked Per Topic):[/cyan]")
            for topic, count in coverage[:10]:
                bar = "█" * min(count, 20)
                console.print(f"  {bar:20} {topic[:40]} ({count} questions)")
        
        # Weak topics with clear labeling
        weak_topics = self.engine.get_weak_topics()
        if weak_topics:
            console.print("\n[yellow]⚠️  Topics Needing More Practice (Lowest Scores):[/yellow]")
            console.print("[dim]Lower score = needs more attention[/dim]\n")
            for topic, score in weak_topics[:5]:
                # Visual indicator - more red bars = needs more work
                score_pct = int(score * 100)
                
                # Color coding
                if score_pct < 40:
                    color = "red"
                    emoji = "🔴"
                    label = "Priority: HIGH"
                elif score_pct < 70:
                    color = "yellow"  
                    emoji = "🟡"
                    label = "Priority: Medium"
                else:
                    color = "green"
                    emoji = "🟢"
                    label = "Priority: Low"
                
                console.print(f"  {emoji} [{color}]{score_pct:3d}% - {topic[:45]}[/{color}] ({label})")
        else:
            console.print("\n[green]✓ No weak areas identified yet - keep practicing![/green]")
        
        # Strong topics (opposite of weak)
        all_topics = self.engine.get_topic_coverage()
        if all_topics and weak_topics:
            # Get topics with good scores by checking which aren't in weak list
            console.print("\n[green]✅ Strong Topics (Highest Scores):[/green]")
            console.print("[dim]These are your strongest areas[/dim]\n")
            
            # Get answered questions grouped by topic
            from src import get_session_manager
            sm = get_session_manager()
            questions = sm.get_session_questions(self.engine.session.id)
            
            # Calculate average score per topic
            topic_scores = {}
            for q in questions:
                if q.score is not None and q.topic:
                    if q.topic not in topic_scores:
                        topic_scores[q.topic] = []
                    topic_scores[q.topic].append(q.score)
            
            # Get topics with high scores
            strong_topics = []
            for topic, scores in topic_scores.items():
                avg = sum(scores) / len(scores)
                if avg >= 0.7:  # 70% or better
                    strong_topics.append((topic, avg))
            
            # Sort by score descending
            strong_topics.sort(key=lambda x: x[1], reverse=True)
            
            if strong_topics:
                for topic, score in strong_topics[:5]:
                    score_pct = int(score * 100)
                    console.print(f"  🌟 [green]{score_pct:3d}% - {topic[:45]}[/green] (Keep it up!)")
            else:
                console.print("[dim]  No topics above 70% yet - keep practicing![/dim]")
    
    def view_recommendations(self):
        """View study recommendations."""
        if not self.engine:
            console.print("[red]✗ No active session.[/red]")
            return
        
        console.print("\n[bold blue]STUDY RECOMMENDATIONS[/bold blue]")
        console.print("─" * 50)
        
        recommendation = self.engine.get_next_question_recommendation()
        
        console.print(Panel(
            f"[cyan]Recommended Topic:[/cyan]\n"
            f"[bold]{recommendation['recommended_topic']}[/bold]\n\n"
            f"[cyan]Suggested Difficulty:[/cyan]\n"
            f"{recommendation['recommended_difficulty']}\n\n"
            f"[cyan]Why This Topic?[/cyan]\n"
            f"{recommendation['reason']}",
            title="📚 Next Question Recommendation",
            border_style="blue"
        ))
        
        if recommendation.get('weak_topics'):
            console.print("\n[yellow]⚠️  Your Weakest Topics (Practice These!):[/yellow]")
            console.print("[dim]Listed from weakest to strongest[/dim]\n")
            for topic, score in recommendation['weak_topics']:
                score_pct = int(score * 100)
                
                # Visual priority
                if score_pct < 40:
                    priority = "🔴 URGENT"
                    color = "red"
                elif score_pct < 70:
                    priority = "🟡 Important"
                    color = "yellow"
                else:
                    priority = "🟢 Minor"
                    color = "green"
                
                console.print(f"  {priority} [{color}]{score_pct:3d}% - {topic[:50]}[/{color}]")
        else:
            console.print("\n[green]✓ No weak areas identified yet![/green]")
    

    def run(self):
        """Run the CLI application."""
        self.clear_screen()
        self.show_banner()
        
        console.print("\n[dim]Welcome! Let's help you study effectively with AI-powered questions.[/dim]\n")
        
        try:
            self.main_menu()
        except KeyboardInterrupt:
            console.print("\n\n[yellow]Interrupted by user[/yellow]")
            console.print("[green]Goodbye! 👋[/green]\n")
        except Exception as e:
            console.print(f"\n[red]✗ Unexpected error: {e}[/red]")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point for CLI."""
    cli = QuestionAskerCLI()
    cli.run()


if __name__ == "__main__":
    main()