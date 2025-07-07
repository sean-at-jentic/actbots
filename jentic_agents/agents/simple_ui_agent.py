"""
Simple UI agent that provides a graphical interface for goal input and execution.
Uses tkinter for a lightweight, cross-platform GUI experience with modern ChatGPT-like design.
"""

import logging
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, font
from typing import Any
import threading
from datetime import datetime

from .base_agent import BaseAgent
from ..reasoners.base_reasoner import ReasoningResult

logger = logging.getLogger(__name__)


class SimpleUIAgent(BaseAgent):
    """
    Simple GUI agent using tkinter with modern ChatGPT-like design.

    Provides a clean, chat-style interface for goal input and execution,
    with modern styling, message bubbles, and smooth user experience.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the simple UI agent."""
        super().__init__(*args, **kwargs)
        self._running = False
        self._history = []
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Setup the modern tkinter GUI components."""
        # Main window
        self.root = tk.Tk()
        self.root.title("ActBots - AI Assistant")
        self.root.geometry("1000x800")
        self.root.configure(bg="#ffffff")
        self.root.minsize(800, 600)

        # Configure modern fonts
        self.font_title = font.Font(family="SF Pro Display", size=20, weight="bold")
        self.font_body = font.Font(family="SF Pro Text", size=14)
        self.font_small = font.Font(family="SF Pro Text", size=12)
        self.font_mono = font.Font(family="SF Mono", size=11)

        # Try to use system fonts, fallback to defaults
        try:
            self.font_title = font.Font(family="Segoe UI", size=20, weight="bold")
            self.font_body = font.Font(family="Segoe UI", size=14)
            self.font_small = font.Font(family="Segoe UI", size=12)
            self.font_mono = font.Font(family="Consolas", size=11)
        except:
            self.font_title = font.Font(family="Arial", size=20, weight="bold")
            self.font_body = font.Font(family="Arial", size=14)
            self.font_small = font.Font(family="Arial", size=12)
            self.font_mono = font.Font(family="Courier", size=11)

        # Configure grid weights
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self._create_header()
        self._create_chat_area()
        self._create_input_area()

        # Bind close event
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        # Focus on input
        self.root.after(100, lambda: self.input_text.focus())

    def _create_header(self) -> None:
        """Create the modern header section."""
        header_frame = tk.Frame(self.root, bg="#ffffff", height=70)
        header_frame.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        header_frame.grid_propagate(False)
        header_frame.grid_columnconfigure(1, weight=1)

        # Add subtle border
        border_frame = tk.Frame(header_frame, bg="#e5e5e5", height=1)
        border_frame.grid(row=1, column=0, columnspan=3, sticky="ew")

        # Title with icon
        title_frame = tk.Frame(header_frame, bg="#ffffff")
        title_frame.grid(row=0, column=0, padx=25, pady=20, sticky="w")

        icon_label = tk.Label(
            title_frame, text="🤖", font=font.Font(size=24), bg="#ffffff", fg="#2563eb"
        )
        icon_label.pack(side="left", padx=(0, 10))

        title_label = tk.Label(
            title_frame,
            text="ActBots",
            font=self.font_title,
            bg="#ffffff",
            fg="#1f2937",
        )
        title_label.pack(side="left")

        subtitle_label = tk.Label(
            title_frame,
            text="AI Assistant",
            font=self.font_small,
            bg="#ffffff",
            fg="#6b7280",
        )
        subtitle_label.pack(side="left", padx=(10, 0))

        # Control buttons with modern style
        button_frame = tk.Frame(header_frame, bg="#ffffff")
        button_frame.grid(row=0, column=2, padx=25, pady=20, sticky="e")

        # Style for modern buttons
        button_style = {
            "font": self.font_small,
            "relief": "flat",
            "borderwidth": 0,
            "padx": 16,
            "pady": 8,
            "cursor": "hand2",
        }

        self.clear_btn = tk.Button(
            button_frame,
            text="Clear Chat",
            bg="#f3f4f6",
            fg="#4b5563",
            activebackground="#e5e7eb",
            activeforeground="#1f2937",
            command=self._clear_chat,
            **button_style,
        )
        self.clear_btn.pack(side="left", padx=5)

        self.history_btn = tk.Button(
            button_frame,
            text="History",
            bg="#f3f4f6",
            fg="#4b5563",
            activebackground="#e5e7eb",
            activeforeground="#1f2937",
            command=self._show_history,
            **button_style,
        )
        self.history_btn.pack(side="left", padx=5)

        self.help_btn = tk.Button(
            button_frame,
            text="Help",
            bg="#f3f4f6",
            fg="#4b5563",
            activebackground="#e5e7eb",
            activeforeground="#1f2937",
            command=self._show_help,
            **button_style,
        )
        self.help_btn.pack(side="left", padx=5)

    def _create_chat_area(self) -> None:
        """Create the main chat area with message bubbles."""
        # Chat container
        chat_frame = tk.Frame(self.root, bg="#f9fafb")
        chat_frame.grid(row=1, column=0, sticky="nsew", padx=0, pady=0)
        chat_frame.grid_rowconfigure(0, weight=1)
        chat_frame.grid_columnconfigure(0, weight=1)

        # Chat scrollable area
        self.chat_canvas = tk.Canvas(
            chat_frame, bg="#f9fafb", highlightthickness=0, borderwidth=0
        )

        self.chat_scrollbar = ttk.Scrollbar(
            chat_frame, orient="vertical", command=self.chat_canvas.yview
        )

        self.chat_scrollable_frame = tk.Frame(self.chat_canvas, bg="#f9fafb")

        # Configure scrolling
        self.chat_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.chat_canvas.configure(
                scrollregion=self.chat_canvas.bbox("all")
            ),
        )

        self.chat_canvas.create_window(
            (0, 0), window=self.chat_scrollable_frame, anchor="nw"
        )
        self.chat_canvas.configure(yscrollcommand=self.chat_scrollbar.set)

        # Grid the chat components
        self.chat_canvas.grid(row=0, column=0, sticky="nsew")
        self.chat_scrollbar.grid(row=0, column=1, sticky="ns")

        # Mouse wheel scrolling
        self.chat_canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.chat_scrollable_frame.bind("<MouseWheel>", self._on_mousewheel)

        # Welcome message
        self._add_bot_message(
            "👋 Welcome to ActBots! I'm your AI assistant. What can I help you with today?"
        )

    def _create_input_area(self) -> None:
        """Create the modern input area at the bottom."""
        # Input container with shadow effect
        input_container = tk.Frame(self.root, bg="#ffffff")
        input_container.grid(row=2, column=0, sticky="ew", padx=0, pady=0)
        input_container.grid_columnconfigure(0, weight=1)

        # Top border for separation
        border_frame = tk.Frame(input_container, bg="#e5e5e5", height=1)
        border_frame.grid(row=0, column=0, sticky="ew")

        # Input frame
        input_frame = tk.Frame(input_container, bg="#ffffff")
        input_frame.grid(row=1, column=0, sticky="ew", padx=25, pady=20)
        input_frame.grid_columnconfigure(0, weight=1)

        # Text input with modern styling
        input_wrapper = tk.Frame(
            input_frame, bg="#f3f4f6", relief="flat", borderwidth=1
        )
        input_wrapper.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        input_wrapper.grid_columnconfigure(0, weight=1)

        self.input_text = tk.Text(
            input_wrapper,
            height=3,
            wrap=tk.WORD,
            font=self.font_body,
            bg="#f3f4f6",
            fg="#1f2937",
            relief="flat",
            borderwidth=0,
            padx=16,
            pady=12,
            insertbackground="#2563eb",
            selectbackground="#dbeafe",
        )
        self.input_text.grid(row=0, column=0, sticky="ew")

        # Placeholder text
        self.input_text.insert(1.0, "Type your message here... (Ctrl+Enter to send)")
        self.input_text.configure(fg="#9ca3af")

        # Bind events for placeholder behavior
        self.input_text.bind("<FocusIn>", self._on_input_focus_in)
        self.input_text.bind("<FocusOut>", self._on_input_focus_out)
        self.input_text.bind("<KeyPress>", self._on_input_key_press)

        # Button area
        button_area = tk.Frame(input_frame, bg="#ffffff")
        button_area.grid(row=1, column=0, sticky="ew")
        button_area.grid_columnconfigure(0, weight=1)

        # Status and send button
        status_frame = tk.Frame(button_area, bg="#ffffff")
        status_frame.grid(row=0, column=0, sticky="w")

        self.status_dot = tk.Label(
            status_frame, text="●", font=font.Font(size=12), fg="#22c55e", bg="#ffffff"
        )
        self.status_dot.pack(side="left", padx=(0, 8))

        self.status_label = tk.Label(
            status_frame, text="Ready", font=self.font_small, fg="#6b7280", bg="#ffffff"
        )
        self.status_label.pack(side="left")

        # Send button
        self.send_btn = tk.Button(
            button_area,
            text="Send",
            font=font.Font(family=self.font_body["family"], size=14, weight="bold"),
            bg="#2563eb",
            fg="#ffffff",
            activebackground="#1d4ed8",
            activeforeground="#ffffff",
            relief="flat",
            borderwidth=0,
            padx=24,
            pady=10,
            cursor="hand2",
            command=self._send_message,
        )
        self.send_btn.grid(row=0, column=1, sticky="e")

        # Bind Enter key
        self.input_text.bind("<Control-Return>", lambda e: self._send_message())
        self.input_text.bind(
            "<Command-Return>", lambda e: self._send_message()
        )  # For Mac

    def _on_input_focus_in(self, event):
        """Handle input focus in event."""
        if (
            self.input_text.get(1.0, tk.END).strip()
            == "Type your message here... (Ctrl+Enter to send)"
        ):
            self.input_text.delete(1.0, tk.END)
            self.input_text.configure(fg="#1f2937")

    def _on_input_focus_out(self, event):
        """Handle input focus out event."""
        if not self.input_text.get(1.0, tk.END).strip():
            self.input_text.insert(
                1.0, "Type your message here... (Ctrl+Enter to send)"
            )
            self.input_text.configure(fg="#9ca3af")

    def _on_input_key_press(self, event):
        """Handle key press in input."""
        if (
            self.input_text.get(1.0, tk.END).strip()
            == "Type your message here... (Ctrl+Enter to send)"
        ):
            self.input_text.delete(1.0, tk.END)
            self.input_text.configure(fg="#1f2937")

    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling."""
        self.chat_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _add_user_message(self, message: str) -> None:
        """Add a user message bubble to the chat."""
        timestamp = datetime.now().strftime("%H:%M")

        # Message container
        msg_container = tk.Frame(self.chat_scrollable_frame, bg="#f9fafb")
        msg_container.pack(fill="x", padx=25, pady=(10, 5))
        msg_container.grid_columnconfigure(0, weight=1)

        # User message bubble (right side)
        bubble_frame = tk.Frame(msg_container, bg="#f9fafb")
        bubble_frame.grid(row=0, column=0, sticky="e")

        # Message bubble
        bubble = tk.Frame(bubble_frame, bg="#2563eb", relief="flat", borderwidth=0)
        bubble.pack(side="right", padx=(100, 0))

        # Message text
        msg_label = tk.Label(
            bubble,
            text=message,
            font=self.font_body,
            bg="#2563eb",
            fg="#ffffff",
            wraplength=400,
            justify="left",
            anchor="w",
            padx=16,
            pady=12,
        )
        msg_label.pack()

        # Timestamp
        time_label = tk.Label(
            bubble_frame,
            text=timestamp,
            font=font.Font(family=self.font_small["family"], size=10),
            bg="#f9fafb",
            fg="#9ca3af",
        )
        time_label.pack(side="right", padx=(0, 5), pady=(5, 0))

        self._scroll_to_bottom()

    def _add_bot_message(self, message: str, message_type: str = "info") -> None:
        """Add a bot message bubble to the chat."""
        timestamp = datetime.now().strftime("%H:%M")

        # Message container
        msg_container = tk.Frame(self.chat_scrollable_frame, bg="#f9fafb")
        msg_container.pack(fill="x", padx=25, pady=(10, 5))

        # Bot message bubble (left side)
        bubble_frame = tk.Frame(msg_container, bg="#f9fafb")
        bubble_frame.pack(fill="x")

        # Bot avatar
        avatar = tk.Label(
            bubble_frame, text="🤖", font=font.Font(size=20), bg="#f9fafb"
        )
        avatar.pack(side="left", padx=(0, 12), pady=(8, 0), anchor="n")

        # Message bubble
        bubble_color = "#ffffff"
        text_color = "#1f2937"

        if message_type == "error":
            bubble_color = "#fef2f2"
            text_color = "#dc2626"
        elif message_type == "success":
            bubble_color = "#f0fdf4"
            text_color = "#16a34a"
        elif message_type == "warning":
            bubble_color = "#fffbeb"
            text_color = "#d97706"

        bubble = tk.Frame(
            bubble_frame,
            bg=bubble_color,
            relief="flat",
            borderwidth=1,
            highlightbackground="#e5e7eb",
            highlightthickness=1,
        )
        bubble.pack(side="left", fill="x", expand=True, padx=(0, 100))

        # Message text
        msg_label = tk.Label(
            bubble,
            text=message,
            font=self.font_body,
            bg=bubble_color,
            fg=text_color,
            wraplength=500,
            justify="left",
            anchor="w",
            padx=16,
            pady=12,
        )
        msg_label.pack(fill="x")

        # Timestamp
        time_label = tk.Label(
            bubble_frame,
            text=timestamp,
            font=font.Font(family=self.font_small["family"], size=10),
            bg="#f9fafb",
            fg="#9ca3af",
        )
        time_label.pack(side="left", padx=(8, 0), pady=(5, 0))

        self._scroll_to_bottom()

    def _scroll_to_bottom(self) -> None:
        """Scroll chat to bottom."""
        self.root.update_idletasks()
        self.chat_canvas.yview_moveto(1.0)

    def _update_status(self, message: str, status_type: str = "ready") -> None:
        """Update the status indicator."""
        colors = {
            "ready": "#22c55e",
            "processing": "#f59e0b",
            "error": "#ef4444",
            "success": "#22c55e",
        }

        self.status_dot.configure(fg=colors.get(status_type, "#22c55e"))
        self.status_label.configure(text=message)
        self.root.update_idletasks()

    def _clear_chat(self) -> None:
        """Clear the chat area."""
        for widget in self.chat_scrollable_frame.winfo_children():
            widget.destroy()
        self._add_bot_message("Chat cleared. How can I help you?")

    def _send_message(self) -> None:
        """Send the user's message."""
        message = self.input_text.get(1.0, tk.END).strip()

        # Ignore placeholder text
        if not message or message == "Type your message here... (Ctrl+Enter to send)":
            return

        # Add user message to chat
        self._add_user_message(message)

        # Clear input
        self.input_text.delete(1.0, tk.END)
        self._on_input_focus_out(None)  # Restore placeholder

        # Disable send button and update status
        self.send_btn.configure(state=tk.DISABLED, bg="#9ca3af")
        self._update_status("Processing...", "processing")

        # Start processing in thread
        thread = threading.Thread(target=self._process_goal_thread, args=(message,))
        thread.daemon = True
        thread.start()

    def _process_goal_thread(self, goal: str) -> None:
        """Process goal in background thread."""
        try:
            # Add to history
            self._history.append(
                {
                    "goal": goal,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

            # Process the goal
            result = self.process_goal(goal)

            # Handle output on main thread
            self.root.after(0, lambda: self._handle_goal_result(goal, result))

        except Exception as e:
            error_msg = f"Error processing goal: {str(e)}"
            logger.error(error_msg)
            self.root.after(0, lambda: self._handle_goal_error(goal, error_msg))

    def _handle_goal_result(self, goal: str, result: ReasoningResult) -> None:
        """Handle goal processing result on main thread."""
        try:
            # Handle output
            self.handle_output(result)

            # Acknowledge successful processing
            self.inbox.acknowledge_goal(goal)

            # Update status
            self._update_status("Ready", "ready")

        except Exception as e:
            error_msg = f"Error handling result: {str(e)}"
            self._handle_goal_error(goal, error_msg)

        finally:
            # Re-enable send button
            self.send_btn.configure(state=tk.NORMAL, bg="#2563eb")

    def _handle_goal_error(self, goal: str, error_msg: str) -> None:
        """Handle goal processing error on main thread."""
        self._add_bot_message(
            f"I encountered an error while processing your request:\n\n{error_msg}",
            "error",
        )

        # Reject the goal
        try:
            self.inbox.reject_goal(goal, error_msg)
        except Exception as e:
            logger.error(f"Error rejecting goal: {e}")

        # Update status
        self._update_status("Error occurred", "error")

        # Re-enable send button
        self.send_btn.configure(state=tk.NORMAL, bg="#2563eb")

    def _show_help(self) -> None:
        """Show modern help dialog."""
        help_window = tk.Toplevel(self.root)
        help_window.title("Help - ActBots")
        help_window.geometry("600x500")
        help_window.configure(bg="#ffffff")
        help_window.resizable(False, False)

        # Center the window
        help_window.transient(self.root)
        help_window.grab_set()

        # Header
        header_frame = tk.Frame(help_window, bg="#f8fafc")
        header_frame.pack(fill="x", padx=0, pady=0)

        header_label = tk.Label(
            header_frame,
            text="🤖 ActBots Help",
            font=font.Font(family=self.font_title["family"], size=18, weight="bold"),
            bg="#f8fafc",
            fg="#1f2937",
            pady=20,
        )
        header_label.pack()

        # Content
        content_frame = tk.Frame(help_window, bg="#ffffff")
        content_frame.pack(fill="both", expand=True, padx=30, pady=20)

        help_text = """How to Use ActBots:

1. Type your goal or question in the message box below
2. Press Enter or click 'Send' to submit
3. Wait for the AI assistant to process and respond
4. View the conversation history in the chat area above

Example Goals:
• "Find information about machine learning"
• "Search for Python tutorials on GitHub"
• "Help me plan a project roadmap"
• "Summarize recent AI developments"

Tips:
• Be specific with your requests for better results
• Use Ctrl+Enter (or Cmd+Enter on Mac) as a shortcut
• Click 'Clear Chat' to start a fresh conversation
• Check 'History' to see your recent requests

The AI assistant can help with research, planning, coding questions, and much more!"""

        help_label = tk.Label(
            content_frame,
            text=help_text,
            font=self.font_body,
            bg="#ffffff",
            fg="#374151",
            justify="left",
            anchor="nw",
        )
        help_label.pack(fill="both", expand=True)

        # Close button
        close_btn = tk.Button(
            help_window,
            text="Got it",
            font=self.font_body,
            bg="#2563eb",
            fg="#ffffff",
            activebackground="#1d4ed8",
            relief="flat",
            borderwidth=0,
            padx=30,
            pady=10,
            command=help_window.destroy,
        )
        close_btn.pack(pady=20)

    def _show_history(self) -> None:
        """Show modern history dialog."""
        if not self._history:
            messagebox.showinfo("History", "No goals processed yet.")
            return

        history_window = tk.Toplevel(self.root)
        history_window.title("History - ActBots")
        history_window.geometry("700x500")
        history_window.configure(bg="#ffffff")

        # Center the window
        history_window.transient(self.root)
        history_window.grab_set()

        # Header
        header_frame = tk.Frame(history_window, bg="#f8fafc")
        header_frame.pack(fill="x")

        header_label = tk.Label(
            header_frame,
            text="📋 Recent Goals",
            font=font.Font(family=self.font_title["family"], size=18, weight="bold"),
            bg="#f8fafc",
            fg="#1f2937",
            pady=20,
        )
        header_label.pack()

        # History content
        content_frame = tk.Frame(history_window, bg="#ffffff")
        content_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Scrollable text area
        text_frame = tk.Frame(content_frame, bg="#ffffff")
        text_frame.pack(fill="both", expand=True)

        history_text = scrolledtext.ScrolledText(
            text_frame,
            wrap=tk.WORD,
            font=self.font_body,
            bg="#f9fafb",
            fg="#374151",
            relief="flat",
            borderwidth=1,
            highlightbackground="#e5e7eb",
            highlightthickness=1,
            padx=20,
            pady=15,
        )
        history_text.pack(fill="both", expand=True)

        # Populate history
        history_text.insert(tk.END, "Recent Goals (most recent first):\n\n")

        for i, entry in enumerate(reversed(self._history[-10:]), 1):
            history_text.insert(tk.END, f"{i}. {entry['timestamp']}\n")
            history_text.insert(tk.END, f"   Goal: {entry['goal']}\n\n")

        history_text.configure(state=tk.DISABLED)

        # Close button
        close_btn = tk.Button(
            history_window,
            text="Close",
            font=self.font_body,
            bg="#6b7280",
            fg="#ffffff",
            activebackground="#4b5563",
            relief="flat",
            borderwidth=0,
            padx=30,
            pady=10,
            command=history_window.destroy,
        )
        close_btn.pack(pady=20)

    def _on_closing(self) -> None:
        """Handle window closing event."""
        if messagebox.askokcancel("Quit", "Are you sure you want to quit ActBots?"):
            self.stop()

    def spin(self) -> None:
        """
        Main agent loop - starts the tkinter main loop.

        Continues until the window is closed.
        """
        logger.info("Starting SimpleUIAgent")
        self._running = True

        try:
            # Start tkinter main loop
            self.root.mainloop()

        finally:
            self._running = False
            self.inbox.close()
            logger.info("SimpleUIAgent stopped")

    def handle_input(self, input_data: Any) -> str:
        """
        Handle input from the user/environment.

        For UI agent, this processes the goal text from the input area.

        Args:
            input_data: Raw input data from the interface

        Returns:
            Processed goal string
        """
        return str(input_data).strip()

    def handle_output(self, result: ReasoningResult) -> None:
        """
        Handle output to the user/environment.

        Formats and displays the reasoning result in the chat area.

        Args:
            result: Reasoning result to present
        """
        if result.success:
            # Success output
            response = result.final_answer

            if result.tool_calls:
                response += f"\n\n✅ Completed using {len(result.tool_calls)} tool(s) in {result.iterations} iteration(s)"

            self._add_bot_message(response, "success")
        else:
            # Error output
            error_response = (
                f"I wasn't able to complete that request.\n\n{result.final_answer}"
            )

            if result.error_message:
                error_response += f"\n\nError details: {result.error_message}"

            self._add_bot_message(error_response, "error")

    def should_continue(self) -> bool:
        """
        Determine if the agent should continue processing.

        Returns:
            True if agent should keep running, False to stop
        """
        return self._running

    def stop(self) -> None:
        """Stop the agent gracefully."""
        self._running = False
        if hasattr(self, "root"):
            try:
                self.root.quit()
                self.root.destroy()
            except tk.TclError:
                pass  # Window already closed
