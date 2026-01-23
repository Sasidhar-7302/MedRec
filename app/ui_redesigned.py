"""Modern PySide6 UI for Medical Dictation Assistant."""

from __future__ import annotations

import json
import logging
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sounddevice as sd
import soundfile as sf
from PySide6.QtCore import Qt, QTimer, QRectF, Signal, QPropertyAnimation, QEasingCurve, Property
from PySide6.QtGui import QColor, QPainter, QPen, QFont, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLineEdit,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QScrollArea,
    QStackedWidget,
)

from .audio import AudioRecorder
from .config import AppConfig, load_config
from .doctor_assistant import DoctorAssistant
from .doctor_profiles import DoctorProfileManager
from .logging_utils import configure_logging
from .storage import StorageManager
from .summarizer import OllamaSummarizer, SummaryResult
from .prompt_templates import PROMPTS
from .terminology import apply_corrections
from .transcriber import TranscriptionResult, WhisperTranscriber


class AnimatedMicWidget(QWidget):
    """Animated microphone widget with pulse effect."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setFixedSize(180, 180)
        self._recording = False
        self._pulse_value = 0.0
        self._animation = QPropertyAnimation(self, b"pulseValue")
        self._animation.setDuration(1000)
        self._animation.setStartValue(0.0)
        self._animation.setEndValue(1.0)
        self._animation.setEasingCurve(QEasingCurve.InOutSine)
        self._animation.setLoopCount(-1)

    def get_pulse_value(self) -> float:
        return self._pulse_value

    def set_pulse_value(self, value: float) -> None:
        self._pulse_value = value
        self.update()

    pulseValue = Property(float, get_pulse_value, set_pulse_value)

    def set_recording(self, recording: bool) -> None:
        self._recording = recording
        if recording:
            self._animation.start()
        else:
            self._animation.stop()
            self._pulse_value = 0.0
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Background circle
        center = self.rect().center()
        base_radius = 70

        if self._recording:
            # Pulsing effect when recording
            scale = 1.0 + (0.15 * self._pulse_value)
            radius = base_radius * scale
            color = QColor("#EF4444")
            glow_color = QColor("#F87171")
        else:
            radius = base_radius
            color = QColor("#0EA5E9")
            glow_color = QColor("#38BDF8")

        # Draw glow
        painter.setBrush(Qt.NoBrush)
        for i in range(3):
            alpha = 30 - (i * 10)
            glow = QColor(glow_color)
            glow.setAlpha(alpha)
            painter.setPen(QPen(glow, 3))
            painter.drawEllipse(center, radius + (i * 8), radius + (i * 8))

        # Draw main circle
        painter.setBrush(color)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(center, radius, radius)

        # Draw microphone icon
        painter.setBrush(Qt.white)
        mic_width = 24
        mic_height = 36
        mic_rect = QRectF(
            center.x() - mic_width / 2,
            center.y() - mic_height / 2 - 8,
            mic_width,
            mic_height,
        )
        painter.drawRoundedRect(mic_rect, 12, 12)

        # Mic stand
        stand_width = 8
        stand_height = 20
        stand_rect = QRectF(
            center.x() - stand_width / 2,
            center.y() + 12,
            stand_width,
            stand_height,
        )
        painter.drawRoundedRect(stand_rect, 4, 4)

        # Mic base
        base_width = 32
        base_height = 4
        painter.drawRoundedRect(
            center.x() - base_width / 2,
            center.y() + 30,
            base_width,
            base_height,
            2,
            2,
        )


class NavButton(QPushButton):
    """Custom navigation button with icon and text."""

    def __init__(self, text: str, icon_text: str, parent=None):
        super().__init__(parent)
        self.setText(text)
        self._icon_text = icon_text
        self.setCheckable(True)
        self.setFixedHeight(56)
        self.setCursor(Qt.PointingHandCursor)


class FolderCard(QFrame):
    """Folder card widget."""

    clicked = Signal(str)

    def __init__(self, title: str, count: int, color: str, parent=None):
        super().__init__(parent)
        self.title = title
        self._setup_ui(title, count, color)

    def _setup_ui(self, title: str, count: int, color: str) -> None:
        self.setObjectName("FolderCard")
        self.setCursor(Qt.PointingHandCursor)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # Icon
        icon = QFrame()
        icon.setFixedSize(56, 56)
        icon.setStyleSheet(
            f"background-color: {color}; border-radius: 14px; "
            "border: 2px solid rgba(255, 255, 255, 0.5);"
        )
        layout.addWidget(icon)

        # Text content
        text_layout = QVBoxLayout()
        text_layout.setSpacing(4)

        title_label = QLabel(title)
        title_label.setStyleSheet(
            "font-size: 16px; font-weight: 600; color: #0F172A; "
            "letter-spacing: -0.2px;"
        )

        self.count_label = QLabel(f"{count} recordings")
        self.count_label.setStyleSheet(
            "font-size: 13px; color: #64748B; font-weight: 500;"
        )

        text_layout.addWidget(title_label)
        text_layout.addWidget(self.count_label)
        layout.addLayout(text_layout, 1)

        # Arrow
        arrow = QLabel("›")
        arrow.setStyleSheet("font-size: 24px; color: #BDBDBD;")
        layout.addWidget(arrow)

    def update_count(self, count: int) -> None:
        self.count_label.setText(f"{count} recordings")

    def mousePressEvent(self, event) -> None:
        self.clicked.emit(self.title)
        super().mousePressEvent(event)


class RecordingCard(QFrame):
    """Recent recording card."""

    clicked = Signal(str)

    def __init__(
        self,
        title: str,
        time_ago: str,
        duration: str,
        status: str,
        session_path: Optional[str] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.session_path = session_path or ""
        self._setup_ui(title, time_ago, duration, status)

    def _setup_ui(self, title: str, time_ago: str, duration: str, status: str) -> None:
        self.setObjectName("RecordingCard")
        self.setCursor(Qt.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet(
            "font-size: 15px; font-weight: 600; color: #111827; "
            "letter-spacing: -0.1px;"
        )
        title_label.setWordWrap(True)
        layout.addWidget(title_label)

        # Bottom row
        bottom = QHBoxLayout()
        bottom.setSpacing(12)

        time_label = QLabel(time_ago)
        time_label.setStyleSheet("font-size: 12px; color: #6B7280; font-weight: 400;")
        bottom.addWidget(time_label)

        duration_label = QLabel(duration)
        duration_label.setStyleSheet(
            "font-size: 12px; color: #6B7280; font-weight: 400;"
        )
        bottom.addWidget(duration_label)

        bottom.addStretch()

        status_label = QLabel(status)
        if status == "Transcribed":
            status_label.setStyleSheet(
                "background-color: #D1FAE5; color: #065F46; "
                "padding: 4px 12px; border-radius: 10px; font-size: 11px; font-weight: 600; "
                "border: 1px solid #A7F3D0;"
            )
        else:
            status_label.setStyleSheet(
                "background-color: #FED7AA; color: #92400E; "
                "padding: 4px 12px; border-radius: 10px; font-size: 11px; font-weight: 600; "
                "border: 1px solid #FDBA74;"
            )
        bottom.addWidget(status_label)

        layout.addLayout(bottom)

    def mousePressEvent(self, event) -> None:
        target = self.session_path or self.title
        self.clicked.emit(target)
        super().mousePressEvent(event)
class MedRecWindow(QMainWindow):
    """Main application window with modern design."""

    # Signal used to schedule callables on the Qt main thread from worker threads.
    # Signature: (callable, delay_ms)
    schedule_signal = Signal(object, int)

    def __init__(self) -> None:
        super().__init__()
        self.logger = logging.getLogger("medrec.ui")
        self.logger.info("Initializing Medical Dictation window")

        # Initialize components
        self.config: AppConfig = load_config()
        self.audio = AudioRecorder(self.config.audio)
        self.transcriber = WhisperTranscriber(self.config.whisper)
        self.summarizer = OllamaSummarizer(self.config.summarizer)
        self.storage = StorageManager(self.config.storage)
        self.profile_manager = DoctorProfileManager()
        self.doctor_assistant = DoctorAssistant(self.config.summarizer, self.profile_manager)

        if self.config.storage.auto_cleanup:
            self.storage.purge_old_sessions()

        # State variables
        self.active_audio: Optional[Path] = None
        self.transcript_text: str = ""
        self.summary_text: str = ""
        self.last_transcription: Optional[TranscriptionResult] = None
        self.last_summary: Optional[SummaryResult] = None
        self.recent_sessions: List[Dict[str, Any]] = []

        self.executor = ThreadPoolExecutor(max_workers=2)
        self.result_queue: "queue.Queue[Tuple[str, Dict[str, Any]]]" = queue.Queue()
        self.is_recording = False
        self.record_started_at: Optional[float] = None

        self.summarizer_ready = False
        self.auto_launch_attempted = False
        self.auto_started_ollama = False
        self.ollama_process: Optional[subprocess.Popen] = None

        # connect scheduling signal to ensure QTimer usage happens on main thread
        self.schedule_signal.connect(self._on_schedule)
        self._model_pull_inflight: set[str] = set()
        self._model_pull_failed: set[str] = set()

        self.device_map: Dict[str, Optional[int]] = {}
        self.default_device_label: Optional[str] = None
        self._load_devices()
        self.folder_cards: List[FolderCard] = []

        # Timers
        self.record_timer = QTimer(self)
        self.record_timer.timeout.connect(self._tick_timer)
        self.queue_timer = QTimer(self)
        self.queue_timer.timeout.connect(self._process_queue)

        # Chat assistant state/UI placeholders
        self.chat_histories: Dict[str, List[dict]] = {}
        self.chat_history_view: Optional[QTextEdit] = None
        self.chat_input: Optional[QLineEdit] = None
        self.chat_send_btn: Optional[QPushButton] = None
        self.chat_status_label: Optional[QLabel] = None
        self.chat_include_transcript: Optional[QCheckBox] = None
        self.chat_doctor_field: Optional[QLineEdit] = None
        self.summary_style_combo: Optional[QComboBox] = None

        self._init_ui()
        self._refresh_service_status()
        self._render_recent_sessions()
        self.queue_timer.start(200)
        self.logger.info("UI ready")

    def _init_ui(self) -> None:
        self.setWindowTitle("MedRec - Medical Dictation Assistant")
        self.setMinimumSize(1200, 800)
        self._apply_styles()

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Left sidebar
        left_sidebar = self._build_left_sidebar()
        main_layout.addWidget(left_sidebar)

        # Right content area
        right_content = self._build_right_content()
        main_layout.addWidget(right_content, 1)

    def _apply_styles(self) -> None:
        self.setStyleSheet("""
            QMainWindow {
                background-color: #F5F7FA;
            }
            QWidget {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
            }
            QFrame#Sidebar {
                background-color: #FFFFFF;
                border-right: 1px solid #E5E7EB;
            }
            QFrame#ContentArea {
                background-color: #F5F7FA;
            }
            QPushButton#NavButton {
                background-color: transparent;
                border: none;
                text-align: left;
                padding: 16px 24px;
                font-size: 15px;
                font-weight: 500;
                color: #64748B;
                border-radius: 12px;
                margin: 2px 8px;
            }
            QPushButton#NavButton:hover {
                background-color: #F1F5F9;
                color: #334155;
            }
            QPushButton#NavButton:checked {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #E0F2FE, stop:1 #DBEAFE);
                color: #0C4A6E;
                font-weight: 600;
                border-left: 4px solid #0EA5E9;
            }
            QFrame#FolderCard {
                background-color: #FFFFFF;
                border: 1px solid #E5E7EB;
                border-radius: 12px;
            }
            QFrame#FolderCard:hover {
                background-color: #F9FAFB;
                border-color: #D1D5DB;
            }
            QFrame#RecordingCard {
                background-color: #FFFFFF;
                border: 1px solid #E5E7EB;
                border-radius: 12px;
            }
            QFrame#RecordingCard:hover {
                background-color: #F9FAFB;
                border-color: #D1D5DB;
            }
            QFrame#Card {
                background-color: #FFFFFF;
                border: 1px solid #E5E7EB;
                border-radius: 16px;
            }
            QPushButton#PrimaryButton {
                background-color: #3B82F6;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: 600;
                min-height: 44px;
            }
            QPushButton#PrimaryButton:hover {
                background-color: #2563EB;
            }
            QPushButton#PrimaryButton:pressed {
                background-color: #1D4ED8;
            }
            QPushButton#PrimaryButton:disabled {
                background-color: #E5E7EB;
                color: #9CA3AF;
            }
            QPushButton#SecondaryButton {
                background-color: #FFFFFF;
                color: #374151;
                border: 1px solid #E5E7EB;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: 500;
                min-height: 44px;
            }
            QPushButton#SecondaryButton:hover {
                background-color: #F9FAFB;
                border-color: #D1D5DB;
                color: #111827;
            }
            QPushButton#SecondaryButton:pressed {
                background-color: #F3F4F6;
            }
            QPushButton#DangerButton {
                background-color: #EF4444;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: 600;
                min-height: 44px;
            }
            QPushButton#DangerButton:hover {
                background-color: #DC2626;
            }
            QPushButton#DangerButton:pressed {
                background-color: #B91C1C;
            }
            QPushButton#IconButton {
                background-color: #FFFFFF;
                border: 1px solid #E5E7EB;
                border-radius: 8px;
                padding: 8px 16px;
                font-size: 13px;
                font-weight: 500;
                color: #6B7280;
            }
            QPushButton#IconButton:hover {
                background-color: #F9FAFB;
                border-color: #D1D5DB;
                color: #374151;
            }
            QPushButton#IconButton:pressed {
                background-color: #F3F4F6;
            }
            QTextEdit {
                background-color: #FFFFFF;
                border: 1px solid #E5E7EB;
                border-radius: 8px;
                padding: 16px;
                font-size: 14px;
                line-height: 1.6;
                color: #111827;
                selection-background-color: #DBEAFE;
                selection-color: #0C4A6E;
            }
            QTextEdit:focus {
                border-color: #3B82F6;
                background-color: #FFFFFF;
            }
            QComboBox {
                background-color: #FFFFFF;
                border: 1px solid #E5E7EB;
                border-radius: 8px;
                padding: 10px 16px;
                font-size: 14px;
                color: #111827;
                min-height: 20px;
            }
            QComboBox:hover {
                border-color: #D1D5DB;
                background-color: #F9FAFB;
            }
            QComboBox:focus {
                border-color: #3B82F6;
            }
            QComboBox::drop-down {
                border: none;
                width: 30px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid #64748B;
                width: 0px;
                height: 0px;
            }
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                background-color: transparent;
                width: 10px;
                margin: 0;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background-color: #CBD5E1;
                border-radius: 5px;
                min-height: 30px;
                margin: 2px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #94A3B8;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QCheckBox {
                font-size: 14px;
                color: #334155;
                spacing: 10px;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                border: 2px solid #CBD5E1;
                border-radius: 6px;
                background-color: #FFFFFF;
            }
            QCheckBox::indicator:hover {
                border-color: #94A3B8;
            }
            QCheckBox::indicator:checked {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #0EA5E9, stop:1 #0284C7);
                border-color: #0284C7;
            }
            QLineEdit {
                background-color: #FFFFFF;
                border: 1px solid #E5E7EB;
                border-radius: 8px;
                padding: 10px 16px;
                font-size: 14px;
                color: #111827;
                selection-background-color: #DBEAFE;
                selection-color: #0C4A6E;
            }
            QLineEdit:focus {
                border-color: #3B82F6;
                background-color: #FFFFFF;
            }
            QLineEdit:hover {
                border-color: #D1D5DB;
            }
        """)

    def _build_left_sidebar(self) -> QFrame:
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(320)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QWidget()
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(24, 32, 24, 24)

        title = QLabel("MedRec")
        title.setStyleSheet("font-size: 24px; font-weight: 700; color: #1A1A1A;")
        subtitle = QLabel("Medical Dictation")
        subtitle.setStyleSheet("font-size: 13px; color: #757575; margin-top: 4px;")

        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        layout.addWidget(header)

        # Navigation
        self.nav_home = QPushButton("  Record")
        self.nav_home.setObjectName("NavButton")
        self.nav_home.setCheckable(True)
        self.nav_home.setChecked(True)
        self.nav_home.clicked.connect(lambda: self._switch_page(0))

        self.nav_folders = QPushButton("  Folders")
        self.nav_folders.setObjectName("NavButton")
        self.nav_folders.setCheckable(True)
        self.nav_folders.clicked.connect(lambda: self._switch_page(1))

        self.nav_profile = QPushButton("  Profile")
        self.nav_profile.setObjectName("NavButton")
        self.nav_profile.setCheckable(True)
        self.nav_profile.clicked.connect(lambda: self._switch_page(2))

        layout.addWidget(self.nav_home)
        layout.addWidget(self.nav_folders)
        layout.addWidget(self.nav_profile)

        layout.addStretch()

        # System status at bottom
        status_widget = QWidget()
        status_layout = QVBoxLayout(status_widget)
        status_layout.setContentsMargins(24, 16, 24, 24)

        status_title = QLabel("System Status")
        status_title.setStyleSheet("font-size: 13px; font-weight: 600; color: #757575; margin-bottom: 8px;")
        status_layout.addWidget(status_title)

        self.whisper_status = self._create_status_row("Whisper", "Checking...")
        self.summarizer_status = self._create_status_row("Summarizer", "Checking...")

        status_layout.addWidget(self.whisper_status)
        status_layout.addWidget(self.summarizer_status)

        self.launch_ollama_btn = QPushButton("Start Summarizer")
        self.launch_ollama_btn.setObjectName("SecondaryButton")
        self.launch_ollama_btn.clicked.connect(lambda: self._start_ollama_server(manual=True))
        status_layout.addWidget(self.launch_ollama_btn)
        self.launching_summarizer = False

        layout.addWidget(status_widget)

        return sidebar

    def _create_status_row(self, label: str, status: str) -> QWidget:
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 4, 0, 4)

        label_widget = QLabel(label)
        label_widget.setStyleSheet("font-size: 13px; color: #616161;")
        row_layout.addWidget(label_widget)

        row_layout.addStretch()

        status_widget = QLabel(status)
        status_widget.setStyleSheet(
            "font-size: 11px; padding: 3px 10px; "
            "background-color: #FFF3E0; color: #EF6C00; border-radius: 10px;"
        )
        row_layout.addWidget(status_widget)

        row._status_label = status_widget  # Store reference
        return row

    def _build_right_content(self) -> QFrame:
        content = QFrame()
        content.setObjectName("ContentArea")

        layout = QVBoxLayout(content)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(0)

        # Stacked widget for pages
        self.pages = QStackedWidget()

        # Page 0: Record
        self.pages.addWidget(self._build_record_page())

        # Page 1: Folders
        self.pages.addWidget(self._build_folders_page())

        # Page 2: Profile
        self.pages.addWidget(self._build_profile_page())

        layout.addWidget(self.pages)

        return content

    def _build_record_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(20)

        layout.addWidget(self._build_record_header_card())

        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(20)

        # Left column - recorder
        left_col = QVBoxLayout()
        left_col.setContentsMargins(0, 0, 0, 0)
        left_col.setSpacing(16)

        recorder_card = self._build_recorder_card()
        recorder_card.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        left_col.addWidget(recorder_card)

        recent_card = self._build_recent_card()
        recent_card.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        recent_card.setMinimumHeight(200)
        left_col.addWidget(recent_card, 1)

        content_layout.addLayout(left_col, 1)

        # Right column - transcript and summary
        right_col = QVBoxLayout()
        right_col.setContentsMargins(0, 0, 0, 0)
        right_col.setSpacing(16)

        transcript_card = self._build_transcript_card()
        transcript_card.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        transcript_card.setMinimumHeight(200)
        right_col.addWidget(transcript_card, 2)

        summary_card = self._build_summary_card()
        summary_card.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        summary_card.setMinimumHeight(200)
        right_col.addWidget(summary_card, 2)

        doctor_chat_card = self._build_doctor_chat_card()
        doctor_chat_card.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        doctor_chat_card.setMinimumHeight(250)
        right_col.addWidget(doctor_chat_card, 2)

        content_layout.addLayout(right_col, 3)
        layout.addLayout(content_layout, 1)

        return page

    def _build_record_header_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("Card")
        card.setStyleSheet(
            "QFrame#Card {"
            "background-color: #EFF6FF; "
            "border: 1px solid #BFDBFE; "
            "border-radius: 12px;"
            "}"
        )
        layout = QHBoxLayout(card)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        text_col = QVBoxLayout()
        text_col.setSpacing(8)
        title = QLabel("Voice Recorder")
        title.setStyleSheet(
            "font-size: 24px; font-weight: 700; color: #1E40AF; "
            "letter-spacing: -0.3px;"
        )
        subtitle = QLabel("Local-first medical dictation with AI-powered summaries")
        subtitle.setStyleSheet("font-size: 13px; color: #6B7280; font-weight: 400;")
        text_col.addWidget(title)
        text_col.addWidget(subtitle)

        badges = QHBoxLayout()
        badges.setSpacing(8)
        badge_labels = [
            ("✓ HIPAA-safe", "#0A6B4B", "#D1FAE5"),
            ("🎙️ Offline Whisper", "#0C4A6E", "#DBEAFE"),
            ("🤖 AI Summary", "#7C2D12", "#FED7AA"),
        ]
        for label, text_color, bg_color in badge_labels:
            pill = QLabel(label)
            pill.setStyleSheet(
                f"padding: 6px 14px; border-radius: 10px; "
                f"background-color: {bg_color}; color: {text_color}; "
                f"font-size: 12px; font-weight: 600; border: 1px solid {bg_color};"
            )
            badges.addWidget(pill)
        badges.addStretch()
        text_col.addLayout(badges)

        layout.addLayout(text_col, 1)

        hero_icon = QFrame()
        hero_icon.setFixedSize(80, 80)
        hero_icon.setStyleSheet(
            "background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #0EA5E9, stop:1 #0284C7); "
            "border-radius: 20px; border: 3px solid #FFFFFF;"
        )
        layout.addWidget(hero_icon, alignment=Qt.AlignRight | Qt.AlignVCenter)

        return card

    def _build_recorder_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("Card")

        layout = QVBoxLayout(card)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(20)

        # Microphone widget
        mic_layout = QVBoxLayout()
        mic_layout.setAlignment(Qt.AlignCenter)

        self.mic_widget = AnimatedMicWidget()
        mic_layout.addWidget(self.mic_widget, alignment=Qt.AlignCenter)

        # Timer
        self.timer_label = QLabel("00:00")
        self.timer_label.setStyleSheet(
            "font-size: 48px; font-weight: 300; color: #111827; margin-top: 16px; "
            "letter-spacing: 1px;"
        )
        self.timer_label.setAlignment(Qt.AlignCenter)
        mic_layout.addWidget(self.timer_label)

        # Status
        self.status_label = QLabel("Ready to record")
        self.status_label.setStyleSheet(
            "font-size: 13px; color: #6B7280; margin-top: 8px; font-weight: 400;"
        )
        self.status_label.setAlignment(Qt.AlignCenter)
        mic_layout.addWidget(self.status_label)

        layout.addLayout(mic_layout)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)

        self.record_btn = QPushButton("Start Recording")
        self.record_btn.setObjectName("PrimaryButton")
        self.record_btn.clicked.connect(self._handle_record)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("DangerButton")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._handle_stop)

        button_layout.addWidget(self.record_btn)
        button_layout.addWidget(self.stop_btn)
        layout.addLayout(button_layout)

        # Secondary buttons
        secondary_layout = QHBoxLayout()
        secondary_layout.setSpacing(12)

        load_btn = QPushButton("Load Audio")
        load_btn.setObjectName("SecondaryButton")
        load_btn.clicked.connect(self._handle_load)

        settings_btn = QPushButton("Settings")
        settings_btn.setObjectName("SecondaryButton")
        settings_btn.clicked.connect(self._open_settings)

        secondary_layout.addWidget(load_btn)
        secondary_layout.addWidget(settings_btn)
        layout.addLayout(secondary_layout)

        # Device selector
        device_layout = QHBoxLayout()
        device_layout.setSpacing(12)

        device_label = QLabel("Input device:")
        device_label.setStyleSheet("font-size: 13px; color: #757575;")

        self.device_combo = QComboBox()
        for label in sorted(self.device_map.keys()):
            self.device_combo.addItem(label)
        if self.default_device_label:
            index = self.device_combo.findText(self.default_device_label)
            if index >= 0:
                self.device_combo.setCurrentIndex(index)

        device_layout.addWidget(device_label)
        device_layout.addWidget(self.device_combo, 1)
        layout.addLayout(device_layout)

        # Options
        self.auto_summary_check = QCheckBox("Auto-summarize after transcription")
        self.auto_summary_check.setChecked(True)

        layout.addWidget(self.auto_summary_check)

        return card

    def _build_transcript_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("Card")

        layout = QVBoxLayout(card)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        # Header
        header = QHBoxLayout()
        title = QLabel("Transcript")
        title.setStyleSheet(
            "font-size: 18px; font-weight: 600; color: #111827; "
            "letter-spacing: -0.2px;"
        )
        header.addWidget(title)
        header.addStretch()

        self.summarize_btn = QPushButton("✨ Summarize")
        self.summarize_btn.setObjectName("IconButton")
        self.summarize_btn.setEnabled(False)
        self.summarize_btn.clicked.connect(self._handle_summarize)
        header.addWidget(self.summarize_btn)

        layout.addLayout(header)

        # Text edit
        self.transcript_edit = QTextEdit()
        self.transcript_edit.setReadOnly(True)
        self.transcript_edit.setPlaceholderText("Your transcript will appear here...")
        self.transcript_edit.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.transcript_edit.setMinimumHeight(150)
        try:
            self.transcript_edit.setFont(QFont("SF Mono", 13))
        except:
            self.transcript_edit.setFont(QFont("Consolas", 13))

        layout.addWidget(self.transcript_edit, 1)

        return card

    def _build_summary_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("Card")

        layout = QVBoxLayout(card)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        # Header
        header = QHBoxLayout()
        title = QLabel("AI Summary")
        title.setStyleSheet(
            "font-size: 18px; font-weight: 600; color: #111827; "
            "letter-spacing: -0.2px;"
        )
        header.addWidget(title)
        header.addStretch()

        self.copy_btn = QPushButton("📋 Copy")
        self.copy_btn.setObjectName("IconButton")
        self.copy_btn.setEnabled(False)
        self.copy_btn.clicked.connect(self._copy_summary)
        header.addWidget(self.copy_btn)

        layout.addLayout(header)

        # Summary style selector
        format_row = QHBoxLayout()
        format_label = QLabel("Format")
        format_label.setStyleSheet("font-size: 13px; color: #6B7280; font-weight: 500;")
        format_row.addWidget(format_label)

        self.summary_style_combo = QComboBox()
        for style_name in PROMPTS.keys():
            self.summary_style_combo.addItem(style_name)
        default_style = self.config.ui.default_summary_format or self.config.summarizer.prompt_style
        if default_style in PROMPTS:
            index = self.summary_style_combo.findText(default_style)
            if index >= 0:
                self.summary_style_combo.setCurrentIndex(index)
        format_row.addWidget(self.summary_style_combo, 1)
        layout.addLayout(format_row)

        # Text edit
        self.summary_edit = QTextEdit()
        self.summary_edit.setReadOnly(True)
        self.summary_edit.setPlaceholderText("AI-generated summary will appear here...")
        self.summary_edit.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.summary_edit.setMinimumHeight(150)
        try:
            self.summary_edit.setFont(QFont("SF Mono", 13))
        except:
            self.summary_edit.setFont(QFont("Consolas", 13))

        layout.addWidget(self.summary_edit, 1)

        return card

    def _build_doctor_chat_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("Card")

        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        title = QLabel("Doctor Coaching Assistant")
        title.setStyleSheet("font-size: 18px; font-weight: 600; color: #111827;")
        layout.addWidget(title)

        doctor_row = QHBoxLayout()
        doctor_label = QLabel("Doctor ID:")
        doctor_label.setStyleSheet("font-size: 13px; color: #6B7280; font-weight: 500;")
        doctor_row.addWidget(doctor_label)

        self.chat_doctor_field = QLineEdit()
        self.chat_doctor_field.setPlaceholderText("dr_default")
        self.chat_doctor_field.textChanged.connect(self._handle_chat_doctor_changed)
        doctor_row.addWidget(self.chat_doctor_field, 1)

        import_btn = QPushButton("Import Notes")
        import_btn.setObjectName("SecondaryButton")
        import_btn.clicked.connect(self._handle_chat_import_notes)
        doctor_row.addWidget(import_btn)

        layout.addLayout(doctor_row)

        self.chat_include_transcript = QCheckBox("Include current transcript context")
        self.chat_include_transcript.setChecked(True)
        layout.addWidget(self.chat_include_transcript)

        self.chat_history_view = QTextEdit()
        self.chat_history_view.setReadOnly(True)
        self.chat_history_view.setMinimumHeight(150)
        self.chat_history_view.setMaximumHeight(300)
        self.chat_history_view.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.chat_history_view.setStyleSheet("background-color: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 8px;")
        layout.addWidget(self.chat_history_view, 1)

        input_row = QHBoxLayout()
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Ask for guidance or paste a corrected summary…")
        input_row.addWidget(self.chat_input, 1)

        self.chat_send_btn = QPushButton("Send")
        self.chat_send_btn.setObjectName("PrimaryButton")
        self.chat_send_btn.clicked.connect(self._handle_chat_send)
        input_row.addWidget(self.chat_send_btn)

        layout.addLayout(input_row)

        self.chat_status_label = QLabel("Ready for instructions.")
        self.chat_status_label.setStyleSheet("font-size: 12px; color: #6B7280;")
        layout.addWidget(self.chat_status_label)

        return card

    def _build_recent_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("Card")

        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        # Header
        title = QLabel("Recent Recordings")
        title.setStyleSheet("font-size: 16px; font-weight: 600; color: #111827;")
        layout.addWidget(title)

        # Scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        scroll.setMinimumHeight(150)

        scroll_widget = QWidget()
        self.recent_layout = QVBoxLayout(scroll_widget)
        self.recent_layout.setContentsMargins(0, 0, 0, 0)
        self.recent_layout.setSpacing(12)
        self.recent_layout.addStretch()

        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll, 1)

        return card

    # ------------------------------------------------------------------ Doctor chat helpers
    def _active_doctor_id(self) -> str:
        if not self.chat_doctor_field:
            return "default_doctor"
        value = self.chat_doctor_field.text().strip()
        return value or "default_doctor"

    def _handle_chat_doctor_changed(self, _: str) -> None:
        doctor_id = self._active_doctor_id()
        self._render_chat_history(doctor_id)

    def _render_chat_history(self, doctor_id: str) -> None:
        if not self.chat_history_view:
            return
        history = self.chat_histories.get(doctor_id, [])
        if not history:
            self.chat_history_view.setPlainText("No interactions yet.")
            return
        lines: List[str] = []
        for turn in history[-20:]:
            role = "You" if turn.get("role") == "doctor" else "Assistant"
            lines.append(f"{role}: {turn.get('content', '').strip()}")
        self.chat_history_view.setPlainText("\n\n".join(lines))
        self.chat_history_view.verticalScrollBar().setValue(self.chat_history_view.verticalScrollBar().maximum())

    def _handle_chat_import_notes(self) -> None:
        doctor_id = self._active_doctor_id()
        files, _ = QFileDialog.getOpenFileNames(self, "Select approved notes", "", "Text Files (*.txt)")
        if not files:
            return
        imported = 0
        for file_path in files:
            path = Path(file_path)
            try:
                content = path.read_text(encoding="utf-8").strip()
            except OSError as exc:
                self.logger.warning("note_import_failed | path=%s | error=%s", path, exc)
                continue
            if not content:
                continue
            self.profile_manager.add_note(
                doctor_id=doctor_id,
                content=content,
                title=path.stem.replace("_", " "),
            )
            imported += 1
        if self.chat_status_label:
            self.chat_status_label.setText(f"Imported {imported} notes for {doctor_id}")

    def _handle_chat_send(self) -> None:
        if not self.chat_input or not self.chat_send_btn:
            return
        doctor_id = self._active_doctor_id()
        message = self.chat_input.text().strip()
        if not message:
            return

        self._ensure_summarizer_service()
        if not self.summarizer_ready:
            if self.chat_status_label:
                self.chat_status_label.setText("Summarizer offline. Launch Ollama from the status panel.")
            QMessageBox.warning(self, "Summarizer offline", "Start the Ollama service before chatting.")
            return

        include_transcript = bool(self.chat_include_transcript and self.chat_include_transcript.isChecked())
        transcript = self.transcript_text if include_transcript else ""
        history = list(self.chat_histories.get(doctor_id, []))
        history.append({"role": "doctor", "content": message})
        self.chat_histories[doctor_id] = history
        self._render_chat_history(doctor_id)
        self.chat_input.clear()
        self.chat_send_btn.setEnabled(False)
        if self.chat_status_label:
            self.chat_status_label.setText("Assistant is composing a reply…")
        self._submit_task("doctor_chat", self._doctor_chat_worker, doctor_id, message, transcript, history)

    def _doctor_chat_worker(
        self, doctor_id: str, message: str, transcript: str, history: List[dict]
    ) -> Dict[str, Any]:
        result = self.doctor_assistant.respond(doctor_id, message, transcript=transcript, history=history)
        return {"doctor_id": doctor_id, "text": result.summary, "runtime": result.runtime_s}

    def _handle_doctor_chat_result(self, data: Dict[str, Any]) -> None:
        doctor_id = data["doctor_id"]
        reply = data.get("text", "").strip()
        history = list(self.chat_histories.get(doctor_id, []))
        history.append({"role": "assistant", "content": reply})
        self.chat_histories[doctor_id] = history
        if self._active_doctor_id() == doctor_id:
            self._render_chat_history(doctor_id)
        if self.chat_status_label:
            runtime = data.get("runtime", 0.0)
            self.chat_status_label.setText(f"Assistant replied in {runtime:.1f}s")
        if self.chat_send_btn:
            self.chat_send_btn.setEnabled(True)

    def _build_folders_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(24)

        # Header
        header = QHBoxLayout()
        title = QLabel("My Folders")
        title.setStyleSheet("font-size: 28px; font-weight: 700; color: #1A1A1A;")
        header.addWidget(title)
        header.addStretch()

        create_btn = QPushButton("+ Create New Folder")
        create_btn.setObjectName("PrimaryButton")
        create_btn.clicked.connect(self._handle_create_folder)
        header.addWidget(create_btn)

        layout.addLayout(header)

        # Subtitle
        subtitle = QLabel("Organize your recordings by patient type or procedure")
        subtitle.setStyleSheet("font-size: 14px; color: #757575; margin-bottom: 8px;")
        layout.addWidget(subtitle)

        # Folders grid
        folders_layout = QVBoxLayout()
        folders_layout.setSpacing(16)

        folder_specs = [
            {"title": "Patient Consultations", "count": 0, "color": "#E3F2FD"},
            {"title": "Surgery Notes", "count": 0, "color": "#E8F5E9"},
            {"title": "Follow-ups", "count": 0, "color": "#FFF3E0"},
        ]

        for spec in folder_specs:
            card = FolderCard(spec["title"], spec["count"], spec["color"])
            card.clicked.connect(self._handle_folder_click)
            self.folder_cards.append(card)
            folders_layout.addWidget(card)

        layout.addLayout(folders_layout)
        layout.addStretch()

        return page

    def _build_profile_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(32)

        # Header
        title = QLabel("Profile")
        title.setStyleSheet("font-size: 28px; font-weight: 700; color: #1A1A1A;")
        layout.addWidget(title)

        subtitle = QLabel("Manage your account and preferences")
        subtitle.setStyleSheet("font-size: 14px; color: #757575;")
        layout.addWidget(subtitle)

        # Profile card
        profile_card = QFrame()
        profile_card.setObjectName("Card")

        card_layout = QVBoxLayout(profile_card)
        card_layout.setContentsMargins(40, 40, 40, 40)
        card_layout.setSpacing(24)

        # Avatar
        avatar_layout = QVBoxLayout()
        avatar_layout.setAlignment(Qt.AlignCenter)

        avatar = QFrame()
        avatar.setFixedSize(100, 100)
        avatar.setStyleSheet(
            "background-color: #E3F2FD; border-radius: 50px; "
            "border: 3px solid #5C6BC0;"
        )
        avatar_layout.addWidget(avatar, alignment=Qt.AlignCenter)

        name = QLabel("Dr. Sarah Johnson")
        name.setStyleSheet("font-size: 24px; font-weight: 600; color: #1A1A1A; margin-top: 16px;")
        name.setAlignment(Qt.AlignCenter)
        avatar_layout.addWidget(name)

        specialty = QLabel("Gastroenterology")
        specialty.setStyleSheet("font-size: 14px; color: #757575; margin-top: 4px;")
        specialty.setAlignment(Qt.AlignCenter)
        avatar_layout.addWidget(specialty)

        card_layout.addLayout(avatar_layout)

        # Info rows
        info_layout = QVBoxLayout()
        info_layout.setSpacing(20)

        email_row = self._create_info_row("Email", "sarah.johnson@hospital.com")
        license_row = self._create_info_row("License Number", "MD123456")

        info_layout.addWidget(email_row)
        info_layout.addWidget(license_row)

        card_layout.addLayout(info_layout)

        layout.addWidget(profile_card)

        # Settings section
        settings_title = QLabel("Settings")
        settings_title.setStyleSheet("font-size: 20px; font-weight: 600; color: #1A1A1A; margin-top: 16px;")
        layout.addWidget(settings_title)

        settings_card = QFrame()
        settings_card.setObjectName("Card")
        settings_layout = QVBoxLayout(settings_card)
        settings_layout.setContentsMargins(24, 24, 24, 24)
        settings_layout.setSpacing(16)

        app_settings_btn = self._create_settings_button("App Settings", "Customize your experience")
        sign_out_btn = self._create_settings_button("Sign Out", "Log out of your account", danger=True)

        settings_layout.addWidget(app_settings_btn)
        settings_layout.addWidget(sign_out_btn)

        layout.addWidget(settings_card)
        layout.addStretch()

        return page

    def _create_info_row(self, label: str, value: str) -> QWidget:
        row = QWidget()
        layout = QVBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        label_widget = QLabel(label)
        label_widget.setStyleSheet("font-size: 12px; color: #757575; font-weight: 500;")

        value_widget = QLabel(value)
        value_widget.setStyleSheet("font-size: 15px; color: #1A1A1A;")

        layout.addWidget(label_widget)
        layout.addWidget(value_widget)

        return row

    def _create_settings_button(self, title: str, subtitle: str, danger: bool = False) -> QPushButton:
        btn = QPushButton()
        btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                text-align: left;
                padding: 16px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #F5F5F5;
            }
        """)

        btn_layout = QVBoxLayout(btn)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(4)

        title_label = QLabel(title)
        if danger:
            title_label.setStyleSheet("font-size: 15px; font-weight: 600; color: #FF5252;")
        else:
            title_label.setStyleSheet("font-size: 15px; font-weight: 600; color: #1A1A1A;")

        subtitle_label = QLabel(subtitle)
        subtitle_label.setStyleSheet("font-size: 13px; color: #757575;")

        btn_layout.addWidget(title_label)
        btn_layout.addWidget(subtitle_label)

        return btn

    def _switch_page(self, index: int) -> None:
        """Switch between pages."""
        self.pages.setCurrentIndex(index)

        # Update nav buttons
        self.nav_home.setChecked(index == 0)
        self.nav_folders.setChecked(index == 1)
        self.nav_profile.setChecked(index == 2)

    def _handle_folder_click(self, folder_name: str) -> None:
        QMessageBox.information(self, "Folder", f"Opening {folder_name}...")

    def _handle_create_folder(self) -> None:
        QMessageBox.information(
            self,
            "Create Folder",
            "Folder management feature coming soon!",
        )

    # ------------------------------------------------------------------ Recording actions
    def _handle_record(self) -> None:
        self._ensure_summarizer_service()
        temp_dir = Path(self.config.storage.root) / "tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        audio_path = temp_dir / f"recording_{time.strftime('%Y%m%d_%H%M%S')}.wav"

        try:
            device_label = self.device_combo.currentText()
            self.audio.config.input_device = self.device_map.get(device_label)
            self.audio.start(audio_path)
        except Exception as exc:
            QMessageBox.critical(self, "Audio Error", str(exc))
            self.logger.error(f"Record start failed: {exc}")
            return

        self.active_audio = audio_path
        self.record_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.is_recording = True
        self.record_started_at = time.time()
        self.record_timer.start(100)
        self.mic_widget.set_recording(True)
        self.status_label.setText("Recording in progress...")

    def _handle_stop(self) -> None:
        self.audio.stop()
        self.record_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.is_recording = False
        self.record_timer.stop()
        self.mic_widget.set_recording(False)
        self.timer_label.setText("00:00")

        file_size = None
        try:
            file_size = self.audio.last_file_size
        except:
            file_size = None

        if self.active_audio and self.active_audio.exists() and (file_size or 0) > 1024:
            self.status_label.setText("Transcribing audio...")
            self._start_transcription(self.active_audio)
        else:
            self.status_label.setText("Recording cancelled - no audio captured")

    def _handle_load(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "", "Audio Files (*.wav *.mp3 *.m4a)"
        )
        if not file_path:
            return

        self.active_audio = Path(file_path)
        self.status_label.setText("Transcribing audio...")
        self._start_transcription(self.active_audio)

    def _handle_summarize(self) -> None:
        if not self.transcript_text.strip():
            QMessageBox.warning(self, "No Transcript", "Please transcribe audio first.")
            return

        self._ensure_summarizer_service()
        if not self.summarizer_ready:
            self.status_label.setText("Summarizer offline. Launch Ollama from the status panel.")
            QMessageBox.warning(
                self,
                "Summarizer offline",
                "Please start the Ollama service (see System Status) before requesting a summary.",
            )
            return

        self.summarize_btn.setEnabled(False)
        self.copy_btn.setEnabled(False)
        self.summarize_btn.setText("⏳ Processing...")
        self.status_label.setText("🤖 AI is analyzing and generating summary...")

        style = self.config.ui.default_summary_format
        if self.summary_style_combo:
            style = self.summary_style_combo.currentText() or style
        self.logger.info("summary_start | style=%s | transcript_chars=%s", style, len(self.transcript_text))
        self._submit_task("summary", self._summarize_worker, self.transcript_text, style)

    def _copy_summary(self) -> None:
        if not self.summary_text.strip():
            return
        QApplication.clipboard().setText(self.summary_text)
        self.status_label.setText("Summary copied to clipboard")

    def _open_settings(self) -> None:
        QMessageBox.information(
            self,
            "Settings",
            "Edit config.json to customize summary styles, retention policies, and model paths.",
        )

    # ------------------------------------------------------------------ Background tasks
    def _start_transcription(self, path: Path) -> None:
        self.transcript_edit.clear()
        self.summary_edit.clear()
        self.transcript_text = ""
        self.summary_text = ""
        self.summarize_btn.setEnabled(False)
        self.copy_btn.setEnabled(False)
        self.logger.info(
            "transcription_start | path=%s | engine=%s",
            path,
            getattr(self.config.whisper, "engine", "cli"),
        )
        self._submit_task("transcription", self._transcribe_worker, path)

    def _submit_task(self, name: str, func, *args) -> None:
        def runner() -> None:
            try:
                payload = func(*args)
                self.result_queue.put((name, {"ok": True, "data": payload}))
            except Exception as exc:
                self.result_queue.put((name, {"ok": False, "error": str(exc)}))

        self.executor.submit(runner)

    def _transcribe_worker(self, path: Path) -> Dict[str, Any]:
        def progress_callback(partial: str) -> None:
            self.result_queue.put(("transcription_partial", {"text": partial}))

        result = self.transcriber.transcribe(path, progress_cb=progress_callback)
        cleaned = apply_corrections(result.text)
        return {"result": result, "text": cleaned}

    def _summarize_worker(self, transcript: str, style: str) -> SummaryResult:
        return self.summarizer.summarize(transcript, style=style)

    def _process_queue(self) -> None:
        while not self.result_queue.empty():
            kind, payload = self.result_queue.get()
            if kind == "transcription_partial":
                self._handle_transcription_partial(payload.get("text", ""))
                continue

            if not payload.get("ok"):
                message = payload.get("error", "Operation failed")
                QMessageBox.critical(self, "Error", message)
                self.status_label.setText(f"❌ Error: {message}")
                self.logger.error("background_task_error | task=%s | error=%s", kind, message)
                if self.transcript_text.strip():
                    self.summarize_btn.setEnabled(True)
                    self.summarize_btn.setText("✨ Summarize")
                if self.summary_text.strip():
                    self.copy_btn.setEnabled(True)
                if kind == "doctor_chat" and self.chat_send_btn:
                    self.chat_send_btn.setEnabled(True)
                    if self.chat_status_label:
                        self.chat_status_label.setText(f"❌ Assistant error: {message}")
                continue

            data = payload["data"]
            if kind == "transcription":
                self._handle_transcription_result(data)
            elif kind == "summary":
                self._handle_summary_result(data)
            elif kind == "doctor_chat":
                self._handle_doctor_chat_result(data)

    def _handle_transcription_partial(self, raw_text: str) -> None:
        text = apply_corrections(raw_text)
        if not text.strip():
            return
        self.transcript_text = text
        self.transcript_edit.setPlainText(text)
        self.status_label.setText("Transcribing…")

    def _handle_transcription_result(self, data: Dict[str, Any]) -> None:
        result: TranscriptionResult = data["result"]
        text = data["text"]
        self.last_transcription = result
        self.transcript_text = text

        if text.strip():
            self.transcript_edit.setPlainText(text)
            self.status_label.setText(
                f"✓ Transcription complete ({result.runtime_s:.1f}s) - {len(text)} characters"
            )
            self.summarize_btn.setEnabled(True)
            self.logger.info(
                "transcription_complete | runtime=%.2fs | chars=%s",
                result.runtime_s,
                len(text),
            )

            if self.auto_summary_check.isChecked():
                self._handle_summarize()
        else:
            self.status_label.setText("⚠ No speech detected - please check microphone")

    def _handle_summary_result(self, result: SummaryResult) -> None:
        self.last_summary = result
        self.summary_text = result.summary
        self.summary_edit.setPlainText(result.summary)
        self.copy_btn.setEnabled(True)
        self.summarize_btn.setEnabled(True)
        self.summarize_btn.setText("✨ Summarize")
        
        model_name = getattr(result, "model_used", self.config.summarizer.model)
        validation_status = "✓" if getattr(result, "validation_passed", True) else "⚠"
        refinement_info = ""
        if getattr(result, "refinement_count", 0) > 0:
            refinement_info = f" (refined {result.refinement_count}x)"
        
        self.status_label.setText(
            f"{validation_status} Summary ready ({result.runtime_s:.1f}s) via {model_name}{refinement_info}"
        )
        self.logger.info(
            "summary_complete | runtime=%.2fs | chars=%s | model=%s | validation=%s | refinements=%d",
            result.runtime_s,
            len(result.summary),
            model_name,
            getattr(result, "validation_passed", True),
            getattr(result, "refinement_count", 0),
        )
        self._persist_session()

    # ------------------------------------------------------------------ Storage
    def _persist_session(self) -> None:
        if not (self.active_audio and self.active_audio.exists()):
            return

        metadata: Dict[str, Any] = {}
        if self.last_transcription:
            metadata["transcriber_runtime_s"] = self.last_transcription.runtime_s
            if self.last_transcription.command:
                metadata["whisper_command"] = " ".join(str(part) for part in self.last_transcription.command)
        if self.last_summary:
            metadata["summarizer_runtime_s"] = self.last_summary.runtime_s
            metadata["summarizer_model"] = getattr(self.last_summary, "model_used", "")

        self.storage.persist(
            audio_file=self.active_audio,
            transcript=self.transcript_text,
            summary=self.summary_text,
            metadata=metadata,
        )
        self._render_recent_sessions()
        self._refresh_folder_counts()

    def _render_recent_sessions(self) -> None:
        # Clear existing
        while self.recent_layout.count() > 1:  # Keep stretch
            item = self.recent_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        sessions = self._load_recent_sessions()
        self.recent_sessions = sessions
        if not sessions:
            empty = QLabel("No recordings yet")
            empty.setStyleSheet("font-size: 13px; color: #9E9E9E; padding: 20px;")
            empty.setAlignment(Qt.AlignCenter)
            self.recent_layout.insertWidget(0, empty)
            return

        for session in sessions:
            card = RecordingCard(
                session["title"],
                session["age"],
                session["duration"],
                session["status"],
                session_path=str(session.get("path", "")),
            )
            card.clicked.connect(self._handle_recent_card_click)
            self.recent_layout.insertWidget(self.recent_layout.count() - 1, card)
        return sessions

    def _handle_recent_card_click(self, target: str) -> None:
        """Load a previous session into the UI when its card is clicked."""
        session_path = Path(target)
        if not session_path.exists():
            QMessageBox.warning(self, "Session missing", f"Session path not found:\n{session_path}")
            return

        # If a file path was provided, look at its parent directory
        if session_path.is_file():
            session_path = session_path.parent

        transcript_path = session_path / "transcript.txt"
        summary_path = session_path / "summary.txt"

        try:
            transcript_text = transcript_path.read_text(encoding="utf-8") if transcript_path.exists() else ""
            summary_text = summary_path.read_text(encoding="utf-8") if summary_path.exists() else ""
        except OSError as exc:
            QMessageBox.critical(self, "Open session failed", str(exc))
            return

        self.transcript_text = transcript_text
        self.summary_text = summary_text
        if self.transcript_edit:
            self.transcript_edit.setPlainText(transcript_text)
        if self.summary_edit:
            self.summary_edit.setPlainText(summary_text)

        # Enable actions based on loaded content
        self.summarize_btn.setEnabled(bool(transcript_text.strip()))
        self.copy_btn.setEnabled(bool(summary_text.strip()))

        # Track audio path if present
        metadata_path = session_path / "metadata.json"
        audio_path: Optional[Path] = None
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                audio_file = metadata.get("audio_file")
                if audio_file:
                    audio_path = Path(audio_file)
            except json.JSONDecodeError:
                pass
        if not audio_path:
            wavs = list(session_path.glob("*.wav"))
            if wavs:
                audio_path = wavs[0]
        if audio_path and audio_path.exists():
            self.active_audio = audio_path

        if self.status_label:
            self.status_label.setText(f"Loaded session {session_path.name}")

    def _load_recent_sessions(self, limit: int = 5) -> List[Dict[str, str]]:
        sessions: List[Dict[str, str]] = []
        if not self.storage.sessions_dir.exists():
            return sessions

        for session_dir in sorted(
            self.storage.sessions_dir.glob(f"{self.config.storage.session_prefix}_*"),
            reverse=True,
        )[:limit]:
            metadata_path = session_dir / "metadata.json"
            metadata: Dict[str, Any] = {}
            if metadata_path.exists():
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    metadata = {}

            created = metadata.get("created_at")
            title = session_dir.name
            if created:
                try:
                    title = datetime.fromisoformat(created).strftime("%b %d, %I:%M %p")
                except ValueError:
                    pass
            duration_s = metadata.get("audio_duration_s")
            if duration_s is None:
                duration_s = self._probe_audio_duration(metadata.get("audio_file"))
            sessions.append(
                {
                    "title": title,
                    "age": self._format_age(created),
                    "duration": self._format_duration(duration_s),
                    "status": "Transcribed" if (session_dir / "summary.txt").exists() else "Processing",
                    "path": session_dir,
                }
            )
        return sessions

    def _probe_audio_duration(self, audio_file: Optional[str]) -> Optional[float]:
        """Probe audio file duration in seconds."""
        if not audio_file:
            return None
        try:
            audio_path = Path(audio_file)
            if not audio_path.exists():
                return None
            with sf.SoundFile(str(audio_path)) as f:
                return len(f) / float(f.samplerate)
        except Exception as exc:
            self.logger.debug("audio_duration_probe_failed | path=%s | error=%s", audio_file, exc)
            return None

    def _format_age(self, created_at: Optional[str]) -> str:
        if not created_at:
            return "Unknown"
        try:
            created = datetime.fromisoformat(created_at)
        except ValueError:
            return created_at

        delta = datetime.now() - created
        if delta.days >= 1:
            return f"{delta.days}d ago"
        hours = delta.seconds // 3600
        if hours >= 1:
            return f"{hours}h ago"
        minutes = delta.seconds // 60
        if minutes >= 1:
            return f"{minutes}m ago"
        return "Just now"

    def _format_duration(self, seconds: Optional[float]) -> str:
        if not seconds or seconds <= 0:
            return "—"
        minutes, secs = divmod(int(seconds), 60)
        return f"{minutes:d}:{secs:02d}"

    def _refresh_folder_counts(self) -> None:
        if not self.folder_cards:
            return

        session_dirs = list(
            self.storage.sessions_dir.glob(f"{self.config.storage.session_prefix}_*")
        )
        total = len(session_dirs)

        counts = [
            total,
            max(total // 3, 0),
            max(total // 4, 0),
        ]

        for card, count in zip(self.folder_cards, counts):
            card.update_count(count)

    # ------------------------------------------------------------------ Utilities
    def _tick_timer(self) -> None:
        if not self.is_recording or not self.record_started_at:
            return
        elapsed = int(time.time() - self.record_started_at)
        minutes, seconds = divmod(elapsed, 60)
        self.timer_label.setText(f"{minutes:02d}:{seconds:02d}")

    def _load_devices(self) -> None:
        self.device_map = {}
        self.default_device_label = None
        try:
            devices = sd.query_devices()
            hostapis = sd.query_hostapis()
        except Exception:
            self.device_map["Default microphone"] = None
            return
        preferred_hosts = self._preferred_host_indices(hostapis)
        seen_counts: Dict[str, int] = {}
        default_input: Optional[int] = None
        try:
            default_input = sd.default.device[0]
        except Exception:
            default_input = None
        for idx, dev in enumerate(devices):
            if dev.get("max_input_channels", 0) <= 0:
                continue
            host_idx = dev.get("hostapi")
            if preferred_hosts and host_idx not in preferred_hosts:
                continue
            label = self._friendly_device_name(dev.get("name", ""))
            if not label:
                continue
            count = seen_counts.get(label, 0)
            seen_counts[label] = count + 1
            if count:
                label = f"{label} #{count + 1}"
            if not self._device_is_available(idx, dev):
                continue
            self.device_map[label] = idx
            if default_input is not None and idx == default_input:
                self.default_device_label = label
        if not self.device_map:
            self.device_map["Default microphone"] = None
            self.default_device_label = "Default microphone"
        elif not self.default_device_label:
            self.default_device_label = next(iter(self.device_map.keys()))

    def _device_is_available(self, idx: int, info: Dict[str, Any]) -> bool:
        try:
            sd.check_input_settings(
                device=idx,
                samplerate=self.config.audio.sample_rate,
                channels=self.config.audio.channels,
            )
            return True
        except sd.PortAudioError:
            rate = info.get("default_samplerate")
            if rate:
                try:
                    sd.check_input_settings(device=idx, samplerate=rate, channels=self.config.audio.channels)
                    return True
                except sd.PortAudioError:
                    return False
            return False
        except Exception:
            return False
        return False

    def _preferred_host_indices(self, hostapis: Any) -> set[int]:
        indices: set[int] = set()
        if not isinstance(hostapis, list):
            return indices
        preferred_names = [
            "Windows WASAPI",
            "Core Audio",
            "ALSA",
            "PulseAudio",
        ]
        for name in preferred_names:
            for idx, info in enumerate(hostapis):
                if info.get("name") == name:
                    return {idx}
        # fallback to default host api if present
        try:
            default_host = sd.default.hostapi
            if default_host is not None:
                return {default_host}
        except Exception:
            pass
        return indices

    def _friendly_device_name(self, raw: str) -> str:
        name = raw.strip()
        if not name:
            return ""
        # Windows Bluetooth devices include brand name after last ';'
        if ";(" in name:
            tail = name.rsplit(";", 1)[-1].strip()
            tail = tail.strip("() ")
            tail = re.sub(r"\)+$", "", tail).strip()
            if tail:
                name = tail
        if "(@" in name:
            name = name.split("(@", 1)[0].strip()
        name = re.sub(r"\s{2,}", " ", name)
        return name

    def _refresh_service_status(self) -> None:
        whisper_ready = Path(self.config.whisper.binary_path).exists() and Path(
            self.config.whisper.model_path
        ).exists()
        self._set_status(self.whisper_status, whisper_ready)

        sum_ready = self.summarizer.health_check()
        self._set_status(self.summarizer_status, sum_ready)
        self.summarizer_ready = sum_ready

        if sum_ready:
            self._model_pull_failed.clear()

        if hasattr(self, "launch_ollama_btn"):
            if sum_ready:
                self.launch_ollama_btn.setText("Summarizer Running")
                self.launch_ollama_btn.setEnabled(False)
                self.launching_summarizer = False
            else:
                if not self.launching_summarizer:
                    self.launch_ollama_btn.setText("Start Summarizer")
                    self.launch_ollama_btn.setEnabled(True)
                else:
                    self.launch_ollama_btn.setEnabled(False)

        if not sum_ready and not self.auto_launch_attempted:
            self.auto_launch_attempted = True
            self._start_ollama_server()

    def _set_status(self, widget: QWidget, ready: bool) -> None:
        status_label = getattr(widget, "_status_label", None)
        if not status_label:
            return

        if ready:
            status_label.setText("✓ Ready")
            status_label.setStyleSheet(
                "font-size: 11px; padding: 4px 12px; font-weight: 600; "
                "background-color: #D1FAE5; color: #065F46; border-radius: 12px; "
                "border: 1px solid #A7F3D0;"
            )
        else:
            status_label.setText("⚠ Offline")
            status_label.setStyleSheet(
                "font-size: 11px; padding: 4px 12px; font-weight: 600; "
                "background-color: #FEE2E2; color: #991B1B; border-radius: 12px; "
                "border: 1px solid #FECACA;"
            )

    def _on_schedule(self, fn: object, delay_ms: int) -> None:
        """Slot to schedule callables on the Qt main thread via QTimer.singleShot.

        Emitted from background threads using schedule_signal.emit(callable, delay_ms).
        """
        try:
            # delay_ms might be 0
            QTimer.singleShot(int(delay_ms), fn)
        except Exception:
            self.logger.exception("Failed to schedule callable on main thread")

    def _ensure_summarizer_service(self) -> None:
        if self.summarizer_ready:
            return
        if not self.auto_launch_attempted:
            self.auto_launch_attempted = True
            self._start_ollama_server()

    def _start_ollama_server(self, manual: bool = False) -> None:
        if self.launching_summarizer:
            return
        if shutil.which("ollama") is None:
            if manual:
                QMessageBox.warning(
                    self,
                    "Summarizer",
                    "Ollama CLI not found. Install it from https://ollama.com/download and retry.",
                )
            return

        self.launching_summarizer = True
        if manual and hasattr(self, "launch_ollama_btn"):
            self.launch_ollama_btn.setText("Starting…")
            self.launch_ollama_btn.setEnabled(False)

        try:
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            proc = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=flags,
            )
            self.ollama_process = proc
            self.auto_started_ollama = True
            self.logger.info("ollama_serve_launched | manual=%s", manual)
            self._ensure_ollama_models()
            QTimer.singleShot(2000, self._refresh_service_status)
        except Exception as exc:
            self.launching_summarizer = False
            self.logger.error(f"Ollama launch failed: {exc}")
            if manual:
                QMessageBox.critical(self, "Summarizer", f"Failed to start Ollama: {exc}")
    def _ensure_ollama_models(self) -> None:
        models = {self.config.summarizer.model}
        fallback = getattr(self.config.summarizer, "fallback_model", None)
        if fallback:
            models.add(fallback)
        for model in models:
            if model:
                self._pull_model_async(model)

    def _pull_model_async(self, model: str, attempt: int = 1) -> None:
        if model in self._model_pull_inflight or model in self._model_pull_failed:
            return
        if shutil.which("ollama") is None:
            return

        self._model_pull_inflight.add(model)

        def worker() -> None:
            try:
                if self._is_model_available(model):
                    self.logger.info("ollama_model_present | model=%s", model)
                    return
                self.logger.info("ollama_pull_start | model=%s | attempt=%s", model, attempt)
                subprocess.run(
                    ["ollama", "pull", model],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True,
                )
                self.logger.info("ollama_pull_complete | model=%s", model)
            except subprocess.CalledProcessError as exc:
                self.logger.warning("ollama_pull_failed | model=%s | error=%s", model, exc)
                if attempt < 3:
                    delay_ms = 2000 * attempt
                    self.schedule_signal.emit(
                        lambda m=model, a=attempt + 1: self._pull_model_async(m, a),
                        delay_ms,
                    )
                else:
                    self._model_pull_failed.add(model)
                    self.schedule_signal.emit(lambda m=model: self._notify_model_pull_failure(m), 0)
            finally:
                self._model_pull_inflight.discard(model)
                self.schedule_signal.emit(self._refresh_service_status, 0)

        threading.Thread(target=worker, daemon=True).start()

    def _is_model_available(self, model: str) -> bool:
        if shutil.which("ollama") is None:
            return False
        try:
            proc = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            self.logger.debug("ollama_list_failed | error=%s", exc)
            return False
        output = proc.stdout or ""
        if model in output:
            return True
        output = output.strip()
        if not output:
            return False
        try:
            data = json.loads(output)
            if isinstance(data, list):
                for entry in data:
                    if isinstance(entry, dict) and model in entry.get("name", ""):
                        return True
        except json.JSONDecodeError:
            pass
        return False

    def _notify_model_pull_failure(self, model: str) -> None:
        message = (
            f"Ollama could not download '{model}'. "
            f"Run `ollama pull {model}` manually when you have network access."
        )
        self.status_label.setText(message)
        self.logger.warning(message)

    def closeEvent(self, event) -> None:
        self.audio.stop()
        self.executor.shutdown(wait=False, cancel_futures=True)
        if self.auto_started_ollama and self.ollama_process:
            if self.ollama_process.poll() is None:
                self.ollama_process.terminate()
        super().closeEvent(event)


def run() -> None:
    configure_logging()
    app = QApplication.instance() or QApplication(sys.argv)
    window = MedRecWindow()
    window.show()
    sys.exit(app.exec())
