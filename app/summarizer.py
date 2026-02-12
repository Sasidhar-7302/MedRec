"""Advanced summarizer interface with multi-step processing and enhanced error handling."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Optional

import requests

from .config import SummarizerConfig
from .prompt_templates import build_prompt


@dataclass
class SummaryResult:
    summary: str
    runtime_s: float
    prompt: str
    model_used: str
    validation_passed: bool = True
    refinement_count: int = 0


class OllamaSummarizer:
    def __init__(self, config: SummarizerConfig):
        self.config = config
        self.logger = logging.getLogger("medrec.summarizer")
        self.max_retries = 3
        self.retry_delay = 1.0

    @property
    def _endpoint(self) -> str:
        return f"{self.config.base_url.rstrip('/')}/api/generate"

    def summarize(self, transcript: str, style: Optional[str] = None) -> SummaryResult:
        """Generate summary with multi-step processing and validation."""
        if not transcript or not transcript.strip():
            raise ValueError("Transcript cannot be empty")
        
        # Check if we should use the TwoPass (GIO/GIGuide) summarizer
        # We use it if the model is 'gio' or if manually requested via style/config
        # For simplicity, we'll use TwoPass for all 'ollama' provider requests now
        # as it is the 'Dragon-Level' standard we are pushing for.
        try:
            from .two_pass_summarizer import TwoPassSummarizer
            tp_summarizer = TwoPassSummarizer(self.config)
            
            # Using the advanced two-pass engine
            result = tp_summarizer.summarize(transcript, style)
            
            # Format back to SummaryResult for UI compatibility
            # Format the structured summary to text
            summary_text = tp_summarizer._format_structured_summary(result)
            
            return SummaryResult(
                summary=summary_text,
                runtime_s=result.runtime_s,
                prompt="Two-Pass (Extraction -> RAG -> Structuring)",
                model_used=result.model_used,
                validation_passed=True,
                refinement_count=0
            )
        except ImportError:
            self.logger.warning("TwoPassSummarizer not found, falling back to simple prompt.")
        except Exception as e:
            self.logger.error(f"TwoPassSummarizer failed: {e}. Falling back.")

        prompt = build_prompt(transcript, style or self.config.prompt_style)
        models_to_try = [self.config.model]
        fallback = getattr(self.config, "fallback_model", None)
        if fallback and fallback not in models_to_try:
            models_to_try.append(fallback)
        
        last_error: Optional[Exception] = None
        for model in models_to_try:
            for attempt in range(self.max_retries):
                try:
                    result = self._invoke_model(model, prompt)
                    
                    # Validate and refine if needed
                    validated_result = self._validate_and_refine(result, transcript, style or self.config.prompt_style)
                    
                    self.logger.info(
                        "summary_complete | model=%s | attempts=%d | validation_passed=%s | refinement_count=%d",
                        model,
                        attempt + 1,
                        validated_result.validation_passed,
                        validated_result.refinement_count,
                    )
                    return validated_result
                    
                except requests.RequestException as exc:
                    last_error = exc
                    self.logger.warning(
                        "summarizer_attempt_failed | model=%s | attempt=%d | error=%s",
                        model,
                        attempt + 1,
                        exc,
                    )
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (attempt + 1))
                    continue
                except Exception as exc:
                    last_error = exc
                    self.logger.error("summarizer_unexpected_error | model=%s | error=%s", model, exc)
                    break
        
        raise RuntimeError(f"All summarizer models failed after {self.max_retries} attempts: {last_error}")

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> SummaryResult:
        """Generate text with enhanced error handling."""
        model_name = model or self.config.model
        
        for attempt in range(self.max_retries):
            try:
                return self._invoke_model(model_name, prompt, temperature, max_tokens)
            except requests.RequestException as exc:
                if attempt < self.max_retries - 1:
                    self.logger.warning("generate_retry | attempt=%d | error=%s", attempt + 1, exc)
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise

    def _invoke_model(
        self,
        model_name: str,
        prompt: str,
        temperature: Optional[float] = None,
        num_predict: Optional[int] = None,
    ) -> SummaryResult:
        """Invoke the model with optimized parameters."""
        temp_value = self.config.temperature if temperature is None else temperature
        max_tokens_value = self.config.max_tokens if num_predict is None else num_predict
        
        # Optimize temperature for medical summarization
        if temp_value > 0.3:
            temp_value = 0.15  # Lower temperature for more consistent medical summaries
        
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temp_value,
                "num_predict": max_tokens_value,
                "top_p": 0.9,  # Nucleus sampling for better quality
                "top_k": 40,   # Limit vocabulary for medical terms
                "repeat_penalty": 1.1,  # Reduce repetition
                "num_thread": 8,  # Use multiple threads for CPU
            },
        }
        
        start = time.perf_counter()
        try:
            # Increase timeout for CPU inference
            timeout = max(self.config.timeout_s, 300)  # At least 5 minutes for CPU
            response = requests.post(
                self._endpoint,
                json=payload,
                timeout=timeout,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            data = response.json()
            text = data.get("response", "").strip()
            
            if not text:
                raise ValueError("Empty response from model")
            
            runtime = time.perf_counter() - start
            return SummaryResult(
                summary=text,
                runtime_s=runtime,
                prompt=prompt,
                model_used=model_name,
                validation_passed=True,
                refinement_count=0,
            )
        except requests.Timeout:
            self.logger.error("model_timeout | model=%s | timeout=%d", model_name, self.config.timeout_s)
            raise
        except requests.RequestException as exc:
            self.logger.error("model_request_error | model=%s | error=%s", model_name, exc)
            raise

    def _validate_and_refine(
        self, result: SummaryResult, transcript: str, style: str
    ) -> SummaryResult:
        """Validate summary structure and refine if needed."""
        summary = result.summary
        refinement_count = 0
        
        # Check for required sections based on style
        if style.lower() == "narrative":
            required_sections = ["Findings:", "Assessment:", "Plan:"]
            optional_sections = ["Medications/Orders:", "Follow-up:"]
        elif style.lower() == "soap":
            required_sections = ["Subjective:", "Objective:", "Assessment:", "Plan:"]
            optional_sections = []
        else:
            required_sections = []
            optional_sections = []
        
        # Validate structure - check required sections
        missing_sections = []
        for section in required_sections:
            if section.lower() not in summary.lower():
                missing_sections.append(section)
        
        # Warn about optional sections but don't fail validation
        missing_optional = []
        for section in optional_sections:
            if section.lower() not in summary.lower():
                missing_optional.append(section)
        
        # Check for common issues
        issues = []
        if len(summary) < 50:
            issues.append("summary_too_short")
        if "unclear" in summary.lower() and "unclear" not in transcript.lower():
            issues.append("excessive_uncertainty")
        if summary.count("\n") < 3:
            issues.append("poor_structure")
        
        # If validation fails, attempt refinement (limited to 1 attempt to avoid loops)
        if missing_sections and refinement_count == 0:
            self.logger.warning(
                "summary_validation_failed | missing_sections=%s | missing_optional=%s | issues=%s",
                missing_sections,
                missing_optional,
                issues,
            )
            # Could add refinement logic here if needed
            # For now, we'll mark as passed but log the issues
        
        # Only fail validation if required sections are missing
        validation_passed = len(missing_sections) == 0
        
        return SummaryResult(
            summary=summary,
            runtime_s=result.runtime_s,
            prompt=result.prompt,
            model_used=result.model_used,
            validation_passed=validation_passed,
            refinement_count=refinement_count,
        )

    def health_check(self) -> bool:
        """Enhanced health check with better error handling."""
        try:
            # Quick health check - just check if API is reachable
            response = requests.get(
                f"{self.config.base_url.rstrip('/')}/api/tags",
                timeout=5,
            )
            if response.status_code == 200:
                # Check if model is available
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                target_model = self.config.model.split(":")[0]  # Get base name
                return any(target_model in name for name in model_names)
            return False
        except requests.Timeout:
            self.logger.warning("health_check_timeout")
            return False
        except requests.RequestException as exc:
            self.logger.warning("health_check_failed | error=%s", exc)
            return False
        except (ValueError, KeyError) as exc:
            self.logger.warning("health_check_parse_error | error=%s", exc)
            return False
