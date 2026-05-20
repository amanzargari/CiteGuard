from __future__ import annotations
import asyncio
from datetime import datetime, timezone

from citeguard.config import Settings
from citeguard.models import CitationRecord, ClaimRecord, Verdict, VerificationResult
from citeguard.retrieval.base import RetrieverBase
from citeguard.cache.manager import CacheManager
from citeguard.tools.session import VerificationSession
from citeguard.tools.retrieve import make_retrieve_tools
from citeguard.tools.verdict import make_verdict_tools
from citeguard.tools.metadata import make_metadata_tools

_SYSTEM_PROMPTS: dict[str, str] = {
    "lenient": """\
You are a citation accuracy reviewer helping researchers verify academic claims.

Verification guidelines (LENIENT mode):
- SUPPORTED: Claim is a reasonable paraphrase or compression of what the paper says.
  Standard academic simplification is acceptable.
- PARTIAL: Claim contains a minor misstatement alongside correct information.
- UNSUPPORTED: Claim directly contradicts the paper, or no related content exists at all.
- EXAGGERATED: Numbers or scope are substantially inflated (e.g., paper says 60%, claim says 90%).
- FABRICATED: Claim describes something clearly absent from the paper.
- AMBIGUOUS: Evidence is present but genuinely unclear.
- UNVERIFIABLE: No PDF is available or retrieval returns nothing relevant.

Use retrieve_chunks() first. If initial evidence is weak, use re_query() with a different angle.
After reviewing evidence (up to 3 rounds total), call mark_verdict() then save_checkpoint().
""",
    "balanced": """\
You are a citation accuracy reviewer helping researchers verify academic claims.

Verification guidelines (BALANCED mode):
- SUPPORTED: Claim accurately represents the paper, including acceptable paraphrase.
- PARTIAL: Claim is partially correct but omits important qualifications or caveats.
- UNSUPPORTED: Claim cannot be found in or directly inferred from the paper's content.
- EXAGGERATED: Claim overstates findings — inflated numbers, broader scope, stronger conclusions.
- FABRICATED: Claim describes something the paper clearly does not contain or contradicts.
- AMBIGUOUS: Evidence is present but genuinely unclear whether it supports the claim.
- UNVERIFIABLE: No PDF available or retrieval yields nothing relevant.

Distinguishing acceptable vs. problematic:
ACCEPTABLE: reasonable paraphrase, compression of multiple findings, standard domain terminology
PROBLEMATIC: attributing results from specific conditions to general case, overstated effect sizes,
  missing critical caveats (e.g., "works on small datasets only" omitted from claim)

Use retrieve_chunks(), then re_query() if needed (max 3 rounds total).
End every verification with mark_verdict() followed by save_checkpoint().
""",
    "strict": """\
You are a rigorous citation accuracy auditor.

Verification guidelines (STRICT mode):
- SUPPORTED: Claim matches the paper's stated findings with high fidelity.
  Paraphrase is acceptable ONLY if meaning is preserved exactly.
- PARTIAL: Any important qualification, condition, or caveat in the paper is omitted.
- UNSUPPORTED: Claim cannot be traced to specific text in the paper.
- EXAGGERATED: Any inflation of numbers, scope, certainty, or generalizability.
- FABRICATED: Claim not found in paper.
- AMBIGUOUS: Only when you genuinely cannot determine truth due to evidence quality.

In strict mode, when uncertain, prefer PARTIAL or AMBIGUOUS over SUPPORTED.
Always end with mark_verdict() then save_checkpoint().
""",
}


class VerificationAgent:
    """Wraps Google ADK LlmAgent for per-citation verification."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def _build_prompt(self, citation: CitationRecord, claim: ClaimRecord) -> str:
        return (
            f"Citation marker: {citation.raw_marker}\n"
            f"Matched PDF: {citation.matched_pdf or 'NONE — mark as UNVERIFIABLE'}\n"
            f"Reference entry: {citation.reference_text or 'not parsed'}\n\n"
            f"Manuscript claim:\n{claim.claim_text}\n\n"
            f"Context before: {claim.context_before}\n"
            f"Context after: {claim.context_after}\n\n"
            "Please verify this claim using the available tools, "
            "then call mark_verdict() and save_checkpoint()."
        )

    async def verify_async(
        self,
        citation: CitationRecord,
        claim: ClaimRecord,
        retriever: RetrieverBase | None,
        cache: CacheManager,
        pdf_paths: list[str],
    ) -> VerificationSession:
        session = VerificationSession(citation, claim)

        # Handle UNVERIFIABLE case without LLM call
        if citation.matched_pdf is None:
            session.verdict = Verdict.UNVERIFIABLE
            session.reasoning = "No matching PDF found for this citation."
            session.confidence = 1.0
            _, save_checkpoint = make_verdict_tools(session, cache)
            save_checkpoint()
            return session

        retrieve_chunks, re_query = make_retrieve_tools(
            retriever=retriever,
            session=session,
            max_requery_rounds=self._settings.retrieval.max_requery_rounds,
        )
        mark_verdict, save_checkpoint = make_verdict_tools(session, cache)
        (get_paper_metadata,) = make_metadata_tools(pdf_paths)

        system_prompt = _SYSTEM_PROMPTS[self._settings.strictness]
        tools = [retrieve_chunks, re_query, mark_verdict, save_checkpoint, get_paper_metadata]
        prompt = self._build_prompt(citation, claim)

        try:
            from google.adk.agents import LlmAgent
            from google.adk.runners import Runner
            from google.adk.sessions import InMemorySessionService
            from google.adk.models.lite_llm import LiteLlm

            model = LiteLlm(
                model=f"openrouter/{self._settings.models.strong}",
                api_key=self._settings.openrouter_api_key,
                api_base=self._settings.openrouter_base_url,
            )

            agent = LlmAgent(
                name=f"verifier_{citation.id}",
                model=model,
                instruction=system_prompt,
                tools=tools,
            )

            runner = Runner(
                agent=agent,
                app_name="citeguard",
                session_service=InMemorySessionService(),
            )

            from google.genai import types as genai_types
            new_message = genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=prompt)],
            )

            async for _event in runner.run_async(
                user_id="citeguard",
                session_id=f"verify_{citation.id}",
                new_message=new_message,
            ):
                pass  # Tools execute via side effects on session

        except Exception as e:
            # If ADK fails for any reason, mark as ERROR
            if session.verdict is None:
                session.verdict = Verdict.ERROR
                session.reasoning = f"Agent error: {str(e)[:200]}"
                session.confidence = 0.0
                save_checkpoint()

        # Ensure checkpoint exists even if agent didn't call save_checkpoint
        if session.verdict is None:
            session.verdict = Verdict.AMBIGUOUS
            session.reasoning = "Agent did not produce a verdict."
            save_checkpoint()

        return session

    def verify(
        self,
        citation: CitationRecord,
        claim: ClaimRecord,
        retriever: RetrieverBase | None,
        cache: CacheManager,
        pdf_paths: list[str],
    ) -> VerificationSession:
        """Synchronous wrapper around verify_async."""
        return asyncio.run(
            self.verify_async(citation, claim, retriever, cache, pdf_paths)
        )
