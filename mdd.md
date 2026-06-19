
# Multi-Agent Architecture Design: The Silicon Jury

This document outlines the concrete technical architecture, framework selection, and a programmatic Python blueprint for running the 12-agent "Secret Ballot to Deliberation" jury experiment.

## 1. Framework Evaluation

Criteria

Microsoft AutoGen

LangGraph

PettingZoo

**Primary Use Case**

Conversational, autonomous LLM group chats.

State-machine controlled agent workflows (DAGs).

Multi-Agent Reinforcement Learning (RL).

**Suitability for this Project**

**High (for debate)**. Great at natural conversation, but hard to strictly enforce a "3-round limit" or "secret ballot" without hacky wrappers.

**Very High (for experiment control)**. Allows precise state management, making it impossible for agents to cheat or see private memos.

**None**. Designed for mathematical environments (gaming/physics), not natural language.

**API Cost Control**

Moderate (agents can get talkative if not restricted).

High (each transition is explicitly defined and deterministic).

N/A

### The Verdict

We recommend using **LangGraph** or a **Custom AsyncIO Python Framework** for the orchestration layer, utilizing standard LLM clients (like OpenAI or Anthropic).

Because your academic paper relies on **strict control of variables** (preventing agents from leaking information before the ballot, terminating precisely after Round 3), a state-machine or custom async wrapper is much easier to debug and statistically measure than a fully autonomous AutoGen group chat.

## 2. System State Diagram

To run this experiment deterministically, the system must transition through five distinct states:

```
[State 0: Initialization] ──> Generates 12 random personas from Databank
          │
          ▼
[State 1: Secret Ballot]  ──> Runs 12 isolated parallel LLM calls (Private Memos)
          │
          ▼
[State 2: Synthesis]      ──> Orchestrator extracts and summarizes points of conflict
          │
          ▼
[State 3: Deliberation]   ──> Runs 3 rounds of structured, turn-based conversation
          │
          ▼
[State 4: Final Ballot]   ──> Runs 12 isolated parallel LLM calls for final votes
          │
          ▼
[State 5: Analysis]       ──> Computes FPR/FNR and saves trial execution JSON

```

## 3. Technical Blueprint (Python Conceptual Skeleton)

Below is a production-ready architectural skeleton in Python. It uses asynchronous programming to run the 12 independent analysis steps in parallel, saving significant execution time, while maintaining a clean state object for data logging.

```
import asyncio
import json
from typing import List, Dict, Any
from pydantic import BaseModel, Field

# --- 1. DATA STRUCTURES ---

class Persona(BaseModel):
    id: str
    cognitive_style: str
    systemic_bias: str
    communication_style: str

class PrivateMemo(BaseModel):
    juror_id: str
    initial_verdict: str  # "Guilty" or "Not Guilty"
    confidence: float     # 0.0 to 1.0
    incriminating_evidence: List[str]
    exculpatory_evidence: List[str]

class DebateTurn(BaseModel):
    round_number: int
    juror_id: str
    statement: str

class FinalVote(BaseModel):
    juror_id: str
    final_verdict: str
    confidence: float
    reason_for_shift: str

class TrialState(BaseModel):
    trial_id: str
    ground_truth: str  # "Guilty" or "Innocent" (Exonerated)
    human_verdict: str  # Real-world jury outcome
    personas: Dict[str, Persona] = {}
    private_memos: Dict[str, PrivateMemo] = {}
    debate_history: List[DebateTurn] = []
    final_votes: Dict[str, FinalVote] = {}

# --- 2. THE JUROR AGENT DEFINITION ---

class JurorAgent:
    def __init__(self, persona: Persona):
        self.persona = persona

    async def analyze_transcript(self, transcript: str) -> PrivateMemo:
        """
        Stage 1: Reads transcript in total isolation. No communication with other agents.
        """
        prompt = f"""
        You are a juror with the following profile:
        - Cognitive Style: {self.persona.cognitive_style}
        - Bias: {self.persona.systemic_bias}
        - Communication: {self.persona.communication_style}

        Analyze the following trial transcript and output a JSON matching the schema:
        {{
            "initial_verdict": "Guilty" or "Not Guilty",
            "confidence": 0.0 to 1.0,
            "incriminating_evidence": ["point 1", ...],
            "exculpatory_evidence": ["point 1", ...]
        }}
        
        Transcript:
        {transcript[:10000]}... (Truncated for blueprint)
        """
        # In practice, call LLM client here
        # response = await openai_client.chat.completions.create(...)
        
        # Mocking response for structural integrity:
        return PrivateMemo(
            juror_id=self.persona.id,
            initial_verdict="Not Guilty",
            confidence=0.8,
            incriminating_evidence=["Lack of physical proof"],
            exculpatory_evidence=["Alibi witness testimony"]
        )

    async def deliberate(self, round_num: int, conflict_summary: str, history: List[DebateTurn]) -> DebateTurn:
        """
        Stage 3: Speak in the group chat based on the conflict summary and chat history.
        """
        # Construct conversation history context for the LLM
        formatted_history = "\n".join([f"Juror {t.juror_id}: {t.statement}" for t in history])
        
        prompt = f"""
        You are Juror {self.persona.id} ({self.persona.cognitive_style}).
        We are in Round {round_num} of deliberation. 
        
        The primary points of conflict are: {conflict_summary}
        
        Here is the conversation history so far:
        {formatted_history}
        
        Provide your short contribution (under 100 words) defending your stance or responding to others. Keep character.
        """
        # response = await openai_client.chat.completions.create(...)
        return DebateTurn(
            round_number=round_num,
            juror_id=self.persona.id,
            statement="I still don't see any DNA tying him to the weapon. We cannot convict."
        )

    async def cast_final_vote(self, history: List[DebateTurn]) -> FinalVote:
        """
        Stage 4: Cast secret final ballot after deliberation.
        """
        prompt = f"Based on the deliberation history, cast your final secret ballot..."
        return FinalVote(
            juror_id=self.persona.id,
            final_verdict="Not Guilty",
            confidence=0.9,
            reason_for_shift="The discussion highlighted the unreliability of the eyewitness."
        )

# --- 3. THE ORCHESTRATOR / WORKFLOW ENGINE ---

class TrialOrchestrator:
    def __init__(self, trial_id: str, ground_truth: str, human_verdict: str):
        self.state = TrialState(
            trial_id=trial_id,
            ground_truth=ground_truth,
            human_verdict=human_verdict
        )
        self.jurors: List[JurorAgent] = []

    def initialize_jury(self, persona_pool: List[Persona]):
        # Select 12 random distinct personas
        import random
        selected_personas = random.sample(persona_pool, 12)
        for i, p in enumerate(selected_personas):
            p.id = f"Juror_{i+1}"
            self.state.personas[p.id] = p
            self.jurors.append(JurorAgent(p))

    async def run_stage_1_secret_ballot(self, transcript: str):
        """Runs all 12 juror analyses in parallel to minimize API bottlenecking."""
        tasks = [juror.analyze_transcript(transcript) for juror in self.jurors]
        memos = await asyncio.gather(*tasks)
        for memo in memos:
            self.state.private_memos[memo.juror_id] = memo

    def generate_conflict_points(self) -> str:
        """Helper logic for Stage 2: Summarizes the split."""
        guilty_votes = sum(1 for m in self.state.private_memos.values() if m.initial_verdict == "Guilty")
        not_guilty_votes = 12 - guilty_votes
        
        summary = f"The initial split is {guilty_votes} Guilty to {not_guilty_votes} Not Guilty. "
        summary += "Key dispute: Eyewitness credibility vs. Lack of direct physical forensic evidence."
        return summary

    async def run_stage_3_deliberation(self, conflict_summary: str, num_rounds: int = 3):
        """Orchestrates turn-taking over dynamic conversation rounds."""
        for r in range(1, num_rounds + 1):
            # To simulate a natural debate, we can loop through the agents.
            # In a more advanced build, only agents with high conviction or dissenting opinions speak.
            for juror in self.jurors:
                turn = await juror.deliberate(r, conflict_summary, self.state.debate_history)
                self.state.debate_history.append(turn)

    async def run_stage_4_final_ballot(self):
        """Runs final secret ballot in parallel."""
        tasks = [juror.cast_final_vote(self.state.debate_history) for juror in self.jurors]
        votes = await asyncio.gather(*tasks)
        for vote in votes:
            self.state.final_votes[vote.juror_id] = vote

    def save_results(self, output_path: str):
        with open(output_path, "w") as f:
            json.dump(self.state.dict(), f, indent=4)

# --- 4. EXECUTION SIMULATION ENTRYPOINT ---

async def main():
    # Setup mock environment
    persona_pool = [
        Persona(id="", cognitive_style="Analytical", systemic_bias="Skeptical of eyewitnesses", communication_style="Blunt"),
        Persona(id="", cognitive_style="Empathetic", systemic_bias="Deconstructs systemic bias", communication_style="Collaborative"),
        # ... Imagine 100+ personas loaded here
    ] * 6 # Multiplying to have enough for selection
    
    orchestrator = TrialOrchestrator(
        trial_id="TX-1994-MCVEIGH",
        ground_truth="Innocent", # DNA Exoneration
        human_verdict="Guilty"   # Real human jury made a false-positive error
    )
    
    print("[1/5] Selecting 12 Random Personas...")
    orchestrator.initialize_jury(persona_pool)
    
    print("[2/5] Running Stage 1: Isolated Secret Ballot (Parallelized)...")
    mock_transcript = "This is the transcript of a robbery trial..."
    await orchestrator.run_stage_1_secret_ballot(mock_transcript)
    
    print("[3/5] Compiling Points of Conflict...")
    conflict_points = orchestrator.generate_conflict_points()
    
    print("[4/5] Running Stage 3: Structured 3-Round Deliberation Chat...")
    await orchestrator.run_stage_3_deliberation(conflict_summary=conflict_points, num_rounds=3)
    
    print("[5/5] Running Stage 4: Secret Final Vote (Parallelized)...")
    await orchestrator.run_stage_4_final_ballot()
    
    # Save the execution state for statistical compilation later
    orchestrator.save_results("simulation_results_trial_1.json")
    print("Simulation Complete. Results saved.")

if __name__ == "__main__":
    asyncio.run(main())

```
