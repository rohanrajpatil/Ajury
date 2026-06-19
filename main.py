import asyncio
import random
import json
import logging
import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, conint
import aiohttp

# Configure minimal, clean logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Gemini API Configuration (As per guidelines)
API_KEY = ""  # The execution environment provides this key at runtime
MODEL_NAME = "gemini-2.5-flash-preview-09-2025"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"

# =====================================================================
# DATA STRUCTURES & SCHEMAS
# =====================================================================

class OceanProfile(BaseModel):
    """
    Scientifically validated Big Five Personality Traits (OCEAN Model).
    All scores are constrained between 1 and 100.
    """
    openness: int = Field(..., ge=1, le=100, description="Intellectual curiosity and processing of complex alternative theories.")
    conscientiousness: int = Field(..., ge=1, le=100, description="Dutifulness, attention to legal procedure, and literal instructions.")
    extraversion: int = Field(..., ge=1, le=100, description="Assertiveness, vocal output, and dominance in debate settings.")
    agreeableness: int = Field(..., ge=1, le=100, description="Empathy, value of consensus, and susceptibility to peer pressure.")
    neuroticism: int = Field(..., ge=1, le=100, description="Sensitivity to graphic evidence, emotional rhetoric, or logical stress.")

class JurorPersona(BaseModel):
    """Represents a unique juror profile with objective traits and demographics."""
    id: str
    age: int
    occupation: str
    ocean: OceanProfile

class JurorVote(BaseModel):
    """Represents an individual juror's ballot vote and their rationale."""
    juror_id: str
    vote: str = Field(..., description="Must be either 'GUILTY' or 'NOT GUILTY'")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the decision.")
    rationale: str = Field(..., description="The internal reasoning mapping back to their OCEAN scores and legal instructions.")

class DebateContribution(BaseModel):
    """Represents a vocal statement made by a juror during the deliberation rounds."""
    juror_id: str
    statement: str
    target_juror_id: Optional[str] = None

class TrialState(BaseModel):
    """Tracks the entire lifecycle state of a single trial simulation."""
    trial_id: str
    transcript: str
    jurors: List[JurorPersona] = []
    initial_votes: Dict[str, JurorVote] = {}
    debate_rounds: List[List[DebateContribution]] = []
    final_votes: Dict[str, JurorVote] = {}

# =====================================================================
# PERSONA GENERATION ENGINE
# =====================================================================

class PersonaDatabank:
    """Generates scientifically sound juror personas using the OCEAN scale."""
    
    OCCUPATIONS = [
        "Software Engineer", "High School Teacher", "Registered Nurse", "Accountant", 
        "Mechanic", "Sales Director", "Graphic Designer", "Librarian", "Construction Manager",
        "Retired Postal Worker", "Paralegal", "Electrician", "Small Business Owner"
    ]

    @classmethod
    def generate_random_persona(cls, juror_id: str) -> JurorPersona:
        """Generates a randomized, scientifically balanced juror persona."""
        return JurorPersona(
            id=juror_id,
            age=random.randint(18, 75),
            occupation=random.choice(cls.OCCUPATIONS),
            ocean=OceanProfile(
                openness=random.randint(1, 100),
                conscientiousness=random.randint(1, 100),
                extraversion=random.randint(1, 100),
                agreeableness=random.randint(1, 100),
                neuroticism=random.randint(1, 100)
            )
        )

    @classmethod
    def create_jury_pool(cls, size: int = 12) -> List[JurorPersona]:
        """Creates a jury pool of distinct human-like personas."""
        return [cls.generate_random_persona(f"Juror_{i+1:02d}") for i in range(size)]

# =====================================================================
# ROBUST GEMINI API CONNECTOR
# =====================================================================

async def call_gemini_api(system_instruction: str, user_prompt: str, session: aiohttp.ClientSession) -> str:
    """
    Calls the Gemini API with standard exponential backoff (up to 5 retries).
    Does not log retries to console to keep outputs clean.
    """
    payload = {
        "contents": [{
            "parts": [{"text": user_prompt}]
        }],
        "systemInstruction": {
            "parts": [{"text": system_instruction}]
        }
    }
    
    # We will enforce structured JSON returns if possible, or parse raw text responses
    payload["generationConfig"] = {
        "responseMimeType": "application/json"
    }

    backoff_delays = [1, 2, 4, 8, 16]
    
    for attempt, delay in enumerate(backoff_delays):
        try:
            async with session.post(API_URL, json=payload, timeout=30) as response:
                if response.status == 200:
                    result = await response.json()
                    # Parse the standard response payload
                    text_content = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                    if text_content:
                        return text_content
                elif response.status == 429:
                    # Rate limit hit, let backoff handle it
                    pass
                else:
                    # Other status code error
                    pass
        except Exception:
            # Connection errors handled silently in retry loop
            pass
        
        await asyncio.sleep(delay)
        
    raise RuntimeError("Gemini API call repeatedly failed after 5 retry attempts. Please verify your environment and API configuration.")

# =====================================================================
# MULTI-AGENT STATE-CONTROLLED PROTOCOL ENGINE
# =====================================================================

class JurySimulationEngine:
    """Orchestrates the entire multi-agent jury system lifecycle."""
    
    def __init__(self, transcript: str, trial_id: str = "TRIAL_001"):
        self.state = TrialState(trial_id=trial_id, transcript=transcript)
        self.state.jurors = PersonaDatabank.create_jury_pool(12)

    def _build_system_instruction(self, persona: JurorPersona) -> str:
        """Constructs the system instruction prompt strictly framing the Big Five traits."""
        return f"""You are Juror {persona.id}, a citizen serving on a criminal trial jury.
Your demographics: Age {persona.age}, Occupation: {persona.occupation}.

As a juror, you must uphold the fundamental Western legal standard of "Beyond a Reasonable Doubt."
You must strictly respect the Blackstone Standard: "It is better that ten guilty persons escape than that one innocent suffer."

Your decision-making process is strictly governed by your unique, scientifically validated Big Five (OCEAN) personality scores.
These scores range from 1 (extremely low) to 100 (extremely high):

[YOUR COGNITIVE TRAITS - OCEAN SCORES]
- Openness: {persona.ocean.openness}/100 (High score: processes complex forensic patterns; Low score: relies heavily on literal physical evidence)
- Conscientiousness: {persona.ocean.conscientiousness}/100 (High score: strictly complies with judicial instructions; Low score: relies on intuition)
- Extraversion: {persona.ocean.extraversion}/100 (High score: highly vocal, initiates debate, seeks to persuade; Low score: quiet, holds ground internally)
- Agreeableness: {persona.ocean.agreeableness}/100 (High score: highly empathetic, fears social isolation, susceptible to consensus; Low score: adversarial, critical)
- Neuroticism: {persona.ocean.neuroticism}/100 (High score: emotionally sensitive to victim descriptions; Low score: cold, mathematically logical)

[INSTRUCTIONS]
1. You must think, write, vote, and speak strictly in accordance with these numeric variables.
2. Do not break character. Do not list your numeric scores in conversation, simply let them dictate your cognitive strategy.
3. You must output all JSON targets in their exact formats.
"""

    async def run_stage_1_secret_ballot(self, session: aiohttp.ClientSession):
        """Stage 1: Jurors analyze trial transcript in absolute isolation."""
        logging.info("Initiating Stage 1: Private Juror Analysis & Initial Secret Ballot...")
        
        user_prompt = f"""You have just listened to the entire trial. Here is the trial transcript:
---
{self.state.transcript}
---

TASK:
Examine the evidence based on your system instructions, legal standards, and personal cognitive OCEAN traits.
Submit your initial secret ballot containing:
- Your vote (GUILTY or NOT GUILTY).
- Your level of confidence (a float between 0.0 and 1.0).
- A concise summary of your private logical rationale (maximum 3 sentences).

Return your decision in the following raw JSON format:
{{
  "juror_id": "{self.state.jurors[0].id}",
  "vote": "GUILTY or NOT GUILTY",
  "confidence": 0.85,
  "rationale": "Your detailed rationale here..."
}}
"""
        tasks = []
        for juror in self.state.jurors:
            system_inst = self._build_system_instruction(juror)
            # Personalize the prompt with the exact juror id
            customized_prompt = user_prompt.replace(self.state.jurors[0].id, juror.id)
            tasks.append(call_gemini_api(system_inst, customized_prompt, session))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for juror, result in zip(self.state.jurors, results):
            if isinstance(result, Exception):
                logging.error(f"Failed to fetch initial vote for {juror.id}: {result}")
                # Fallback to avoid complete execution failure
                self.state.initial_votes[juror.id] = JurorVote(
                    juror_id=juror.id, vote="NOT GUILTY", confidence=0.5, rationale="Error parsing response. Indubitable default set to innocent."
                )
                continue
            
            try:
                data = json.loads(result)
                self.state.initial_votes[juror.id] = JurorVote(**data)
            except Exception as e:
                logging.error(f"JSON parsing error on initial ballot for {juror.id}: {e}. Response received: {result}")
                # String cleanup attempt
                clean_vote = "GUILTY" if "GUILTY" in result.upper() and "NOT GUILTY" not in result.upper() else "NOT GUILTY"
                self.state.initial_votes[juror.id] = JurorVote(
                    juror_id=juror.id, vote=clean_vote, confidence=0.5, rationale=f"Parser recovery from: {result[:100]}"
                )

    async def run_stage_2_debate(self, rounds: int, session: aiohttp.ClientSession):
        """Stage 2: Orchestrated deliberation rounds inside a state-controlled loop."""
        logging.info(f"Initiating Stage 2: Deliberation ({rounds} rounds of controlled debate)...")
        
        for round_idx in range(rounds):
            logging.info(f"Running Deliberation Round {round_idx + 1}...")
            round_contributions: List[DebateContribution] = []
            
            # Step 1: Collect points from all active jurors sequentially to maintain conversation context
            for juror in self.state.jurors:
                # Compile preceding debate history for this round to give context
                history_text = ""
                for r in self.state.debate_rounds:
                    for contribution in r:
                        history_text += f"Juror {contribution.juror_id}: \"{contribution.statement}\"\n"
                for contribution in round_contributions:
                    history_text += f"Juror {contribution.juror_id}: \"{contribution.statement}\"\n"
                
                # Check initial stance to keep reasoning stable but dynamic
                my_initial_vote = self.state.initial_votes[juror.id]
                
                debate_prompt = f"""The jury is deliberating. 
Your initial private opinion was: {my_initial_vote.vote} (Confidence: {my_initial_vote.confidence}) with rationale: "{my_initial_vote.rationale}"

Here is the transcript of statements made by other jurors in the room so far:
---
{history_text if history_text else "[The room is silent. You are the first to speak this round.]"}
---

TASK:
Make a concise, logical argument supporting your current stance or reacting to previous claims.
Do not repeat yourself. Adjust your tone based on your:
- Extraversion score ({juror.ocean.extraversion}): If high, be highly assertive and try to lead. If low, speak briefly and wait.
- Agreeableness score ({juror.ocean.agreeableness}): If high, seek to build consensus and compromise. If low, be highly analytical and critical.
- Neuroticism score ({juror.ocean.neuroticism}): If high, express moral concern. If low, keep it cold and logical.

Return your response in the following raw JSON format:
{{
  "juror_id": "{juror.id}",
  "statement": "Your vocal argument in maximum two sentences.",
  "target_juror_id": "Juror_XX" (or null if addressing the room)
}}
"""
                system_inst = self._build_system_instruction(juror)
                try:
                    result = await call_gemini_api(system_inst, debate_prompt, session)
                    data = json.loads(result)
                    round_contributions.append(DebateContribution(**data))
                except Exception as e:
                    logging.error(f"Error gathering debate statement for {juror.id}: {e}")
                    # Passive fallback contribution
                    round_contributions.append(DebateContribution(
                        juror_id=juror.id, 
                        statement="I am carefully reviewing the points raised by my fellow jurors.",
                        target_juror_id=None
                    ))
            
            self.state.debate_rounds.append(round_contributions)

    async def run_stage_3_final_ballot(self, session: aiohttp.ClientSession):
        """Stage 3: Jurors cast final secret ballot after analyzing the group deliberations."""
        logging.info("Initiating Stage 3: Post-Deliberation Secret Final Ballot...")
        
        # Build comprehensive history of debate
        history_text = ""
        for round_idx, r in enumerate(self.state.debate_rounds):
            history_text += f"--- Deliberation Round {round_idx + 1} ---\n"
            for contribution in r:
                history_text += f"Juror {contribution.juror_id}: \"{contribution.statement}\"\n"

        user_prompt = f"""You have concluded jury deliberations. Here is the entire debate history:
---
{history_text}
---

TASK:
Assess all legal arguments, counterarguments, and evidence. 
Apply the Blackstone Standard and determine if there remains a reasonable doubt of guilt.
Submit your final, absolute secret ballot containing your final vote and updated confidence.

Return your decision in the following raw JSON format:
{{
  "juror_id": "{self.state.jurors[0].id}",
  "vote": "GUILTY or NOT GUILTY",
  "confidence": 0.95,
  "rationale": "Explain how the debate influenced, shifted, or reinforced your final stance in relation to your OCEAN personality."
}}
"""
        tasks = []
        for juror in self.state.jurors:
            system_inst = self._build_system_instruction(juror)
            customized_prompt = user_prompt.replace(self.state.jurors[0].id, juror.id)
            tasks.append(call_gemini_api(system_inst, customized_prompt, session))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for juror, result in zip(self.state.jurors, results):
            if isinstance(result, Exception):
                logging.error(f"Failed to fetch final vote for {juror.id}: {result}")
                self.state.final_votes[juror.id] = self.state.initial_votes[juror.id] # Fallback to initial
                continue
            
            try:
                data = json.loads(result)
                self.state.final_votes[juror.id] = JurorVote(**data)
            except Exception as e:
                logging.error(f"JSON parsing error on final ballot for {juror.id}: {e}")
                self.state.final_votes[juror.id] = self.state.initial_votes[juror.id]

    def compile_simulation_metrics(self) -> Dict[str, Any]:
        """Calculates quantitative shifts in consensus, polarization, and traits correlation."""
        initial_guilty_count = sum(1 for v in self.state.initial_votes.values() if v.vote == "GUILTY")
        final_guilty_count = sum(1 for v in self.state.final_votes.values() if v.vote == "GUILTY")
        
        # Calculate flips
        flipped_jurors = []
        for j_id in self.state.initial_votes:
            init_v = self.state.initial_votes[j_id].vote
            final_v = self.state.final_votes[j_id].vote
            if init_v != final_v:
                juror = next(j for j in self.state.jurors if j.id == j_id)
                flipped_jurors.append({
                    "id": j_id,
                    "from": init_v,
                    "to": final_v,
                    "agreeableness": juror.ocean.agreeableness,
                    "extraversion": juror.ocean.extraversion
                })

        return {
            "trial_id": self.state.trial_id,
            "initial_guilty_ratio": f"{initial_guilty_count}/12",
            "final_guilty_ratio": f"{final_guilty_count}/12",
            "unanimous_consensus_reached": final_guilty_count in [0, 12],
            "total_flips": len(flipped_jurors),
            "flipped_jurors_profiles": flipped_jurors
        }

# =====================================================================
# SIMULATION HARNESS / EXECUTION PIPELINE
# =====================================================================

# Mock trial script representing a realistic circumstantial evidence case
MOCK_TRANSCRIPT = """
[CASE FILE NO. 2026-CR-012]
CHARGE: Grand Larceny (Theft of Diamond Ring valued at $45,000)
DEFENDANT: Arthur Vance (Former Store Clerk)

PROSECUTION CASE:
Arthur Vance was the only clerk present at "Vance & Sterling Jewellers" when the security safe was opened. 
The safe's code was known by three employees. Security footage shows Arthur walking past the safe at 11:42 PM. 
The ring was found missing the next morning. When arrested, Arthur had $8,000 in cash in his safe, for which 
he has no tax records or proof of legal origin. 

DEFENSE CASE:
Arthur Vance has a completely clean record. The cash in his home safe was saved over seven years from cash-only side 
gigs (landscaping). The security footage shows him walking past the safe, but at no point does it show him opening 
it or carrying any merchandise. The safe had been malfunctioning, and any of the other two employees could have returned 
to the building after hours off-camera via the rear loading dock door, which has no security camera coverage.
"""

async def main():
    print("==========================================================")
    print("      AI JURY SIMULATION ENGINE - EXPERIMENTAL PIPELINE   ")
    print("==========================================================")
    
    # 1. Initialize simulation with mock trial transcript
    engine = JurySimulationEngine(transcript=MOCK_TRANSCRIPT)
    
    # Print the scientific persona pool for validation
    print("\n--- GENERATED PERSOONAL JURY POOL (BIG FIVE OCEAN MODEL) ---")
    for juror in engine.state.jurors:
        o = juror.ocean
        print(f"[{juror.id}] Age {juror.age}, {juror.occupation:23s} | O:{o.openness:2d} | C:{o.conscientiousness:2d} | E:{o.extraversion:2d} | A:{o.agreeableness:2d} | N:{o.neuroticism:2d}")
    print("------------------------------------------------------------\n")
    
    # 2. Run simulation pipeline asynchronously
    async with aiohttp.ClientSession() as session:
        try:
            # Stage 1: Initial secret ballot
            await engine.run_stage_1_secret_ballot(session)
            print(">>> Stage 1 (Initial Ballots) Completed successfully.")
            for j_id, v in engine.state.initial_votes.items():
                print(f" - {j_id}: voted {v.vote:10s} | Confidence: {v.confidence:.2f} | Rationale: {v.rationale[:90]}...")

            # Stage 2: Debate
            print("\n>>> Stage 2: Deliberation Starting...")
            await engine.run_stage_2_debate(rounds=2, session=session)
            
            # Print highlights of the debate to standard output
            for round_idx, contributions in enumerate(engine.state.debate_rounds):
                print(f"\n--- Debate Round {round_idx + 1} Highlight ---")
                for c in contributions[:3]:  # Print first 3 to avoid flooding the console
                    print(f" [{c.juror_id}]: {c.statement}")
                print(" [...]")

            # Stage 3: Final Secret Ballot
            print("\n>>> Stage 3 (Final Secret Ballot) Starting...")
            await engine.run_stage_3_final_ballot(session)
            print(">>> Stage 3 Completed successfully.")
            for j_id, v in engine.state.final_votes.items():
                print(f" - {j_id}: voted {v.vote:10s} | Confidence: {v.confidence:.2f} | Rationale: {v.rationale[:90]}...")

            # 3. Compile Metrics and Statistical Outputs
            metrics = engine.compile_simulation_metrics()
            print("\n==========================================================")
            print("                 SIMULATION METRICS SUMMARY               ")
            print("==========================================================")
            print(f"Trial ID:                        {metrics['trial_id']}")
            print(f"Initial Verdict Split (Guilty): {metrics['initial_guilty_ratio']}")
            print(f"Final Verdict Split (Guilty):   {metrics['final_guilty_ratio']}")
            print(f"Unanimous Consensus Reached:    {metrics['unanimous_consensus_reached']}")
            print(f"Total Jurors Switched (Flips):   {metrics['total_flips']}")
            
            if metrics["total_flips"] > 0:
                print("\nJurors who changed their minds:")
                for flip in metrics["flipped_jurors_profiles"]:
                    print(f" - {flip['id']}: {flip['from']} -> {flip['to']} | Agreeableness Score: {flip['agreeableness']}/100 | Extraversion: {flip['extraversion']}/100")
            print("==========================================================")

        except Exception as e:
            print(f"\n[FATAL ERROR] Simulation stopped: {e}")
            print("Please ensure your API_KEY variable is correctly configured if trying to connect to Gemini cloud live services.")

if __name__ == "__main__":
    asyncio.run(main())