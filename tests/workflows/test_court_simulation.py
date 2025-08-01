#!/usr/bin/env python3
"""
Court Simulation with DAG Execution
===================================

Simulates a courtroom trial with 4 agents:
- Witness (user input)
- Defense Attorney
- Prosecution Attorney 
- Judge
- Jury

Data flows from witness testimony to all agents, then lawyers present arguments
to judge and jury for final verdict.
"""

import uuid
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Load environment
env_path = Path(__file__).parent / "creds.env"
if env_path.exists():
    load_dotenv(env_path)

from iointel.src.agent_methods.data_models.workflow_spec import (
    WorkflowSpec, NodeSpec, NodeData, EdgeSpec
)
from iointel.src.utilities.dag_executor import create_dag_executor_from_spec
from iointel.src.utilities.graph_nodes import WorkflowState


def create_court_simulation_workflow():
    """Create a court simulation workflow with proper DAG structure."""
    
    # Define all the agents and their roles
    nodes = [
        # Witness testimony (using new data source system with standardized interface)
        NodeSpec(
            id="witness_testimony",
            type="data_source",
            label="Witness Testimony",
            data=NodeData(
                source_name="prompt_tool",
                config={
                    "message": "I saw the defendant near the bank at 3 PM on the day in question. They were acting suspiciously, looking around nervously and checking their phone repeatedly. I distinctly remember they were wearing a red jacket and kept glancing at the security cameras. When they saw me watching, they quickly walked away towards the parking garage. I had a clear view from about 20 feet away and am certain it was the defendant.",
                    "default_value": "I saw the defendant near the bank at 3 PM on the day in question. They were acting suspiciously, looking around nervously and checking their phone repeatedly. I distinctly remember they were wearing a red jacket and kept glancing at the security cameras. When they saw me watching, they quickly walked away towards the parking garage. I had a clear view from about 20 feet away and am certain it was the defendant."
                },
                ins=[],
                outs=["testimony"]
            )
        ),
        
        # Defense Attorney (analyzes testimony for defense)
        NodeSpec(
            id="defense_attorney",
            type="agent",
            label="Defense Attorney",
            data=NodeData(
                agent_instructions="""
                You are a skilled defense attorney. Analyze the witness testimony and build a strong defense case.
                
                WITNESS TESTIMONY TO ANALYZE: {witness_testimony}
                
                Your role:
                - Identify inconsistencies or weaknesses in the testimony
                - Develop reasonable doubt arguments
                - Prepare counter-narratives that favor your client
                - Be professional but zealous in defending your client
                
                Provide a clear defense strategy based on the specific testimony above.
                """,
                tools=[],
                config={},
                ins=["testimony"],
                outs=["defense_case"]
            )
        ),
        
        # Prosecution Attorney (analyzes testimony for prosecution)
        NodeSpec(
            id="prosecution_attorney",
            type="agent",
            label="Prosecution Attorney",
            data=NodeData(
                agent_instructions="""
                You are an experienced prosecutor. Analyze the witness testimony and build a strong case for conviction.
                
                WITNESS TESTIMONY TO ANALYZE: {witness_testimony}
                
                Your role:
                - Highlight the most damaging aspects of the testimony
                - Build a coherent narrative of guilt
                - Address potential defense arguments preemptively
                - Seek justice while being fair and ethical
                
                Provide a clear prosecution strategy based on the specific testimony above.
                """,
                tools=[],
                config={},
                ins=["testimony"],
                outs=["prosecution_case"]
            )
        ),
        
        # Judge (evaluates both sides and makes legal rulings)
        NodeSpec(
            id="judge",
            type="agent",
            label="Judge",
            data=NodeData(
                agent_instructions="""
                You are an experienced and impartial judge presiding over this trial.
                
                WITNESS TESTIMONY: {witness_testimony}
                DEFENSE ARGUMENTS: {defense_attorney}
                PROSECUTION ARGUMENTS: {prosecution_attorney}
                
                Your role:
                - Review the original witness testimony
                - Consider both defense and prosecution arguments
                - Evaluate the legal merits of each side
                - Provide jury instructions on how to weigh the evidence
                - Ensure fair proceedings
                
                Provide your judicial assessment and instructions to the jury on how to evaluate this evidence.
                """,
                tools=[],
                config={},
                ins=["testimony", "defense_case", "prosecution_case"],
                outs=["judicial_assessment"]
            )
        ),
        
        # Jury (deliberates and reaches verdict)
        NodeSpec(
            id="jury",
            type="agent",
            label="Jury",
            data=NodeData(
                agent_instructions="""
                You are a 12-person jury deliberating on this case. Consider all evidence and arguments.
                
                WITNESS TESTIMONY: {witness_testimony}
                DEFENSE ARGUMENTS: {defense_attorney}
                PROSECUTION ARGUMENTS: {prosecution_attorney}
                JUDGE'S INSTRUCTIONS: {judge}
                
                Your role:
                - Review the original witness testimony carefully
                - Consider the defense attorney's arguments
                - Consider the prosecution attorney's arguments  
                - Follow the judge's instructions
                - Deliberate and reach a verdict based on evidence
                - Apply the standard of "beyond reasonable doubt"
                
                Provide your final verdict (Guilty/Not Guilty) with detailed reasoning explaining how you weighed the evidence and arguments from both sides.
                """,
                tools=[],
                config={},
                ins=["testimony", "defense_case", "prosecution_case", "judicial_assessment"],
                outs=["verdict"]
            )
        )
    ]
    
    # Define the data flow edges
    edges = [
        # Witness testimony goes to both attorneys (parallel processing)
        EdgeSpec(id="testimony_to_defense", source="witness_testimony", target="defense_attorney"),
        EdgeSpec(id="testimony_to_prosecution", source="witness_testimony", target="prosecution_attorney"),
        
        # All inputs go to judge (testimony + both attorney cases)
        EdgeSpec(id="testimony_to_judge", source="witness_testimony", target="judge"),
        EdgeSpec(id="defense_to_judge", source="defense_attorney", target="judge"),
        EdgeSpec(id="prosecution_to_judge", source="prosecution_attorney", target="judge"),
        
        # All inputs go to jury (testimony + both attorney cases + judge instructions)
        EdgeSpec(id="testimony_to_jury", source="witness_testimony", target="jury"),
        EdgeSpec(id="defense_to_jury", source="defense_attorney", target="jury"),
        EdgeSpec(id="prosecution_to_jury", source="prosecution_attorney", target="jury"),
        EdgeSpec(id="judge_to_jury", source="judge", target="jury"),
    ]
    
    return WorkflowSpec(
        id=uuid.uuid4(),
        rev=1,
        title="Court Simulation",
        description="A realistic courtroom trial simulation with witness testimony, attorneys, judge, and jury",
        nodes=nodes,
        edges=edges,
        metadata={
            "simulation_type": "court_trial",
            "participants": ["witness", "defense", "prosecution", "judge", "jury"],
            "workflow_pattern": "complex_dag_with_parallel_and_convergence"
        }
    )


async def run_court_simulation():
    """Run the court simulation and display results."""
    print("🏛️  COURT SIMULATION STARTING")
    print("=" * 60)
    
    # Create the workflow
    workflow = create_court_simulation_workflow()
    
    # Create DAG executor
    executor = create_dag_executor_from_spec(workflow)
    
    # Show execution plan
    summary = executor.get_execution_summary()
    print(f"📋 Execution Plan:")
    print(f"   Total nodes: {summary['total_nodes']}")
    print(f"   Total batches: {summary['total_batches']}")
    print(f"   Max parallelism: {summary['max_parallelism']}")
    print(f"   Execution order: {summary['execution_order']}")
    print()
    
    # Execute the simulation
    initial_state = WorkflowState(conversation_id="court_sim", initial_text="", results={})
    print("🚀 Starting trial proceedings...")
    print()
    
    final_state = await executor.execute_dag(initial_state)
    
    # Display results in courtroom format
    print("\n" + "=" * 60)
    print("📋 TRIAL PROCEEDINGS SUMMARY")
    print("=" * 60)
    
    # Witness Testimony
    if "witness_testimony" in final_state.results:
        print("\n👤 WITNESS TESTIMONY:")
        print("-" * 40)
        testimony = final_state.results["witness_testimony"]
        if hasattr(testimony, 'result') and testimony.result:
            print(f"'{testimony.result}'\n")
        else:
            print("'I saw the defendant near the bank at 3 PM. They were acting suspiciously, looking around nervously and checking their phone repeatedly. I distinctly remember they were wearing a red jacket and kept glancing at the security cameras.'\n")
    
    # Defense Case
    if "defense_attorney" in final_state.results:
        print("⚖️  DEFENSE ATTORNEY ARGUMENTS:")
        print("-" * 40)
        defense = final_state.results["defense_attorney"]
        if hasattr(defense, 'agent_response'):
            print(f"{defense.agent_response.result}\n")
    
    # Prosecution Case  
    if "prosecution_attorney" in final_state.results:
        print("⚔️  PROSECUTION ATTORNEY ARGUMENTS:")
        print("-" * 40)
        prosecution = final_state.results["prosecution_attorney"]
        if hasattr(prosecution, 'agent_response'):
            print(f"{prosecution.agent_response.result}\n")
    
    # Judge's Assessment
    if "judge" in final_state.results:
        print("👨‍⚖️ JUDGE'S INSTRUCTIONS TO JURY:")
        print("-" * 40)
        judge = final_state.results["judge"]
        if hasattr(judge, 'agent_response'):
            print(f"{judge.agent_response.result}\n")
    
    # Final Verdict
    if "jury" in final_state.results:
        print("🏛️  JURY VERDICT:")
        print("-" * 40)
        jury = final_state.results["jury"]
        if hasattr(jury, 'agent_response'):
            print(f"{jury.agent_response.result}\n")
    
    print("=" * 60)
    print("⚖️  TRIAL CONCLUDED")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_court_simulation())