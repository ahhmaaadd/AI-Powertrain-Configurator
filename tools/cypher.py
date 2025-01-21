import streamlit as st
from langchain_core.prompts import PromptTemplate
from llm import llm
from graph import graph
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain

CYPHER_GENERATION_TEMPLATE = """
You are a Neo4j Developer tasked with generating Cypher queries to construct vehicle powertrain configurations.

Rules for Generating Powertrains:
1. Start with a component that has no input energy.
2. Components can connect only if the energy output of one matches the energy input of the next.
3. Stop when you reach a component with no output energy.

Schema Overview:
Nodes:
- Component: Represents powertrain components. Properties:
    - name (e.g., "FuelTank")
    - type (Actuator, EnergyStorage, Transmission)
    - description (e.g., "Stores chemical energy as fuel for the engine")

- EnergyType: Represents types of energy. Properties:
    - name (ChemicalEnergy, ElectricalEnergy, MechanicalEnergy)

Relationships:
- HAS_INPUT: Links a Component to an EnergyType it accepts as input.
- HAS_OUTPUT: Links a Component to an EnergyType it produces as output.

Question: {question}

Thought Process:
1. Identify the starting component (no input energy).
2. Iteratively find components with matching energy outputs and inputs.
3. Stop when no further components can be connected.
4. Return the powertrain configuration as a sequence of connections.

Select one component at a time
Cypher Query:
"""


cypher_generation_prompt = PromptTemplate(
    template=CYPHER_GENERATION_TEMPLATE,
    input_variables=["schema", "question"],
)


# Create the Cypher QA chain
cypher_qa = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    cypher_prompt=cypher_generation_prompt,
    verbose=True,
    allow_dangerous_requests=True
)

