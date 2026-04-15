from langchain_community.llms import Ollama
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun
import json

# Function to generate travel advice with budgeting
def travel_advice(destination: str):
    llm = Ollama(model="mistral")
    prompt = f"""Suggest the top tourist attractions, local foods, and travel tips for {destination}.
    Include an estimated budget for budget travelers, mid-range travelers, and luxury travelers."""
    return llm.invoke(prompt)

# Budget estimation function
def estimate_budget(destination: str):
    llm = Ollama(model="mistral")
    prompt = f"""Estimate the cost of travel to {destination} for:
    - Budget travelers
    - Mid-range travelers
    - Luxury travelers
    Include accommodation, food, transportation, and activities."""
    return llm.invoke(prompt)

# Internet search tool
search_tool = DuckDuckGoSearchRun()

# Formatted response function
def formatted_response(destination: str):
    advice = travel_advice(destination)
    budget = estimate_budget(destination)
    result = {
        "Destination": destination,
        "Travel Advice": advice,
        "Estimated Budget": budget
    }
    return json.dumps(result, indent=4)

# Create tools
travel_tool = Tool(
    name="Travel Advisor",
    func=travel_advice,
    description="Provide a city or country name to get travel recommendations."
)

budget_tool = Tool(
    name="Budget Estimator",
    func=estimate_budget,
    description="Estimate budget for budget, mid-range, and luxury travelers."
)

formatted_tool = Tool(
    name="Formatted Travel Guide",
    func=formatted_response,
    description="Get a structured travel guide with advice and budget."
)

# Initialize the agent
agent = initialize_agent(
    tools=[travel_tool, budget_tool, search_tool, formatted_tool],
    llm=Ollama(model="mistral"),
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

# Run the agent
destination = "Kerala"
response = agent.invoke(f"Give me a structured travel guide for {destination} with budget estimates.")
print(response)
