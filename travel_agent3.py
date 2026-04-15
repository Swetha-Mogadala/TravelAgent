from langchain_community.llms import Ollama
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun
import json

# Function to generate travel advice
def travel_advice(destination: str):
    llm = Ollama(model="mistral")
    prompt = f"""Suggest the top tourist attractions, local foods, and travel tips for {destination}.
    Include safety tips and cultural etiquette."""
    return llm.invoke(prompt)

# Budget estimation function
def estimate_budget(destination: str):
    llm = Ollama(model="mistral")
    prompt = f"""Estimate the cost of travel to {destination} for:
    - Budget travelers
    - Mid-range travelers
    - Luxury travelers
    Break down the costs for accommodation, food, transportation, and activities."""
    return llm.invoke(prompt)

# Trip planning function
def plan_trip(destination: str):
    llm = Ollama(model="mistral")
    prompt = f"""Plan a 7-day trip to {destination}. Include:
    - Daily activities
    - Accommodation recommendations
    - Local food options
    - Transport details
    - Estimated budget per day for budget, mid-range, and luxury travelers."""
    return llm.invoke(prompt)

# Live travel offers search
def search_travel_offers(destination: str):
    search_tool = DuckDuckGoSearchRun()
    query = f"best travel deals to {destination} including flights and hotels"
    return search_tool.run(query)

# Formatted response function
def formatted_response(destination: str):
    advice = travel_advice(destination)
    budget = estimate_budget(destination)
    itinerary = plan_trip(destination)
    offers = search_travel_offers(destination)

    result = {
        "Destination": destination,
        "Travel Advice": advice,
        "Estimated Budget": budget,
        "Itinerary": itinerary,
        "Live Travel Offers": offers
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

trip_planner_tool = Tool(
    name="Trip Planner",
    func=plan_trip,
    description="Plan a 7-day trip including itinerary, accommodation, and food."
)

travel_offers_tool = Tool(
    name="Live Travel Offers",
    func=search_travel_offers,
    description="Fetches live travel deals including flights and hotels."
)

formatted_tool = Tool(
    name="Formatted Travel Guide",
    func=formatted_response,
    description="Get a structured travel guide with advice, itinerary, and budget."
)

# Initialize the agent
agent = initialize_agent(
    tools=[travel_tool, budget_tool, trip_planner_tool, travel_offers_tool, formatted_tool],
    llm=Ollama(model="mistral"),
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

# Run the agent
destination = "Kerala"
response = agent.invoke(f"Plan a structured 7-day trip to {destination} with budget and live travel offers.")
print(response)
