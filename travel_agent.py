from langchain_community.llms import Ollama
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool

# Function to generate travel advice
def travel_advice(destination: str):
    llm = Ollama(model="mistral")
    prompt = (f"Suggest the top tourist attractions, local foods, and travel tips for {destination}. "
        f"Also, include a budget estimate for a budget traveler, mid-range traveler, and luxury traveler."
        )
    return llm.invoke(prompt)

# Create the tool
travel_tool = Tool(
    name="Travel Advisor",
    func=travel_advice,
    description="Provide a city or country name to get travel recommendations with budget estimates."
)

# Initialize the agent
agent = initialize_agent(
    tools=[travel_tool],
    llm=Ollama(model="mistral"),
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

# Run the agent
destination = "Kerala"
response = agent.invoke(f"Suggest travel destinations in {destination} with budget estimates.")
print(response)
