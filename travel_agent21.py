import json
import requests
from datetime import datetime
from langchain_community.llms import Ollama
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool

# Load the Ollama model
llm = Ollama(model="mistral")

# OpenWeatherMap API Configuration
WEATHER_API_KEY = "0a7d6d34385cb649b96ffdb5f9de495c"  # Replace with your actual API key
WEATHER_API_URL = "https://api.openweathermap.org/data/2.5/forecast"

# Function to fetch multi-day weather forecast
def get_weather_forecast(destination, days):
    try:
        params = {
            "q": destination,
            "appid": WEATHER_API_KEY,
            "units": "metric",
            "cnt": days * 8  # OpenWeatherMap provides data in 3-hour intervals (8 per day)
        }
        response = requests.get(WEATHER_API_URL, params=params)
        weather_data = response.json()

        if response.status_code == 200:
            daily_forecast = {}
            for forecast in weather_data["list"]:
                date = forecast["dt_txt"].split(" ")[0]  # Extract date
                temp = forecast["main"]["temp"]
                feels_like = forecast["main"]["feels_like"]
                weather_desc = forecast["weather"][0]["description"].capitalize()
                wind_speed = forecast["wind"]["speed"]

                if date not in daily_forecast:
                    formatted_date = datetime.strptime(date, "%Y-%m-%d").strftime("%B %d")
                    daily_forecast[date] = (f"🌤 Weather: {weather_desc}, {temp}°C "
                                            f"(Feels Like {feels_like}°C), "
                                            f"Wind Speed: {wind_speed} m/s")
            return daily_forecast
        else:
            return {"Error": "Weather data unavailable."}
    except Exception:
        return {"Error": "Weather data unavailable."}

# Function to generate weather-aware travel advice
def travel_advice(destination: str, days: int, budget: int):
    weather_forecast = get_weather_forecast(destination, days)
    
    if "Error" in weather_forecast:
        return weather_forecast["Error"]

    weather_info = "\n".join([f"{datetime.strptime(date, '%Y-%m-%d').strftime('%B %d')} - {forecast}"
                               for date, forecast in weather_forecast.items()])

    prompt = (
        f"I am planning a {days}-day trip to {destination} with a budget of ₹{budget}. "
        f"Here is the weather forecast:\n{weather_info}\n\n"
        f"Generate a detailed daily itinerary for this trip, including activities, attractions, meals, "
        f"and estimated costs. Ensure outdoor activities match the weather conditions."
    )

    response = llm.invoke(prompt)

    itinerary_lines = response.split("\n")
    formatted_itinerary = ""

    day_counter = 1
    for date, weather in weather_forecast.items():
        formatted_date = datetime.strptime(date, "%Y-%m-%d").strftime("%B %d")
        formatted_itinerary += f"\nDay {day_counter} ({formatted_date})\n{weather}\n"
        
        for line in itinerary_lines:
            if f"Day {day_counter}" in line:
                formatted_itinerary += f"{line}\n"
        
        day_counter += 1
    
    return formatted_itinerary.strip()

# Travel Planning Tool
def travel_planner_tool(input_text):
    try:
        input_data = json.loads(input_text)
        destination = input_data.get("destination", "Unknown")
        days = int(input_data.get("days", 1))
        budget = int(input_data.get("budget", 10000))
        
        return travel_advice(destination, days, budget)
    except Exception as e:
        return f"Error processing request: {str(e)}"

# Define the tool for the agent
travel_tool = Tool(
    name="Travel Planner",
    func=travel_planner_tool,
    description="Provide a structured trip plan including itinerary, budget, travel tips, and live weather updates."
)

# Initialize the agent
agent = initialize_agent(
    tools=[travel_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

# === CLI User Input ===
destination = input("Enter your travel destination(s): ")
days = int(input("Enter number of days for the trip: "))
budget = int(input("Enter your budget in INR: "))

# Creating structured input for AI agent
user_input = {"destination": destination, "days": days, "budget": budget}

# Generate the travel plan
response = agent.invoke({"input": json.dumps(user_input)})

# Extract and format the response
response_text = response["output"] if isinstance(response, dict) and "output" in response else str(response)

# Get real-time weather data
weather_info = get_weather_forecast(destination, days)

# === Print Structured Travel Plan ===
print("\n=== Travel Plan ===\n")
print(f"**Destination:** {destination}")
print(f"**Trip Duration:** {days} days")
print(f"**Budget:** ₹{budget}")
print("\n**Weather Forecast:**")
for date, forecast in weather_info.items():
    formatted_date = datetime.strptime(date, "%Y-%m-%d").strftime("%B %d")
    print(f"{formatted_date}: {forecast}")

print("\n**Generated Itinerary:**\n")
print(response_text)
