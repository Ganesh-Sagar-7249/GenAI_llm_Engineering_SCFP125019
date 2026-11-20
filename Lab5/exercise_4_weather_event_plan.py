from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
import requests
import os

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

class EventDetails(BaseModel):
    event_type: str
    event_date: str
    location: str
    indoor_outdoor_preference: str

class WeatherData(BaseModel):
    temperature: float
    precipitation_chance: int
    conditions: str

parser = PydanticOutputParser(pydantic_object=EventDetails)

prompt = PromptTemplate(
    template="Extract the event type, date, location, and indoor/outdoor preference from this description:\n{description}\n{format_instructions}",
    input_variables=["description"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | llm | parser

description = input("Enter event description: ")
event = chain.invoke({"description": description})

print("\nStep 1: Event Details")
print(event.model_dump_json(indent=2))

geo = requests.get(
    "https://geocoding-api.open-meteo.com/v1/search",
    params={"name": event.location, "count": 1},
    timeout=20
)
geo.raise_for_status()
places = geo.json().get("results", [])

if not places:
    raise ValueError("Location not found. Enter a more specific city or town.")

place = places[0]

weather = requests.get(
    "https://api.open-meteo.com/v1/forecast",
    params={
        "latitude": place["latitude"],
        "longitude": place["longitude"],
        "daily": "temperature_2m_max,precipitation_probability_max,weather_code",
        "timezone": "auto",
        "start_date": event.event_date,
        "end_date": event.event_date
    },
    timeout=20
)
weather.raise_for_status()
daily = weather.json().get("daily", {})

if not daily.get("time"):
    raise ValueError("Weather forecast is unavailable for that date.")

code = daily["weather_code"][0]
conditions = (
    "Clear" if code == 0 else
    "Partly cloudy" if code in [1, 2] else
    "Cloudy or foggy" if code in [3, 45, 48] else
    "Rain or showers" if code in range(51, 83) else
    "Thunderstorm or other severe conditions" if code >= 95 else
    "Snow or other precipitation"
)

data = WeatherData(
    temperature=daily["temperature_2m_max"][0],
    precipitation_chance=daily["precipitation_probability_max"][0],
    conditions=conditions
)

print("\nStep 2: Weather Data")
print(data.model_dump_json(indent=2))

prompt3 = ChatPromptTemplate.from_messages([
    ("system", "You are an event planning assistant. Recommend whether to proceed, reschedule, or move indoors. Use the event and weather data, and explain the reason briefly."),
    ("human", "Event: {event}\nWeather: {weather}\nGive a practical recommendation.")
])

chain3 = prompt3 | llm | StrOutputParser()

recommendation = chain3.invoke({
    "event": event.model_dump_json(),
    "weather": data.model_dump_json()
})

print("\nStep 3: Event Recommendation")
print(recommendation)