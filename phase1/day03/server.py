# A mock weather server — tools: get_current_weather, get_forecast, get_location_info

import sys
import httpx
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

# 1. Initialize FastMCP with custom error details masked 
# This prevents unintended runtime errors from leaking traces to stdout/stderr
mcp = FastMCP("Weather Intelligence Server", mask_error_details=True)

# Replace with your actual key or use an environment variable loader
API_KEY = "YOUR_WEATHERAPI_KEY"

# Ensure all logging strings strictly print to stderr to preserve clean stdio transport
print("Initializing Weather MCP Server over stdio transport...", file=sys.stderr)


# --- TOOL 1: LOCATION LOOKUP ---
@mcp.tool(
    name="lookup_location",
    description="Detects geographic location coordinates, city, and country metadata using an IP address. Use 'auto' to check the current client location."
)
async def lookup_location(ip: str) -> str:
    """
    Looks up location data for an IP address.
    
    :param ip: The target IP address or 'auto' for the client's current location. (Required)
    """
    # Explicit payload validation for invalid or blank strings
    if not ip or not ip.strip():
        raise ToolError("Invalid input: The 'ip' parameter cannot be blank or empty.")

    url = "https://api.weatherapi.com/v1/ip.json"
    
    async with httpx.AsyncClient() as client:
        try:
            print(f"Executing API request to geolocate IP: {ip}", file=sys.stderr)
            response = await client.get(url, params={"key": API_KEY, "q": ip.strip()})
            
            if response.status_code == 400:
                raise ToolError(f"MCP Validation Error: The provided IP '{ip}' is invalid or unrecognized by the geolocation registry.")
            
            response.raise_for_status()
            data = response.json()
            
            return (
                f"Location Detected:\n"
                f"- City: {data.get('city', 'Unknown')}\n"
                f"- Region: {data.get('region', 'Unknown')}\n"
                f"- Country: {data.get('country_name', 'Unknown')} ({data.get('country_code', '??')})\n"
                f"- Coordinates: {data.get('lat')}, {data.get('lon')}\n"
                f"- Timezone: {data.get('tz_id', 'Unknown')}"
            )
            
        except httpx.HTTPStatusError as e:
            raise ToolError(f"MCP Backend Error: Remote Geolocation API returned status {e.response.status_code}")
        except httpx.RequestError as e:
            raise ToolError(f"MCP Network Error: Unable to reach upstream geolocation service. Details: {str(e)}")


# --- TOOL 2: CURRENT WEATHER ---
@mcp.tool(
    name="get_current_weather",
    description="Fetches real-time weather metrics including temperatures, current conditions, and humidity for a specified city and country."
)
async def get_current_weather(city: str, country: str = "") -> str:
    """
    Fetches real-time weather details.
    
    :param city: The name of the target city (e.g., 'London' or 'Tokyo'). (Required)
    :param country: Optional country name or 2-letter ISO code to narrow down search results.
    """
    if not city or not city.strip():
        raise ToolError("Invalid input: The 'city' parameter is required and cannot be empty.")

    url = "https://api.weatherapi.com/v1/current.json"
    query = f"{city.strip()}, {country.strip()}".strip(", ")
    
    async with httpx.AsyncClient() as client:
        try:
            print(f"Executing API request for current weather: {query}", file=sys.stderr)
            response = await client.get(url, params={"key": API_KEY, "q": query})
            
            if response.status_code == 400:
                raise ToolError(f"MCP Validation Error: Location matching '{query}' could not be found.")
                
            response.raise_for_status()
            data = response.json()
            
            loc = data['location']
            current = data['current']
            
            return (
                f"Current Conditions for {loc['name']}, {loc['country']}:\n"
                f"- Weather status: {current['condition']['text']}\n"
                f"- Temperature: {current['temp_c']}°C ({current['temp_f']}°F)\n"
                f"- Humidity: {current['humidity']}%\n"
                f"- Wind: {current['wind_kph']} km/h coming from {current['wind_dir']}"
            )
            
        except httpx.HTTPStatusError as e:
            raise ToolError(f"MCP Backend Error: Upstream service responded with code {e.response.status_code}")
        except httpx.RequestError as e:
            raise ToolError(f"MCP Network Error: Failed to resolve weather service connection.")


# --- TOOL 3: WEATHER FORECAST ---
@mcp.tool(
    name="get_weather_forecast",
    description="Retrieves a structured multi-day forecast (1 to 3 days maximum) tracking temperatures, precipitation probability, and anticipated conditions."
)
async def get_weather_forecast(city: str, country: str = "", days: int = 3) -> str:
    """
    Generates a multi-day weather forecast outlook.
    
    :param city: The target city name. (Required)
    :param country: Optional country name/code modifier to limit scope.
    :param days: Number of forecast days to pull (Min: 1, Max: 3). Default is 3.
    """
    if not city or not city.strip():
        raise ToolError("Invalid input: The 'city' parameter is required and cannot be empty.")
        
    if not (1 <= days <= 3):
        raise ToolError("Validation Error: The 'days' parameter must be an integer between 1 and 3 inclusive.")

    url = "https://api.weatherapi.com/v1/forecast.json"
    query = f"{city.strip()}, {country.strip()}".strip(", ")
    
    async with httpx.AsyncClient() as client:
        try:
            print(f"Executing API request for {days}-day forecast: {query}", file=sys.stderr)
            response = await client.get(url, params={"key": API_KEY, "q": query, "days": days, "aqi": "no"})
            
            if response.status_code == 400:
                raise ToolError(f"MCP Validation Error: Unable to compile forecast. Location '{query}' not found.")
                
            response.raise_for_status()
            data = response.json()
            
            output = f"Weather Forecast Outlook for {data['location']['name']}, {data['location']['country']}:\n"
            for day in data["forecast"]["forecastday"]:
                day_info = day["day"]
                output += (
                    f"📅 {day['date']}:\n"
                    f"  - Summary: {day_info['condition']['text']}\n"
                    f"  - Range: High {day_info['maxtemp_c']}°C | Low {day_info['mintemp_c']}°C\n"
                    f"  - Rain Probability: {day_info['daily_chance_of_rain']}%\n"
                )
            return output
            
        except httpx.HTTPStatusError as e:
            raise ToolError(f"MCP Backend Error: Forecast endpoint returned status code {e.response.status_code}")
        except httpx.RequestError as e:
            raise ToolError(f"MCP Network Error: Network request dropped during execution.")


if __name__ == "__main__":
    # Bootstraps server strictly over stdio protocol
    mcp.run()