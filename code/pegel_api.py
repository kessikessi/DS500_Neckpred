import requests
import json
from datetime import datetime

API_URL = "https://www.hochwasserzentralen.de/webservices/get_infoservice_geojson.php"
STATION_ID = "00086"  # Kirchentellinsfurt

def get_pegel_data(station_id=STATION_ID):
    headers = {
        "User-Agent": "PegelMonitor/1.0"
    }
    params = {
        "bundesland": "BW"
    }

    try:
        response = requests.get(API_URL, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Search for Kirchentellinsfurt by pgnr or name
        for feature in data.get("features", []):
            props = feature.get("properties", {})
            
            # The station ID in the feed matches pgnr
            if props.get("pgnr") == station_id or props.get("id") == station_id:
                return {
                    "station_name": props.get("pegelname"),
                    "river": props.get("gewaesser"),
                    "station_id": props.get("pgnr"),
                    "water_level_cm": props.get("w"),
                    "discharge_m3s": props.get("q"),
                    "date": props.get("datum"),
                    "time": props.get("uhrzeit"),
                    "trend": props.get("trend"),
                    "warning_level": props.get("warnstufe"),
                    "fetched_at": datetime.now().isoformat()
                }
                
        print(f"Station {station_id} not found in the response.")
        return None

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

if __name__ == "__main__":
    result = get_pegel_data()
    if result:
        print(json.dumps(result, indent=4, ensure_ascii=False))