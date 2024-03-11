import requests

def fetch_food_data(query):
    base_url = "https://api.nal.usda.gov/fdc/v1/foods/search"
    
    params = {
        "query": query,
        "dataType": "",
        "pageSize": 1,
        "pageNumber": 1,
        "sortBy": "dataType.keyword",
        "sortOrder": "asc",
        "api_key": "FymlvKnmRGKYrN3PafDZcPH4Body21TNsfuhDrh9"
    }
    
    try:
        response = requests.get(base_url, params=params)

        if response.status_code == 200:
            data = response.json()
            
            if 'foods' in data and len(data['foods']) > 0:
                food_item = data['foods'][0]
                
                calories = None
                protein = None
                serving_size = None
                
                for nutrient in food_item.get('foodNutrients', []):
                    nutrient_id = nutrient.get('nutrientId')
                    if nutrient_id == 1003:
                        protein = nutrient.get('value')
                    elif nutrient_id == 1008:
                        calories = nutrient.get('value')
                
                # Extract serving size
                serving_size = food_item.get('servingSize')
                
                return {
                    "calories": calories,
                    "protein": protein,
                    "serving_size": serving_size
                }
            else:
                print("No food data found for query:", query)
                return None
        else:
            print("Error: Unable to fetch data. Status code:", response.status_code)
            return None
    except Exception as e:
        print("Error:", e)
        return None

# # Example usage
# search_term = "Baked Potato"  # Specify the search term here
# food_data = fetch_food_data(search_term)
# if food_data:
#     print("Calories:", food_data["calories"])
#     print("Protein:", food_data["protein"])
#     print("Serving Size:", food_data["serving_size"])
