import os
import gradio as gr
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import sqlite3
from pathlib import Path
import requests
import json
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Configuration class with proper boolean handling"""
    mistral_api_key: str
    exa_api_key: str
    firecrawl_api_key: Optional[str]
    database_url: str
    gradio_host: str
    gradio_port: int
    gradio_share: bool
    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection: str
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Load configuration from environment variables with proper type conversion"""
        # Fix for boolean handling - convert string to actual boolean
        gradio_share = os.getenv('GRADIO_SHARE', 'false').lower()
        gradio_share_bool = gradio_share in ['true', '1', 'yes', 'on']  # This is correct
        
        return cls(
            mistral_api_key=os.getenv('MISTRAL_API_KEY', ''),
            exa_api_key=os.getenv('EXA_API_KEY', ''),
            firecrawl_api_key=os.getenv('FIRECRAWL_API_KEY'),
            database_url=os.getenv('DATABASE_URL', 'nutrisense_data.db'),
            gradio_host=os.getenv('GRADIO_HOST', '127.0.0.1'),
            gradio_port=int(os.getenv('GRADIO_PORT', '7860')),
            gradio_share=gradio_share_bool,
            qdrant_url=os.getenv('QDRANT_URL', ''),
            qdrant_api_key=os.getenv('QDRANT_API_KEY', ''),
            qdrant_collection=os.getenv('QDRANT_COLLECTION', 'nutrisense_memories')
        )

class NutrisenseAssistant:
    """Main assistant class with proper error handling"""
    
    def __init__(self, config: Config):
        self.config = config
        self.db_path = Path(config.database_url)
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with user preferences support"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Existing nutrition logs table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS nutrition_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT,
                        food_item TEXT,
                        calories REAL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # User preferences table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS user_preferences (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT UNIQUE,
                        dietary_restrictions TEXT,  -- JSON array of restrictions
                        food_allergies TEXT,        -- JSON array of allergies
                        cuisine_preferences TEXT,   -- JSON array of preferred cuisines
                        health_goals TEXT,          -- JSON object with goals
                        weight_goal REAL,           -- Target weight
                        current_weight REAL,        -- Current weight
                        activity_level TEXT,        -- sedentary, light, moderate, active, very_active
                        age INTEGER,
                        gender TEXT,
                        height_cm REAL,
                        daily_calorie_target INTEGER,
                        protein_target REAL,
                        carb_target REAL,
                        fat_target REAL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.commit()
                logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
    
    def validate_input(self, user_input: str) -> bool:
        """Validate user input with proper boolean handling"""
        if not user_input or not isinstance(user_input, str):
            return False
        
        # Common fix for boolean iteration error
        forbidden_words = ['spam', 'abuse', 'illegal']  # This should be a list
        
        # WRONG: if 'spam' in True  # This would cause the error
        # RIGHT: Check if any forbidden word is in the input
        return not any(word in user_input.lower() for word in forbidden_words)
    
    def save_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> str:
        """Save or update user preferences"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Convert lists/dicts to JSON strings
                dietary_restrictions = json.dumps(preferences.get('dietary_restrictions', []))
                food_allergies = json.dumps(preferences.get('food_allergies', []))
                cuisine_preferences = json.dumps(preferences.get('cuisine_preferences', []))
                health_goals = json.dumps(preferences.get('health_goals', {}))
                
                # Use INSERT OR REPLACE to handle updates
                conn.execute('''
                    INSERT OR REPLACE INTO user_preferences 
                    (user_id, dietary_restrictions, food_allergies, cuisine_preferences, 
                     health_goals, weight_goal, current_weight, activity_level, age, 
                     gender, height_cm, daily_calorie_target, protein_target, 
                     carb_target, fat_target, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (
                    user_id, dietary_restrictions, food_allergies, cuisine_preferences,
                    health_goals, preferences.get('weight_goal'), preferences.get('current_weight'),
                    preferences.get('activity_level'), preferences.get('age'),
                    preferences.get('gender'), preferences.get('height_cm'),
                    preferences.get('daily_calorie_target'), preferences.get('protein_target'),
                    preferences.get('carb_target'), preferences.get('fat_target')
                ))
                conn.commit()
                return "✅ Preferences saved successfully!"
        except Exception as e:
            logger.error(f"Error saving preferences: {e}")
            return f"❌ Error saving preferences: {str(e)}"
    
    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Retrieve user preferences"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT * FROM user_preferences WHERE user_id = ?", (user_id,)
                )
                row = cursor.fetchone()
                
                if not row:
                    return {}
                
                # Convert row to dict
                columns = [desc[0] for desc in cursor.description]
                preferences = dict(zip(columns, row))
                
                # Parse JSON fields
                try:
                    preferences['dietary_restrictions'] = json.loads(preferences.get('dietary_restrictions', '[]'))
                    preferences['food_allergies'] = json.loads(preferences.get('food_allergies', '[]'))
                    preferences['cuisine_preferences'] = json.loads(preferences.get('cuisine_preferences', '[]'))
                    preferences['health_goals'] = json.loads(preferences.get('health_goals', '{}'))
                except json.JSONDecodeError:
                    logger.warning(f"JSON decode error for user {user_id} preferences")
                
                return preferences
                
        except Exception as e:
            logger.error(f"Error retrieving preferences: {e}")
            return {}
    
    def _generate_smart_search_queries(self, location: str, user_prefs: Dict[str, Any], cuisine: str = "") -> List[str]:
        """Generate intelligent search queries using LLM based on user preferences, with fallback"""
        
        # Try AI-powered query generation first
        if self.config.mistral_api_key:
            try:
                # Build context about user preferences
                context_parts = [f"Location: {location}"]
                
                if cuisine:
                    context_parts.append(f"Cuisine preference: {cuisine}")
                
                if user_prefs.get('dietary_restrictions'):
                    dietary_list = ', '.join(user_prefs['dietary_restrictions'])
                    context_parts.append(f"Dietary restrictions: {dietary_list}")
                
                if user_prefs.get('food_allergies'):
                    allergy_list = ', '.join(user_prefs['food_allergies'])
                    context_parts.append(f"Food allergies: {allergy_list}")
                
                if user_prefs.get('health_goals'):
                    goals = [k.replace('_', ' ') for k, v in user_prefs['health_goals'].items() if v]
                    if goals:
                        context_parts.append(f"Health goals: {', '.join(goals)}")
                
                context = '. '.join(context_parts)
                
                # Create prompt for LLM
                prompt = f"""Based on the following user context, generate 3 diverse and effective search queries to find restaurants that match their needs:

{context}

Generate 3 different search queries that would find relevant restaurants. Each query should:
1. Be specific and targeted for web search
2. Include the location
3. Incorporate the user's dietary needs naturally
4. Use different search strategies (broad, specific, health-focused)

Format: Return only the 3 queries, one per line, no numbering or bullets.

Example format:
best vegetarian restaurants in Mumbai for weight loss
Mumbai healthy dining gluten-free options reviews
vegetarian restaurants Mumbai nutritious meals"""

                # Call Mistral AI
                headers = {
                    "Authorization": f"Bearer {self.config.mistral_api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": "mistral-small-latest",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 200,
                    "temperature": 0.7
                }
                
                response = requests.post(
                    "https://api.mistral.ai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=15
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result['choices'][0]['message']['content'].strip()
                    queries = [q.strip() for q in content.split('\n') if q.strip()]
                    
                    if len(queries) >= 3:
                        logger.info(f"✅ Generated {len(queries)} smart search queries using Mistral AI")
                        return queries[:3]
                    else:
                        logger.warning(f"⚠️ AI generated only {len(queries)} queries, falling back to basic queries")
                else:
                    logger.warning(f"⚠️ Mistral AI failed with status {response.status_code}, falling back to basic queries")
                    
            except Exception as e:
                logger.warning(f"⚠️ Mistral AI error: {e}, falling back to basic queries")
        
        # Fallback to basic query generation when AI is not available or fails
        logger.info("🔄 Using fallback query generation (AI not available)")
        return self._generate_fallback_queries(location, user_prefs, cuisine)
    
    def _generate_fallback_queries(self, location: str, user_prefs: Dict[str, Any], cuisine: str = "") -> List[str]:
        """Generate basic search queries when AI is not available"""
        queries = []
        
        # Build dietary restriction string
        dietary_terms = []
        if user_prefs.get('dietary_restrictions'):
            for restriction in user_prefs['dietary_restrictions']:
                if 'vegetarian' in restriction.lower():
                    dietary_terms.append('vegetarian')
                elif 'vegan' in restriction.lower():
                    dietary_terms.append('vegan')
                elif 'pescatarian' in restriction.lower():
                    dietary_terms.append('pescatarian')
                elif 'keto' in restriction.lower():
                    dietary_terms.append('keto')
                elif 'gluten' in restriction.lower():
                    dietary_terms.append('gluten-free')
                elif 'halal' in restriction.lower():
                    dietary_terms.append('halal')
                elif 'kosher' in restriction.lower():
                    dietary_terms.append('kosher')
        
        # Build health goal terms
        health_terms = []
        if user_prefs.get('health_goals'):
            if user_prefs['health_goals'].get('weight_loss'):
                health_terms.append('healthy')
            if user_prefs['health_goals'].get('muscle_gain'):
                health_terms.append('protein-rich')
        
        # Generate Query 1: Basic location + dietary
        dietary_str = ' '.join(dietary_terms) if dietary_terms else ''
        cuisine_str = cuisine if cuisine else 'restaurants'
        query1 = f"best {dietary_str} {cuisine_str} in {location}".strip()
        queries.append(query1)
        
        # Generate Query 2: Location + health focus
        health_str = ' '.join(health_terms) if health_terms else 'good'
        query2 = f"{health_str} restaurants {location} reviews".strip()
        queries.append(query2)
        
        # Generate Query 3: Platform-specific search
        platform_query = f"{location} restaurants"
        if dietary_terms:
            platform_query += f" {dietary_terms[0]}"
        platform_query += " zomato swiggy"
        queries.append(platform_query)
        
        logger.info(f"🔄 Generated {len(queries)} fallback queries: {queries}")
        return queries
    

    
    def _get_ai_nutrition_advice(self, query: str, user_prefs: Dict[str, Any]) -> str:
        """Get personalized nutrition advice using Mistral AI"""
        try:
            if not self.config.mistral_api_key:
                return "⚠️ AI nutrition advice requires Mistral API configuration. Please set MISTRAL_API_KEY in your environment variables."
            
            # Build user context
            context_parts = []
            
            if user_prefs.get('age'):
                context_parts.append(f"Age: {user_prefs['age']}")
            if user_prefs.get('gender'):
                context_parts.append(f"Gender: {user_prefs['gender']}")
            if user_prefs.get('current_weight') and user_prefs.get('height_cm'):
                context_parts.append(f"Current weight: {user_prefs['current_weight']}kg, Height: {user_prefs['height_cm']}cm")
            if user_prefs.get('weight_goal'):
                context_parts.append(f"Weight goal: {user_prefs['weight_goal']}kg")
            if user_prefs.get('activity_level'):
                context_parts.append(f"Activity level: {user_prefs['activity_level']}")
            if user_prefs.get('daily_calorie_target'):
                context_parts.append(f"Daily calorie target: {user_prefs['daily_calorie_target']} calories")
            if user_prefs.get('dietary_restrictions'):
                restrictions = ', '.join(user_prefs['dietary_restrictions'])
                context_parts.append(f"Dietary restrictions: {restrictions}")
            if user_prefs.get('food_allergies'):
                allergies = ', '.join(user_prefs['food_allergies'])
                context_parts.append(f"Food allergies: {allergies}")
            if user_prefs.get('health_goals'):
                goals = [k.replace('_', ' ').title() for k, v in user_prefs['health_goals'].items() if v]
                if goals:
                    context_parts.append(f"Health goals: {', '.join(goals)}")
            
            # Create personalized prompt
            if context_parts:
                user_context = "User Profile: " + " | ".join(context_parts)
            else:
                user_context = "User Profile: No specific preferences provided"
            
            prompt = f"""You are a certified nutritionist and registered dietitian providing personalized nutrition advice.

{user_context}

User Question: {query}

Provide comprehensive, evidence-based nutrition advice tailored to this user's specific profile. Include:
1. Specific recommendations based on their goals and restrictions
2. Practical meal suggestions and food choices
3. Portion guidance and timing if relevant
4. Any special considerations for their dietary restrictions or health goals
5. Scientific rationale when appropriate

Keep the response helpful, actionable, and professional. Use emojis to make it engaging but maintain credibility."""

            # Call Mistral AI
            headers = {
                "Authorization": f"Bearer {self.config.mistral_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "mistral-small-latest",
                "messages": [
                    {"role": "system", "content": "You are an expert nutritionist and registered dietitian providing personalized, evidence-based nutrition advice."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 800,
                "temperature": 0.7
            }
            
            response = requests.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result['choices'][0]['message']['content'].strip()
                logger.info("✅ Generated personalized AI nutrition advice")
                return ai_response
            else:
                logger.error(f"Mistral AI error: {response.status_code}")
                return f"⚠️ Unable to generate AI advice at the moment. Status: {response.status_code}. Please try again or check your API configuration."
                
        except Exception as e:
            logger.error(f"AI nutrition advice error: {e}")
            return f"⚠️ Error generating AI advice: {str(e)}. Please check your Mistral API configuration and try again."
    
    def generate_workout_plan(self, user_prefs: Dict[str, Any], query: str) -> str:
        """Generate personalized workout plan using AI - no fallbacks"""
        if not self.config.mistral_api_key:
            raise Exception("🔑 MISTRAL_API_KEY required for personalized workout plans. Please configure your API key.")
        
        # Build comprehensive user context
        context_parts = []
        if user_prefs.get('age'): 
            context_parts.append(f"Age: {user_prefs['age']} years")
        if user_prefs.get('gender'): 
            context_parts.append(f"Gender: {user_prefs['gender']}")
        if user_prefs.get('current_weight') and user_prefs.get('height_cm'):
            context_parts.append(f"Weight: {user_prefs['current_weight']}kg, Height: {user_prefs['height_cm']}cm")
        if user_prefs.get('activity_level'): 
            context_parts.append(f"Activity Level: {user_prefs['activity_level']}")
        if user_prefs.get('health_goals'): 
            goals = [k.replace('_', ' ').title() for k, v in user_prefs['health_goals'].items() if v]
            if goals:
                context_parts.append(f"Goals: {', '.join(goals)}")
        if user_prefs.get('dietary_restrictions'):
            context_parts.append(f"Diet: {', '.join(user_prefs['dietary_restrictions'])}")
        
        user_context = " | ".join(context_parts) if context_parts else "General fitness guidance needed"
        
        prompt = f'''You are a certified personal trainer and sports nutritionist. Create a comprehensive, personalized workout and nutrition plan.

User Profile: {user_context}
User Request: {query}

Provide a detailed plan including:

🏋️ **WORKOUT PLAN:**
- Specific exercises tailored to their fitness level and goals
- Sets, reps, and rest periods
- Weekly training frequency and schedule
- Progression plan for next 4-8 weeks

🍎 **NUTRITION TIMING:**  
- Pre-workout meal recommendations (timing and foods)
- Post-workout recovery nutrition
- Daily macro distribution based on their goals
- Hydration strategy

💪 **RECOVERY & PROGRESSION:**
- Rest day activities
- Sleep recommendations
- Signs of overtraining to watch for
- How to progress safely

Make it specific, actionable, and completely tailored to their profile. Include scientific rationale where helpful.'''

        headers = {
            "Authorization": f"Bearer {self.config.mistral_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "mistral-small-latest",
            "messages": [
                {"role": "system", "content": "You are an expert personal trainer and sports nutritionist providing evidence-based, personalized fitness and nutrition advice."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1200,
            "temperature": 0.7
        }
        
        response = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=20
        )
        
        if response.status_code != 200:
            raise Exception(f"Mistral AI failed with status {response.status_code}: {response.text}")
        
        result = response.json()
        ai_response = result['choices'][0]['message']['content'].strip()
        logger.info("✅ Generated personalized workout plan using AI")
        return ai_response

    
    def search_restaurants(self, location: str, cuisine: str = "", user_id: str = None) -> Dict[str, Any]:
        """Search for restaurants using Exa API with personalized search based on user preferences"""
        try:
            if not self.config.exa_api_key:
                return {"error": "🔑 EXA_API_KEY not configured. Please set your Exa API key in environment variables to enable restaurant search."}
            
            logger.info(f"🔍 Starting restaurant search for '{location}' with cuisine '{cuisine}' for user '{user_id}'")
            
            headers = {
                "X-API-Key": self.config.exa_api_key,
                "Content-Type": "application/json"
            }
            
            # Get user preferences for personalized search
            user_prefs = {}
            if user_id:
                user_prefs = self.get_user_preferences(user_id)
                logger.info(f"📋 User preferences: {len(user_prefs)} items loaded")
            
            # Use preferred cuisine from user preferences if none specified
            if not cuisine and user_prefs.get('cuisine_preferences'):
                cuisine = user_prefs['cuisine_preferences'][0] if user_prefs['cuisine_preferences'] else ""
                logger.info(f"🍽️ Using preferred cuisine from profile: {cuisine}")
            
            # Generate AI-powered search queries (no fallbacks - force real AI usage)
            search_queries = self._generate_smart_search_queries(location, user_prefs, cuisine)
            logger.info(f"🤖 AI generated {len(search_queries)} smart queries: {search_queries}")
            
            logger.info(f"🔍 Total queries to try: {len(search_queries)}")
            
            # Try multiple search strategies to get more comprehensive results
            all_results = []
            successful_queries = 0
            
            for i, query in enumerate(search_queries[:6], 1):  # Limit to 6 queries max
                try:
                    logger.info(f"Query {i}: '{query}'")
                    
                    payload = {
                        "query": query,
                        "type": "keyword",
                        "useAutoprompt": True,
                        "numResults": 4,
                        "contents": {
                            "text": True
                        },
                        "includeOrigin": ["zomato.com", "tripadvisor.com", "opentable.com", "yelp.com"]  # Focus on restaurant sites
                    }
                    
                    response = requests.post(
                        "https://api.exa.ai/search",
                        headers=headers,
                        json=payload,
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        results = data.get("results", [])
                        all_results.extend(results)
                        successful_queries += 1
                        logger.info(f"✅ Query '{query}' found {len(results)} results")
                        
                        # If we have enough results, stop searching
                        if len(all_results) >= 10:
                            logger.info(f"🎯 Got enough results ({len(all_results)}), stopping search")
                            break
                    else:
                        logger.warning(f"❌ Query '{query}' failed: {response.status_code} - {response.text[:100]}")
                        
                except Exception as e:
                    logger.warning(f"❌ Query '{query}' exception: {str(e)}")
                    continue
            
            # Remove duplicates based on URL
            seen_urls = set()
            unique_results = []
            for result in all_results:
                url = result.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    unique_results.append(result)
            
            logger.info(f"🎯 Found {len(unique_results)} unique restaurant results for {location} from {successful_queries} successful queries")
            
            if not unique_results:
                logger.warning(f"⚠️ No results found for {location} after trying {len(search_queries)} queries")
                return {
                    "results": [],
                    "total_found": 0,
                    "debug_info": f"Tried {len(search_queries)} search queries, {successful_queries} successful API calls, but got 0 unique results"
                }
            
            return {
                "results": unique_results[:8],  # Return top 8 unique results
                "total_found": len(unique_results),
                "debug_info": f"Successfully found {len(unique_results)} restaurants using {successful_queries} successful queries"
            }
                
        except Exception as e:
            logger.error(f"❌ Restaurant search error: {e}")
            return {"error": f"Search failed: {str(e)}. Please check your EXA_API_KEY configuration."}
    
    def format_restaurant_results(self, search_results: Dict[str, Any], location: str, user_prefs: Dict[str, Any] = None) -> str:
        """Format restaurant search results with personalized recommendations"""
        try:
            if "error" in search_results:
                error_msg = search_results['error']
                if "EXA_API_KEY" in error_msg:
                    return f"🔑 **Restaurant Search Not Available**\n\n{error_msg}\n\n**To enable real-time restaurant search:**\n1. Get an API key from https://exa.ai\n2. Add `EXA_API_KEY=your_key_here` to your .env file\n3. Restart the application\n\n**Meanwhile, here are dining tips for {location}:**\n• Check Zomato, Google Maps, or TripAdvisor for reviews\n• Look for restaurants with healthy menu options\n• Consider the cooking methods - grilled, baked, or steamed are better choices"
                else:
                    return f"⚠️ **Search Issue:** {error_msg}\n\n**General dining tips for {location}:**\n• Look for restaurants with good reviews on Google Maps or Zomato\n• Check for healthy options like grilled proteins and fresh vegetables\n• Consider the cooking methods - grilled, baked, or steamed are better choices\n• Watch portion sizes and consider sharing dishes"
            
            results = search_results.get("results", [])
            total_found = search_results.get("total_found", len(results))
            debug_info = search_results.get("debug_info", "")
            
            if not results:
                debug_msg = f"\n\n🔍 **Debug Info:** {debug_info}" if debug_info else ""
                return f"🤔 **No Restaurant Results Found**\n\nI searched extensively but couldn't find specific restaurant recommendations for **{location}** right now. This could be due to:\n\n• **Location specificity**: Try a broader area (e.g., 'Mumbai' instead of 'Bandra West')\n• **Search timing**: Restaurant databases might be updating\n• **API limitations**: Some regions have limited coverage\n\n**Meanwhile, here are proven ways to find great restaurants:**\n• 🔍 **Zomato/Swiggy**: Best for Indian locations with reviews and ratings\n• 🗺️ **Google Maps**: Search 'restaurants near [location]' with photos and reviews\n• ✈️ **TripAdvisor**: Great for tourist areas and popular spots\n• 👥 **Ask locals**: Social media groups or friends in the area{debug_msg}"
            
            # Header with count
            response = f"🍽️ **Found {total_found} restaurant recommendations for {location}:**\n\n"
            
            # Show up to 5 results for better user experience
            for i, result in enumerate(results[:5], 1):
                title = result.get("title", "Restaurant")
                url = result.get("url", "")
                text = result.get("text", "")
                
                response += f"**{i}. {title}**\n"
                if url:
                    response += f"🔗 {url}\n"
                
                # Extract and clean relevant info from text snippet
                if text:
                    # Clean up the text and take meaningful snippet
                    clean_text = text.replace('\n', ' ').replace('\r', ' ')
                    # Look for sentences that contain useful info
                    sentences = clean_text.split('. ')
                    useful_info = []
                    
                    for sentence in sentences[:3]:  # Check first 3 sentences
                        sentence = sentence.strip()
                        if len(sentence) > 20 and any(word in sentence.lower() for word in 
                            ['restaurant', 'food', 'cuisine', 'dining', 'menu', 'serves', 'offers', 'location', 'rating']):
                            useful_info.append(sentence)
                    
                    if useful_info:
                        snippet = '. '.join(useful_info[:2])  # Use up to 2 useful sentences
                        if len(snippet) > 250:
                            snippet = snippet[:250] + "..."
                        response += f"ℹ️ {snippet}\n"
                    else:
                        # Fallback to first 200 chars if no useful sentences found
                        snippet = clean_text[:200].strip()
                        if len(clean_text) > 200:
                            snippet += "..."
                        response += f"ℹ️ {snippet}\n"
                
                response += "\n"
            
            # Show if there are more results
            if total_found > 5:
                response += f"_... and {total_found - 5} more recommendations found!_\n\n"
            
            # Add personalized nutrition advice based on user preferences
            response += "**💡 Personalized Dining Tips for Your Visit:**\n"
            
            # Base healthy tips
            response += "• **Look for grilled, baked, or steamed dishes** instead of fried options\n"
            
            # Personalized dietary advice
            if user_prefs and user_prefs.get('dietary_restrictions'):
                restrictions = user_prefs['dietary_restrictions']
                if any('vegan' in r.lower() for r in restrictions):
                    response += "• **Ask about vegan alternatives** and dairy-free options\n"
                elif any('pescatarian' in r.lower() for r in restrictions):
                    response += "• **Look for seafood and fish dishes** plus vegetarian options\n"
                elif any('vegetarian' in r.lower() and 'non-vegetarian' not in r.lower() for r in restrictions):
                    response += "• **Look for vegetarian options** like plant-based proteins and veggie-packed dishes\n"
                elif any('non-vegetarian' in r.lower() for r in restrictions):
                    response += "• **Choose from meat, poultry, and seafood options** for complete proteins\n"
                
                if any('gluten' in r.lower() for r in restrictions):
                    response += "• **Confirm gluten-free preparation** and ask about cross-contamination\n"
                if any('keto' in r.lower() for r in restrictions):
                    response += "• **Focus on high-fat, low-carb options** and avoid bread/rice\n"
                if any('halal' in r.lower() for r in restrictions):
                    response += "• **Ensure halal certification** and ask about halal preparation methods\n"
                if any('kosher' in r.lower() for r in restrictions):
                    response += "• **Look for kosher options** and confirm kosher preparation standards\n"
            else:
                response += "• **Load up on vegetables and lean proteins** (chicken, fish, tofu)\n"
            
            # Weight management advice
            if user_prefs and user_prefs.get('health_goals'):
                goals = user_prefs['health_goals']
                if isinstance(goals, dict) and goals.get('weight_loss'):
                    response += "• **Portion control**: Consider sharing dishes or taking half home\n"
                    response += "• **Start with a salad** or vegetable soup to feel fuller\n"
                elif isinstance(goals, dict) and goals.get('muscle_gain'):
                    response += "• **Prioritize protein-rich dishes** to support your muscle goals\n"
                    response += "• **Don't skip carbs** - they fuel your workouts\n"
            
            # Food allergy warnings
            if user_prefs and user_prefs.get('food_allergies'):
                allergies = user_prefs['food_allergies']
                if allergies:
                    allergy_list = ', '.join(allergies[:3])  # Show up to 3 allergies
                    response += f"• **⚠️ Allergy Alert**: Inform staff about your {allergy_list} allergies\n"
            
            # General tips
            response += "• **Ask for dressings and sauces on the side** to control portions\n"
            response += "• **Stay hydrated** with water throughout your meal\n"
            response += "• **Check online menus** beforehand to plan healthier choices"
            
            return response
            
        except Exception as e:
            logger.error(f"Error formatting results: {e}")
            return f"Found some restaurants for {location}, but had trouble formatting the results. Try searching on Zomato or Google Maps for local recommendations!"
    
    def detect_restaurant_query(self, query: str) -> Optional[str]:
        """Detect if the query is asking for restaurant recommendations and extract location"""
        query_lower = query.lower()
        restaurant_keywords = ['restaurant', 'dining', 'eat out', 'dinner', 'lunch', 'breakfast', 'food place', 'cafe', 'eatery', 'bar', 'bistro', 'dine', 'meal']
        food_keywords = ['serving', 'food', 'cuisine', 'dish', 'fish', 'chicken', 'vegetarian', 'vegan', 'italian', 'chinese', 'indian', 'pizza', 'burger', 'sushi', 'seafood']
        location_indicators = ['in ', 'at ', 'near ', 'around ', 'for ', 'at ']
        
        # Check for explicit restaurant keywords OR food + location patterns
        has_restaurant_keyword = any(keyword in query_lower for keyword in restaurant_keywords)
        has_food_location_pattern = (
            any(food_word in query_lower for food_word in food_keywords) and
            any(place in query_lower for place in ['pune', 'mumbai', 'delhi', 'bangalore', 'hyderabad', 'chennai', 'kolkata', 'gurgaon', 'noida', 'bandra', 'andheri', 'powai'])
        )
        
        if has_restaurant_keyword or has_food_location_pattern:
            logger.info(f"🍽️ Restaurant query detected: '{query}'")
            
            # Try to extract location using multiple strategies
            location = None
            
            # Strategy 1: Look for location indicators
            for indicator in location_indicators:
                if indicator in query_lower:
                    parts = query_lower.split(indicator)
                    if len(parts) > 1:
                        # Extract location after the indicator
                        location_words = []
                        remaining_text = parts[1].strip()
                        words = remaining_text.split()
                        
                        # Take words that look like location names (capitalized or known places)
                        for word in words[:4]:  # Check up to 4 words
                            clean_word = word.strip('.,!?')
                            if clean_word and (clean_word.istitle() or len(clean_word) > 3):
                                location_words.append(clean_word)
                            else:
                                break  # Stop at first non-location word
                        
                        if location_words:
                            location = ' '.join(location_words).title()
                            logger.info(f"📍 Extracted location using indicator '{indicator}': '{location}'")
                            break
            
            # Strategy 2: Check for common city/area names
            if not location:
                common_places = [
                    'pune', 'mumbai', 'delhi', 'bangalore', 'hyderabad', 'chennai', 'kolkata', 'gurgaon', 'noida',
                    'bandra', 'andheri', 'powai', 'koregaon park', 'viman nagar', 'whitefield', 'indiranagar',
                    'connaught place', 'karol bagh', 'cyber city', 'sector 29', 'mg road', 'brigade road'
                ]
                
                for place in common_places:
                    if place in query_lower:
                        location = place.title()
                        logger.info(f"📍 Found location from common places: '{location}'")
                        break
            
            # Strategy 3: Extract any capitalized words that might be locations
            if not location:
                import re
                # Look for sequences of capitalized words
                matches = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
                if matches:
                    # Take the longest match as it's likely a place name
                    location = max(matches, key=len)
                    logger.info(f"📍 Extracted location from capitalized words: '{location}'")
            
            return location
        
        return None
    
    def process_nutrition_query(self, query: str, user_id: str = None) -> str:
        """Process nutrition-related queries with AI-powered personalized recommendations"""
        try:
            if not self.validate_input(query):
                return "Invalid input. Please provide a valid nutrition question."
            
            # Get user preferences for personalized responses
            user_prefs = {}
            if user_id:
                user_prefs = self.get_user_preferences(user_id)
            
            # Check if this is a restaurant search query
            location = self.detect_restaurant_query(query)
            if location:
                logger.info(f"Restaurant query detected for location: {location}")
                search_results = self.search_restaurants(location, user_id=user_id)
                return self.format_restaurant_results(search_results, location, user_prefs)
            
            # Route queries to appropriate AI-powered handlers (no fallbacks)
            query_lower = query.lower()
            
            # Workout/fitness queries get dedicated workout plan generation
            if any(word in query_lower for word in ['workout', 'exercise', 'fitness', 'training', 'gym', 'strength', 'cardio', 'muscle']):
                logger.info(f"Processing workout query with dedicated AI trainer: {query[:50]}...")
                return self.generate_workout_plan(user_prefs, query)
            
            # All other nutrition queries use general AI nutrition advice
            logger.info(f"Processing nutrition query with AI nutritionist: {query[:50]}...")
            return self._get_ai_nutrition_advice(query, user_prefs)
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"⚠️ Sorry, I encountered an error processing your query: {str(e)}"
    
    def log_food_intake(self, user_id: str, food_item: str, calories: float) -> str:
        """Log food intake to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO nutrition_logs (user_id, food_item, calories) VALUES (?, ?, ?)",
                    (user_id, food_item, calories)
                )
                conn.commit()
            return f"Logged: {food_item} ({calories} calories)"
        except Exception as e:
            logger.error(f"Error logging food: {e}")
            return f"Error logging food: {str(e)}"

def create_gradio_interface(assistant: NutrisenseAssistant) -> gr.Blocks:
    """Create Gradio interface with proper error handling"""
    
    with gr.Blocks(title="Nutrisense Nutrition Assistant", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🥗 Nutrisense Nutrition & Fitness Assistant")
        gr.Markdown("Get personalized nutrition advice based on your preferences, dietary restrictions, and health goals!")
        
        # Add system status indicator
        try:
            exa_status = "✅ Connected" if assistant.config.exa_api_key else "⚠️ Not configured"
            mistral_status = "✅ Connected" if assistant.config.mistral_api_key else "⚠️ Not configured"
            db_status = "✅ Ready" if assistant.db_path.exists() else "⚠️ Initializing"
            
            gr.Markdown(f"**System Status:** Restaurant Search: {exa_status} | Smart Queries: {mistral_status} | Database: {db_status}")
        except Exception as status_error:
            gr.Markdown("**System Status:** ⚠️ Checking system health...")
        
        with gr.Tab("🎯 Set Your Preferences"):
            gr.Markdown("## 🆔 User Identity")
            gr.Info("👆 **IMPORTANT:** Set your unique User ID first! This will save all your preferences and allow personalized AI advice.")
            
            user_id_pref = gr.Textbox(
                label="🆔 Your Unique User ID", 
                placeholder="e.g., john_doe, sarah123, or your_name", 
                value="demo_user",
                info="💡 This identifies you in the system. Use the same ID each time to access your saved preferences!"
            )
            
            gr.Markdown("## 👤 Personal Information")
            with gr.Row():
                age_input = gr.Number(label="Age", minimum=13, maximum=100)
                gender_input = gr.Dropdown(choices=["Male", "Female", "Other"], label="Gender")
            
            with gr.Row():
                current_weight = gr.Number(label="Current Weight (kg)", minimum=30, maximum=300)
                target_weight = gr.Number(label="Target Weight (kg)", minimum=30, maximum=300)
                height_input = gr.Number(label="Height (cm)", minimum=100, maximum=250)
            
            activity_level = gr.Dropdown(
                choices=["Sedentary", "Light", "Moderate", "Active", "Very Active"],
                label="Activity Level"
            )
            
            gr.Markdown("## 🍽️ Dietary Preferences")
            dietary_restrictions = gr.CheckboxGroup(
                choices=["Vegetarian", "Vegan", "Pescatarian", "Non-Vegetarian", "Gluten-Free", "Dairy-Free", "Keto", "Paleo", "Low-Carb", "Mediterranean", "Halal", "Kosher"],
                label="Dietary Restrictions/Preferences"
            )
            
            food_allergies = gr.CheckboxGroup(
                choices=["Nuts", "Shellfish", "Eggs", "Dairy", "Soy", "Fish", "Wheat/Gluten", "Sesame"],
                label="Food Allergies"
            )
            
            cuisine_preferences = gr.CheckboxGroup(
                choices=["Italian", "Chinese", "Indian", "Mexican", "Mediterranean", "Japanese", "Thai", "American"],
                label="Favorite Cuisines"
            )
            
            gr.Markdown("## 🎯 Health Goals")
            with gr.Row():
                daily_calories = gr.Number(label="Daily Calorie Target", minimum=1000, maximum=4000)
                protein_target = gr.Number(label="Protein Target (g)", minimum=20, maximum=300)
            
            with gr.Row():
                carb_target = gr.Number(label="Carb Target (g)", minimum=20, maximum=500)
                fat_target = gr.Number(label="Fat Target (g)", minimum=20, maximum=200)
            
            health_goals = gr.CheckboxGroup(
                choices=["Weight Loss", "Weight Gain", "Muscle Gain", "General Health", "Athletic Performance"],
                label="Primary Health Goals"
            )
            
            save_prefs_btn = gr.Button("💾 Save My Preferences", variant="primary", size="large")
            prefs_output = gr.Textbox(label="Status", interactive=False)
            
            # Connect the save button (moved outside to ensure proper scoping)
            def save_user_prefs(user_id, age, gender, current_wt, target_wt, height, activity, 
                               diet_restrictions, allergies, cuisines, calories, protein, carb, fat, goals):
                """Save user preferences to database"""
                try:
                    if not user_id or not user_id.strip():
                        return "❌ Please enter a valid User ID"
                    
                    preferences = {
                        'age': age,
                        'gender': gender,
                        'current_weight': current_wt,
                        'weight_goal': target_wt,
                        'height_cm': height,
                        'activity_level': activity,
                        'dietary_restrictions': diet_restrictions or [],
                        'food_allergies': allergies or [],
                        'cuisine_preferences': cuisines or [],
                        'daily_calorie_target': int(calories) if calories else None,
                        'protein_target': protein,
                        'carb_target': carb,
                        'fat_target': fat,
                        'health_goals': {goal.lower().replace(' ', '_'): True for goal in (goals or [])}
                    }
                    return assistant.save_user_preferences(user_id.strip(), preferences)
                except Exception as e:
                    logger.error(f"Preferences save error: {e}")
                    return f"❌ Error saving preferences: {str(e)}"
        
        with gr.Tab("💬 Nutrition Chat"):
            gr.Markdown("### 🆔 Your Identity")
            user_id_chat = gr.Textbox(
                label="🆔 User ID", 
                placeholder="Enter your user ID (same as in preferences)", 
                value="demo_user",
                info="⚠️ Use the same User ID as in preferences to get personalized AI advice!"
            )
            
            gr.Markdown("### 🤖 AI Nutrition Assistant")
            chatbot = gr.Chatbot(height=400, type="messages")
            user_input = gr.Textbox(
                placeholder="Ask me about nutrition, calories, fitness, or restaurants...",
                label="Your Question"
            )
            submit_btn = gr.Button("🚀 Get AI Advice", variant="primary")
            
            def respond(message, history, user_id):
                """Handle chat responses with personalized recommendations"""
                try:
                    if not message.strip():
                        return history, ""
                    
                    # Add user context to the response
                    user_prefs = assistant.get_user_preferences(user_id) if user_id else {}
                    if user_prefs and len(history) > 0:
                        # Check recent messages for personalization context
                        recent_messages = history[-3:] if len(history) >= 3 else history
                        if not any("personalized" in msg.get("content", "").lower() for msg in recent_messages if isinstance(msg, dict)):
                            # Show preferences reminder occasionally
                            pref_summary = []
                            if user_prefs.get('dietary_restrictions'):
                                pref_summary.append(f"Dietary: {', '.join(user_prefs['dietary_restrictions'][:2])}")
                            if user_prefs.get('health_goals'):
                                goals = [k.replace('_', ' ').title() for k, v in user_prefs['health_goals'].items() if v]
                                if goals:
                                    pref_summary.append(f"Goals: {', '.join(goals[:2])}")
                            
                            if pref_summary:
                                context_note = f"*Using your preferences: {' | '.join(pref_summary)}*\n\n"
                            else:
                                context_note = ""
                        else:
                            context_note = ""
                    else:
                        context_note = ""
                    
                    response = assistant.process_nutrition_query(message, user_id)
                    full_response = context_note + response
                    
                    # Add message in the new format
                    history.append({"role": "user", "content": message})
                    history.append({"role": "assistant", "content": full_response})
                    return history, ""
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    history.append({"role": "user", "content": message})
                    history.append({"role": "assistant", "content": error_msg})
                    return history, ""
            
            submit_btn.click(
                respond,
                inputs=[user_input, chatbot, user_id_chat],
                outputs=[chatbot, user_input]
            )
            user_input.submit(
                respond,
                inputs=[user_input, chatbot, user_id_chat],
                outputs=[chatbot, user_input]
            )
        
        with gr.Tab("🍎 Food Logger"):
            gr.Markdown("## 📊 Track Your Daily Food Intake")
            gr.Markdown("Log your meals and snacks to track calories and monitor your nutrition goals!")
            
            with gr.Row():
                user_id_food = gr.Textbox(label="User ID", placeholder="Enter your user ID", value="demo_user")
            
            with gr.Row():
                food_input = gr.Textbox(label="Food Item", placeholder="e.g., Apple, Grilled Chicken Breast, Brown Rice")
                calories_input = gr.Number(label="Calories", minimum=0, placeholder="Enter calories")
            
            with gr.Row():
                protein_input = gr.Number(label="Protein (g)", minimum=0, placeholder="Optional")
                carbs_input = gr.Number(label="Carbs (g)", minimum=0, placeholder="Optional")
                fat_input = gr.Number(label="Fat (g)", minimum=0, placeholder="Optional")
            
            meal_type = gr.Dropdown(
                choices=["Breakfast", "Lunch", "Dinner", "Snack"],
                label="Meal Type",
                value="Breakfast"
            )
            
            with gr.Row():
                log_btn = gr.Button("🍽️ Log Food", variant="primary", size="large")
                view_btn = gr.Button("📈 View Today's Log", variant="secondary")
            
            log_output = gr.Textbox(label="Status", interactive=False, lines=3)
            
            def enhanced_log_food(user_id, food_item, calories, protein, carbs, fat, meal_type):
                """Enhanced food logging with macronutrients and meal type"""
                try:
                    # Basic validation
                    if not food_item or not calories:
                        return "❌ Please enter both food item and calories."
                    
                    # Log to database with enhanced info
                    result = assistant.log_food_intake(user_id, food_item, calories)
                    
                    # Get user preferences for context
                    user_prefs = assistant.get_user_preferences(user_id)
                    
                    # Calculate daily progress if user has targets
                    progress_info = ""
                    if user_prefs.get('daily_calorie_target'):
                        # Get today's total (simplified - in real app, you'd query today's entries)
                        target = user_prefs['daily_calorie_target']
                        progress_info = f"\n📊 Progress: Added {calories} calories toward your {target} calorie target."
                        
                        # Add macro breakdown if provided
                        if protein or carbs or fat:
                            macro_info = []
                            if protein:
                                macro_info.append(f"Protein: {protein}g")
                            if carbs:
                                macro_info.append(f"Carbs: {carbs}g")
                            if fat:
                                macro_info.append(f"Fat: {fat}g")
                            progress_info += f"\n🥗 Macros: {', '.join(macro_info)}"
                    
                    return f"✅ {result}\n🕐 Meal: {meal_type}{progress_info}"
                    
                except Exception as e:
                    return f"❌ Error logging food: {str(e)}"
            
            def get_daily_summary(user_id):
                """Get summary of today's food intake"""
                try:
                    with sqlite3.connect(assistant.db_path) as conn:
                        cursor = conn.execute("""
                            SELECT food_item, calories, timestamp 
                            FROM nutrition_logs 
                            WHERE user_id = ? AND date(timestamp) = date('now')
                            ORDER BY timestamp DESC
                        """, (user_id,))
                        
                        entries = cursor.fetchall()
                        
                        if not entries:
                            return "📝 No food entries logged today. Start tracking your meals!"
                        
                        total_calories = sum(entry[1] for entry in entries)
                        
                        summary = f"📊 **Today's Food Log Summary:**\n\n"
                        summary += f"🔥 **Total Calories:** {total_calories}\n"
                        summary += f"📱 **Meals Logged:** {len(entries)}\n\n"
                        
                        summary += "**Recent Entries:**\n"
                        for i, (food, cals, timestamp) in enumerate(entries[:5], 1):
                            # Parse timestamp to show time
                            time_str = timestamp.split()[1][:5] if ' ' in timestamp else 'Unknown'
                            summary += f"{i}. {food} - {cals} cal ({time_str})\n"
                        
                        if len(entries) > 5:
                            summary += f"... and {len(entries) - 5} more entries\n"
                        
                        # Show progress toward goal if user has target
                        user_prefs = assistant.get_user_preferences(user_id)
                        if user_prefs.get('daily_calorie_target'):
                            target = user_prefs['daily_calorie_target']
                            percentage = round((total_calories / target) * 100, 1)
                            summary += f"\n🎯 **Goal Progress:** {percentage}% of {target} calorie target"
                        
                        return summary
                        
                except Exception as e:
                    return f"❌ Error retrieving daily summary: {str(e)}"
            
            log_btn.click(
                enhanced_log_food,
                inputs=[user_id_food, food_input, calories_input, protein_input, carbs_input, fat_input, meal_type],
                outputs=[log_output]
            )
            
            view_btn.click(
                get_daily_summary,
                inputs=[user_id_food],
                outputs=[log_output]
            )
        
        # Connect preference save button (placed at end to ensure all components are in scope)
        save_prefs_btn.click(
            save_user_prefs,
            inputs=[user_id_pref, age_input, gender_input, current_weight, target_weight, height_input,
                   activity_level, dietary_restrictions, food_allergies, cuisine_preferences,
                   daily_calories, protein_target, carb_target, fat_target, health_goals],
            outputs=[prefs_output]
        )
    
    return interface

def create_fallback_interface() -> gr.Blocks:
    """Create a simple fallback interface if main initialization fails"""
    with gr.Blocks(title="Nutrisense Nutrition Assistant", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🥗 Nutrisense Nutrition Assistant")
        gr.Markdown("⚠️ **System is initializing...** Some features may be temporarily unavailable.")
        
        with gr.Tab("🎯 Set Your Preferences"):
            gr.Markdown("## System Status")
            gr.Markdown("The preference system is currently initializing. Please refresh the page in a moment.")
            
        with gr.Tab("💬 Nutrition Chat"):
            gr.Markdown("## Chat Temporarily Unavailable")
            gr.Markdown("The nutrition chat system is starting up. Please try again shortly.")
            
        with gr.Tab("🍎 Food Logger"):
            gr.Markdown("## Food Logger Initializing")
            gr.Markdown("The food logging system is being prepared. Please wait a moment and refresh.")
    
    return interface

# Health check endpoint for Hugging Face Spaces
def health_check():
    """Simple health check endpoint"""
    return "✅ Nutrisense AI is healthy and running!"

def main():
    """Main function with comprehensive error handling and fallbacks"""
    try:
        # Load environment variables from .env file
        from dotenv import load_dotenv
        load_dotenv()
        
        logger.info("🚀 Starting Nutrisense Nutrition AI...")
        
        # Load configuration
        config = Config.from_env()
        
        # Make all APIs optional for faster startup
        if not config.exa_api_key:
            logger.info("EXA_API_KEY not configured - restaurant search will use fallback")
        
        if not config.mistral_api_key:
            logger.info("MISTRAL_API_KEY not configured - will use basic responses")
        
        # Skip Qdrant check to avoid import issues
        logger.info("Starting with minimal dependencies for faster deployment")
        
        # Initialize assistant with error handling
        try:
            assistant = NutrisenseAssistant(config)
            logger.info("Assistant initialized successfully")
        except Exception as init_error:
            logger.error(f"Assistant initialization failed: {init_error}")
            # Create a minimal assistant for fallback
        assistant = NutrisenseAssistant(config)
        
        # Create and launch interface
        try:
            interface = create_gradio_interface(assistant)
            logger.info("Interface created successfully")
        except Exception as interface_error:
            logger.error(f"Interface creation failed: {interface_error}")
            logger.info("Creating fallback interface")
            interface = create_fallback_interface()
        
        # Add health check route
        interface.health_check = health_check
        
        # Optimized launch for Hugging Face Spaces
        logger.info("🌐 Launching interface...")
        interface.launch(
            server_name="0.0.0.0",  # Bind to all interfaces for HF Spaces
            server_port=7860,       # Standard port for HF Spaces
            share=True,            # Disabled for HF deployment
            show_error=True,
            prevent_thread_lock=False,
            quiet=False,            # Show startup logs
            show_api=False,         # Reduce API overhead
            favicon_path=None,      # Reduce resource loading
            ssl_verify=False        # Skip SSL verification for faster startup
        )
        
    except Exception as e:
        logger.error(f"Application startup error: {e}")
        print(f"❌ Error starting application: {e}")
        
        # Last resort: create a basic interface
        try:
            logger.info("🆘 Creating emergency fallback interface...")
            fallback_interface = create_fallback_interface()
            fallback_interface.launch(
                server_name="0.0.0.0",
                server_port=7860,
                share=False,
                show_error=True
            )
        except Exception as fallback_error:
            logger.error(f"Even fallback interface failed: {fallback_error}")
            # Final fallback - minimal text output
            print("🚨 CRITICAL: All interfaces failed. Check API keys and dependencies.")
        raise

if __name__ == "__main__":
    main() 