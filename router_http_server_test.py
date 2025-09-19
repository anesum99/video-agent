# router_http_server.py - simplified to call external Ollama service
import requests
import json
import time
from flask import Flask, request, jsonify

app = Flask(__name__)

class RouterService:
    def __init__(self):
        self.ollama_url = "http://ollama-test:11435"  # Call ollama service, not localhost
        self.model = "phi3:mini"
        self._ensure_model_ready()
    
    def _ensure_model_ready(self):
        """Ensure Ollama service is ready and model is pulled"""
        print("⏳ Waiting for Ollama service...")
        
        # Wait for Ollama service to be ready
        max_retries = 30
        for i in range(max_retries):
            try:
                response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
                if response.status_code == 200:
                    models = [m["name"] for m in response.json().get("models", [])]
                    if self.model in models:
                        print(f"✅ Ollama ready with {self.model}")
                        return
                    else:
                        print(f"📥 Pulling {self.model}...")
                        # Pull model via API call
                        pull_response = requests.post(
                            f"{self.ollama_url}/api/pull",
                            json={"name": self.model},
                            timeout=300  # 5 minutes for model download
                        )
                        if pull_response.status_code == 200:
                            print(f"✅ Model {self.model} pulled successfully")
                            return
            except Exception as e:
                print(f"⏳ Waiting for Ollama... ({i+1}/{max_retries}) - {e}")
                time.sleep(3)
        
        print("❌ Ollama service not available")

    def extract_count_target(self, question: str) -> dict:
        """Extract what to count using phi3 with comprehensive fallback"""
        
        prompt = f"""Extract what object to count from this question. Return ONLY the object name from this list.

VALID OBJECTS (return exact match):
person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush

MAPPING RULES:
- people/humans/individuals → person
- bikes/bicycles/cycles → bicycle  
- cars/vehicles/automobiles → car
- phones/smartphones/mobiles → cell phone
- computers/laptops/notebooks → laptop
- tables/desks → dining table
- screens/monitors → tv
- balls/footballs/basketballs → sports ball

Examples:
Q: "How many people are in the video?" A: person
Q: "Count the cars" A: car
Q: "Total vehicles in the scene" A: car
Q: "Number of dogs visible" A: dog
Q: "How many bikes can you see" A: bicycle
Q: "Count smartphones" A: cell phone
Q: "How many tables" A: dining table
Q: "Total balls in the video" A: sports ball

Question: "{question}"

Return ONLY the exact object name from the list above:"""

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 15}
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()["response"].strip().lower()
                
                # Clean up response (remove any extra text)
                words = result.split()
                if words:
                    # Take first few words that might form a valid class
                    for i in range(min(3, len(words))):
                        candidate = ' '.join(words[:i+1])
                        if self._is_valid_yolo_class(candidate):
                            print(f"🎯 TARGET: '{question}' → {candidate} (confidence: 0.9)")
                            return {"target": candidate, "confidence": 0.9}
                
        except Exception as e:
            print(f"Target extraction error: {e}")
        
        # Fallback to comprehensive keyword matching
        fallback = self._comprehensive_fallback_extract_target(question)
        print(f"🎯 TARGET: '{question}' → {fallback['target']} (fallback, confidence: {fallback['confidence']})")
        return fallback
    
    def _is_valid_yolo_class(self, candidate: str) -> bool:
        """Check if candidate is valid YOLO class"""
        yolo_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        return candidate in yolo_classes
    
    def _comprehensive_fallback_extract_target(self, question: str) -> dict:
        """Comprehensive fallback target extraction covering all 80 YOLO classes"""
        q_lower = question.lower()
        
        # Comprehensive keyword to YOLO class mapping
        keyword_map = {
            # People
            'person': ['people', 'person', 'human', 'humans', 'individual', 'individuals', 'man', 'men', 'woman', 'women', 'child', 'children', 'adult', 'adults', 'guy', 'guys', 'girl', 'girls', 'boy', 'boys'],
            
            # Vehicles - Land
            'car': ['car', 'cars', 'vehicle', 'vehicles', 'automobile', 'automobiles', 'sedan', 'sedans', 'suv', 'suvs', 'hatchback', 'hatchbacks'],
            'bicycle': ['bike', 'bikes', 'bicycle', 'bicycles', 'cycle', 'cycles', 'cycling'],
            'motorcycle': ['motorcycle', 'motorcycles', 'motorbike', 'motorbikes', 'scooter', 'scooters', 'moped', 'mopeds'],
            'truck': ['truck', 'trucks', 'lorry', 'lorries', 'pickup', 'pickups', 'van', 'vans', 'delivery'],
            'bus': ['bus', 'buses', 'coach', 'coaches'],
            'train': ['train', 'trains', 'locomotive', 'locomotives', 'subway', 'metro'],
            
            # Vehicles - Air/Water
            'boat': ['boat', 'boats', 'ship', 'ships', 'vessel', 'vessels', 'yacht', 'yachts', 'sailboat'],
            'airplane': ['airplane', 'airplanes', 'plane', 'planes', 'aircraft', 'jet', 'jets', 'flight'],
            
            # Animals
            'dog': ['dog', 'dogs', 'puppy', 'puppies', 'canine', 'canines', 'pup', 'pups'],
            'cat': ['cat', 'cats', 'kitten', 'kittens', 'feline', 'felines'],
            'bird': ['bird', 'birds', 'pigeon', 'pigeons', 'crow', 'crows', 'seagull', 'seagulls'],
            'horse': ['horse', 'horses', 'pony', 'ponies', 'mare', 'stallion'],
            'sheep': ['sheep', 'lamb', 'lambs', 'ewe', 'ram'],
            'cow': ['cow', 'cows', 'cattle', 'bull', 'bulls', 'beef'],
            'elephant': ['elephant', 'elephants'],
            'bear': ['bear', 'bears'],
            'zebra': ['zebra', 'zebras'],
            'giraffe': ['giraffe', 'giraffes'],
            
            # Electronics
            'tv': ['tv', 'tvs', 'television', 'televisions', 'monitor', 'monitors', 'screen', 'screens', 'display', 'displays'],
            'laptop': ['laptop', 'laptops', 'computer', 'computers', 'notebook', 'notebooks', 'pc', 'pcs'],
            'cell phone': ['phone', 'phones', 'cellphone', 'cellphones', 'mobile', 'mobiles', 'smartphone', 'smartphones', 'iphone', 'android'],
            'remote': ['remote', 'remotes', 'controller', 'controllers'],
            'keyboard': ['keyboard', 'keyboards', 'keys'],
            'mouse': ['mouse', 'mice', 'computer mouse'],
            
            # Furniture
            'chair': ['chair', 'chairs', 'seat', 'seats', 'stool', 'stools'],
            'couch': ['couch', 'couches', 'sofa', 'sofas', 'loveseat'],
            'bed': ['bed', 'beds', 'mattress', 'mattresses'],
            'dining table': ['table', 'tables', 'desk', 'desks', 'counter', 'counters'],
            'toilet': ['toilet', 'toilets'],
            'bench': ['bench', 'benches'],
            
            # Kitchen Items
            'bottle': ['bottle', 'bottles', 'water bottle'],
            'cup': ['cup', 'cups', 'mug', 'mugs', 'coffee cup'],
            'wine glass': ['wine glass', 'wine glasses', 'glass', 'glasses', 'wineglass'],
            'bowl': ['bowl', 'bowls'],
            'fork': ['fork', 'forks'],
            'knife': ['knife', 'knives'],
            'spoon': ['spoon', 'spoons'],
            'microwave': ['microwave', 'microwaves'],
            'oven': ['oven', 'ovens'],
            'refrigerator': ['fridge', 'fridges', 'refrigerator', 'refrigerators', 'freezer', 'freezers'],
            'toaster': ['toaster', 'toasters'],
            'sink': ['sink', 'sinks'],
            
            # Food
            'banana': ['banana', 'bananas'],
            'apple': ['apple', 'apples'],
            'orange': ['orange', 'oranges'],
            'pizza': ['pizza', 'pizzas'],
            'sandwich': ['sandwich', 'sandwiches', 'sub', 'subs'],
            'cake': ['cake', 'cakes'],
            'donut': ['donut', 'donuts', 'doughnut', 'doughnuts'],
            'hot dog': ['hot dog', 'hotdog', 'hot dogs', 'hotdogs'],
            'broccoli': ['broccoli'],
            'carrot': ['carrot', 'carrots'],
            
            # Personal Items
            'book': ['book', 'books', 'novel', 'novels'],
            'clock': ['clock', 'clocks', 'watch', 'watches'],
            'backpack': ['backpack', 'backpacks', 'bag', 'bags', 'schoolbag', 'bookbag'],
            'umbrella': ['umbrella', 'umbrellas'],
            'handbag': ['handbag', 'handbags', 'purse', 'purses', 'bag'],
            'suitcase': ['suitcase', 'suitcases', 'luggage'],
            'scissors': ['scissors', 'shears'],
            'vase': ['vase', 'vases'],
            'tie': ['tie', 'ties', 'necktie', 'neckties'],
            'teddy bear': ['teddy bear', 'teddy bears', 'bear', 'toy bear', 'stuffed animal'],
            'hair drier': ['hair dryer', 'hair drier', 'blow dryer'],
            'toothbrush': ['toothbrush', 'toothbrushes'],
            
            # Sports Equipment
            'sports ball': ['ball', 'balls', 'football', 'footballs', 'basketball', 'basketballs', 'soccer ball', 'volleyball', 'tennis ball'],
            'tennis racket': ['racket', 'rackets', 'racquet', 'racquets', 'tennis racket'],
            'baseball bat': ['bat', 'bats', 'baseball bat'],
            'baseball glove': ['glove', 'gloves', 'mitt', 'baseball glove'],
            'skateboard': ['skateboard', 'skateboards', 'skate', 'skates'],
            'surfboard': ['surfboard', 'surfboards', 'surf board'],
            'skis': ['ski', 'skis', 'skiing'],
            'snowboard': ['snowboard', 'snowboards'],
            'frisbee': ['frisbee', 'frisbees'],
            'kite': ['kite', 'kites'],
            
            # Traffic/Street Items
            'traffic light': ['light', 'lights', 'signal', 'signals', 'traffic light', 'traffic lights', 'stoplight'],
            'stop sign': ['sign', 'signs', 'stop sign', 'stop signs'],
            'fire hydrant': ['hydrant', 'hydrants', 'fire hydrant'],
            'parking meter': ['meter', 'meters', 'parking meter'],
            
            # Misc Objects
            'potted plant': ['plant', 'plants', 'pot', 'pots', 'flower', 'flowers', 'potted plant']
        }
        
        # Find best match with scoring
        matches = []
        
        for yolo_class, keywords in keyword_map.items():
            for keyword in keywords:
                if keyword in q_lower:
                    # Score based on keyword length and specificity
                    score = len(keyword) * 0.1
                    # Exact word boundary matches get higher score
                    if f' {keyword} ' in f' {q_lower} ' or q_lower.startswith(keyword) or q_lower.endswith(keyword):
                        score += 0.3
                    matches.append((yolo_class, score))
        
        if matches:
            # Return highest scoring match
            best_match = max(matches, key=lambda x: x[1])
            confidence = min(0.7 + best_match[1], 0.85)
            return {"target": best_match[0], "confidence": confidence}
        else:
            return {"target": "person", "confidence": 0.4}

    def classify_intent(self, question: str, has_video: bool = True) -> dict:
        """Classify user intent"""
        if not has_video:
            print(f"🔄 ROUTE: No video provided → RESPOND")
            return {"intent": "RESPOND", "confidence": 1.0}
       
        prompt = f"""You are a video analysis router. Classify this question into exactly ONE category.

DECISION TREE:
1. Asks for quantity/count? → COUNT
2. Asks about video file properties? → METADATA  
3. Asks to read visible text? → OCR
4. Asks about audio/speech? → ASR
5. Asks about video structure/cuts? → SCENES
6. Asks for step-by-step sequence? → TIMELINE
7. Everything else → VISUAL_ANALYSIS

CATEGORIES WITH EXAMPLES:

COUNT:
- "how many people are in the video"
- "count the number of cars"
- "how many dogs do you see"
- "total number of bikes"
- "quantity of objects on the table"
- "how many times does someone wave"
- "count the red items"
- "number of people wearing hats"

METADATA:
- "how long is this video"
- "what's the video duration"
- "video frame rate"
- "video resolution"
- "file size of video"
- "video dimensions"
- "how many frames per second"
- "technical specs"

OCR:
- "what text is visible in the video"
- "read the street sign"
- "what does the banner say"
- "text on the screen"
- "words on the building"
- "read the license plate"
- "what's written on the shirt"
- "caption text shown"
- "billboard content"

ASR:
- "what did they say"
- "transcribe the dialogue"
- "what sounds do you hear"
- "audio content"
- "speech transcription"
- "what music is playing"
- "background noise"
- "spoken words"
- "conversation transcript"

SCENES:
- "how many different scenes"
- "scene changes in video"
- "different camera shots"
- "video transitions"
- "scene cuts"
- "shot composition"
- "camera angles used"
- "editing structure"

TIMELINE:
- "step by step what happens"
- "sequence of events"
- "chronological breakdown"
- "what happens first then next"
- "timeline of actions"
- "order of events"
- "process shown in video"
- "progression of activities"
- "break down the sequence"

VISUAL_ANALYSIS:
- "what's happening in the video"
- "describe what you see"
- "who is in the video"
- "what color is the car"
- "what are they doing"
- "where does this take place"
- "what's the mood"
- "describe the setting"
- "what objects are visible"
- "analyze the scene"
- "what's the main activity"

Question: "{question}"

Answer with ONLY the category name:"""

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 15}
                },
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()["response"].strip().upper()
                valid_intents = ["COUNT", "TIMELINE", "VISUAL_ANALYSIS", "OCR", "ASR", "SCENES", "METADATA"]
                
                intent = result if result in valid_intents else "VISUAL_ANALYSIS"
                print(f"🔄 ROUTE: '{question}' → {intent} (confidence: 0.95)")
                return {"intent": intent, "confidence": 0.95}
            
        except Exception as e:
            print(f"Ollama classification error: {e}")
        
        # Improved fallback classification
        fallback_result = self._fallback_classify(question)
        print(f"🔄 ROUTE: '{question}' → {fallback_result['intent']} (fallback, confidence: {fallback_result['confidence']})")
        return fallback_result
    
    def _fallback_classify(self, question: str) -> dict:
        """Improved fallback classification"""
        q_lower = question.lower()
        
        if "how many" in q_lower or "count" in q_lower:
            return {"intent": "COUNT", "confidence": 0.8}
        elif any(phrase in q_lower for phrase in [
            "step by step", "break down", "break this down", "break it down",
            "timeline", "sequence", "chronological", "step-by-step"
        ]):
            return {"intent": "TIMELINE", "confidence": 0.8}
        elif any(phrase in q_lower for phrase in ["text", "read", "sign", "label"]):
            return {"intent": "OCR", "confidence": 0.8}
        elif any(phrase in q_lower for phrase in ["say", "speech", "audio", "transcribe"]):
            return {"intent": "ASR", "confidence": 0.8}
        elif "scene" in q_lower:
            return {"intent": "SCENES", "confidence": 0.8}
        elif any(phrase in q_lower for phrase in ["duration", "fps", "resolution"]):
            return {"intent": "METADATA", "confidence": 0.8}
        else:
            return {"intent": "VISUAL_ANALYSIS", "confidence": 0.7}

router_service = RouterService()

@app.route('/health')
def health():
    return {"status": "healthy", "service": "router"}

@app.route('/tools/status', methods=['POST'])
def tools_status():
    """Status endpoint for HTTP tool manager compatibility"""
    return jsonify({"status": "healthy", "tools": ["classify"], "model": "phi3:mini"})

@app.route('/classify', methods=['POST'])
def classify():
    data = request.json
    question = data.get('question', '')
    has_video = data.get('has_video', True)
    
    result = router_service.classify_intent(question, has_video)
    return jsonify(result)

@app.route('/extract_target', methods=['POST'])
def extract_target():
    data = request.json
    question = data.get('question', '')

    result = router_service.extract_count_target(question)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8016, debug=False)

