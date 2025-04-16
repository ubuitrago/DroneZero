"""
Agent for processing text and image inputs to generate drone flight plans.
This module uses langroid for language model integration and vision capabilities.
"""

from typing import List, Tuple, Dict, Optional
import numpy as np
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from PIL import Image
import base64
from io import BytesIO

class FlightPlanAgent:
    def __init__(self, 
                 model_name: str = "gpt-4-vision-preview",
                 system_message: str = None):
        """
        Initialize the flight plan agent.
        
        Args:
            model_name (str): Name of the language model to use
            system_message (str): Custom system message for the agent
        """
        # Configure the language model
        llm_config = OpenAIGPTConfig(
            chat_model=model_name,
            temperature=0.7,
            max_output_tokens=1000
        )
        
        # Default system message if none provided
        if system_message is None:
            system_message = """You are an expert drone flight planner. Your task is to:
            1. Analyze text descriptions and images of environments
            2. Generate safe and efficient flight plans
            3. Consider obstacles, no-fly zones, and mission objectives
            4. Output waypoints in a format suitable for drone control"""
            
        # Configure the chat agent
        agent_config = ChatAgentConfig(
            llm=llm_config,
            system_message=system_message
        )
        
        self.agent = ChatAgent(agent_config)
        self.task = Task(self.agent)
        
    def _encode_image(self, image: Image.Image) -> str:
        """
        Encode an image to base64 string.
        
        Args:
            image (Image.Image): PIL Image to encode
            
        Returns:
            str: Base64 encoded image string
        """
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
        
    def process_input(self, 
                     text_input: str,
                     image_input: Optional[Image.Image] = None) -> Dict:
        """
        Process text and image inputs to generate a flight plan.
        
        Args:
            text_input (str): Text description of the mission or environment
            image_input (Optional[Image.Image]): Optional image of the environment
            
        Returns:
            Dict: Flight plan containing waypoints and other parameters
        """
        # Prepare the message for the language model
        messages = [{"role": "user", "content": text_input}]
        
        # Add image if provided
        if image_input is not None:
            image_str = self._encode_image(image_input)
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_str}"
                }
            })
            
        # Get response from the language model
        response = self.task.run(messages)
        
        # Parse the response into a flight plan
        # This is a placeholder - you'll need to implement proper parsing
        # based on your specific flight plan format
        flight_plan = self._parse_response(response)
        
        return flight_plan
        
    def _parse_response(self, response: str) -> Dict:
        """
        Parse the language model response into a structured flight plan.
        
        Args:
            response (str): Raw response from the language model
            
        Returns:
            Dict: Structured flight plan
        """
        # This is a placeholder implementation
        # You'll need to implement proper parsing based on your needs
        return {
            "waypoints": [],  # List of (x, y, z) coordinates
            "speed": 5.0,     # Default speed in m/s
            "altitude": 10.0, # Default altitude in meters
            "notes": response # Raw response for reference
        }
        
    def generate_waypoints(self, flight_plan: Dict) -> List[Tuple[float, float, float]]:
        """
        Convert flight plan into a list of waypoints.
        
        Args:
            flight_plan (Dict): Structured flight plan
            
        Returns:
            List[Tuple[float, float, float]]: List of (x, y, z) waypoints
        """
        # This is a placeholder implementation
        # You'll need to implement proper waypoint generation
        return flight_plan["waypoints"]
        
    def validate_plan(self, flight_plan: Dict) -> bool:
        """
        Validate a flight plan for safety and feasibility.
        
        Args:
            flight_plan (Dict): Flight plan to validate
            
        Returns:
            bool: True if the plan is valid
        """
        # This is a placeholder implementation
        # You'll need to implement proper validation
        return True 