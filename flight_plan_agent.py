"""
Agent for generating flight plans based on ZSLAgent insight and output.
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
                 model_name: str = "gpt-4.1-mini",
                 system_message: str = ""):
        """
        Initialize the flight plan agent.
        
        Args:
            model_name (str): Name of the language model to use
            system_message (str): Custom system message for the agent
        """
        # Configure the language model
        llm_config = OpenAIGPTConfig(
            chat_model=model_name,
            temperature=0.6,
            max_output_tokens=1000
        )
        
        # Default system message if none provided
        if not system_message:
            system_message = """
            PURPOSE:
            You are an expert drone flight planner. 

            TASK:
            1. Generate accurate flight plans based on the ZSLAgent insight and output
            2. Consider obstacles, no-fly zones, and mission objectives
            3. Output drone_control methods and necessary parameters
            
            GUIDELINES:
            - You are only allowed to use the functions I have defined for you.
            - You are not to use any other hypothetical functions that you think might exist.
            
            EXAMPLES:
            Example output 1:
            {
                "flight_plan": [
                    ("takeoff", (5.0,)),    
                    ("move_by_velocity", (1.0, 0.0, 0.0, 2.0)),
                    ("land", ())
                ]
            }

            Example output 2:
            {
                "flight_plan": [
                    ("takeoff", (5.0,)),  # Take off to 5 meters
                    ("move_by_velocity", (1.0, 0.0, 0.0, 2.0)),  # Move forward at 1 m/s for 2 seconds
                    ("move_by_velocity", (-1.0, 0.0, 0.0, 2.0)),  # Move backward at 1 m/s for 2 seconds
                    ("land", ())  # Land safely
                ]
            }

            """
            
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
        
    def generate_plan(self, zsl_agent_output: str):
        """
        Generate a flight plan based on inputs.
        
        Args:
            text_input (str): Text description of the mission or environment
            
        Returns:
            List[Tuple[str, tuple]]: List of (method_name, args) tuples for drone control
        """
            
        # Get response from the language model
        response = self.task.run(zsl_agent_output)
        
        # Parse the response into a flight plan
        flight_plan = self._parse_response(response)
        
        return flight_plan
        
    def _parse_response(self, response: str) -> List[Tuple[str, tuple]]:
        """
        Parse the language model response into a structured flight plan.
        
        Args:
            response (str): Raw response from the language model
            
        Returns:
            List[Tuple[str, tuple]]: List of (method_name, args) tuples for drone control
        """
        pass