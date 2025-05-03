"""
LLM Intent Inference Agent for generating flight plans based on ZSLAgent insight and output.
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
import json
import re

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
            You are an expert drone flight planner whom collaborates with an external Zero-Shot-Learning (ZSL) Agent.
            Do not worry about the ZSLAgent, they will prompt you with input.
            Your outputs ALWAYS begin with -- ("takeoff", (5.0,)) -- and end with -- ("land", ()). 

            TASK:
            1. Generate accurate flight plans based on the provided insight
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
            # Concatenate method prototypes
            fns = open("airsim_methods.txt", "r")
            methods = fns.read()
            fns.close()
            
            system_message += methods
        # Configure the chat agent
        agent_config = ChatAgentConfig(
            llm=llm_config,
            system_message=system_message
        )
        
        self.agent = ChatAgent(agent_config)
        self.task = Task(self.agent, interactive=False)
        
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
        
    def generate_plan(self, zsl_agent_output: dict):
        """
        Generate a flight plan based on inputs.
        
        Args:
            zsl_agent_output (dict): Output from the ZSL agent containing gesture analysis
            
        Returns:
            List[Tuple[str, tuple]]: List of (method_name, args) tuples for drone control
        """
        if not zsl_agent_output:
            print("Warning: Empty ZSL agent output received")
            return [
                ("takeoff", (5.0,)),
                ("land", ())
            ]
            
        # Check if the ZSL agent output indicates an error
        if zsl_agent_output.get('predicted_gesture') == 'unknown':
            print(f"Warning: ZSL agent reported an error: {zsl_agent_output.get('reasoning', 'Unknown error')}")
            return [
                ("takeoff", (5.0,)),
                ("land", ())
            ]
            
        # Get response from the language model
        try:
            response = self.task.run(zsl_agent_output)
            
            if response is None:
                print("Warning: No response received from language model")
                return [
                    ("takeoff", (5.0,)),
                    ("land", ())
                ]
            
            # Parse the response into a flight plan
            try:
                # Try to find JSON object in the response using regex
                json_match = re.search(r'\{[^{}]*\}', response)
                if json_match:
                    json_str = json_match.group(0)
                    parsed_response = json.loads(json_str)
                    
                    if 'flight_plan' in parsed_response:
                        # Convert string representation of tuples into actual tuples
                        flight_plan = []
                        for cmd in parsed_response['flight_plan']:
                            if isinstance(cmd, str):
                                # Parse string representation of tuple
                                method, args = eval(cmd)
                            else:
                                # Already in correct format
                                method, args = cmd
                            flight_plan.append((method, args))
                        return flight_plan
                        
                # If JSON parsing fails, try to extract information from text
                flight_plan = []
                lines = response.split('\n')
                for line in lines:
                    if '(' in line and ')' in line:
                        try:
                            # Try to evaluate the line as a tuple
                            cmd = eval(line.strip())
                            if isinstance(cmd, tuple) and len(cmd) == 2:
                                flight_plan.append(cmd)
                        except:
                            continue
                        
                if flight_plan:
                    return flight_plan
                    
            except Exception as e:
                print(f"Error parsing flight plan: {str(e)}")
                
        except Exception as e:
            print(f"Error generating flight plan: {str(e)}")
            
        # If all parsing fails, return a safe default plan
        return [
            ("takeoff", (5.0,)),
            ("land", ())
        ]
        
    def _parse_response(self, response: str) -> List[Tuple[str, tuple]]:
        """
        Parse the language model response into a structured flight plan.
        
        Args:
            response (str): Raw response from the language model
            
        Returns:
            List[Tuple[str, tuple]]: List of (method_name, args) tuples for drone control
        """
        pass