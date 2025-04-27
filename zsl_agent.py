"""
Agent for zero-shot learning tasks in drone control.
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

class ZSLAgent:
    def __init__(self, 
                 model_name: str = "gpt-4.1-mini",
                 system_message: str = ""):
        """
        Initialize the zero-shot learning agent.
        
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
        if not system_message:
            system_message = """
                PURPOSE:
                You perform zero-shot learning on provided human gestures. 
                We need your help in prediciting what someones 'body language' means.

                TASK:
                1. Analyze text descriptions, images, and sensor data provided by a BlazePose model
                2. Identify patterns and relationships in the data
                3. Contextualize the human gesture command
                4. Provide insights for flight planning
                5. Output the contextualized interpretation of the human gesture command in a structured format
                
                EXAMPLES:
                Example output 1:
                {
                    "predicted_gesture": "fly in a circular motion",
                    "reasoning": "The provided input has a strong classification of a circular motion",
                    "insight": "The human hand is pointing up and rotating in a circular motion, inidication some kind of cocentric turn"
                }

                Example output 2:
                {
                    "predicted_gesture": "fly in a straight line",
                    "reasoning": "The predicted gesture command is based on the analysis of the text, image, and sensor data",
                    "insight": "The human gesture command has one static hand pointing forward, indicating a straight line motion"
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
        
    def analyze_inputs(self, 
                      text_input: str,
                      image_input: Optional[Image.Image] = None,
                      sensor_data: Optional[List[float]] = None) -> Dict:
        """
        Analyze inputs using zero-shot learning.
        
        Args:
            text_input (str): Text description or command
            image_input (Optional[Image.Image]): Optional image input
            sensor_data (Optional[List[float]]): Optional sensor readings
            
        Returns:
            Dict: Analysis results containing insights and predictions
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
            
        # Add sensor data if provided
        if sensor_data is not None:
            messages[0]["content"].append({
                "type": "text",
                "text": f"Sensor data: {sensor_data}"
            })
            
        # Get response from the language model
        response = self.task.run(messages)
        
        # # Parse the response into structured insights
        # analysis = self._parse_response(response)
        
        # return analysis
        return response
        
    def _parse_response(self, response: str) -> Dict:
        """
        Parse the language model response into structured insights.
        
        Args:
            response (str): Raw response from the language model
            
        Returns:
            Dict: Structured analysis containing insights and predictions
        """
        # This is a placeholder implementation
        # You'll need to implement proper parsing based on your needs
        return {
            "environment_analysis": {
                "obstacles": [],
                "safe_zones": [],
                "hazards": []
            },
            "behavior_prediction": {
                "expected_movements": [],
                "potential_risks": [],
                "recommended_actions": []
            },
            "sensor_interpretation": {
                "patterns": [],
                "anomalies": [],
                "trends": []
            }
        } 