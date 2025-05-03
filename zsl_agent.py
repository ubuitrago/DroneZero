"""
Agent for zero-shot learning tasks in drone control.
This module uses langroid for language model integration and vision capabilities.
"""
import time
import asyncio, threading
from typing import Dict
import numpy as np
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from PIL import Image
import base64
from io import BytesIO
import json
import re

class ZSLAgent:
    def __init__(self, 
                 model_name: str = "gpt-4.1",
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
                CONTEXT:
                You are an expert in zero-shot learning.
                You are given a text description and a gif.
                The data shared with you serves as context, but is not the main source of information.

                PURPOSE:
                Pretend you are a drone flying at 12 ft high and 10ft away from the person in focus.
                When someones 'body language' is detected in the gif, you must predict what action the drone should take.

                TASK:
                1. Analyze text descriptions AND gif provided.
                2. Identify patterns and relationships in the data.
                3. Contextualize the human gesture command by identifying how the body language would translate to a deaf person.
                4. Provide insights for flight planning.
                5. Output the contextualized interpretation of the human gesture command in a structured format.
                
                EXAMPLES:
                Example output 1:
                {
                    "predicted_gesture": "fly in a circular motion",
                    "reasoning": "The provided input has a strong classification of a circular motion",
                    "insight": "The human hand is pointing up and rotating in a circular motion, 
                    indicating some kind of concentric turn"
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
        self.task = Task(self.agent, interactive=False, max_stalled_steps=10)
        self.result = {}
        
    def run_task_in_thread(self, messages):
        response_doc = self.task.run(messages)
        if response_doc:
            self.result['response'] = response_doc.content
        else:
            self.result['response'] = None
            
    def analyze_inputs(self, 
                      text_input: str,
                      image_input: str) -> Dict:
        """
        Analyze inputs using zero-shot learning.
        
        Args:
            text_input (str): Text description or command
            image_input (Optional[Image.Image]): Optional image input
            
        Returns:
            Dict: Analysis results containing insights and predictions
        """
        if not text_input:
            return {
                'predicted_gesture': 'unknown',
                'reasoning': 'No text input provided',
                'insight': 'Error: Missing text input'
            }
            
        # Prepare the message for the language model
        messages = [{"role": "user", "content": [text_input]}]
        
        # Add image if provided
        if image_input is not None:
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": image_input
                }
            })
            
        try:
            # Get response from the language model
            self.run_task_in_thread(messages)

            # Access the result
            response = self.result.get('response')

            # if response_doc is None:
            #     # fallback: try last agent message
            #     response_doc = self.agent.last_message()
            #     print("Fallback to agent.last_message():", response_doc)

            # if response_doc is not None:
            #     response = response_doc.content
            #     print("Parsed Response:", response)
            #     return self._parse_response(response)

            # Parse and return the response as a dictionary
            return self._parse_response(response)
            
        except Exception as e:
            return {
                'predicted_gesture': 'unknown',
                'reasoning': f'Error during analysis: {str(e)}',
                'insight': 'Error in analysis process'
            }
        
    def _parse_response(self, response: str) -> Dict:
        """
        Parse the language model response into structured insights.
        
        Args:
            response (str): Raw response from the language model
            
        Returns:
            Dict: Structured analysis containing predicted gesture, reasoning, and insight
        """
        try:
            # Try to find JSON object in the response using regex
            json_match = re.search(r'\{[^{}]*\}', response)
            if json_match:
                json_str = json_match.group(0)
                parsed_response = json.loads(json_str)
                
                # Verify required fields are present
                required_fields = ['predicted_gesture', 'reasoning', 'insight']
                if all(field in parsed_response for field in required_fields):
                    return parsed_response
                
            # If JSON parsing fails or required fields are missing, 
            # try to extract information from text
            lines = response.split('\n')
            parsed_response = {
                'predicted_gesture': '',
                'reasoning': '',
                'insight': ''
            }
            
            current_field = None
            for line in lines:
                line = line.strip()
                if 'predicted gesture' in line.lower():
                    current_field = 'predicted_gesture'
                    parsed_response[current_field] = line.split(':', 1)[1].strip() if ':' in line else ''
                elif 'reasoning' in line.lower():
                    current_field = 'reasoning'
                    parsed_response[current_field] = line.split(':', 1)[1].strip() if ':' in line else ''
                elif 'insight' in line.lower():
                    current_field = 'insight'
                    parsed_response[current_field] = line.split(':', 1)[1].strip() if ':' in line else ''
                elif current_field and line:
                    parsed_response[current_field] += ' ' + line
            
            # Clean up the extracted text
            for key in parsed_response:
                parsed_response[key] = parsed_response[key].strip()
            
            return parsed_response
            
        except Exception as e:
            # If all parsing attempts fail, return a default structure
            return {
                'predicted_gesture': 'unknown',
                'reasoning': f'Failed to parse response: {str(e)}',
                'insight': 'Error in response parsing'
            } 