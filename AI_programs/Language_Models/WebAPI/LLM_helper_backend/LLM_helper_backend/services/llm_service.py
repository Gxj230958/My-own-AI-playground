from openai import OpenAI
import json
from .function_tools import function_registry, function_desc
from typing import Generator, Dict, Any, List, Tuple

# Initialize OpenAI client
client = OpenAI(
    api_key="sk-0b870c21ee08424da20f00e028041b99",
    base_url="https://api.deepseek.com"
)

# Set system message
system_message = {"role": "system", "content": "你是一个用于对话场景的智能助手，请正确、简洁、比较口语化地回答问题。你能够使用提供的tools（函数）来回答问题，有必要时需要从用户提问中抽取函数所需要的参数"}

def get_llm_response(question: str, conversation_history=None) -> tuple[str, list]:
    """
    Get response from LLM for a given question with function calling support
    
    Args:
        question: The question from the user
        conversation_history: Optional conversation history to maintain context
    
    Returns:
        A tuple containing (response_text, updated_conversation_history)
    """
    # Initialize messages with system message and optional history
    if conversation_history:
        messages = conversation_history
    else:
        messages = [system_message]
    
    # Add user question to messages
    user_message = {"role": "user", "content": question}
    messages.append(user_message)
    
    try:
        # Create request with function calling capability
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=function_desc,
            stream=False
        )
        
        # Check if the response contains a function call
        message = response.choices[0].message
        
        # If there's a function call in the response
        if message.tool_calls:
            # Add the assistant's message with the tool calls to the conversation
            messages.append(message)
            
            # Process all function calls
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                try:
                    function_arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    function_arguments = {}
                
                # Call the function and get the result
                function_result = function_registry.get(function_name)(function_arguments)
                
                # Add the function result to the conversation
                tool_message = {"role": "tool", "tool_call_id": tool_call.id, "content": function_result}
                messages.append(tool_message)
            
            # Get the final response after function calls
            final_response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                stream=False
            )
            
            # Get the final answer
            final_answer = final_response.choices[0].message.content
            
            # Add the final answer to the conversation
            messages.append({"role": "assistant", "content": final_answer})
            
            return final_answer, messages
        else:
            # For regular responses without function calls
            answer = message.content
            
            # Add the response to the conversation
            messages.append({"role": "assistant", "content": answer})
            
            return answer, messages
    
    except Exception as e:
        return f"Error: {str(e)}", messages

def get_llm_response_stream(question: str, conversation_history=None) -> Generator[Dict[str, Any], None, None]:
    """
    Get streaming response from LLM for a given question with function calling support
    
    Args:
        question: The question from the user
        conversation_history: Optional conversation history to maintain context
    
    Yields:
        Chunks of the response as they are received
    """
    # Initialize messages with system message and optional history
    if conversation_history:
        messages = conversation_history.copy()
    else:
        messages = [system_message]
    
    # Add user question to messages
    user_message = {"role": "user", "content": question}
    messages.append(user_message)
    
    full_response = ""
    
    try:
        # First check if we need to call a function
        initial_response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=function_desc,
            stream=False
        )
        
        message = initial_response.choices[0].message
        
        # If there's a function call in the response
        if message.tool_calls:
            # 告知用户正在处理函数调用
            yield {"content": "正在处理您的请求，请稍候..."}
            
            # Add the assistant's message with the tool calls to the conversation
            messages.append(message)
            
            # Process all function calls
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                try:
                    function_arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    function_arguments = {}
                
                # Call the function and get the result
                function_result = function_registry.get(function_name)(function_arguments)
                
                # Add the function result to the conversation
                tool_message = {"role": "tool", "tool_call_id": tool_call.id, "content": function_result}
                messages.append(tool_message)
            
            # Get the final response after function calls - this time in streaming mode
            final_response_stream = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                stream=True
            )
            
            # Stream the final answer
            for chunk in final_response_stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield {"content": content}
            
            # Add the final answer to the conversation
            messages.append({"role": "assistant", "content": full_response})
        else:
            # For regular responses without function calls, stream directly
            stream_response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                stream=True
            )
            
            for chunk in stream_response:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield {"content": content}
            
            # Add the response to the conversation
            messages.append({"role": "assistant", "content": full_response})
    
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        yield {"content": error_msg}
        full_response = error_msg 