from .llm_service import get_llm_response, get_llm_response_stream
from typing import Generator, Dict, Any

# Store conversation history for maintaining context across requests
conversation_history = None

def answer(question: str) -> str:
    """
    Process user question and return answer from LLM
    
    Args:
        question: The question from the user
    
    Returns:
        The answer as a string
    """
    global conversation_history
    
    # Get response from LLM service
    response, updated_history = get_llm_response(question, conversation_history)
    
    # Update conversation history for future requests
    conversation_history = updated_history
    
    return response

def answer_stream(question: str) -> Generator[Dict[str, Any], None, None]:
    """
    Process user question and return streaming answer from LLM
    
    Args:
        question: The question from the user
    
    Yields:
        Chunks of the response as they are received
    """
    global conversation_history
    
    # 创建一个内部生成器函数来处理流式响应和更新会话历史
    def stream_with_history_update():
        global conversation_history
        
        # 获取流式响应生成器
        response_stream = get_llm_response_stream(question, conversation_history)
        
        # 收集完整响应以便稍后更新会话历史
        full_response = ""
        
        # 遍历流式响应的每个块
        try:
            for chunk in response_stream:
                yield chunk
                
            # 在流式响应完成后，使用非流式方法获取更新后的会话历史
            # 这样可以确保会话历史被正确更新
            _, updated_history = get_llm_response(question, conversation_history)
            conversation_history = updated_history
        except Exception as e:
            print(f"Error in streaming response: {e}")
    
    # 返回内部生成器
    yield from stream_with_history_update()