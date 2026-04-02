from openai import OpenAI
import sys
import signal

# 初始化OpenAI客户端
client = OpenAI(
    api_key="sk-0b870c21ee08424da20f00e028041b99",  # 请替换为您的API密钥
    base_url="https://api.deepseek.com"
)

# 设置系统消息
system_message = {"role": "system", "content": "You are a helpful assistant"}

# 初始化消息列表，保存对话历史
messages = [system_message]

def handle_interrupt(signal, frame):
    """处理Ctrl+C中断"""
    print("\n程序已终止")
    sys.exit(0)

# 注册中断处理函数
signal.signal(signal.SIGINT, handle_interrupt)

def chat_with_model():
    """实现与模型的连续对话"""
    try:
        while True:
            # 获取用户输入
            user_input = input("Q: ")
            
            # 将用户消息添加到消息列表（保存上下文）
            user_message = {"role": "user", "content": user_input}
            messages.append(user_message)
            
            print("A: ", end="", flush=True)
            
            # 创建流式请求
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                stream=True
            )
            
            # 存储完整回答，用于添加到消息历史
            full_response = ""
            
            # 处理流式响应
            for chunk in response:
                # 检查此块是否包含content
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    full_response += content
            
            # 添加助手回答到消息列表（保存上下文）
            assistant_message = {"role": "assistant", "content": full_response}
            messages.append(assistant_message)
            
            print("\n")
    
    except Exception as e:
        print(f"\n发生错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    chat_with_model()
