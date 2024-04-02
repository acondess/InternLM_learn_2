# 课时二 轻松分钟玩转书生·浦语大模型趣味 Demo

![Helloword](image.png)

[飞书地址](https://aicarrier.feishu.cn/wiki/Vv4swUFMni5DiMkcasUczUp9nid#LSBkd2cTHorhsAx5jZAcO0B3nqe)






## 提交的作业结果

[作业要求](https://github.com/InternLM/Tutorial/blob/camp2/helloworld/homework.md)

- 使用 InternLM2-Chat-1.8B 模型生成 300 字的小故事（截图）

![寓言故事](image-11.png)

- 熟悉 huggingface 下载功能，使用 huggingface_hub python 包，下载 InternLM2-Chat-7B 的 config.json 文件到本地（截图下载过程）
- 完成 浦语·灵笔2 的 图文创作 及 视觉问答 部署（截图）
- 完成 Lagent 工具调用 数据分析 Demo 部署（截图）

### 作业笔记

#### 模型生成小故事

#### huggingface下载模型

- [模型地址](https://huggingface.co/internlm/internlm2-chat-1_8b)

- [huggingface hub python文档](https://huggingface.co/docs/huggingface_hub/quick-start)

#### 灵笔2部署

- 图文创作

- 视觉问答

#### Lagent 数据分析

## 视频笔记

[视频链接](https://www.bilibili.com/video/BV1AH4y1H78d/)

## 文档复现笔记

[文档链接](https://github.com/InternLM/Tutorial/blob/camp2/helloworld/hello_world.md)

### 部署InternLM2-Chat-1.8B 模型进行智能对话

#### 配置基础环境

- 创建开发机

  进入[算力平台](https://studio.intern-ai.org.cn/)点击创建开发机，选择算力——>开发机命名——>选择镜像（cuda11.7-conda）——>设置算力用时。

  ![alt text](image-2.png)


- 配置开发机开发环境
  
  进入开发机终端，输入如下命令安装conda开发环境（实测运行12分钟左右）
  
  ``` bash
  studio-conda -o internlm-base -t demo
  # 与 studio-conda 等效的配置方案
  # conda create -n demo python==3.10 -y
  # conda activate demo
  # conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
  ```
  ![开发机终端](image-1.png)

  - conda create -n demo python==3.10 -y: 这个命令使用conda包管理器创建一个名为"demo"的环境，并指定使用Python 3.10版本。-y选项表示在创建环境时自动确认所有提示，无需手动确认。

  - conda activate demo: 这个命令用于激活名为"demo"的环境。激活环境后，所有后续的命令和操作都将在该环境中进行。

  - conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia: 这个命令使用conda安装特定版本的PyTorch及其相关库。pytorch==2.0.1表示安装PyTorch 2.0.1版本，torchvision==0.15.2表示安装torchvision库的0.15.2版本，torchaudio==2.0.2表示安装torchaudio库的2.0.2版本。pytorch-cuda=11.7表示安装支持CUDA 11.7的PyTorch版本。-c pytorch -c nvidia指定了从pytorch和nvidia这两个渠道进行安装。

  ![conda环境安装结果1](image-3.png)
  
  ![conda环境安装结果2](image-4.png)

  查看conda环境list

  ![alt text](image-5.png)

  进入开发环境
  ``` bash
    conda activate demo
  ```

  

  进行环境依赖包安装

  ``` bash
   pip install huggingface-hub==0.17.3
   pip install transformers==4.34 
   pip install psutil==5.9.8
   pip install accelerate==0.24.1
   pip install streamlit==1.32.2 
   pip install matplotlib==3.8.3 
   pip install modelscope==1.9.5
   pip install sentencepiece==0.1.99
  ```

    - huggingface-hub: 提供了与Hugging Face模型和数据集库的交互功能。

    - transformers: 提供了用于自然语言处理任务的预训练模型和相关工具。

    - psutil: 提供了一个跨平台的库，用于获取系统信息和进程管理。

    - accelerate: 提供了用于加速深度学习训练的工具和API。

    - streamlit: 提供了一个用于构建交互式Web应用程序的Python库。

    - matplotlib: 提供了一个用于绘制图表和可视化数据的Python库。

    - modelscope: 提供了一个用于分析和比较机器学习模型的Python库。

    - sentencepiece: 提供了一个用于分词和生成子词单元的工具和库。

#### 下载大模型

  进入jupyter终端，进入demo环境（conda），cd到当前jupyter路径下。

``` bash
  cd /root
  conda activate demo
```
![下载大模型环境](image-6.png)

  创建文件夹&python文件

  ``` bash
   mkdir demo
   cd demo
   tourch cli_demo.py
   tourch download_mini.py
  ```

  ![创建文件夹&文件](image-7.png)

  编写脚本——download_mini.py

  ``` python
  import os  # 导入os模块，用于操作系统相关的操作
  from modelscope.hub.snapshot_download import snapshot_download  # 从modelscope.hub模块导入snapshot_download函数，用于下载模型

  # 创建保存模型目录
  os.system("mkdir /root/models")  # 使用os.system执行命令行命令，创建一个名为models的目录在/root路径下

  # save_dir是模型保存到本地的目录
  save_dir="/root/models"  # 定义变量save_dir，其值为模型保存的目录路径

  # 使用snapshot_download函数下载模型，参数包括模型的名字，缓存目录和版本号
  snapshot_download("Shanghai_AI_Laboratory/internlm2-chat-1_8b", 
                    cache_dir=save_dir, 
                    revision='v1.1.0')
  ```

  执行命令下载模型

  ``` bash
    python download_mini.py
  ```
  ![下载大模型](image-8.png)

  ![大模型文件](image-9.png)

  #### 基于大模型对话

    编写cli_demo.py脚本
  
  ```python  linenums="1"
        import torch  # 导入torch库，用于进行深度学习模型的操作
        from transformers import AutoTokenizer, AutoModelForCausalLM  # 从transformers库中导入AutoTokenizer和AutoModelForCausalLM，用于处理自然语言和加载模型

        model_name_or_path = "/root/models/Shanghai_AI_Laboratory/internlm2-chat-1_8b"  # 模型的名称或路径

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, device_map='cuda:0')  # 加载预训练的tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='cuda:0')  # 加载预训练的模型
        model = model.eval()  # 将模型设置为评估模式

        system_prompt = """You are an AI assistant whose name is InternLM (书生·浦语).
        - InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
        - InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
        """  # 系统提示信息

        messages = [(system_prompt, '')]  # 初始化消息列表

        print("=============Welcome to InternLM chatbot, type 'exit' to exit.=============")  # 打印欢迎信息

        while True:  # 循环接收用户输入
          input_text = input("\nUser  >>> ")  # 获取用户输入
          input_text = input_text.replace(' ', '')  # 去除用户输入的空格
          if input_text == "exit":  # 如果用户输入'exit'，则退出循环
            break

          length = 0
          for response, _ in model.stream_chat(tokenizer, input_text, messages):  # 使用模型进行聊天
            if response is not None:  # 如果响应不为空
              print(response[length:], flush=True, end="")  # 打印响应
              length = len(response)  # 更新响应长度
  ```

  ![运行截图](image-10.png)

  输入提示词：创作一个300字的寓言故事，要求有趣
  ![寓言故事](image-11.png)