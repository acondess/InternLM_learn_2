# 课时五 LMDeploy量化部署LLM实践
![alt text](image-113.png)

[飞书地址](https://aicarrier.feishu.cn/wiki/Vv4swUFMni5DiMkcasUczUp9nid#LSBkd2cTHorhsAx5jZAcO0B3nqe)

[算力平台](https://studio.intern-ai.org.cn/)

## 1. 提交的作业结果

[作业要求地址](https://github.com/InternLM/Tutorial/blob/camp2/lmdeploy/homework.md)

### 1.1 基础作业

#### 1.1.1 配置 LMDeploy 运行环境

- 结果截图

  ![alt text](image-118.png)

  ![alt text](image-119.png)

- 复现步骤
  
  [lmdeploy环境部署](#21-lmdeploy环境部署)
  

#### 1.1.2 以命令行方式与 InternLM2-Chat-1.8B 模型对话

- 结果截图
- 复现步骤

### 1.2 进阶作业

### 1.2.1 设置KV Cache最大占用比例为0.4，开启W4A16量化，以命令行方式与模型对话。

- 结果截图
- 复现步骤

### 1.2.2 以API Server方式启动 lmdeploy，开启 W4A16量化，调整KV Cache的占用比例为0.4，分别使用命令行客户端与Gradio网页客户端与模型对话。

- 结果截图
- 复现步骤

### 1.2.3 使用W4A16量化，调整KV Cache的占用比例为0.4，使用Python代码集成的方式运行internlm2-chat-1.8b模型。

- 结果截图
- 复现步骤

### 1.2.4 使用 LMDeploy 运行视觉多模态大模型 llava gradio demo。

- 结果截图
- 复现步骤

### 1.2.5 将 LMDeploy Web Demo 部署到 OpenXLab 。

- 结果截图
- 复现步骤

## 2. 文档复现

[文档地址](https://github.com/InternLM/Tutorial/blob/camp2/lmdeploy/README.md)

### 2.1 LMDeploy环境部署

- 创建开发机

>选择镜像Cuda12.2-conda；选择10% A100*1GPU；点击“立即创建”。

>***注意***   请不要选择Cuda11.7-conda的镜像，新版本的lmdeploy会出现兼容性问题。

![alt text](image-114.png)

- 终端模式

切换到命令行模式。

![alt text](image-115.png)

- conda环境

```bash
studio-conda -t lmdeploy -o pytorch-2.1.2
```

![alt text](image-116.png)

![alt text](image-117.png)

- 安装依赖

  - 激活lmdeploy环境

  ```bash
  conda activate lmdeploy
  ```

  - 安装依赖

  ```bash
  pip install lmdeploy[all]==0.3.0
  ```

  ![alt text](image-118.png)

  ![alt text](image-119.png)

### 2.2 LMDeploy模型对话（chat）

- internlm2-chat-1_8b模型下载

开发机已有模型软链接方式：

```bash
ls
ls /root/share/new_models/Shanghai_AI_Laboratory/
ln -s /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b /root/
ls
```
![alt text](image-120.png)

- 直接对话（省略Transform对比——我要速通-_-` ）

```bash
lmdeploy chat /root/internlm2-chat-1_8b
```
![alt text](image-121.png)

![alt text](image-122.png)

> 确实很快 有种grok的感觉