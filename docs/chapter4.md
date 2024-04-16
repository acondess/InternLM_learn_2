# 课时四 XTuner微调多模态Agent

![alt text](image-93.png)

[飞书地址](https://aicarrier.feishu.cn/wiki/Vv4swUFMni5DiMkcasUczUp9nid#LSBkd2cTHorhsAx5jZAcO0B3nqe)

[算力平台](https://studio.intern-ai.org.cn/)

## 1. 提交的作业结果

[作业要求](https://github.com/InternLM/Tutorial/blob/camp2/xtuner/homework.md)

### 1.1 训练自己的小助手认知（记录复现过程并截图）

- 训练完小助手对话截图：
- 复现笔记：

### 1.2 将自我认知的模型上传到 OpenXLab，并将应用部署到 OpenXLab

### 1.3  复现多模态微调

## 2. 文档复现笔记

[文档地址](https://github.com/InternLM/tutorial/blob/main/xtuner/README.md)

### 2.1 前置知识

### 2.2 复现步骤

#### 2.2.1 环境配置

- 创建开发机

![alt text](image-94.png)

- 进入开发机创建conda环境

```bash
cd /root
/share/install_conda_env_internlm_base.sh xtuner0.1.9
```

![alt text](image-95.png)

- 激活xtuner0.1.9环境&创建文件夹

```bash
conda activate xtuner0.1.9
cd /root
mkdir xtuner019 && cd xtuner019
```

![alt text](image-96.png)

- 获取0.1.9版本源码

```bash
git clone -b v0.1.9  https://github.com/InternLM/xtuner
```
![alt text](image-97.png)

- 从源码安装Xtuner

```bash
cd xtuner
pip install -e '.[all]'
```

![alt text](image-98.png)

- 创建数据集文件夹地址

```bash
# 创建一个微调 oasst1 数据集的工作路径，进入
mkdir ~/ft-oasst1 && cd ~/ft-oasst1
```

![alt text](image-99.png)

## 3. 视频总结