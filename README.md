# smart-report-agent
基于 MCP 的智能财报分析 Agent
该agent用于读取财报并且回答用户相关的问题
亦可读取其他行业的PDF文件（如教育、法律、医疗等等）

**重点**：可解析PDF文件中的表格、图片等等大模型难以解析的文件内容，此方法适用于教育、法律、医疗等多个行业

## 解析文本&表格
以下操作可以在谷歌Colab上使用免费GPU进行安装
### 库的准备
先安装必要的库，此处解析主要靠unstructured.io

`pip install -q unstructured[all-docs] typing-extensions pydantic`

还要安装其他辅助的库

`pip install poppler-utils tesseract-ocr`

poppler和tesseract这两个库，用于解析非纯文本型的PDF，比如说扫描版或者图片型PDF

前者用于转换PDF中的每一页为图像，后者是OCR引擎，用于读取图像中的文字（也可以读取本来就是图片格式的PDF中的内容）

### 开始解析
1. 使用unstructured库，解析PDF文件
    ```from unstructured.partition.pdf import partition_pdf
    
    raw_pdf_elements = partition_pdf(
        filename="./文件名.pdf",
        # 使用布局模型（YOLOX）获取边界框（用于识别表格）并检测标题
        # 标题指文档中的任意子章节
        infer_table_structure=True,
        # 在识别出标题后，进行后处理以聚合文本
        chunking_strategy="by_title",
        # 文本分块参数，用于合并文本块：
        # 尝试在达到约 3800 个字符时创建新块
        # 尝试确保每个块至少包含 2000 个字符
        # 单个文本块的最大字符数硬性上限
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000
    )
    ```

2. 通过以上方式提取出CompositeElements和表格后，构建基于Pydantic的数据结构Element，存储类型（type）和文本（text）
   方便之后提取文件内容并分类：到底是文本、表格，还是图片
   ```
   from pydantic import BaseModel
   from typing import Any
    
   class Element(BaseModel):
       type: str
       text: Any
   ```
   
4. 遍历所有提取出来的元素，保存在一个列表中
   ```categorized_elements = []
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            categorized_elements.append(Element(type="table", text=str(element)))
            # 类型为Table的时候，str(element)返回结果不含边框线，不同列用\t隔开，不同行用\n隔开
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            categorized_elements.append(Element(type="text", text=str(element)))
            # 上面Table的方式，加上Table前后的文本内容
   ```

## 解析图片
**基本思路**：
将PDF转换为图片，遍历每一页，让多模态模型判断是否检测到图片，无图则返回空数组
### 库的准备
安装 `pdf2image`（依赖已装好的 `poppler`），用于遍历每一页：

`pip install -q pdf2image`

### 开始解析
1. 将每页 PDF 转为 PNG 图片：
    ```python
    import os
    from pdf2image import convert_from_path  # 调用pdf2image的convert_from_path方法，迭代返回的实例，将每一个元素（也就是一页）保存为png
    
    os.mkdir("./pages")
    convertor = convert_from_path('./文件名.pdf')
    
    for idx, image in enumerate(convertor):
        image.save(f"./pages/page-{idx}.png")  # 保存为png，保存在pages目录
    
    pages_png = [file for file in os.listdir("./pages") if file.endswith('.png')]
    # 记录每张图的path，供之后使用
    ```

2. 准备请求头和提示词，要求模型以 JSON 格式返回图表描述（忽略表格）：
    ```python
    import os
    os.environ["OPENAI_API_KEY"] = "<Your_OpenAI_Key>"  # 设置openai密钥，方便后续使用，避免硬编码造成错误
    
    headers = {
      "Content-Type": "application/json",  # 说明是什么类型，让openai用json格式读取载荷
      "Authorization": "Bearer " + str(os.environ["OPENAI_API_KEY"])
    }
    
    payload = {
      "model": "gpt-4o",  # 有效载荷带有模型名称
      "messages": [
        {
          "role": "user",
          "content": [  # content就是提示词，因为有限制role为user，所以这里的content就是提示词的意思。这里叫content，方便与role为assistant的content区别开来：那是AI生成的内容
            {
              "type": "text",
              "text": "你是一个助手，负责从图像中识别charts, graphs或示意图并总结信息。若有多张图，请分别说明。忽略表格。"
            },  # 多个type和text对，有助于后期插入图片
            {
              "type": "text",
              "text": '返回格式必须为：{"graphs": [<chart_1>, <chart_2>, ...]}，每个元素是该图的描述。'
            },
            {
              "type": "text",
              "text": '若无图表，返回：{"graphs": []}。不要添加额外内容，不要用 ``` 或 “json” 字样。'
            },  # 模型会自动合并这些type和text对，不影响最终输出结果
            {
              "type": "text",
              "text": "请查看附图，用 JSON 格式描述所有图表内容，简洁明了。"
            }  # 多个type和text对，避免长文本截断风险
          ]
        }
      ],
      "max_tokens": 1000  # 该参数——“最大token限制”，为有效载荷中基础参数
    }
    
    def encode_image(image_path):
      with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    		# 传入图像的path，编码为base64格式——多模态大模型的标准输入格式
    ```

3. 逐页发送图片请求，并保存结果：
   ```python
    from tqdm import tqdm
    import requests
    
    graphs_description = []
    for idx, page in tqdm(enumerate(pages_png)):  
    # 根据每张图的路径，将图片编码为base64这种大模型能够读取的格式
        base64_image = encode_image(f"./pages/{page}")
    
        tmp_payload = copy.deepcopy(payload)  # 把上面的载荷复制为tmp_payload临时载荷
        tmp_payload['messages'][0]['content'].append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{base64_image}"}
        })  # 在临时载荷messages的content里加入图像（content的type里不是text就是image_url）
        # openai的多模态大模型规定按照上面这种“data:[file_type];base64,”的格式传递image_url
    
        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=tmp_payload)  # 大模型返回的是json格式的数据
            response = response.json()  # 读取json，并且后台使用json.loads()将之转换为py字典
            graph_data = json.loads(response['choices'][0]['message']['content'])['graphs']
    				# json.loads()：加载json数据，转换为字典。
    				# 加载大模型回复里的graph字段内容：可能是一张图，可能是两张图
            desc = [f"{page}\n" + '\n'.join(f"{key}: {item[key]}" for key in item.keys()) for item in graph_data]  # 列出来这一页里每张图的内容
            graphs_description.extend(desc)  # 把desc里的每一个元素逐个加进列表，而不是像append那样会把元素全部内容一整个塞进列表
    				# 这里desc是一个列表，所以意思是说把列表里的所有元素逐个加入graphs_description
        except:
            print("解码过程发生错误。")
            continue
    
    graphs_description = [Element(type="graph", text=str(item)) for item in graphs_description]  # 把列表中所有的元素都变成Element实例，文字信息就保存在text里面
    ```
> ⚠️ 此方法每页都调用模型，成本较高。可手动标记含图页面以节省开销。

## 创建智能体
### 基本思路
因为遇到诸如“计算环比增长率”之类问题的话，LLM可能算不准
所以需要使用agent智能体来处理这些特定的问题

### 构建智能体过程
1. 自定义函数calculate_growth_rate，将之作为智能体可调用的工具
   用于解决诸如环比增长率之类LLM可能算不准的问题
   
    ```
    from langchain.tools import tool
    from langchain_openai import ChatOpenAI
    
    @tool
    def calculate_growth_rate(current_value: float, previous_value: float) -> dict:
        """计算环比增长率：(当前值 - 上期值) / 上期值 * 100%"""
        if previous_value == 0:
            return {"error": "上期值为0，无法计算增长率"}
        
        growth_rate = ((current_value - previous_value) / previous_value) * 100
        return round(growth_rate, 2)
    ```
    
2. 构建React系统，搭建智能体体系
    ```
    from langgraph.prebuilt import create_react_agent
    from langchain_core.messages import HumanMessage
    
    # 创建工具列表
    tools = [calculate_growth_rate]
    
    # 创建LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # 创建React Agent
    agent = create_react_agent(llm, tools)
    ```

## 存入向量数据库
1. **准备并存储文档**：将处理好的数据转换成LlamaIndex的Document格式，并保存到Chroma中。
   
  ```python
  import chromadb
  from llama_index.core import VectorStoreIndex, Document
  from llama_index.vector_stores.chroma import ChromaVectorStore
  from llama_index.core.storage.storage_context import StorageContext
  
  chroma_client = chromadb.PersistentClient(path='./VECTOR_DB_DIR')
  chroma_collection = chroma_client.get_or_create_collection(
      name='pdf_report_reader',  
      metadata={"hnsw:space": "cosine"}
  )
  
  vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
  storage_context = StorageContext.from_defaults(vector_store=vector_store)
  ```

2. **生成索引**：使用`VectorStoreIndex.from_documents()`方法创建索引。
  ```
  all_docs = categorized_elements + graphs_description  
  # 之前分步分析文本/表格和图片并且建立各自的列表，目的在此
  # 放在这里遍历，整合成包含了许多Document的列表
  
  documents = [Document(text=t.text, metadata={"category": t.type}) for t in all_docs]
  # 一开始构建了Element类，就是用于这里直接调用text和type的——对应了一开始说的“方便”整合
  # 整合种类元素+图片描述，并且将这些信息打包成Document，用于让VectorStoreIndex直接导入、建立索引
  
  index = VectorStoreIndex.from_documents(
  documents, storage_context=storage_context
  )
  ```

3. **查询和利用索引**：建立查询引擎并执行查询操作。
  ```
  query_engine = index.as_query_engine()  # 用index进行query
  
  user_question = input("请针对公司财务情况，输入问题：")
  
  if any(keyword in user_question for keyword in ["增长率", "环比", "增长", "变化率", "growth rate", "环比增长"]):
      # 如果有涉及环比增长率的问题，则调用react，让智能体针对性回答
      response = agent.invoke([HumanMessage(content=user_question)])
      print(response['messages'][-1].content)
  else:  # 不涉及环比增长之类的问题的话，则直接交给大模型输出结果
      response = query_engine.query(user_question)
      print(response)
  ```

大模型还会引用输入的PDF文件中相对应的图片（如下图）
<img width="640" height="707" alt="image" src="https://github.com/user-attachments/assets/c634ae0d-313a-4508-af55-1f7f3480d148" />

因为chatGPT多模态的模型具有这样的能力

## 评估效果
### 基本思路
没有评估，一切都是玄学
通过评估，证明智能体的效果比人眼看报告效果好
实际评估过程需要量更大的数据集，此处问答对仅作为展示参考用

### 评估过程
1. 准备评估数据集（问题和参考答案）
  ```
  eval_questions = [
      "A公司第三季度营收是多少？环比增长情况如何？",
      "交付量的变化趋势是怎样的？",
      "公司的毛利率表现如何？",
      "研发支出占营收的比例是多少？",
      "自由现金流的变化情况如何？"
  ]
  
  # 参考答案（需人工标注）
  eval_answers = [
      ["A公司第三季度总营收为230亿美元，环比增长约15%"],
      ["第三季度交付量达到45万台，环比增长约12%"],
      ["第三季度毛利率为18.5%，保持稳定"],
      ["研发支出占营收比例约为5%"],
      ["自由现金流为25亿美元，环比增长8%"]
  ]
  ```

2. 使用ragas框架，根据忠实度和问答相关性，计算准确率
  ```
  from ragas.metrics import faithfulness, answer_relevancy
  from ragas import evaluate
  metrics = [
          faithfulness,        # 忠实度：答案是否基于上下文
          answer_relevancy,    # 回答相关性：答案与问题的相关程度
          ]
      
  # 执行评估
  result = evaluate(
      metrics=metrics,
      questions=eval_questions,
      ground_truths=eval_answers
  )
  
  print((result['answer_relevance'] + result['faithfulness']) / 2)  # 取faithfulness和answer_relevancy的平均值为准确率
  ```

通过以上方式，确保agent智能体的效果比人力更好
