from unstructured.partition.pdf import partition_pdf

raw_pdf_elements = partition_pdf(
    filename="./file_name-3.pdf",
    # Use layout model (YOLOX) to get bounding boxes (for tables) and find titles
    # Titles are any sub-section of the document
    infer_table_structure=True,
    # Post processing to aggregate text once we have the title
    chunking_strategy="by_title",
    # Chunking params to aggregate text blocks
    # Attempt to create a new chunk 3800 chars
    # Attempt to keep chunks > 2000 chars
    # Hard max on chunks
    max_characters=4000,
    new_after_n_chars=3800,
    combine_text_under_n_chars=2000
)

from pydantic import BaseModel
from typing import Any

# Define data structure
class Element(BaseModel):
    type: str
    text: Any

# Categorize by type
categorized_elements = []
for element in raw_pdf_elements:
    if "unstructured.documents.elements.Table" in str(type(element)):
        categorized_elements.append(Element(type="table", text=str(element)))
        # 类型为Table的时候，str(element)返回结果不含边框线，不同列用\t隔开，不同行用\n隔开
    elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
        categorized_elements.append(Element(type="text", text=str(element)))
        # 上面Table的方式，加上Table前后的文本内容


import os
from pdf2image import convert_from_path

os.environ["OPENAI_API_KEY"] = "<Your_OPENAI_API_Key>"


os.mkdir("./pages")
convertor = convert_from_path('./file_name.pdf')

for idx, image in enumerate(convertor):
    image.save(f"./pages/page-{idx}.png")  # 保存为png，保存在pages目录

pages_png = [file for file in os.listdir("./pages") if file.endswith('.png')]
# 记录每张图的path

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
        },  # 多个type和text对，有助于后期插入图片——GPT-4V的话，是这样的
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
  "max_tokens": 1000
}

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
		# 传入图像的path，编码为base64格式——多模态大模型的标准输入格式
      

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
        print("暂时跳过...解码过程发生错误。")
        continue

graphs_description = [Element(type="graph", text=str(item)) for item in graphs_description]  # 把列表中所有的元素都变成Element实例，文字信息就保存在text里面


from langchain.tools import tool
from langchain_openai import ChatOpenAI

@tool
def calculate_growth_rate(current_value: float, previous_value: float) -> dict:
    """计算环比增长率：(当前值 - 上期值) / 上期值 * 100%"""
    if previous_value == 0:
        return {"error": "上期值为0，无法计算增长率"}
    
    growth_rate = ((current_value - previous_value) / previous_value) * 100
    return round(growth_rate, 2)


from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

# 创建工具列表
tools = [calculate_growth_rate]

# 创建LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 创建React Agent
agent = create_react_agent(llm, tools)


import chromadb
from llama_index.core import VectorStoreIndex, Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext

chroma_client = chromadb.PersistentClient(path='./chroma_index')
chroma_collection = chroma_client.get_or_create_collection(
    name='your_collection_name',  # 红色部分需要自己补上
    metadata={"hnsw:space": "cosine"}
)

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

all_docs = categorized_elements + graphs_description  
# 之前分步分析文本/表格和图片并且建立各自的列表，目的在此
# 放在这里遍历，整合成包含了许多Document的列表

documents = [Document(text=t.text, metadata={"category": t.type}) for t in all_docs]
# 一开始构建了Element类，就是用于这里直接调用text和type的——对应了一开始说的“方便”整合
# 整合种类元素+图片描述，并且将这些信息打包成Document，用于让VectorStoreIndex直接导入、建立索引

index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

query_engine = index.as_query_engine()  # 用index进行query

user_question = input("请针对公司财务情况，输入问题：")

if any(keyword in user_question for keyword in ["增长率", "环比", "增长", "变化率", "growth rate", "环比增长"]):
    response = agent.invoke([HumanMessage(content=user_question)])
    print(response['messages'][-1].content)
else:
    response = query_engine.query(user_question)
    print(response)


from ragas.metrics import faithfulness, answer_relevancy
from ragas import evaluate

# 准备评估数据集（问题和参考答案）
eval_questions = [
    "特斯拉第三季度营收是多少？环比增长情况如何？",
    "车辆交付量的变化趋势是怎样的？",
    "公司的毛利率表现如何？",
    "研发支出占营收的比例是多少？",
    "自由现金流的变化情况如何？"
]

# 参考答案（实际应用中需要人工标注）
eval_answers = [
    ["特斯拉第三季度总营收为230亿美元，环比增长约15%"],
    ["第三季度车辆交付量达到45万辆，环比增长约12%"],
    ["第三季度毛利率为18.5%，保持稳定"],
    ["研发支出占营收比例约为5%"],
    ["自由现金流为25亿美元，环比增长8%"]
]

metrics = [
        faithfulness,        # 忠实度：答案是否基于上下文
        answer_relevancy,    # 答案相关性：答案与问题的相关程度
        ]
    
# 执行评估
result = evaluate(
    metrics=metrics,
    questions=eval_questions,
    ground_truths=eval_answers
)

print((result['answer_relevance'] + result['faithfulness']) / 2)
