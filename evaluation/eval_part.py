
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
