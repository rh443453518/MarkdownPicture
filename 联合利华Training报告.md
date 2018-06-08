----------


# 联合利华Training报告

标签（空格分隔）： 戎昊 Training Report

---
## 项目概述
本次联合利华Training的目标是通过联合利华的历史订单、产品信息、促销信息的数据来对某个仓库`特定产品未来的销量进行预测。任务主要包括数据的清理整合，特征提取，建模预测以及结果评价。

最终使用的建模方法有机器学习(Linear Regression (Ridge + Lasso), Kernel Ridge Regression (poly=1 kernel), Random Forests Regressor, SVR, Gradient Boosting Regressor)与深度学习(LSTM)。

---
> **Step 1. 数据清理与整合**

首先通过sku_id和spu_id对sku, order, promotion三个表格进行连接（Inner Join）。然后将stock_id和spu_id进行粘合获得所需的Index。最终有20个不同的Index。

然后提取自然周、联合利华月、月中周（weekofmonth）。三者均需要将具体的某周按联合利华月的规则分配到某月。由于我对于每周时间的记录方式是记录下这周最后一天，我只需要判断这周最后一天是否超出该月的3号或该年的1月3号就可以判断这周的归属。具体代码如下：
```python
def calcUniMonth(dateItem):
    month = dateItem.month
    if (dateItem.day <= 3):
        month = (month - 1) if (month != 1) else 12
    year = dateItem.year - 1 if (dateItem.dayofyear <= 3) else dateItem.year
    return str(year) + '-' + ('0'+str(month))[-2:]


def calcWeekOfYear(dateItem):
    dayofyear = dateItem.dayofyear
    year = (dateItem.year - 1) if (dateItem.dayofyear <= 3) else dateItem.year
    nMonth = int(dayofyear/7) if ((dayofyear % 7) <= 3) else int(dayofyear/7) + 1
    if (dayofyear <= 3):
        dayofyear = (dateItem - pd.DateOffset(7)).dayofyear
        nMonth = int(dayofyear/7) + 1 if ((dayofyear % 7) <= 3) else int(dayofyear/7) + 2
    return str(year) + '-' + ('0'+str(nMonth))[-2:]


def calWeekOfMonth(dateItem):
    day = dateItem.day
    if ((day % 7) <= 3 and day <= 3):
        wom = calWeekOfMonth(dateItem - pd.DateOffset(7)) + 1.0
    elif ((day % 7) <= 3 and day > 3):
        wom = float(int(day/7))
    else:
        wom = float(int(day/7) + 1)
    return wom
```

---
> **Step 2. 数据分析**

我先对order_count按自然周、联合利华月、月中周进行可视化：

![订单量by自然周][1]
$$Figure1. 订单量by自然周$$

![订单量by联合利华月][2]
$$Figure2. 订单量by联合利华月$$

![订单量by月中周][3]
$$Figure3. 订单量by月中周$$

* 可以看出某些Index之间的订单量变化有较明显的关联性。

![订单量by品牌＋包装量][4]
$$Figure4. 订单量by品牌＋包装量$$

* 可以看出不同品牌和包装的订单量会有明显的区别。

![订单量与促销量散点图][5]
$$Figure5. 订单量与促销量散点图$$

* 可以看出促销量与订单量有正相关关系。

> **Step 3. 特征提取**

对于机器学习模型，我提取了以下特征。对于其中categorical的特征，我使用了pandas dummy的one hot encoding方式将单列特征展开。

| Feature        | Description                    |
|----------------|--------------------------------|
| weekofmonth    | 该月第几周                     |
| time           | 总体第几月（已有数据从头到尾） |
| week_ofyear    | 该年第几月                     |
| year           | 年                             |
| unilever_month | 联合利华月                     |
| brand          | 金纺／多芬                     |
| product_code   | Spu_id前段                     |
| package_size   | 商品包装容量                   |
| stock_id       | 仓库编号                       |
| quota          | 促销量                         |
| lookback_1     | （当前月－1）月order量         |
| lookback_2     | （当前月－2）月order量         |
| lookback_3     | （当前月－3）月order量         |
| lookback_4     | （当前月－4）月order量         |
| lookback_5     | （当前月－5）月order量         |


在深度学习的模型中，我仅使用了order_count作为特征。在不同的look back month设定下，会使用不同的特征。例如，look back month = a的情况下，会使用该月之前a个月的order_count作为特征，最终的Training Set为一个n乘a的矩阵。




----------


> **Step 4. 建模预测**

* 机器学习使用全部Index数据组成的大数据集建模。Training Set (2/3) 和Test Set (1/3) 在全部数据中随机分配。最终使用Test Adjusted MAPE作为最终Average Adjusted MAPE。

* 深度学习（LSTM－RNN, Long short-term memory是一种循环神经网络模型单元）分别对20个Index单独建模。Training Set和Test Set以2017年7月为分界线分配。最终使用20个模型的平均Adjusted MAPE作为Average Adjusted MAPE。

![LSTM模型][6]
$$Figure6. 某Index的LSTM模型（3M-Lookback） $$

---
## 主要结论
* Random Forest Regressor 和 LSTM （1M-Lookback）效果最好。
* 在机器学习模型中，当前机器学习模型尽管集合了所有Index的信息但是效果并不好。我怀疑使用现有的lookback数据进行所有Index数据的建模可能反而会比单独建模的效果差。因为各个商品应该有不同的销售时序规律。需要更有效的时序特征来反映这些各个商品的规律。


---
## 问题
* Ridge Kernel Polynomial Degree >= 2时（使用二阶以上交互项）会出现预测错误，原因暂时不明。
* 当前的Look Back Month结果不太符合逻辑，5以内较小的Look Back Month反而效果更好，可能需要更大一点的Look Back Month（比如10）？可能要求更久的数据量。
* 不同Index之间有Correlation，不知道可不可以利用下？


---
## 具体分析结果
> **Step 5. 结果评价**

机器学习模型中，可以看出Random Forest Regressor的训练结果最好。

| Model                       | Average Adjusted MAPE |
|-----------------------------|-----------------------|
| Linear Regression           | 0.464260905322259     |
| LR Ridge                    | 0.46552959332241767   |
| LR Lasso                    | 0.46581763027078038   |
| Kernel Ridge                | 0.45697584321687518   |
| Random Forest Regressor     | 0.39583673343991571   |
| Support Vector Regression   | 0.59386562318678249   |
| Gradient Boosting Regressor | 0.43663471275431759   |

深度学习（LSTM－RNN）模型中，可以看出使用过去1个月历史数据的LSTM模型的训练结果最好。

| Model              | Average Adjusted MAPE |
|--------------------|-----------------------|
| LSTM (1M Lookback) | 0.3951827654191061    |
| LSTM (2M Lookback) | 0.40384434798525765   |
| LSTM (3M Lookback) | 0.4143217307103642    |
| LSTM (4M Lookback) | 0.41245143427877257   |
| LSTM (5M Lookback) | 0.4143217307103642    |


---
## 未来优化思路

* 未来可以将现有的机器学习与深度学习模型结果结合。（网上有介绍将LSTM和其他机器学习的预测作为神经网络的节点结合，再进行训练。）
* 使用tsfresh提取更多时间序列的特征。现阶段时间序列的特征很少，只有过去几个月的历史数据，忽视了历史上其他数据变化的特征。使用tsfresh可以提取更多关于历史数据的时间序列上的特征。（如峰值，周期等）
* 提取更多Feature。现阶段各个模型表现都比较一般（包括可以组合特征的Polynomial Kernel Ridge Regression模型），原因可能是缺乏解释性Feature。如果可以获得更多关于产品销售方面的信息，会对建模有很大帮助。


  [1]: https://raw.githubusercontent.com/rh443453518/MarkdownPicture/master/orderByWeek.png
  [2]: https://raw.githubusercontent.com/rh443453518/MarkdownPicture/master/orderByMonth.png
  [3]: https://raw.githubusercontent.com/rh443453518/MarkdownPicture/master/orderByWeekofmonth.png
  [4]: https://raw.githubusercontent.com/rh443453518/MarkdownPicture/master/brand_size.png
  [5]: https://raw.githubusercontent.com/rh443453518/MarkdownPicture/master/quota.png
  [6]: https://raw.githubusercontent.com/rh443453518/MarkdownPicture/master/LSTMpredict.png