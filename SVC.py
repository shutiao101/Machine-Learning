import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score  # sklearn.model_selection： 提供分割数据集和执行交叉验证的函数
from sklearn.svm import SVC  # sklearn.svm： 包含 SVC 类，用于实现支持向量分类
from sklearn.metrics import confusion_matrix, roc_curve, auc  # sklearn.metrics： 提供评估模型性能的函数
import seaborn as sns  # 热力图
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv('krkopt.DATA', header=None)
data.dropna(inplace=True)  # 删除缺失值 inplace=True表示在原数据上进行修改

# 将样本数值化
for i in [0, 2, 4]:  # a,1,b,3,c,2,draw (one example from "krkopt.data")
    data.loc[data[i] == 'a', i] = 1
    data.loc[data[i] == 'b', i] = 2
    data.loc[data[i] == 'c', i] = 3
    data.loc[data[i] == 'd', i] = 4
    data.loc[data[i] == 'e', i] = 5
    data.loc[data[i] == 'f', i] = 6
    data.loc[data[i] == 'g', i] = 7
    data.loc[data[i] == 'h', i] = 8

# 将标签数值化
data.loc[data[6] != 'draw', 6] = -1  # loc[6]: 第七列
data.loc[data[6] == 'draw', 6] = 1

# Convert the label column to int (or float) to avoid any issues
data[6] = data[6].astype(int)

# 位置（用数值表示）归一化处理
for i in range(6):
    data[i] = (data[i] - data[i].mean()) / data[i].std()

# 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :6], data[6], test_size=0.82178500142572)
# 前六列表示数据，第七列是标签,test_size=0.82178500142572表示测试集占总样本的82.1785%

# 寻找C和gamma的粗略范围
CScale = [i for i in range(100, 201, 10)]
gammaScale = [i / 10 for i in range(1, 11)]
cv_scores = 0

for i in CScale:
    for j in gammaScale:
        model = SVC(kernel='rbf', C=i, gamma=j)
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        # cross_val_score：Evaluate a score by cross-validation, parameters:5折交叉验证,scoring='accuracy'表示用准确率作为评估指标
        if scores.mean() > cv_scores:
            cv_scores = scores.mean()
            savei = i
            savej = j * 100

# 找到更精确的C和gamma
CScale = [i for i in range(savei - 5, savei + 5)]  # 再savei的左右各取5个值形成新的取值范围
gammaScale = [i / 100 + 0.01 for i in range(int(savej) - 5, int(savej) + 5)]
# 再savej的左右各取5个值形成新的取值范围，+0.01是为了避免除以0
cv_scores = 0

for i in CScale:
    for j in gammaScale:
        model = SVC(kernel='rbf', C=i, gamma=j)
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        if scores.mean() > cv_scores:
            cv_scores = scores.mean()
            savei = i
            savej = j

# 将确定好的参数重新建立svm模型
model = SVC(kernel='rbf', C=savei, gamma=savej)
model.fit(X_train, y_train)  # 
pre = model.predict(X_test)
model_score = model.score(X_test, y_test)
print(f'Model accuracy on test set: {model_score}')

# 绘制AUC和EER图形
cm = confusion_matrix(y_test, pre, labels=[-1, 1])  # 计算混淆矩阵 labels表示标签的类别
sns.set()  # 设置seaborn的默认风格
f, ax = plt.subplots()
sns.heatmap(cm, annot=True, ax=ax)  # 画热力图 ax表示在哪个图上画
ax.set_title('Confusion Matrix')  # 标题
ax.set_xlabel('Predicted Labels')  # x轴
ax.set_ylabel('True Labels')  # y轴

fpr, tpr, threshold = roc_curve(y_test, pre)  # 计算真正率和假正率
roc_auc = auc(fpr, tpr)  # 计算auc的值，auc就是曲线包围的面积，越大越好

plt.figure()
lw = 2  # 线宽
plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  # 假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [1, 0], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")  # 显示图例（右下角）
plt.show()
