# MiniNN - مكتبة تعليمية للشبكات العصبية

مكتبة خفيفة الوزن وتعليمية للشبكات العصبية مبنية من الصفر باستخدام NumPy. تطبق MiniNN المفاهيم الأساسية للتعلم العميق بكود واضح وسهل القراءة، مثالية للتعلم والتجربة.

## المميزات

### الطبقات
- **Dense**: طبقات متصلة بالكامل مع تهيئة He/Xavier
- **التفعيلات**: ReLU، Sigmoid، Tanh، Linear
- **التنظيم**: Dropout، Batch Normalization
- **دوال الخسارة**: Mean Squared Error، Softmax Cross-Entropy

### المُحسِّنات (Optimizers)
- **SGD**: الانحدار التدريجي العشوائي
- **Momentum**: SGD مع momentum
- **AdaGrad**: learning rate تكيفي
- **Adam**: تقدير اللحظات التكيفي

### مميزات التدريب
- حلقة تدريب مدمجة مع التحقق
- الانحدار التدريجي بالدفعات الصغيرة
- تتبع مقاييس التدريب والتحقق
- ضبط المعاملات الفائقة عبر البحث الشبكي

## التثبيت

```bash
# استنساخ المستودع
git clone https://github.com/Mo-Ha-0/BeyondersTensorflow.git
cd BeyondersTENSORFLOW

# تثبيت المتطلبات
pip install numpy scikit-learn
```

## البدء السريع

```python
import numpy as np
from BeyondersTENSORFLOW import (
    Dense, ReLU, SoftmaxCrossEntropy, 
    NeuralNetwork, Trainer, Adam
)

# تحضير البيانات
X_train, y_train = ...  # بيانات التدريب
X_test, y_test = ...    # بيانات الاختبار

# بناء شبكة عصبية
layers = [
    Dense(input_size, 64),
    ReLU(),
    Dense(64, 32),
    ReLU(),
    Dense(32, num_classes)
]

# إنشاء الشبكة والمُحسِّن
network = NeuralNetwork(layers, SoftmaxCrossEntropy())
optimizer = Adam(learning_rate=0.001)

# تدريب الشبكة
trainer = Trainer(network, optimizer)
trainer.fit(
    X_train, y_train, 
    X_test, y_test,
    epochs=100, 
    batch_size=32,
    verbose=True
)

# التقييم
accuracy = network.accuracy(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")
```

## أمثلة

### تصنيف ثنائي الأبعاد بسيط

```python
# انظر examples/simple_example.py
layers = [
    Dense(2, 10),
    ReLU(),
    Dense(10, 3)
]
network = NeuralNetwork(layers, SoftmaxCrossEntropy())
trainer = Trainer(network, SGD(learning_rate=0.1))
trainer.fit(X_train, y_train, epochs=500, batch_size=16)
```

### تصنيف مجموعة بيانات Iris

```python
# انظر examples/iris_example.py
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# تحميل وتطبيع البيانات
iris = load_iris()
X, y = iris.data, iris.target
scaler = StandardScaler()
X = scaler.fit_transform(X)

# بناء شبكة مع التطبيع الدفعي
layers = [
    Dense(4, 16),
    Sigmoid(),
    BatchNormalization(16),
    Dense(16, 8),
    ReLU(),
    Dense(8, 3)
]

network = NeuralNetwork(layers, SoftmaxCrossEntropy())
optimizer = Adam(learning_rate=0.01)
```

### تصنيف أرقام MNIST

```python
# انظر examples/mnist_example.py
from sklearn.datasets import fetch_openml

# تحميل MNIST
mnist = fetch_openml("mnist_784", version=1)
X, y = mnist.data / 255.0, mnist.target.astype(int)

# بناء شبكة أعمق
layers = [
    Dense(784, 256),
    ReLU(),
    BatchNormalization(256),
    Dense(256, 128),
    ReLU(),
    BatchNormalization(128),
    Dense(128, 10)
]

network = NeuralNetwork(layers, SoftmaxCrossEntropy())
optimizer = Adam(learning_rate=0.001)
```

## مرجع API

### الطبقات

#### طبقة Dense
```python
Dense(input_size: int, output_size: int)
```
طبقة متصلة بالكامل مع تهيئة He.

#### طبقات التفعيل
```python
ReLU()        # وحدة خطية مصححة
Sigmoid()     # تفعيل سيجمويد
Tanh()        # الظل الزائدي
Linear()      # دالة الهوية
```

#### طبقات التنظيم
```python
Dropout(drop_rate: float = 0.5)
BatchNormalization(num_features: int, momentum: float = 0.9)
```

### دوال الخسارة

```python
MeanSquaredError()           # للانحدار
SoftmaxCrossEntropy()       # للتصنيف
```

### المُحسِّنات

```python
SGD(learning_rate: float = 0.01)
Momentum(learning_rate: float = 0.01, momentum: float = 0.9)
AdaGrad(learning_rate: float = 0.01, eps: float = 1e-8)
Adam(learning_rate: float = 0.001, beta1: float = 0.9, 
     beta2: float = 0.999, eps: float = 1e-8)
```

### الشبكة العصبية

```python
network = NeuralNetwork(layers: List[Layer], loss_layer: Layer)

# التمرير الأمامي
predictions = network.predict(x, training=False)

# حساب الخسارة
loss = network.loss(x, y, training=True)

# حساب الدقة
accuracy = network.accuracy(x, y)

# الحصول على التدرجات
params_grads = network.gradient()
```

### المدرب (Trainer)

```python
trainer = Trainer(network: NeuralNetwork, optimizer: Optimizer)

trainer.fit(
    x_train, y_train,
    x_val=None, y_val=None,
    epochs: int = 100,
    batch_size: int = 32,
    verbose: bool = True,
    print_every: int = 10
)
```

### ضبط المعاملات الفائقة

```python
tuner = HyperparameterTuner()

param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'hidden_size': [32, 64, 128]
}

def network_builder(params):
    layers = [
        Dense(input_size, params['hidden_size']),
        ReLU(),
        Dense(params['hidden_size'], output_size)
    ]
    network = NeuralNetwork(layers, SoftmaxCrossEntropy())
    optimizer = Adam(learning_rate=params['learning_rate'])
    return network, optimizer

best_params = tuner.grid_search(
    X_train, y_train, X_val, y_val,
    param_grid, network_builder,
    epochs=50, batch_size=32
)
```

## هيكل المشروع

```
BeyondersTENSORFLOW/
├── __init__.py              # تهيئة الحزمة الرئيسية
├── base.py                  # فئة الطبقة الأساسية
├── network.py               # فئة الشبكة العصبية
├── trainer.py               # تطبيق حلقة التدريب
├── tuning.py                # ضبط المعاملات الفائقة
├── layers/
│   ├── __init__.py
│   ├── dense.py            # طبقة Dense/المتصلة بالكامل
│   ├── activations.py      # دوال التفعيل
│   ├── losses.py           # دوال الخسارة
│   └── regularization.py   # Dropout، BatchNorm
├── optimizers/
│   ├── __init__.py
│   └── optimizers.py       # SGD، Adam، إلخ
└── examples/
    ├── simple_example.py   # تصنيف ثنائي الأبعاد الأساسي
    ├── iris_example.py     # مجموعة بيانات Iris
    └── mnist_example.py    # أرقام MNIST
```

## تفاصيل التنفيذ الرئيسية

### إصلاح المُحسِّن
تستخدم المُحسِّنات (Momentum، AdaGrad، Adam) دالة `id()` في Python لتتبع المعاملات حسب عنوان الذاكرة، مما يضمن التتبع الصحيح عبر طبقات متعددة بنفس أسماء المعاملات.

### التطبيع الدفعي
يحافظ على إحصائيات التشغيل للاستنتاج ويستخدم حساب التدرج المناسب أثناء التدريب.

### الاستقرار العددي
- يستخدم Softmax طرح القيمة القصوى للاستقرار
- يقص Sigmoid قيم الإدخال لمنع الفيض
- قيم إبسيلون الصغيرة تمنع القسمة على صفر

## المتطلبات

- Python 3.7+
- NumPy
- scikit-learn (للأمثلة وتحميل البيانات)

## المساهمة

هذا مشروع تعليمي. لا تتردد في:
- الإبلاغ عن الأخطاء
- اقتراح التحسينات
- إضافة ميزات جديدة
- تحسين التوثيق

## الترخيص

هذا المشروع لأغراض تعليمية.

## المؤلف

محمد - الإصدار 1.0.0

## شكر وتقدير

تم بناؤه كأداة تعليمية لفهم أساسيات الشبكات العصبية والانتشار العكسي من المبادئ الأولى.