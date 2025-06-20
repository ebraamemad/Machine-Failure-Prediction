

# مشروع التنبؤ بفشل الآلات باستخدام PyTorch و Optuna و MLflow

##  وصف المشروع

يهدف هذا المشروع إلى بناء نموذج تنبؤي لاكتشاف احتمالية فشل الآلات باستخدام بيانات أجهزة صناعية. تم استخدام تقنيات التعلم العميق مع تحسين الهايبر-باراميترز وتتبع التجارب.

---

## خطوات العمل

### 1. تحميل وتحليل البيانات
- تم تحميل البيانات من ملف CSV.
- تحليل البيانات إحصائيًا وبصريًا باستخدام `pandas`, `matplotlib`, و `seaborn`.
- توحيد البيانات باستخدام `StandardScaler`.
- تصفية بعض البيانات حسب شروط محددة لتحسين جودة التدريب.

### 2. بناء النموذج
- تم استخدام مكتبة `PyTorch` لبناء نموذج بسيط للتصنيف الثنائي.
- النموذج يحتوي على:
  - طبقة إدخال
  - طبقة مخفية مع `ReLU`
  - طبقة إخراج مع `Sigmoid`

### 3. تحسين Hyperparameters
- تم استخدام `Optuna` لاختيار أفضل القيم لـ:
  - معدل التعلم (`learning_rate`)
  - عدد الوحدات في الطبقة المخفية (`hidden_dim`)
  - نسبة الإسقاط (`dropout_rate`)

### 4. تتبع التجارب وحفظ النموذج
- تم استخدام `MLflow` لتسجيل:
  - القيم التجريبية
  - المقاييس مثل Accuracy و Loss
  - النموذج النهائي

---

## الأدوات والمكتبات المستخدمة

- Python 3
- PyTorch
- Optuna
- MLflow
- Matplotlib & Seaborn
- Pandas & Scikit-learn

---
