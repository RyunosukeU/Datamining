import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# ハイパラメータの設定
Cs = [0.5, 1.0, 1.5]

# 10分割検定
k_folds = 10

# データセットを読み込み
feature_names = ['Area_Integer', 'MajorAxisLength_Real', 'MinorAxisLength_Real', 'Eccentricity_Real', 'ConvexArea_Integer', 'Extent_Real', 'Perimeter_Real', 'Class']
df = pd.read_csv('Raisin_Dataset/Raisin_Dataset.data',names=feature_names)

# 学習準備
# 学習データと教師データの準備
X = df[['Area_Integer', 'MajorAxisLength_Real', 'MinorAxisLength_Real', 'Eccentricity_Real', 'ConvexArea_Integer', 'Extent_Real', 'Perimeter_Real']]
y = df['Class']

# 学習
for c in Cs:
    model = LinearSVC(C=c)
    scores = cross_val_score(model, X, y, cv=KFold(n_splits=k_folds, shuffle=True))
    average = scores.mean()
    print(f'C = {c}: scores={scores}, average={average:.3f}')