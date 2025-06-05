from datasets import Dataset, Features, Array2D, ClassLabel

data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]

'''
데이터셋이 N차원 배열로 구성된 경우 shape이 고정되어 있으면, 동일한 텐서로 간주한다(default).
하지만 느린 shape comparison과 데이터 복사를 방지하기 위해서, 
Array feature 타입을 명시적으로 사용하고 텐서의 shape을 지정해야 한다.
shape은 배열의 차원 크기를 나타냄. 2행 2열 => (2, 2)
'''
features = Features({"data": Array2D(shape=(2, 2), dtype="int32")})

ds = Dataset.from_dict({"data": data}, features=features)
ds = ds.with_format("torch") # To get PyTorch tensors
print("<ds>")
print(ds[:2])
print(ds)

'''
ClassLabel은 데이터를 적절히 텐서로 변환한다.

0D 스칼라
1D 벡터
2D 행렬
3D 텐서(예: 이미지 등)
ND ...
'''
labels = [0, 1, 2]
features2 = Features({"label": ClassLabel(names=["negative", "positive", "neutral"])})
ds2 = Dataset.from_dict({"label": labels}, features=features2)
ds2 = ds2.with_format("torch")
print("\n\n<ds2>")
print(ds2[:3])
print(ds2.features["label"].int2str(0))
print(ds2.features["label"].int2str(1))
print(ds2.features["label"].int2str(2))
print(ds2)