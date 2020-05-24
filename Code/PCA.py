from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
pca_model = PCA(n_components=100).fit(x_train)
x_trainPCA = pca_model.transform(x_train)
print(x_trainPCA)
pca = np.array(x_trainPCA)
data = pd.DataFrame(pca)
writer = pd.ExcelWriter('pca.xlsx')  # 写入Excel文件
data.to_excel(writer, 'page_1', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
writer.save()

writer.close()